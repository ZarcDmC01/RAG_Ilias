"""
FastAPI Weather Agent API
LangChain + Mistral + Weather Tool
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import os

# --- LangChain imports ---
from langchain_mistralai import ChatMistralAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Weather Agent API",
    description="LangChain Agent with Mistral and weather tool",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Tool ──────────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    Args:
        city: Name of the city to get weather for
    """
    # Mock data — remplace par OpenWeatherMap, wttr.in, etc.
    weather_db = {
        "paris": "☀️ Paris: 22°C, ensoleillé, vent 15 km/h",
        "london": "🌧️ London: 14°C, pluie légère, humidité 80%",
        "new york": "⛅ New York: 18°C, partiellement nuageux",
        "tokyo": "🌤️ Tokyo: 26°C, beau temps, légère brise",
        "dubai": "🔥 Dubai: 38°C, ensoleillé, très chaud",
    }
    city_lower = city.lower()
    if city_lower in weather_db:
        return weather_db[city_lower]
    return f"🌡️ {city}: 20°C, conditions normales (données simulées)"

# ── LLM + Agent ───────────────────────────────────────────────────
def build_agent() -> AgentExecutor:
    llm = ChatMistralAI(
        model="mistral-large-latest",       # ou "mistral-small-latest"
        mistral_api_key=os.environ["MISTRAL_API_KEY"],
        temperature=0.3,
    )

    tools = [get_weather]

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Tu es un assistant météo intelligent et sympathique. "
            "Utilise l'outil get_weather pour répondre aux questions météo. "
            "Réponds toujours en français sauf si l'utilisateur parle une autre langue."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor = build_agent()

# ── Schemas ───────────────────────────────────────────────────────
class Message(BaseModel):
    role: str          # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    response: str
    tool_used: Optional[str] = None

# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "agent": "Weather Agent (Mistral + LangChain)"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Convertir l'historique en messages LangChain
    lc_history = []
    for msg in req.history:
        if msg.role == "user":
            lc_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_history.append(AIMessage(content=msg.content))

    try:
        result = agent_executor.invoke({
            "input": req.message,
            "chat_history": lc_history,
        })

        # Détecte si un tool a été utilisé
        tool_used = None
        if hasattr(result, "intermediate_steps") and result.get("intermediate_steps"):
            tool_used = result["intermediate_steps"][0][0].tool

        return ChatResponse(
            response=result["output"],
            tool_used=tool_used,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming version — retourne les tokens au fur et à mesure"""
    lc_history = []
    for msg in req.history:
        if msg.role == "user":
            lc_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_history.append(AIMessage(content=msg.content))

    async def event_generator():
        async for chunk in agent_executor.astream({
            "input": req.message,
            "chat_history": lc_history,
        }):
            if "output" in chunk:
                yield f"data: {json.dumps({'token': chunk['output']})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")