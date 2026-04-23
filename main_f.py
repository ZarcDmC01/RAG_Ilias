import json
import os
import httpx
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Langchain core pour la structure
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from rag_agent_mistral import create_rag_agent

load_dotenv()
agent = create_rag_agent()
# ── App Configuration ─────────────────────────────────────────────
app = FastAPI(title="Weather Agent API (Mistral AI)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas & Routes ───────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

@app.post("/chat")
def chat(req: ChatRequest):
    # Conversion historique simple
    hist = []
    for m in req.history:
        if m.get('role') == 'user':
            hist.append(HumanMessage(content=m['content']))
        else:
            hist.append(AIMessage(content=m['content']))
            
    try:
        result = agent.invoke({"messages": [HumanMessage(content=req.message)]})
        return {"response": result["messages"][-1].content}
    except Exception as e:
        print(f"[SERVER ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
