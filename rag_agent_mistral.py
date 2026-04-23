"""
Étape 3 : Agent RAG avec Mistral et LangGraph
Fichier : rag_agent_mistral.py
"""

from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain.agents import create_agent

import os
from dotenv import load_dotenv

load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY")
)

vectorstore = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print(vectorstore)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

class RetrieverInput(BaseModel):
    query: str = Field(description="La question ou requête pour rechercher dans la base de connaissances")

@tool("retriever", args_schema=RetrieverInput)
def retrieval_tool(query: str) -> str:
    """Récupère les documents pertinents de la base de connaissances"""
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

def create_rag_agent():
    model = ChatMistralAI(
        model="mistral-large-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.7,
    
    )
    
    tools = [retrieval_tool]
    
    agent_executor = create_agent(model, tools)
    
    return agent_executor

# if __name__ == "__main__":
#     agent = create_rag_agent()
#     result = agent.invoke({"messages": [{"role": "user", "content": "quel sont les entreprises sur besancon?"}]})
#     print(result)