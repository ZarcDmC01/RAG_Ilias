"""
Étape 2 : Créer et indexer les documents avec Mistral
Fichier : setup_vectorstore.py
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

documents = [
    "Document 1 placeholder",
    "Document 2 placeholder",
    "Document 3 placeholder",
]

def setup_vectorstore():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    
    texts = []
    for doc in documents:
        texts.extend(splitter.split_text(doc.strip()))
    
    print(f"✓ {len(texts)} chunks créés")
    
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY")
    )
    
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local("./vectorstore_index")
    print("✓ Base vectorielle sauvegardée dans ./vectorstore_index")
    
    return vectorstore

if __name__ == "__main__":
    setup_vectorstore()