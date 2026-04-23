"""
Étape 2 : Créer et indexer les documents avec Mistral
Fichier : setup_vectorstore.py
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

documents = [
    """Les pommes sont des fruits riches en vitamines C et en fibres.
    Elles aident à réduire le cholestérol et à maintenir une bonne digestion.
    Les pommes peuvent être mangées crues ou cuites.""",
    
    """Les bananes contiennent du potassium et du magnésium.
    Elles sont excellentes pour l'énergie et la santé musculaire.
    Les bananes sont faciles à digérer et conviennent aux enfants.""",
    
    """Les oranges sont riches en vitamine C et en antioxydants.
    Elles renforcent le système immunitaire et la peau.
    Le jus d'orange frais est meilleur que le jus transformé.""",
    
    """Les raisins contiennent des polyphénols et des antioxydants puissants.
    Ils aident à prévenir les maladies cardiaques et le cancer.
    Les raisins rouges sont plus bénéfiques que les raisins blancs.""",
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
    print("✓ Base vectorielle sauvegardée")
    
    return vectorstore

if __name__ == "__main__":
    setup_vectorstore()
