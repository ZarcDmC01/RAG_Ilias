import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

# ---------------- CONFIG ----------------
load_dotenv()

URLS = [
    "https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=2","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=3","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=4","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=5","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=6","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=7","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=8","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=9","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=10","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=11","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=12","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=13","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=14","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=15","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=16","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=17","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=18","https://www.pple.fr/recherche//Region-Bourgogne-Franche-Comte?page=19"
]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
INDEX_PATH = "faiss_index"

# ---------------- LOAD + CLEAN ----------------
def load_and_clean(urls):
    docs = []

    for url in urls:
        try:
            headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                    }

            res = requests.get(url, headers=headers, timeout=100)
            res.raise_for_status()

            soup = BeautifulSoup(res.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            text = soup.get_text(" ", strip=True)

            docs.append(Document(
                page_content=text,
                metadata={"source": url}
            ))

            print(f"[OK] {url}")

        except Exception as e:
            print(f"[ERROR] {url} -> {e}")

    return docs

# ---------------- PIPELINE ----------------
def run():
    print("🚀 Ingestion (Mistral embeddings)...")

    docs = load_and_clean(URLS)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    print(f"[CHUNKS] {len(chunks)}")

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_PATH)

    print(f"✅ Index saved -> {INDEX_PATH}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    run()