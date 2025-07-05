from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import httpx
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

# to easily change model via docker-compose variable
MODEL = os.getenv("OLLAMA_MODEL", "gemma3n:e4b-it-q4_K_M")
# MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b-it-qat")

# distance metric so 0 means exacltly the same
SIM_THRESHOLD = 0.8 


embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedder
)


def retrieve(query):
    docs = db.similarity_search_with_score(query, k=1)
    # 1) no hits?
    if len(docs) == 0:
        return None
    # 2) unpack the top hit
    doc, distance = docs[0]
    # 3) distance too large?
    if distance > SIM_THRESHOLD: # if distance is higher return none (.7)
        print(f"[retrieve] Fallback: distance {distance:.4f} > threshold {SIM_THRESHOLD}")
        return None
    # 4) otherwise, good match
    return doc.page_content, doc.metadata["answer"]



async def call_ollama(prompt):
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(OLLAMA_URL, json=payload, timeout=180) # timeout ollama is slow
        result = resp.json()
        return result["response"].strip()
