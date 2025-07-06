import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import httpx
from dotenv import load_dotenv
load_dotenv()

# local testing
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# in rail way 
# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "gemma3n:e4b-it-q4_K_M")
# SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD"))
SIM_THRESHOLD = 1

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("rag-about-me")
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, top_k=1):
    vec = model.encode(query).tolist()
    result = index.query(vector=vec, top_k=top_k, include_metadata=True)

    if not result or not result.matches:
        print(f"[retrieve] Fallback: no matches")
        return None

    match = result.matches[0]
    similarity = match.score
    SIM_THRESHOLD = 0.8  # adjust as needed
    if similarity < SIM_THRESHOLD:
        print(f"[retrieve] Fallback: similarity {similarity:.4f} < threshold {SIM_THRESHOLD}")
        return None
    print(f"[retrieve] pass: similarity {similarity:.4f}")

    text = match.metadata.get("text", "")
    answer = match.metadata.get("answer", "")
    return text, answer


async def call_ollama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(OLLAMA_URL, json=payload, timeout=180)
        result = resp.json()
        return result["response"].strip()
