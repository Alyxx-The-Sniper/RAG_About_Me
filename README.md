# Alyx — RAG + Streamlit + FastAPI + Ollama

This project is an AI Chat Bot called **Alyx**, representing Alexis Mandario.  
It uses a RAG (Retrieval Augmented Generation) pipeline powered by Ollama + FastAPI backend and a Streamlit frontend.

---

✅ backend (FastAPI + retrieval)
✅ frontend (Streamlit)
✅ ollama (LLM server)
…all in one repo, running via docker-compose.

###  Usig docker-compose

```bash
docker-compose up --build
• Backend: http://localhost:8000
• Frontend: http://localhost:8501
• Ollama: http://localhost:11434

---

###  Local (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt


# Ingest vector database
# run this if using different dataset (jsonl) in rag_aboute_me/data
python scripts/ingest.py

# start backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# in another terminal
cd frontend
streamlit run app.py

Note: must have ollama installed and a gemma model

---

