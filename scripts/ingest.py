import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Small, fast, local

docs = []
metadatas = []

with open("data/me.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        q = obj["messages"][0]["content"]
        a = obj["messages"][1]["content"]
        docs.append(q)
        metadatas.append({"answer": a})

# if rerun will delete the existing vectors and replace with new vectors base on the new re run dataset
db = Chroma.from_texts(
    docs, 
    embedder, 
    persist_directory="./chroma_db", 
    metadatas=metadatas
)


# This code is for adding new vectors to db instead of overwriting it
# db = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embedder
# )

# db.add_texts(
#     docs, 
#     metadatas=metadatas
# )

print("Ingest complete. ChromaDB created.")


