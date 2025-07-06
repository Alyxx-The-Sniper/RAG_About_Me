import os
import json
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "rag-about-me"
dimension = 384  # adjust if your embedding model outputs differently
metric = "euclidean"
cloud = "aws"
region = os.environ.get("PINECONE_ENV", "us-east-1")

# Create index if it does not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region)
    )

index = pc.Index(index_name)

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_and_upsert_jsonl(filepath):
    id_counter = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:100]}...")
                continue

            messages = obj.get("messages", [])
            if not messages:
                continue

            # Build full conversation text
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

            # Optional: split into chunks if too long
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]

            for chunk in chunks:
                vec = model.encode(chunk).tolist()
                index.upsert([
                    (f"id-{id_counter}", vec, {"text": chunk})
                ])
                print(f"Upserted id-{id_counter}")
                id_counter += 1

    print(f"Finished upserting {id_counter} vectors.")

if __name__ == "__main__":
    embed_and_upsert_jsonl("../data/me.jsonl")
