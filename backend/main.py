from fastapi import FastAPI, Request
from retrieval import retrieve, call_ollama


app = FastAPI()


def messages_to_prompt(messages):
    prompt = ""
    for m in messages:
        if m["role"] == "system":
            prompt += f"<|system|>\n{m['content']}\n"
        elif m["role"] == "user":
            prompt += f"<|user|>\n{m['content']}\n"
        elif m["role"] == "assistant":
            prompt += f"<|assistant|>\n{m['content']}\n"
    return prompt



@app.post("/chat")
async def chat(request: Request):

    data = await request.json()
    user_query = data["query"]
    ret = retrieve(user_query)
        
    # handle fallback here
    if not ret:
            return {
            "answer": (
                """
    Sorry, I can’t answer that—it’s beyond my current knowledge base right now.
    I’m an AI bot trained to answer questions on a variety of topics, including (but not limited to):\n
    - personal data\n
    - artificial intelligence\n
    - machine learning\n
    - food, travel, movies, and more.

    Please feel free to connect with me on LinkedIn for more info:
    https://www.linkedin.com/in/alexis-mandario-b546881a8/

    Note: To ensure my all answer stays on ground truth, I’ve set the similarity search threshold to a certain value.
    """.lstrip()
            )
        }

    context_q, context_a = ret

    # Build messages
    messages = [
        {
        "role": "system",
        "content": (
            "You are Ayx, an AI-agent and ML/AI enthusiast. "
            "Answer **only** using the CONTEXT provided. "
            "Be friendly, approachable, and authentic; you may insert emojis when it adds to the tone. "
            "Keep your answers engaging and grounded in best practices or real-world knowledge. "
            "Always share my LinkedIn profile at the end: "
            "https://www.linkedin.com/in/alexis-mandario-b546881a8"
                
        )
        },
        {
            "role": "system",
            "content": f"CONTEXT:\nQ: {context_q}\nA: {context_a}"
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    prompt = messages_to_prompt(messages) # call messages_to_prompt so that the mesasges will be the same structure our model required (gemma 3n)
    answer = await call_ollama(prompt)
    return {"answer": answer}


        