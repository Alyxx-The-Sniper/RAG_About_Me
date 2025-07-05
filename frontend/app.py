import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")

st.title("Hi I'm Alyx the vampire bot Rarw!! 🦇    ")

# User input box
user_query = st.text_input("Ask me anything about Alexis:", "")

if user_query:
    with st.spinner("Thinking..."):
        # Send query to FastAPI backend
        response = requests.post(
            BACKEND_URL,
            json={"query": user_query}
        )
        if response.ok:
            answer = response.json()["answer"]
            st.markdown(
                f"**🧛‍♀️ Alyx the bot vampire:**  \n"
                f"{answer}"
            )
        else:
            st.error("Sorry, something went wrong. Please try again.")

