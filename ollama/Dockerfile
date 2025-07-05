FROM ollama/ollama:latest

# pull your model once at build time
RUN ollama pull llama3

# make sure Railway knows which port to expose
EXPOSE 11434

CMD ["ollama", "serve"]