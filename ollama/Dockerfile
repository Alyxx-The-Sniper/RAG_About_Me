FROM ollama/ollama:latest

# pull your model once at build time
RUN ollama pull gemma3n:e4b-it-q4_K_M

# make sure Railway knows which port to expose
EXPOSE 11434

CMD ["ollama", "serve"]