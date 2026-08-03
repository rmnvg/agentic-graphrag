from fastapi import FastAPI

app = FastAPI(title="Agentic GraphRAG API")


@app.get("/")
def home():
    return {"message": "Welcome to Agentic GraphRAG"}