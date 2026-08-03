from fastapi import FastAPI
from app.api.document_routes import router as document_router
from app.api.routes import router

app = FastAPI(
    title="Agentic GraphRAG",
    version="0.1.0"
)

app.include_router(router)
app.include_router(document_router)
