from fastapi import FastAPI

from app.api.document_routes import router as document_router
from app.api.routes import router
from app.api.search_routes import router as search_router

app = FastAPI(
    title="Agentic GraphRAG",
    version="0.1.0"
)

app.include_router(router)
app.include_router(document_router)
app.include_router(search_router)
