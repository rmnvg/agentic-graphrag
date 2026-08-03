from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():
    return {"message": "Agentic GraphRAG API is running 🚀"}


@router.get("/health")
def health():
    return {"status": "healthy"}