import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings
import logging
import os
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Downloading model and dataset from Hugging Face Hub...")

HF_TOKEN = settings.hf_token
print(f"Loaded HF_TOKEN: {HF_TOKEN!r}")

FAISS_INDEX_PATH = snapshot_download(
    repo_id="negi2725/dataRag",
    repo_type="dataset",
    token=HF_TOKEN
)

MODEL_PATH = snapshot_download(
    repo_id="negi2725/legalBert",
    token=HF_TOKEN
)

logger.info(f"FAISS index files downloaded to: {FAISS_INDEX_PATH}")
logger.info(f"Model files downloaded to: {MODEL_PATH}")

os.environ["FAISS_INDEX_PATH"] = FAISS_INDEX_PATH
os.environ["MODEL_PATH"] = MODEL_PATH

app = FastAPI(
    title="Legal RAG Analysis API",
    description="FastAPI backend for legal case analysis using RAG system with LegalBERT predictions and Gemini AI evaluation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Legal RAG Analysis API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )
