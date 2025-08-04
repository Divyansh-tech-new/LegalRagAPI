import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    hf_token: Optional[str] = None 

    api_title: str = "Legal RAG Analysis API"
    api_version: str = "1.0.0"
    debug: bool = False
    
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = "gemini-2.5-flash"

    legal_bert_model_path: str = os.getenv("LEGAL_BERT_MODEL_PATH", "./models/legalbert_model")

    faiss_indexes_base_path: str = os.getenv("FAISS_INDEXES_PATH", "./faiss_indexes")

    constitution_index_path: str = f"{faiss_indexes_base_path}/constitution_bgeLarge.index"
    constitution_chunks_path: str = f"{faiss_indexes_base_path}/constitution_chunks.json"

    ipc_index_path: str = f"{faiss_indexes_base_path}/ipc_bgeLarge.index"
    ipc_chunks_path: str = f"{faiss_indexes_base_path}/ipc_chunks.json"

    ipc_case_index_path: str = f"{faiss_indexes_base_path}/ipc_case_flat.index"
    ipc_case_chunks_path: str = f"{faiss_indexes_base_path}/ipc_case_chunks.json"

    statute_index_path: str = f"{faiss_indexes_base_path}/statute_index.faiss"
    statute_chunks_path: str = f"{faiss_indexes_base_path}/statute_chunks.pkl"

    qa_index_path: str = f"{faiss_indexes_base_path}/qa_faiss_index.idx"
    qa_chunks_path: str = f"{faiss_indexes_base_path}/qa_text_chunks.json"

    case_law_index_path: str = f"{faiss_indexes_base_path}/case_faiss.index"
    case_law_chunks_path: str = f"{faiss_indexes_base_path}/case_chunks.pkl"

    sentence_transformer_model: str = "BAAI/bge-large-en-v1.5"

    top_k_results: int = 5
    max_unique_chunks: int = 10
    confidence_threshold: float = 0.6

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
