import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    VECTOR_DB_PATH: str = str(DATA_DIR / "chroma_db")
    
    # Model settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "llama2:7b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Chunking settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval settings
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.5
    
    # UI settings
    PAGE_TITLE: str = "Local RAG System"
    PAGE_ICON: str = "🤖"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Create directories
settings.DATA_DIR.mkdir(exist_ok=True)
settings.DOCUMENTS_DIR.mkdir(exist_ok=True)