import os
from pathlib import Path

class Config:
    """Configuration settings for the Study Assistant"""
    
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    
    # Embeddings settings
    EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDINGS_DEVICE = "cpu"  # Force CPU usage
    
    # Text processing settings
    CHUNK_SIZE = 800  # Smaller chunks for CPU processing
    CHUNK_OVERLAP = 150
    MAX_CHUNKS_FOR_CONTEXT = 3  # Reduce context to speed up CPU processing
    
    # Vector store settings
    VECTOR_SEARCH_K = 3  # Reduce search results for faster processing
    
    # Memory settings
    CONVERSATION_MEMORY_K = 8  # Reduce memory window for CPU
    
    # File settings
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc']
    MAX_FILE_SIZE_MB = 50
    TEMP_DIR = Path.cwd() / "temp"
    
    # UI settings
    PAGE_TITLE = "Study Assistant with RAG (CPU-Optimized)"
    PAGE_ICON = "📚"
    
    # Ollama timeout settings (important for CPU)
    OLLAMA_REQUEST_TIMEOUT = 300  # 5 minutes for CPU processing
    OLLAMA_NUM_PREDICT = 512  # Limit response length for speed
