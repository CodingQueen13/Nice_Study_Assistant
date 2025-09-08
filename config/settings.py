import os
from pathlib import Path
from dotenv import load_dotenv

 
  
class Config:
    """Configuration settings for the Study Assistant"""
    
    
    # Try loading .env file from config/ or project root
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    
    # LLM settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "gemini"

    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    OLLAMA_NUM_PREDICT = 512  # short responses for CPU
    OLLAMA_REQUEST_TIMEOUT = 300  # 5 minutes timeout for CPU processing


    # Gemini settings

    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    
    # Embeddings settings (uses the same offline cpu-friendly embeddings for both Ollama and Gemini API.)
    EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDINGS_DEVICE = "cpu"  # Force CPU usage
    
    
    # Default values (will be overridden below based on provider)
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    MAX_CHUNKS_FOR_CONTEXT = 3
    VECTOR_SEARCH_K = 3
    CONVERSATION_MEMORY_K = 8  # will expand for Gemini

    # CPU / API optimized configs
    @classmethod
    def apply_dynamic_settings(cls):
        """Adjust settings dynamically based on LLM provider"""
        if cls.LLM_PROVIDER == "ollama":
            # CPU-friendly
            cls.CHUNK_SIZE = 800
            cls.CHUNK_OVERLAP = 150
            cls.MAX_CHUNKS_FOR_CONTEXT = 3
            cls.VECTOR_SEARCH_K = 3
            cls.CONVERSATION_MEMORY_K = 8
        elif cls.LLM_PROVIDER == "gemini":
            # API-friendly (bigger chunks, more memory)
            cls.CHUNK_SIZE = 1200
            cls.CHUNK_OVERLAP = 200
            cls.MAX_CHUNKS_FOR_CONTEXT = 6
            cls.VECTOR_SEARCH_K = 6
            cls.CONVERSATION_MEMORY_K = 20  # keep much more history
        else:
            raise ValueError(f"Unknown LLM provider: {cls.LLM_PROVIDER}")
 
    
    # File settings
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc']
    MAX_FILE_SIZE_MB = 50
    TEMP_DIR = Path.cwd() / "temp"
    
    # UI settings
    PAGE_TITLE = "Study Assistant with RAG (CPU-Optimized)"
    PAGE_ICON = "📚"
    
 # Apply settings on import so any module that imports Config (LLMManager, EmbeddingsManager, TutorEngine,...) gets the right values automatically.
Config.apply_dynamic_settings()
   