# models/embeddings_manager.py
# =============================================================================
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import Config
import torch

class EmbeddingsManager:
    """Manages embeddings with CPU optimization"""
    
    def __init__(self):
        self.embeddings = None
        self.config = Config()
        
    def initialize_embeddings(self):
        """Initialize HuggingFace embeddings optimized for CPU"""
        try:
            # Force CPU usage and optimize for speed
            print("Starting embeddings initialization...")  # Debug
            model_kwargs = {
                'device': self.config.EMBEDDINGS_DEVICE,
                'trust_remote_code': True
            }
            
            encode_kwargs = {
                'normalize_embeddings': True,
                'batch_size': 16  # Smaller batch size for CPU
            }
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDINGS_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            print("Embeddings created successfully")  # Debug
            return True
            
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
            return False
    
    def get_embeddings(self):
        """Get the embeddings instance"""
        return self.embeddings