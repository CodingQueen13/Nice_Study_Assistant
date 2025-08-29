# models/llm_manager.py
# =============================================================================
from typing import Optional
import streamlit as st
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config.settings import Config

class LLMManager:
    """Manages LLM interactions with Ollama"""
    
    def __init__(self):
        self.llm: Optional[Ollama] = None
        self.config = Config()
        
    def initialize_llm(self) -> bool:
        """Initialize Ollama LLM with CPU-optimized settings"""
        try:
            self.llm = Ollama(
                model=self.config.OLLAMA_MODEL,
                base_url=self.config.OLLAMA_BASE_URL,
                num_predict=self.config.OLLAMA_NUM_PREDICT,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                callbacks=[StreamingStdOutCallbackHandler()],
                verbose=False
            )
            
            # Test the connection
            test_response = self.llm.invoke("Hello")
            if test_response:
                return True
            return False
            
        except Exception as e:
            st.error(f"Failed to initialize Ollama: {str(e)}")
            st.error("Make sure Ollama is running with: ollama serve")
            st.error(f"And that you have pulled the model: ollama pull {self.config.OLLAMA_MODEL}")
            return False
    
    def get_llm(self) -> Optional[Ollama]:
        """Get the initialized LLM instance"""
        return self.llm
    
    def is_ready(self) -> bool:
        """Check if LLM is ready for use"""
        return self.llm is not None