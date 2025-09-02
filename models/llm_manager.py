# models/llm_manager.py
# =============================================================================
from typing import Optional
import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config.settings import Config

# Import Ollama + Gemini wrappers
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI




class LLMManager:
    """Manages LLM interactions with Ollama or Gemini"""

    def __init__(self):
        self.llm: Optional[object] = None
        self.config = Config()

    def initialize_llm(self) -> bool:
        """Initialize LLM based on provider"""
        try:
            if self.config.LLM_PROVIDER.lower() == "ollama":
                self.llm = Ollama(
                    model=self.config.OLLAMA_MODEL,
                    base_url=self.config.OLLAMA_BASE_URL,
                    num_predict=self.config.OLLAMA_NUM_PREDICT,
                    temperature=0.7,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    callbacks=[StreamingStdOutCallbackHandler()],
                    verbose=False,
                )
                test_response = self.llm.invoke("Hello")
                return bool(test_response)

            elif self.config.LLM_PROVIDER.lower() == "gemini":
                if not self.config.GEMINI_API_KEY:
                    st.error("Gemini API key not set! Please set GEMINI_API_KEY in env.")
                    return False

                self.llm = ChatGoogleGenerativeAI(
                    model=self.config.GEMINI_MODEL,
                    temperature=0.7,
                    google_api_key=self.config.GEMINI_API_KEY,
                )
                # Quick sanity check
                test_response = self.llm.invoke("Hello")
                return bool(test_response)

            else:
                st.error(f"Unknown LLM provider: {self.config.LLM_PROVIDER}")
                return False

        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            return False

    def get_llm(self) -> Optional[object]:
        """Get the initialized LLM instance"""
        return self.llm

    def is_ready(self) -> bool:
        """Check if LLM is ready for use"""
        return self.llm is not None