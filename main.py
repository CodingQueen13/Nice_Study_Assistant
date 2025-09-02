# main.py
# =============================================================================
"""
Main entry point for the Study Assistant application.

Before running, make sure to:
1. Install Ollama: https://ollama.ai/
2. Run: ollama serve
3. Run: ollama pull llama2
4. Install requirements: pip install -r requirements.txt
"""

import sys
from pathlib import Path
import streamlit as st

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from ui.streamlit_app import StudyAssistantUI
from utils.helpers import SystemUtils

def check_prerequisites():
    st.sidebar.header("🔍 System Check")

    from config.settings import Config

    if Config.LLM_PROVIDER == "ollama":
        if SystemUtils.check_ollama_running():
            st.sidebar.success("✅ Ollama is running")
        else:
            st.sidebar.error("❌ Ollama not detected")
            st.sidebar.markdown("""
            **To start Ollama:**
            1. Install from https://ollama.ai/
            2. Run `ollama serve` in terminal
            3. Run `ollama pull llama2`
            """)
    elif Config.LLM_PROVIDER == "gemini":
        if Config.GEMINI_API_KEY:
            st.sidebar.success("✅ Gemini API key detected")
        else:
            st.sidebar.error("❌ Gemini API key not set")
    
    # System resources
    with st.sidebar.expander("System Resources"):
        SystemUtils.display_system_info()

def main():
    """Main application function"""
    
    # Check prerequisites
    check_prerequisites()
    
    # Create and run the UI
    app = StudyAssistantUI()
    app.run()

if __name__ == "__main__":
    main()