"""
Main entry point for the Study Assistant application.

Run with:
    streamlit run main.py
"""

import sys
from pathlib import Path
import streamlit as st

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from ui.streamlit_app import StudyAssistantUI


def main():
    """Main application function"""

    # Create and run the UI
    app = StudyAssistantUI()
    app.run()


if __name__ == "__main__":
    main()
