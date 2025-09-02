# Nice_Study_Assistant
RAG-powered intelligent tutoring system that allows uploading of textbooks, and help the student study by provides explanations, generating summaries and questions on demand, etc.

 Uses LangChain, Faiss, Gemini API (requiring an internet connection) or Ollama with Llama2 (offline) for the LLM, and HuggingFace sentence-transformers for embeddings (always offline). To choose between the online and offline LLM should change the default LLM_PROVIDER in "/config/settings.py".
 Gemini API key should be set in "/config/."env".
