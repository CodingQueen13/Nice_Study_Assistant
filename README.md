# Nice_Study_Assistant
RAG-powered intelligent tutoring system that allows uploading of textbooks, and help the student study by providing explanations, generating summaries and questions on demand, etc.

 Uses LangChain, Faiss, Gemini API (requiring an internet connection) or Ollama with Llama2 (offline) for the LLM, and HuggingFace sentence-transformers for embeddings (always offline). 
 
 Allows choosing the LLM provider in the UI before starting a chat (changing the LLM provider mid-chat doesn't work.) The default LLM_PROVIDER is set in "/config/settings.py".
 Gemini API key should be set in "/config/."env".
