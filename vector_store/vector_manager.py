# vector_store/vector_manager.py
# =============================================================================
from typing import List, Optional, Tuple
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config.settings import Config

class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self, embeddings_manager):
        self.embeddings_manager = embeddings_manager
        self.vector_store: Optional[FAISS] = None
        self.config = Config()
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """Create FAISS vector store from documents"""
        if not documents:
            st.error("No documents to process")
            return False
        
        if not self.embeddings_manager.get_embeddings():
            st.error("Embeddings not initialized")
            return False
        
        try:
            with st.spinner("Creating vector embeddings... This may take a while on CPU."):
                # Process in smaller batches for CPU efficiency
                batch_size = 10
                all_vectors = []
                
                progress_bar = st.progress(0)
                
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    
                    if i == 0:
                        # Create initial vector store with first batch
                        self.vector_store = FAISS.from_documents(
                            batch, 
                            self.embeddings_manager.get_embeddings()
                        )
                    else:
                        # Add remaining batches to existing vector store
                        batch_store = FAISS.from_documents(
                            batch,
                            self.embeddings_manager.get_embeddings()
                        )
                        self.vector_store.merge_from(batch_store)
                    
                    # Update progress
                    progress = min((i + batch_size) / len(documents), 1.0)
                    progress_bar.progress(progress)
                
                progress_bar.empty()
                return True
                
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform similarity search"""
        if not self.vector_store:
            return []
        
        k = k or self.config.VECTOR_SEARCH_K
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            st.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def get_retriever(self):
        """Get retriever for RAG chain"""
        if not self.vector_store:
            return None
        
        return self.vector_store.as_retriever(
            search_kwargs={"k": self.config.VECTOR_SEARCH_K}
        )
    
    def is_ready(self) -> bool:
        """Check if vector store is ready"""
        return self.vector_store is not None
