# document_processing/text_processor.py
# =============================================================================
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config.settings import Config

class TextProcessor:
    """Handles text processing and chunking"""
    
    def __init__(self):
        self.config = Config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and chunk documents for optimal CPU performance"""
        if not documents:
            return []
        
        # Split documents into smaller chunks
        processed_docs = self.text_splitter.split_documents(documents)
        
        # Add metadata for source tracking
        for i, doc in enumerate(processed_docs):
            doc.metadata['chunk_id'] = i
            doc.metadata['chunk_size'] = len(doc.page_content)
        
        return processed_docs
    
    def get_chunk_stats(self, documents: List[Document]) -> dict:
        """Get statistics about the processed chunks"""
        if not documents:
            return {}
        
        chunk_sizes = [len(doc.page_content) for doc in documents]
        
        return {
            'total_chunks': len(documents),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }
