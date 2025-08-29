# document_processing/document_loader.py
# =============================================================================
from typing import List, Optional
from pathlib import Path
import tempfile
import os
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader
)
from langchain.schema import Document
from config.settings import Config

class DocumentLoader:
    """Handles loading documents from various file formats"""
    
    def __init__(self):
        self.config = Config()
        self.supported_extensions = self.config.SUPPORTED_EXTENSIONS
    
    def validate_file(self, file_path: str) -> bool:
        """Validate file format and size"""
        path = Path(file_path)
        
        # Check extension
        if path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.MAX_FILE_SIZE_MB:
            st.warning(f"File {path.name} is too large: {file_size_mb:.1f}MB")
            return False
        
        return True
    
    def load_single_document(self, file_path: str) -> Optional[List[Document]]:
        """Load a single document based on its extension"""
        if not self.validate_file(file_path):
            return None
        
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            else:
                st.warning(f"Unsupported file format: {file_extension}")
                return None
            
            documents = loader.load()
            return documents
            
        except Exception as e:
            st.error(f"Error loading {Path(file_path).name}: {str(e)}")
            return None
    
    def load_documents_from_uploads(self, uploaded_files) -> List[Document]:
        """Load documents from Streamlit file uploads"""
        all_documents = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load document
                documents = self.load_single_document(file_path)
                if documents:
                    all_documents.extend(documents)
                    st.success(f"✓ Loaded: {uploaded_file.name}")
        
        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return all_documents
