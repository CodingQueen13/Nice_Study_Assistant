# utils/helpers.py
# =============================================================================
import streamlit as st
import psutil
import os
from pathlib import Path

class SystemUtils:
    """Utility functions for system monitoring and helpers"""
    
    @staticmethod
    def check_system_resources():
        """Check available system resources"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu_usage': cpu_percent,
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_usage_percent': memory.percent
        }
    
    @staticmethod
    def display_system_info():
        """Display system information in Streamlit"""
        resources = SystemUtils.check_system_resources()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Usage", f"{resources['cpu_usage']:.1f}%")
        
        with col2:
            st.metric("Memory Available", f"{resources['memory_available_gb']:.1f} GB")
        
        with col3:
            st.metric("Memory Usage", f"{resources['memory_usage_percent']:.1f}%")
    
    @staticmethod
    def check_ollama_running():
        """Check if Ollama is running"""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def create_temp_directory():
        """Create temporary directory for file processing"""
        temp_dir = Path.cwd() / "temp"
        temp_dir.mkdir(exist_ok=True)
        return temp_dir
    
    @staticmethod
    def cleanup_temp_files(temp_dir: Path):
        """Clean up temporary files"""
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

class DocumentUtils:
    """Utilities for document processing"""
    
    @staticmethod
    def estimate_processing_time(num_documents: int, total_size_mb: float) -> int:
        """Estimate processing time for CPU systems"""
        # Rough estimates for CPU processing
        base_time = 30  # Base overhead in seconds
        per_doc_time = 15  # Seconds per document
        per_mb_time = 10  # Seconds per MB
        
        estimated_time = base_time + (num_documents * per_doc_time) + (total_size_mb * per_mb_time)
        return int(estimated_time)
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
