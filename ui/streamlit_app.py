# ui/streamlit_app.py
# =============================================================================
import streamlit as st
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Config
from models.llm_manager import LLMManager
from models.embeddings_manager import EmbeddingsManager
from document_processing.document_loader import DocumentLoader
from document_processing.text_processor import TextProcessor
from vector_store.vector_manager import VectorStoreManager
from tutoring.question_generator import QuestionGenerator
from tutoring.tutor_engine import TutorEngine

class StudyAssistantUI:
    """Main UI class for the Study Assistant"""
    
    def __init__(self):
        self.config = Config()
        self.setup_page_config()
        self.initialize_session_state()
        self.initialize_components()
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title=self.config.PAGE_TITLE,
            page_icon=self.config.PAGE_ICON,
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        session_vars = {
            'llm_manager': None,
            'embeddings_manager': None,
            'document_loader': None,
            'text_processor': None,
            'vector_manager': None,
            'question_generator': None,
            'tutor_engine': None,
            'chat_history': [],
            'system_ready': False,
            'documents_processed': False
        }
        
        for var, default_value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default_value
    
    def initialize_components(self):
        """Initialize all system components"""
        if not st.session_state.system_ready:
            # Initialize managers
            st.session_state.llm_manager = LLMManager()
            st.session_state.embeddings_manager = EmbeddingsManager()
            st.session_state.document_loader = DocumentLoader()
            st.session_state.text_processor = TextProcessor()
    
    def render_sidebar(self):
        """Render sidebar with system controls"""
        with st.sidebar:
            st.header("🔧 System Control")
            
            # System status
            st.subheader("System Status")
            
            # Initialize embeddings
            if st.button("Initialize Embeddings"):
                with st.spinner("Initializing embeddings..."):
                    if st.session_state.embeddings_manager.initialize_embeddings():
                        st.success("✅ Embeddings ready!")
                    else:
                        st.error("❌ Embeddings failed!")
            
            # Initialize LLM
            if st.button("Initialize Ollama LLM"):
                with st.spinner("Connecting to Ollama..."):
                    if st.session_state.llm_manager.initialize_llm():
                        st.success("✅ Ollama connected!")
                        
                        # Initialize vector manager
                        st.session_state.vector_manager = VectorStoreManager(
                            st.session_state.embeddings_manager
                        )
                        
                        # Initialize tutoring components
                        st.session_state.question_generator = QuestionGenerator(
                            st.session_state.llm_manager,
                            st.session_state.vector_manager
                        )
                        st.session_state.tutor_engine = TutorEngine(
                            st.session_state.llm_manager,
                            st.session_state.vector_manager
                        )
                        
                        st.session_state.system_ready = True
                    else:
                        st.error("❌ Ollama connection failed!")
            
            # Status indicators
            if st.session_state.embeddings_manager and st.session_state.embeddings_manager.get_embeddings():
                st.success("🟢 Embeddings Ready")
            else:
                st.error("🔴 Embeddings Not Ready")
            
            if st.session_state.llm_manager and st.session_state.llm_manager.is_ready():
                st.success("🟢 Ollama Ready")
            else:
                st.error("🔴 Ollama Not Ready")
            
            if st.session_state.documents_processed:
                st.success("🟢 Documents Processed")
            else:
                st.warning("🟡 No Documents Loaded")
            
            # System info
            st.subheader("Configuration")
            st.info(f"Model: {self.config.OLLAMA_MODEL}")
            st.info(f"Embeddings: CPU-optimized")
            st.info(f"Chunk Size: {self.config.CHUNK_SIZE}")
            
            # Clear conversation
            if st.button("Clear Conversation"):
                st.session_state.chat_history = []
                if st.session_state.tutor_engine:
                    st.session_state.tutor_engine.clear_memory()
    
    def render_document_upload(self):
        """Render document upload section"""
        st.header("📚 Upload Study Materials")
        
        uploaded_files = st.file_uploader(
            "Upload your textbooks, notes, or study materials",
            type=['pdf', 'txt', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, DOC"
        )
        
        if uploaded_files and st.session_state.system_ready:
            if st.button("Process Documents", type="primary"):
                self.process_documents(uploaded_files)
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents"""
        with st.spinner("Processing documents..."):
            # Load documents
            documents = st.session_state.document_loader.load_documents_from_uploads(uploaded_files)
            
            if not documents:
                st.error("No documents could be loaded!")
                return
            
            # Process documents
            processed_docs = st.session_state.text_processor.process_documents(documents)
            
            # Show processing stats
            stats = st.session_state.text_processor.get_chunk_stats(processed_docs)
            st.info(f"Created {stats['total_chunks']} chunks with average size {stats['avg_chunk_size']:.0f} characters")
            
            # Create vector store
            if st.session_state.vector_manager.create_vector_store(processed_docs):
                st.session_state.documents_processed = True
                st.success("✅ Documents processed successfully!")
            else:
                st.error("❌ Failed to process documents!")
    
    def render_chat_interface(self):
        """Render chat interface"""
        st.header("💬 Study Chat")
        
        if not st.session_state.system_ready:
            st.warning("Please initialize the system first (see sidebar)")
            return
        
        if not st.session_state.documents_processed:
            st.warning("Please upload and process documents first")
            return
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📖 Sources"):
                        for i, source in enumerate(message["sources"][:2]):
                            st.write(f"**Source {i+1}:**")
                            st.write(source.page_content[:200] + "...")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your study materials..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                response, sources = st.session_state.tutor_engine.get_response(prompt)
                st.write(response)
                
                # Show sources
                if sources:
                    with st.expander("📖 Sources"):
                        for i, source in enumerate(sources[:2]):
                            st.write(f"**Source {i+1}:**")
                            st.write(source.page_content[:200] + "...")
            
            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })
    
    def render_question_generator(self):
        """Render question generation interface"""
        st.header("❓ Generate Study Questions")
        
        if not st.session_state.system_ready or not st.session_state.documents_processed:
            st.warning("Please initialize system and upload documents first")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            difficulty = st.selectbox(
                "Difficulty Level:",
                ["easy", "medium", "hard"],
                index=1
            )
        
        with col2:
            question_type = st.selectbox(
                "Question Type:",
                ["mixed", "multiple choice", "short answer", "conceptual"],
                index=0
            )
        
        topic = st.text_input(
            "Focus Topic (optional):",
            placeholder="Enter a specific topic or leave blank for random questions"
        )
        
        if st.button("Generate Questions", type="primary"):
            questions = st.session_state.question_generator.generate_questions(
                topic=topic,
                difficulty=difficulty,
                question_type=question_type
            )
            
            st.markdown("### Generated Questions:")
            st.write(questions)
    
    def render_instructions(self):
        """Render instructions"""
        with st.expander("📖 How to Use (CPU-Optimized Version)"):
            st.markdown("""
            ### Setup Instructions:
            1. **Install Ollama**: Download from https://ollama.ai/
            2. **Start Ollama**: Run `ollama serve` in terminal
            3. **Pull Model**: Run `ollama pull llama2`
            4. **Initialize System**: Click buttons in sidebar to initialize components
            5. **Upload Documents**: Upload your study materials
            6. **Start Studying**: Use chat or generate questions
            
            ### Performance Notes:
            - ⏱️ **CPU Processing**: Responses may take 30-90 seconds
            - 📊 **Batch Processing**: Documents processed in small batches
            - 🔄 **Patience Required**: First response takes longer as model loads
            - 💾 **Memory Usage**: Optimized for lower memory consumption
            
            ### Tips for Better Performance:
            - Keep documents under 50MB total
            - Use shorter, focused questions
            - Process documents in smaller batches
            - Clear conversation history periodically
            
            ### Example Questions:
            - "Explain the main concept from chapter 2"
            - "I think the answer is X, am I correct?"
            - "What should I focus on for the exam?"
            - "Generate practice questions about photosynthesis" 
        """)
    
    def run(self):
        """Main application runner"""
        st.title(self.config.PAGE_TITLE)
        st.markdown("🖥️ **CPU-Optimized Version** - Upload your study materials and get AI tutoring assistance!")
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["📁 Upload Documents", "💬 Study Chat", "❓ Generate Questions"])
        
        with tab1:
            self.render_document_upload()
        
        with tab2:
            self.render_chat_interface()
        
        with tab3:
            self.render_question_generator()
        
        # Instructions at bottom
        self.render_instructions()
