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
from utils.helpers import SystemUtils

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
        """Initialize all system components if they are missing.

        Important: it does NOT recreate the managers on every rerun — because that will erase
        already-initialized objects during the multi-step init sequence.
        """
        # Create each manager only if it's not already present in session_state
        if st.session_state.get("llm_manager") is None:
            st.session_state.llm_manager = LLMManager()
        if st.session_state.get("embeddings_manager") is None:
            st.session_state.embeddings_manager = EmbeddingsManager()
        if st.session_state.get("document_loader") is None:
            st.session_state.document_loader = DocumentLoader()
        if st.session_state.get("text_processor") is None:
            st.session_state.text_processor = TextProcessor()
        # vector_manager, question_generator, tutor_engine are created later,
        # only after documents are processed (so don't touch them here)

    
    def render_sidebar(self):
        """Render sidebar with system controls"""
        with st.sidebar:
            st.header("🔧 System Control")

            # --- LLM Provider Selection ---
            provider = st.radio(
                "Choose LLM Provider:",
                ["ollama", "gemini"],
                index=0 if self.config.LLM_PROVIDER == "ollama" else 1,
                key="llm_provider",
                disabled=st.session_state.get("ui_disabled", False)
            )

            # If provider changes, force re-init
            if provider != self.config.LLM_PROVIDER:
                self.config.LLM_PROVIDER = provider
                self.config.apply_dynamic_settings()
                st.session_state.system_ready = False
                st.session_state.documents_processed = False
                st.session_state["init_status"] = "initializing_embeddings"
                st.session_state["ui_disabled"] = True

            # --- Initialization Process ---
            if "init_status" not in st.session_state:
                st.session_state["init_status"] = "initializing_embeddings"
                st.session_state["ui_disabled"] = True

            status = st.session_state["init_status"]

            if status == "initializing_embeddings":
                st.info("⏳ Please wait, Initializing Embeddings…")
                # Check if embeddings are already initialized
                if (st.session_state.embeddings_manager and 
                    st.session_state.embeddings_manager.get_embeddings() is not None):
                    st.session_state["init_status"] = "initializing_llm"
                    st.rerun()
                elif st.session_state.embeddings_manager.initialize_embeddings():
                    # Verify embeddings are properly loaded
                    embeddings = st.session_state.embeddings_manager.get_embeddings()
                    if embeddings is None:
                        st.error("❌ Embeddings failed to load correctly")
                        st.session_state["init_status"] = "failed"
                    else:
                        st.success("✅ Embeddings initialized successfully!")
                        st.session_state["init_status"] = "initializing_llm"
                        st.rerun()
                else:
                    st.error("❌ Failed to initialize embeddings")
                    st.session_state["init_status"] = "failed"

            elif status == "initializing_llm":
                st.info(f"⏳ Please wait, Initializing {self.config.LLM_PROVIDER.capitalize()} LLM…")
                if st.session_state.llm_manager.initialize_llm():
                    if st.session_state.embeddings_manager.get_embeddings() is None:
                        st.error("❌ Embeddings lost during LLM initialization")
                        st.session_state["init_status"] = "failed"
                        st.rerun()

                    # Only mark system ready (wait for documents before creating vector/tutor components)
                    st.session_state.system_ready = True
                    st.session_state["init_status"] = "ready"
                    st.session_state["ui_disabled"] = False
                    st.success("✅ LLM initialized! Ready for document upload.")
                    st.rerun()
                else:
                    st.session_state["init_status"] = "failed"
                    st.session_state["ui_disabled"] = False

            elif status == "ready":
                st.success("✅ Assistant is ready!")
                # Additional verification
                if (st.session_state.embeddings_manager and 
                    st.session_state.embeddings_manager.get_embeddings() is None):
                    st.warning("⚠️ Embeddings may have been lost - reinitializing...")
                    st.session_state["init_status"] = "initializing_embeddings"
                    st.rerun()

            elif status == "failed":
                st.error("❌ Failed to initialize system")
                if st.button("Retry Initialization"):
                    st.session_state["init_status"] = "initializing_embeddings"
                    st.session_state["ui_disabled"] = True
                    st.rerun()

            # --- Documents Status ---
            st.subheader("📄 Documents")
            if st.session_state.documents_processed:
                st.success("🟢 Documents Processed")
            else:
                st.warning("🟡 No Documents Loaded")

            # --- System Resources ---
            st.subheader("💻 System Resources")
            with st.expander("System Resources"):
                if st.button("🔄 Refresh Resources", key="refresh_resources"):
                    st.session_state["resources"] = SystemUtils.check_system_resources()
                if "resources" not in st.session_state:
                    st.session_state["resources"] = SystemUtils.check_system_resources()
                resources = st.session_state["resources"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CPU Usage", f"{resources['cpu_usage']:.1f}%")
                with col2:
                    st.metric("Memory Available", f"{resources['memory_available_gb']:.1f} GB")
                with col3:
                    st.metric("Memory Usage", f"{resources['memory_usage_percent']:.1f}%")

            # --- Configuration ---
            st.subheader("⚙️ Configuration")
            with st.expander("Current Configuration"):
                st.info(f"Provider: {self.config.LLM_PROVIDER}")
                st.info(f"Model: {self.config.OLLAMA_MODEL if self.config.LLM_PROVIDER=='ollama' else self.config.GEMINI_MODEL}")
                st.info(f"Chunk Size: {self.config.CHUNK_SIZE}")
                st.info(f"Conversation Memory: {self.config.CONVERSATION_MEMORY_K}")

            # --- Debug Information ---
            with st.expander("🔍 Debug Information"):
                st.write("Embeddings Manager Status:", 
                         "✅ Ready" if (st.session_state.embeddings_manager and 
                                      st.session_state.embeddings_manager.get_embeddings() is not None) 
                         else "❌ Not Ready")
                st.write("LLM Manager Status:", 
                         "✅ Ready" if (st.session_state.llm_manager and 
                                      st.session_state.llm_manager.is_ready()) 
                         else "❌ Not Ready")
                st.write("Vector Manager Status:", 
                         "✅ Ready" if st.session_state.vector_manager else "❌ Not Ready")

            # --- Conversation ---
            if st.button("Clear Conversation", disabled=st.session_state.get("ui_disabled", False)):
                st.session_state.chat_history = []
                if st.session_state.tutor_engine:
                    st.session_state.tutor_engine.clear_memory()
    
    def render_document_upload(self):
        """Render document upload section"""
        st.header("📚 Upload Study Materials")

        if st.session_state.get("ui_disabled", False):
            st.info("⏳ Uploads disabled while initializing.")
            return        
        
        
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
        # Verify embeddings are still available before processing - fix the check
        if (not st.session_state.embeddings_manager or 
            st.session_state.embeddings_manager.embeddings is None):
            st.error("❌ Embeddings not available! Please reinitialize the system.")
            st.session_state["init_status"] = "initializing_embeddings"
            st.session_state["ui_disabled"] = True
            st.rerun()
            return
            
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
            if st.session_state.vector_manager is None:
                st.session_state.vector_manager = VectorStoreManager(
                    st.session_state.embeddings_manager
                )
            
            if st.session_state.vector_manager.create_vector_store(processed_docs):
                st.session_state.documents_processed = True
                
                # Now initialize tutoring stack
                st.session_state.question_generator = QuestionGenerator(
                    st.session_state.llm_manager,
                    st.session_state.vector_manager
                )
                st.session_state.tutor_engine = TutorEngine(
                    st.session_state.llm_manager,
                    st.session_state.vector_manager
                )
                
                st.success("✅ Documents processed and tutoring engine ready!")
            else:
                st.error("❌ Failed to process documents!")

    
    def render_chat_interface(self):
        """Render chat interface"""
        st.header("💬 Study Chat")
        
        if st.session_state.get("ui_disabled", False):
            st.info("⏳ Chat disabled while initializing.")
            return
        
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
        
        if st.session_state.get("ui_disabled", False):
            st.info("⏳ Question generation disabled while initializing.")
            return
        
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
        if st.session_state.get("init_status") != "ready":
            st.warning("⏳ Assistant is still initializing. Please wait...")
        else:
            tab1, tab2, tab3 = st.tabs(["📁 Upload Documents", "💬 Study Chat", "❓ Generate Questions"])

            with tab1:
                self.render_document_upload()
            with tab2:
                self.render_chat_interface()
            with tab3:
                self.render_question_generator()

        
        # Instructions at bottom
        self.render_instructions()