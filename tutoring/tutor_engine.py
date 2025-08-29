# tutoring/tutor_engine.py
# =============================================================================
from typing import Tuple, List
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from config.settings import Config

class TutorEngine:
    """Main tutoring engine for student interactions"""
    
    def __init__(self, llm_manager, vector_manager):
        self.llm_manager = llm_manager
        self.vector_manager = vector_manager
        self.config = Config()
        self.conversation_chain = None
        self.memory = None
        self.setup_memory()
        self.setup_prompts()
    
    def setup_memory(self):
        """Setup conversation memory"""
        self.memory = ConversationBufferWindowMemory(
            k=self.config.CONVERSATION_MEMORY_K,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def setup_prompts(self):
        """Setup tutoring prompts"""
        self.tutor_template = """You are a patient and helpful tutor. Use the context below to help the student.

Context from study materials:
{context}

Chat History:
{chat_history}

Student: {question}

As a tutor, please:
1. Answer the student's question clearly and concisely
2. If they provided an answer, give constructive feedback
3. Explain concepts in simple terms
4. Provide examples when helpful
5. Ask follow-up questions to check understanding
6. Be encouraging and supportive
7. Keep responses focused and not too long (CPU-friendly)

Tutor:"""

    def initialize_conversation_chain(self):
        """Initialize the conversation chain"""
        if not self.llm_manager.is_ready() or not self.vector_manager.is_ready():
            return False
        
        try:
            # Create custom prompt
            custom_prompt = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=self.tutor_template
            )
            
            # Create conversation chain with custom prompt
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm_manager.get_llm(),
                retriever=self.vector_manager.get_retriever(),
                memory=self.memory,
                return_source_documents=True,
                verbose=False,
                combine_docs_chain_kwargs={"prompt": custom_prompt}
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing conversation chain: {str(e)}")
            return False
    
    def get_response(self, user_input: str) -> Tuple[str, List[Document]]:
        """Get tutor response to user input"""
        if not self.conversation_chain:
            if not self.initialize_conversation_chain():
                return "Tutor not ready. Please check system status.", []
        
        try:
            with st.spinner("Tutor is thinking... (This may take 30-90 seconds on CPU)"):
                response = self.conversation_chain({"question": user_input})
                return response["answer"], response.get("source_documents", [])
                
        except Exception as e:
            error_msg = f"Error getting tutor response: {str(e)}"
            st.error(error_msg)
            return error_msg, []
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            st.success("Conversation history cleared!")
