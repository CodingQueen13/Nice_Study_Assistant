# tutoring/question_generator.py
# =============================================================================
from typing import List
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from config.settings import Config

class QuestionGenerator:
    """Generates study questions based on document content"""
    
    def __init__(self, llm_manager, vector_manager):
        self.llm_manager = llm_manager
        self.vector_manager = vector_manager
        self.config = Config()
        self.setup_prompts()
    
    def setup_prompts(self):
        """Setup question generation prompts"""
        self.question_prompt = PromptTemplate(
            input_variables=["context", "difficulty", "question_type"],
            template="""Based on the study material below, create {difficulty} level {question_type} questions.

Study Material:
{context}

Instructions:
- Create 3-4 questions that test understanding of key concepts
- Make questions clear and educational
- Include a mix of factual and conceptual questions
- Keep questions focused and answerable from the material

Questions:"""
        )
    
    def generate_questions(
        self, 
        topic: str = "", 
        difficulty: str = "medium",
        question_type: str = "mixed"
    ) -> str:
        """Generate questions based on the study material"""
        
        if not self.llm_manager.is_ready():
            return "LLM not ready. Please check Ollama connection."
        
        if not self.vector_manager.is_ready():
            return "Please upload and process documents first."
        
        try:
            # Get relevant documents
            if topic:
                docs = self.vector_manager.similarity_search(topic, k=2)
            else:
                # Get random sample from vector store
                docs = self.vector_manager.similarity_search("", k=2)
            
            if not docs:
                return "No relevant content found for question generation."
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in docs])
            context = context[:2000]  # Limit context for CPU efficiency
            
            # Generate questions
            question_chain = LLMChain(
                llm=self.llm_manager.get_llm(),
                prompt=self.question_prompt
            )
            
            with st.spinner("Generating questions... (This may take 30-60 seconds on CPU)"):
                questions = question_chain.run(
                    context=context,
                    difficulty=difficulty,
                    question_type=question_type
                )
            
            return questions
            
        except Exception as e:
            return f"Error generating questions: {str(e)}"
