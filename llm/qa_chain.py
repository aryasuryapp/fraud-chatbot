"""
QA Chain - combines retriever with LLM for question answering.
"""

from typing import Optional
import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv
from rag.retriever import Retriever

# Load environment variables from .env file
load_dotenv()


class QAChain:
    """Question-answering chain for fraud-related queries."""
    
    def __init__(
        self, 
        llm_provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        db_path: str = "database.db"
    ):
        """
        Initialize QA chain.
        
        Args:
            llm_provider: LLM provider ('openai', 'anthropic', or 'ollama')
            model_name: Model name
            db_path: Path to SQLite database
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.db_path = db_path
        self.retriever = Retriever()
        self.llm = None
        self.has_vector_store = False
        
        # Load RAG settings from environment
        self.relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
        self.max_chunks = int(os.getenv("MAX_CHUNKS", "10"))
        
        # Try to load retriever (non-blocking if it fails)
        try:
            self.has_vector_store = self.retriever.load_vector_store()
            if not self.has_vector_store:
                print("⚠️  Warning: No document embeddings found. RAG retrieval disabled.")
                print("   Run 'python ingestion/load_docs.py' to create embeddings.")
        except Exception as e:
            print(f"⚠️  Warning: Could not load vector store: {e}")
            print("   The chatbot will work with database context only.")
        
        # Initialize LLM
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM based on provider."""
        if self.llm_provider == "openai":
            try:
                from openai import OpenAI
                self.llm = OpenAI()
                print(f"Initialized OpenAI with model: {self.model_name}")
            except ImportError:
                print("openai not installed. Run: pip install openai")
        
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
                self.llm = anthropic.Anthropic()
                print(f"Initialized Anthropic with model: {self.model_name}")
            except ImportError:
                print("anthropic not installed. Run: pip install anthropic")
        
        elif self.llm_provider == "ollama":
            print(f"Using Ollama with model: {self.model_name}")
            print("Make sure Ollama is running locally")
    
    def _get_db_context(self, query: str) -> str:
        """Get relevant data from SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get table schema
            cursor = conn.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='fraud_transactions'")
            schema = cursor.fetchone()
            
            if not schema:
                conn.close()
                return "No fraud data available in database."
            
            # Get sample statistics
            df = pd.read_sql_query(
                "SELECT * FROM fraud_transactions LIMIT 5", 
                conn
            )
            
            context = f"Database Schema:\n{schema[0]}\n\n"
            context += f"Sample Records:\n{df.to_string()}\n"
            
            conn.close()
            return context
            
        except Exception as e:
            return f"Error accessing database: {str(e)}"
    
    def _build_prompt(self, question: str, context: str, db_context: str) -> str:
        """Build prompt for LLM."""
        prompt = f"""You are a fraud detection expert assistant. Answer the following question about fraud transactions using the provided context.

Question: {question}

Database Context:
{db_context}

Document Context:
{context}

Provide a clear, accurate answer based on the context. If the context doesn't contain enough information, say so.

Answer:"""
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        if self.llm_provider == "openai":
            try:
                print(prompt)
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a fraud detection expert assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling OpenAI: {str(e)}"
        
        elif self.llm_provider == "anthropic":
            try:
                message = self.llm.messages.create(
                    model=self.model_name,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
            except Exception as e:
                return f"Error calling Anthropic: {str(e)}"
        
        elif self.llm_provider == "ollama":
            try:
                import requests
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                return response.json().get("response", "No response from Ollama")
            except Exception as e:
                return f"Error calling Ollama: {str(e)}"
        
        return "LLM not initialized"
    
    def ask(self, question: str, max_chunks: int = None, relevance_threshold: float = None) -> dict:
        """
        Answer a question using RAG with dynamic retrieval.
        
        Args:
            question: User question
            max_chunks: Maximum number of chunks to retrieve initially (default: from env or 10)
            relevance_threshold: Minimum similarity score to include chunk (default: from env or 0.7)
            
        Returns:
            Dictionary with answer and sources
        """
        # Use instance defaults from env if not specified
        if max_chunks is None:
            max_chunks = self.max_chunks
        if relevance_threshold is None:
            relevance_threshold = self.relevance_threshold
        
        # Retrieve relevant documents (if vector store is available)
        sources = []
        if self.has_vector_store:
            # Retrieve top candidates
            raw_results = self.retriever.retrieve(question, k=max_chunks)
            
            # Filter by relevance threshold
            filtered_sources = [(chunk, score) for chunk, score in raw_results if score >= relevance_threshold]
            
            # Use top 5 max for context (cost control)
            sources = filtered_sources[:5]
            
            # Build context from filtered sources
            if sources:
                context_parts = []
                for i, (chunk, score) in enumerate(sources, 1):
                    context_parts.append(f"[Document {i}] (Relevance: {score:.3f})\n{chunk}\n")
                doc_context = "\n".join(context_parts)
            else:
                doc_context = "No highly relevant documents found (all scores below threshold)."
        else:
            doc_context = "No document embeddings available. Answering based on database context only."
        
        # Get database context
        db_context = self._get_db_context(question)
        
        # Build prompt
        prompt = self._build_prompt(question, doc_context, db_context)
        
        # Get answer from LLM
        answer = self._call_llm(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "db_context": db_context,
            "num_chunks_used": len(sources),
            "relevance_threshold": relevance_threshold
        }


if __name__ == "__main__":
    # Example usage
    qa = QAChain(llm_provider="openai", model_name="gpt-3.5-turbo")
    
    question = "What are the most common patterns in fraudulent transactions?"
    result = qa.ask(question)
    
    print(f"Q: {result['question']}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources: {len(result['sources'])} documents retrieved")
