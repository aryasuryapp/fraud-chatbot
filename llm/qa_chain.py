"""
QA Chain - combines retriever with LLM for question answering.
"""

from typing import Optional
import sqlite3
import pandas as pd
import os
import uuid
import time
from dotenv import load_dotenv
from rag.retriever import Retriever
from llm.logging_config import setup_logger
from llm.providers import LLMProviderFactory
from llm.prompt_builder import PromptBuilder, LLMRequestBuilder
from llm.sql_generator import SQLGenerator
from llm.db_manager import DatabaseManager

# Load environment variables from .env file
load_dotenv()

logger = setup_logger(__name__)


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
        self.db_path = db_path
        self.retriever = Retriever()
        self.has_vector_store = False
        
        # Initialize provider through factory
        self.provider = LLMProviderFactory.create_provider(
            provider=llm_provider,
            model_name=model_name
        )
        
        # Initialize utility managers
        self.db_manager = DatabaseManager(db_path)
        self.sql_generator = SQLGenerator(self.provider)
        
        # Log provider info
        logger.info(f"Using {self.provider.provider_name} with {model_name}")
        if self.provider.supports_token_tracking():
            logger.info("Token tracking enabled")
        
        # Load RAG settings from environment
        self.relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
        self.max_chunks = int(os.getenv("MAX_CHUNKS", "10"))
        
        # Try to load retriever (non-blocking if it fails)
        try:
            self.has_vector_store = self.retriever.load_vector_store()
            if not self.has_vector_store:
                logger.warning("No document embeddings found. RAG retrieval disabled.")
                logger.info("Run 'python ingestion/load_docs.py' to create embeddings.")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            logger.info("The chatbot will work with database context only.")
    
    def _get_db_context(self, query: str, request_id: str = None) -> str:
        """Get relevant data from SQLite database with intelligent query generation."""
        log_prefix = f"[{request_id}] " if request_id else ""
        try:
            # Get basic context (schema + stats)
            context = self.db_manager.get_basic_context()
            
            if "No fraud data" in context or "Error" in context:
                return context
            
            # Detect if question needs specific data query
            if self.sql_generator.needs_db_query(query):
                logger.debug(f"{log_prefix}Generating SQL query for specific data...")
                
                # Get schema
                schema = self.db_manager.get_schema()
                if not schema:
                    return context
                
                # Generate SQL query using LLM
                sql_query, sql_usage = self.sql_generator.generate_sql(query, schema, request_id)
                
                # Store SQL usage in context for later aggregation
                if hasattr(self, '_current_sql_usage'):
                    self._current_sql_usage = sql_usage
                
                # Validate query for security
                if self.sql_generator.validate_query(sql_query):
                    # Execute the generated query
                    df = self.db_manager.execute_safe_query(sql_query)
                    
                    if df is not None:
                        context += f"\nGenerated Query:\n{sql_query}\n\n"
                        context += f"Query Results ({len(df)} rows):\n"
                        
                        # Limit output size for context window
                        if len(df) > 50:
                            context += df.head(50).to_string(index=False)
                            context += f"\n... (showing first 50 of {len(df)} rows)"
                        else:
                            context += df.to_string(index=False)
                    else:
                        context += f"\n⚠️  Query execution error\n"
                        context += f"Attempted query: {sql_query}\n"
                else:
                    context += f"\n⚠️  Generated query failed security validation\n"
                    context += f"Query: {sql_query}\n"
            else:
                logger.debug(f"{log_prefix}Using general statistics (no specific query needed)")
            
            return context
            
        except Exception as e:
            return f"Error accessing database: {str(e)}"
    
    def _build_prompt(self, question: str, context: str, db_context: str) -> str:
        """Build prompt for LLM using PromptBuilder."""
        return PromptBuilder.build_qa_prompt(question, context, db_context)
    
    def _call_llm(self, prompt: str, request_id: str = None) -> dict:
        """
        Call LLM with prompt and return answer with token usage.
        
        Returns:
            Dict with 'answer' and 'usage' (tokens and cost)
        """
        log_prefix = f"[{request_id}] " if request_id else ""
        
        # Log the prompt in a pretty, readable format
        logger.debug(f"{log_prefix}" + "=" * 80)
        logger.debug(f"{log_prefix}LLM PROMPT:")
        logger.debug(f"{log_prefix}" + "-" * 80)
        for line in prompt.split('\n'):
            logger.debug(f"{log_prefix}{line}")
        logger.debug(f"{log_prefix}" + "=" * 80)
        
        # Create request using builder
        request = (LLMRequestBuilder()
            .with_prompt(prompt)
            .for_answer_generation()  # Sets temp=0.7, max_tokens=500, system prompt
            .with_request_id(request_id)
            .build())
        
        # Generate using provider
        response = self.provider.generate(request)
        
        # Convert to legacy format for backward compatibility
        result = {
            "answer": response.content if response.is_success else f"Error: {response.error}",
            "usage": response.usage.to_dict() if response.usage else self._empty_usage()
        }
        
        return result
    
    def _empty_usage(self) -> dict:
        """Return empty usage dict for providers without token tracking."""
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0
        }
    
    def ask(self, question: str, max_chunks: int = None, relevance_threshold: float = None) -> dict:
        """
        Answer a question using RAG with dynamic retrieval.
        
        Args:
            question: User question
            max_chunks: Maximum number of chunks to retrieve initially (default: from env or 10)
            relevance_threshold: Minimum similarity score to include chunk (default: from env or 0.7)
            
        Returns:
            Dictionary with answer, sources, and performance metrics
        """
        # Start timing
        start_time = time.perf_counter()
        
        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Processing question: {question}")
        
        # Initialize usage tracking
        total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "sql_generation": {"tokens": 0, "cost": 0.0},
            "answer_generation": {"tokens": 0, "cost": 0.0}
        }
        self._current_sql_usage = None
        
        # Use instance defaults from env if not specified
        if max_chunks is None:
            max_chunks = self.max_chunks
        if relevance_threshold is None:
            relevance_threshold = self.relevance_threshold
        
        # Retrieve relevant documents (if vector store is available)
        retrieval_start = time.perf_counter()
        sources = []
        if self.has_vector_store:
            # Retrieve top candidates
            raw_results = self.retriever.retrieve(question, k=max_chunks)
            
            # Filter by relevance threshold and handle both formats
            filtered_sources = []
            for result in raw_results:
                if len(result) == 3:
                    # New format: (chunk, score, metadata)
                    chunk, score, metadata = result
                    if score >= relevance_threshold:
                        filtered_sources.append((chunk, score, metadata))
                else:
                    # Legacy format: (chunk, score)
                    chunk, score = result
                    if score >= relevance_threshold:
                        filtered_sources.append((chunk, score, None))
            
            # Use top 5 max for context (cost control)
            sources = filtered_sources[:5]
            
            # Build context from filtered sources
            if sources:
                context_parts = []
                for i, source_item in enumerate(sources, 1):
                    if len(source_item) == 3:
                        chunk, score, metadata = source_item
                        meta_str = ""
                        if metadata:
                            meta_str = f" [Source: {metadata.get('source', 'unknown')}, Page: {metadata.get('page', 'N/A')}]"
                        context_parts.append(f"[Document {i}]{meta_str} (Relevance: {score:.3f})\n{chunk}\n")
                    else:
                        chunk, score, _ = source_item
                        context_parts.append(f"[Document {i}] (Relevance: {score:.3f})\n{chunk}\n")
                doc_context = "\n".join(context_parts)
            else:
                doc_context = "No highly relevant documents found (all scores below threshold)."
        else:
            doc_context = "No document embeddings available. Answering based on database context only."
        
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000  # Convert to ms
        
        # Get database context (this may include SQL generation)
        db_start = time.perf_counter()
        db_context = self._get_db_context(question, request_id)
        db_time = (time.perf_counter() - db_start) * 1000  # Convert to ms
        
        # Capture SQL generation usage if it occurred
        if self._current_sql_usage:
            total_usage["sql_generation"]["tokens"] = self._current_sql_usage["total_tokens"]
            total_usage["sql_generation"]["cost"] = self._current_sql_usage["cost"]
            total_usage["input_tokens"] += self._current_sql_usage["input_tokens"]
            total_usage["output_tokens"] += self._current_sql_usage["output_tokens"]
            total_usage["total_tokens"] += self._current_sql_usage["total_tokens"]
            total_usage["cost"] += self._current_sql_usage["cost"]
        
        # Build prompt
        prompt = self._build_prompt(question, doc_context, db_context)
        
        # Get answer from LLM
        llm_start = time.perf_counter()
        llm_result = self._call_llm(prompt, request_id)
        llm_time = (time.perf_counter() - llm_start) * 1000  # Convert to ms
        
        answer = llm_result["answer"]
        
        # Aggregate answer generation usage
        if "usage" in llm_result:
            usage = llm_result["usage"]
            total_usage["answer_generation"]["tokens"] = usage["total_tokens"]
            total_usage["answer_generation"]["cost"] = usage["cost"]
            total_usage["input_tokens"] += usage["input_tokens"]
            total_usage["output_tokens"] += usage["output_tokens"]
            total_usage["total_tokens"] += usage["total_tokens"]
            total_usage["cost"] += usage["cost"]
        
        # Calculate total latency
        total_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Log comprehensive metrics
        logger.info(
            f"[{request_id}] Completed | "
            f"Latency: {total_time:.0f}ms (retrieval: {retrieval_time:.0f}ms, db: {db_time:.0f}ms, llm: {llm_time:.0f}ms) | "
            f"Tokens: {total_usage['total_tokens']} (in: {total_usage['input_tokens']}, out: {total_usage['output_tokens']}) | "
            f"Cost: ${total_usage['cost']:.6f} | "
            f"Model: {self.provider.model_name}"
        )
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "db_context": db_context,
            "num_chunks_used": len(sources),
            "relevance_threshold": relevance_threshold,
            "request_id": request_id,
            "metrics": {
                "latency_ms": total_time,
                "latency_breakdown": {
                    "retrieval_ms": retrieval_time,
                    "database_ms": db_time,
                    "llm_ms": llm_time
                },
                "tokens": total_usage,
                "model": self.provider.model_name
            }
        }
    
    def ask_for_evaluation(self, question: str, max_chunks: int = None, relevance_threshold: float = None) -> dict:
        """
        Answer a question and return result in RAGAS-compatible format.
        
        This method is specifically designed for evaluation purposes, formatting
        the response to work seamlessly with RAGAS metrics.
        
        Args:
            question: User question
            max_chunks: Maximum number of chunks to retrieve initially
            relevance_threshold: Minimum similarity score to include chunk
            
        Returns:
            Dictionary with RAGAS-compatible format:
            - question: str
            - answer: str
            - contexts: List[str] (list of context strings)
            - metadata: dict (sources with scores, db_context, etc.)
        """
        # Get standard result
        result = self.ask(question, max_chunks, relevance_threshold)
        
        # Convert to RAGAS format
        contexts = []
        
        # Add document contexts (from vector store)
        for source_item in result["sources"]:
            if len(source_item) == 3:
                chunk, score, metadata = source_item
                contexts.append(chunk)
            else:
                chunk, score, _ = source_item
                contexts.append(chunk)
        
        # Optionally include database context as a separate context
        if result["db_context"] and "No fraud data" not in result["db_context"]:
            contexts.append(f"Database Context:\n{result['db_context']}")
        
        return {
            "question": result["question"],
            "answer": result["answer"],
            "contexts": contexts,  # RAGAS expects List[str]
            "metadata": {
                "sources": result["sources"],  # Keep original with scores
                "db_context": result["db_context"],
                "num_chunks_used": result["num_chunks_used"],
                "relevance_threshold": result["relevance_threshold"],
                "request_id": result["request_id"]
            }
        }


if __name__ == "__main__":
    # Example usage
    qa = QAChain(llm_provider="openai", model_name="gpt-3.5-turbo")
    
    question = "What are the most common patterns in fraudulent transactions?"
    result = qa.ask(question)
    
    print(f"Q: {result['question']}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources: {len(result['sources'])} documents retrieved")
