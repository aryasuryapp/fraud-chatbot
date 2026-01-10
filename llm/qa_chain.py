"""
QA Chain - combines retriever with LLM for question answering.
"""

from typing import Optional
import sqlite3
import pandas as pd
import os
import logging
import uuid
from dotenv import load_dotenv
from rag.retriever import Retriever

# Load environment variables from .env file
load_dotenv()

# Configure logging with pretty formatting
def setup_logger(name: str) -> logging.Logger:
    """Setup a logger with pretty formatting."""
    logger = logging.getLogger(name)
    
    # Get log level from environment or default to INFO
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler with custom formatting
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    
    # Simple, clean formatter
    formatter = logging.Formatter(
        fmt='%(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

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
                logger.warning("No document embeddings found. RAG retrieval disabled.")
                logger.info("Run 'python ingestion/load_docs.py' to create embeddings.")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            logger.info("The chatbot will work with database context only.")
        
        # Initialize LLM
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM based on provider."""
        if self.llm_provider == "openai":
            try:
                from openai import OpenAI
                self.llm = OpenAI()
                logger.info(f"Initialized OpenAI with model: {self.model_name}")
            except ImportError:
                logger.error("openai not installed. Run: pip install openai")
        
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
                self.llm = anthropic.Anthropic()
                logger.info(f"Initialized Anthropic with model: {self.model_name}")
            except ImportError:
                logger.error("anthropic not installed. Run: pip install anthropic")
        
        elif self.llm_provider == "ollama":
            logger.info(f"Using Ollama with model: {self.model_name}")
            logger.info("Make sure Ollama is running locally")
    
    def _get_table_stats(self, conn) -> str:
        """Get useful aggregate statistics about the fraud dataset."""
        try:
            stats_query = """
            SELECT 
                COUNT(*) as total_records,
                SUM(is_fraud) as total_fraud_cases,
                ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) as fraud_rate_pct,
                ROUND(AVG(amt), 2) as avg_transaction_amt,
                ROUND(MIN(amt), 2) as min_amt,
                ROUND(MAX(amt), 2) as max_amt,
                COUNT(DISTINCT category) as num_categories,
                COUNT(DISTINCT state) as num_states,
                MIN(trans_date_trans_time) as earliest_date,
                MAX(trans_date_trans_time) as latest_date
            FROM fraud_transactions
            """
            df = pd.read_sql_query(stats_query, conn)
            return df.to_string(index=False)
        except Exception as e:
            return f"Error getting stats: {str(e)}"
    
    def _question_needs_db_query(self, question: str) -> bool:
        """Determine if question requires specific database query beyond general stats."""
        question_lower = question.lower()
        
        # Keywords that indicate need for specific data queries
        specific_keywords = [
            'how many', 'what percentage', 'show me', 'list', 'find',
            'which', 'top', 'most', 'least', 'average', 'total',
            'by category', 'by state', 'by merchant', 'between',
            'where', 'in california', 'over', 'under', 'above', 'below',
            'compare', 'breakdown', 'distribution', 'group'
        ]
        
        return any(keyword in question_lower for keyword in specific_keywords)
    
    def _generate_sql_query(self, question: str, schema: str, request_id: str = None) -> str:
        """Use LLM to generate SQL query from natural language question."""
        prompt = f"""Given this SQLite database schema:
{schema}

Generate a VALID SQLite query to answer this question: {question}

IMPORTANT Requirements:
- Return ONLY the SQL query, no explanation or markdown
- Use proper SQLite syntax (e.g., strftime for dates, || for concatenation)
- ALWAYS include 'LIMIT 100' to prevent returning too many rows
- For aggregations, use appropriate GROUP BY clauses
- Use WHERE clauses to filter data efficiently
- Use is_fraud column (0 or 1) to filter fraud cases
- Common categories: gas_transport, grocery_pos, home, shopping_pos, etc.
- trans_date_trans_time is in format 'YYYY-MM-DD HH:MM:SS'

SQL Query:"""
        
        try:
            query = self._call_llm(prompt, request_id).strip()
            # Clean up common formatting issues
            query = query.replace('```sql', '').replace('```', '').strip()
            # Remove any trailing semicolon and add LIMIT if missing
            query = query.rstrip(';')
            if 'LIMIT' not in query.upper():
                query += ' LIMIT 100'
            return query
        except Exception as e:
            return f"SELECT * FROM fraud_transactions LIMIT 100 -- Error generating query: {e}"
    
    def _is_safe_query(self, query: str) -> bool:
        """Validate SQL query to prevent SQL injection and dangerous operations."""
        if not query or not isinstance(query, str):
            return False
        
        query_upper = query.upper().strip()
        
        # Must start with SELECT
        if not query_upper.startswith('SELECT'):
            return False
        
        # Block dangerous SQL keywords and patterns
        dangerous_patterns = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
            'TRUNCATE', 'EXEC', 'EXECUTE', 'PRAGMA', 'ATTACH',
            '--', ';--', '/*', '*/', 'xp_', 'sp_',
            'UNION SELECT', 'UNION ALL SELECT',  # Block common injection
            'INTO OUTFILE', 'INTO DUMPFILE',  # Block file operations
        ]
        
        for pattern in dangerous_patterns:
            if pattern in query_upper:
                return False
        
        # Additional safety: only allow single statement
        if query.count(';') > 1:
            return False
        
        return True
    
    def _get_db_context(self, query: str, request_id: str = None) -> str:
        """Get relevant data from SQLite database with intelligent query generation."""
        log_prefix = f"[{request_id}] " if request_id else ""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table schema
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='fraud_transactions'")
            schema = cursor.fetchone()
            
            if not schema:
                conn.close()
                return "No fraud data available in database."
            
            # Always get basic statistics (useful for any question)
            stats = self._get_table_stats(conn)
            
            context = f"""Database: fraud_transactions

Schema:
{schema[0]}

Dataset Statistics:
{stats}
"""
            
            # Detect if question needs specific data query
            if self._question_needs_db_query(query):
                logger.debug(f"{log_prefix}Generating SQL query for specific data...")
                
                # Generate SQL query using LLM
                sql_query = self._generate_sql_query(query, schema[0], request_id)
                
                # Validate query for security
                if self._is_safe_query(sql_query):
                    try:
                        # Execute the generated query
                        df = pd.read_sql_query(sql_query, conn)
                        
                        context += f"\nGenerated Query:\n{sql_query}\n\n"
                        context += f"Query Results ({len(df)} rows):\n"
                        
                        # Limit output size for context window
                        if len(df) > 50:
                            context += df.head(50).to_string(index=False)
                            context += f"\n... (showing first 50 of {len(df)} rows)"
                        else:
                            context += df.to_string(index=False)
                    except Exception as e:
                        context += f"\n⚠️  Query execution error: {str(e)}\n"
                        context += f"Attempted query: {sql_query}\n"
                else:
                    context += f"\n⚠️  Generated query failed security validation\n"
                    context += f"Query: {sql_query}\n"
            else:
                logger.debug(f"{log_prefix}Using general statistics (no specific query needed)")
            
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
    
    def _call_llm(self, prompt: str, request_id: str = None) -> str:
        """Call LLM with prompt."""
        log_prefix = f"[{request_id}] " if request_id else ""
        if self.llm_provider == "openai":
            try:
                # Log the prompt in a pretty, readable format
                logger.debug(f"{log_prefix}" + "=" * 80)
                logger.debug(f"{log_prefix}LLM PROMPT:")
                logger.debug(f"{log_prefix}" + "-" * 80)
                for line in prompt.split('\n'):
                    logger.debug(f"{log_prefix}{line}")
                logger.debug(f"{log_prefix}" + "=" * 80)
                
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
        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Processing question: {question}")
        
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
        db_context = self._get_db_context(question, request_id)
        
        # Build prompt
        prompt = self._build_prompt(question, doc_context, db_context)
        
        # Get answer from LLM
        answer = self._call_llm(prompt, request_id)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "db_context": db_context,
            "num_chunks_used": len(sources),
            "relevance_threshold": relevance_threshold,
            "request_id": request_id
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
        for chunk, score in result["sources"]:
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
