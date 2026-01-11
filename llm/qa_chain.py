"""
QA Chain - combines retriever with LLM for question answering.
"""

from typing import Optional
import sqlite3
import pandas as pd
import os
import logging
import uuid
import time
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


# Token pricing per 1K tokens (USD)
MODEL_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for LLM API call based on token usage.
    
    Args:
        model_name: Name of the model used
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        
    Returns:
        Cost in USD
    """
    if model_name not in MODEL_PRICING:
        logger.debug(f"No pricing info for model {model_name}, returning 0")
        return 0.0
    
    pricing = MODEL_PRICING[model_name]
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
    return cost


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
    
    def _generate_sql_query(self, question: str, schema: str, request_id: str = None) -> tuple[str, dict]:
        """Use LLM to generate SQL query from natural language question with temperature=0 for consistency.
        
        Returns:
            Tuple of (sql_query, usage_dict) where usage_dict contains tokens and cost
        """
        prompt = f"""Given this SQLite database schema:
{schema}

Generate a VALID SQLite query to answer this question: {question}

IMPORTANT Requirements:
- Return ONLY the SQL query, no explanation or markdown
- Use proper SQLite syntax (e.g., strftime for dates, || for concatenation)
- For percentage calculations, use: COUNT(CASE WHEN condition THEN 1 END) * 100.0 / COUNT(*)
- For percentage queries, use WHERE to filter first, then calculate percentage within filtered set
- Use LIMIT 100 only for queries returning multiple rows (not for aggregations without GROUP BY)
- For aggregations, use appropriate GROUP BY clauses
- Use WHERE clauses to filter data efficiently
- Use is_fraud column (0 or 1) to filter fraud cases
- Common categories: gas_transport, grocery_pos, home, shopping_pos, etc.
- trans_date_trans_time is in format 'YYYY-MM-DD HH:MM:SS'

SQL Query:"""
        
        log_prefix = f"[{request_id}] " if request_id else ""
        usage_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0}
        
        try:
            # Use temperature=0 for deterministic SQL generation
            if self.llm_provider == "openai":
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert SQL developer. Generate only valid, efficient SQL queries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,  # Deterministic for consistent SQL generation
                    max_tokens=300
                )
                query = response.choices[0].message.content.strip()
                
                # Extract token usage from OpenAI response
                if hasattr(response, 'usage') and response.usage:
                    usage_info["input_tokens"] = response.usage.prompt_tokens
                    usage_info["output_tokens"] = response.usage.completion_tokens
                    usage_info["total_tokens"] = response.usage.total_tokens
                    usage_info["cost"] = calculate_cost(
                        self.model_name,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )
                    
            elif self.llm_provider == "anthropic":
                message = self.llm.messages.create(
                    model=self.model_name,
                    max_tokens=300,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )
                query = message.content[0].text.strip()
                
                # Extract token usage from Anthropic response
                if hasattr(message, 'usage') and message.usage:
                    usage_info["input_tokens"] = message.usage.input_tokens
                    usage_info["output_tokens"] = message.usage.output_tokens
                    usage_info["total_tokens"] = message.usage.input_tokens + message.usage.output_tokens
                    usage_info["cost"] = calculate_cost(
                        self.model_name,
                        message.usage.input_tokens,
                        message.usage.output_tokens
                    )
                    
            else:
                # Fallback to _call_llm for other providers
                result = self._call_llm(prompt, request_id)
                query = result["answer"].strip() if isinstance(result, dict) else result.strip()
                if isinstance(result, dict) and "usage" in result:
                    usage_info = result["usage"]
            
            # Clean up common formatting issues
            query = query.replace('```sql', '').replace('```', '').strip()
            query = query.rstrip(';')
            
            # Smart LIMIT handling: only add if query returns multiple rows
            query_upper = query.upper()
            needs_limit = (
                'LIMIT' not in query_upper and (
                    'GROUP BY' in query_upper or  # Multiple groups need LIMIT
                    # Non-aggregation queries need LIMIT (has SELECT but no aggregation functions)
                    (query_upper.count('SELECT') == 1 and 
                     'COUNT(' not in query_upper and 
                     'SUM(' not in query_upper and 
                     'AVG(' not in query_upper and
                     'MAX(' not in query_upper and
                     'MIN(' not in query_upper)
                )
            )
            
            if needs_limit:
                query += ' LIMIT 100'
            
            logger.debug(f"{log_prefix}Generated SQL query: {query}")
            return query, usage_info
        except Exception as e:
            logger.error(f"{log_prefix}Error generating SQL query: {e}")
            return f"SELECT * FROM fraud_transactions LIMIT 100 -- Error generating query: {e}", usage_info
    
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
                sql_query, sql_usage = self._generate_sql_query(query, schema[0], request_id)
                
                # Store SQL usage in context for later aggregation (pass back via special marker)
                if hasattr(self, '_current_sql_usage'):
                    self._current_sql_usage = sql_usage
                
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
        """Build prompt for LLM with proper handling of both database and document context."""
        prompt = f"""You are a fraud detection expert assistant. Answer the question using the provided context.

        Question: {question}

        Database Context:
        {db_context}

        Document Context:
        {context}

        INSTRUCTIONS:
        - Use information from BOTH Database Context and Document Context
        - Database Context (statistics, query results) is valid and authoritative - use it freely
        - Document Context (PDF excerpts) provides additional insights - reference with [Document N]
        - If database has the answer (statistics, counts, percentages), use it directly
        - Only say "information not available" if NEITHER source contains relevant information
        - Be specific and cite your sources when possible

        Answer:"""
        return prompt
    
    def _call_llm(self, prompt: str, request_id: str = None) -> dict:
        """Call LLM with prompt and return answer with token usage.
        
        Returns:
            Dict with 'answer' and 'usage' (tokens and cost)
        """
        log_prefix = f"[{request_id}] " if request_id else ""
        result = {
            "answer": "",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0
            }
        }
        
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
                result["answer"] = response.choices[0].message.content
                
                # Extract token usage
                if hasattr(response, 'usage') and response.usage:
                    result["usage"]["input_tokens"] = response.usage.prompt_tokens
                    result["usage"]["output_tokens"] = response.usage.completion_tokens
                    result["usage"]["total_tokens"] = response.usage.total_tokens
                    result["usage"]["cost"] = calculate_cost(
                        self.model_name,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )
                    
                return result
            except Exception as e:
                result["answer"] = f"Error calling OpenAI: {str(e)}"
                return result
        
        elif self.llm_provider == "anthropic":
            try:
                message = self.llm.messages.create(
                    model=self.model_name,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                result["answer"] = message.content[0].text
                
                # Extract token usage
                if hasattr(message, 'usage') and message.usage:
                    result["usage"]["input_tokens"] = message.usage.input_tokens
                    result["usage"]["output_tokens"] = message.usage.output_tokens
                    result["usage"]["total_tokens"] = message.usage.input_tokens + message.usage.output_tokens
                    result["usage"]["cost"] = calculate_cost(
                        self.model_name,
                        message.usage.input_tokens,
                        message.usage.output_tokens
                    )
                    
                return result
            except Exception as e:
                result["answer"] = f"Error calling Anthropic: {str(e)}"
                return result
        
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
                result["answer"] = response.json().get("response", "No response from Ollama")
                # Note: Ollama doesn't provide token/cost info for local models
                return result
            except Exception as e:
                result["answer"] = f"Error calling Ollama: {str(e)}"
                return result
        
        result["answer"] = "LLM not initialized"
        return result
    
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
            f"Model: {self.model_name}"
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
                "model": self.model_name
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
