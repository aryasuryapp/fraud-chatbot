"""
Prompt building utilities with request builder pattern.
"""

from typing import Optional
from llm.providers.base import LLMRequest


class PromptBuilder:
    """Builds prompts for different use cases."""
    
    @staticmethod
    def build_qa_prompt(question: str, context: str, db_context: str) -> str:
        """
        Build prompt for question answering.
        
        Args:
            question: User question
            context: Document context from vector store
            db_context: Database context
            
        Returns:
            Formatted prompt
        """
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
    
    @staticmethod
    def build_sql_generation_prompt(question: str, schema: str) -> str:
        """
        Build prompt for SQL query generation.
        
        Args:
            question: User question
            schema: Database schema
            
        Returns:
            Formatted prompt for SQL generation
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
        return prompt


class LLMRequestBuilder:
    """Builder pattern for creating LLMRequest objects with fluent API."""
    
    def __init__(self):
        """Initialize builder with default values."""
        self._request = LLMRequest(prompt="")
    
    def with_prompt(self, prompt: str) -> 'LLMRequestBuilder':
        """Set the prompt."""
        self._request.prompt = prompt
        return self
    
    def with_system_prompt(self, system_prompt: str) -> 'LLMRequestBuilder':
        """Set the system prompt."""
        self._request.system_prompt = system_prompt
        return self
    
    def for_sql_generation(self) -> 'LLMRequestBuilder':
        """Configure for SQL generation (temperature=0, focused)."""
        self._request.temperature = 0.0
        self._request.max_tokens = 300
        self._request.system_prompt = "You are an expert SQL developer. Generate only valid, efficient SQL queries."
        return self
    
    def for_answer_generation(self) -> 'LLMRequestBuilder':
        """Configure for answer generation (temperature=0.7, creative)."""
        self._request.temperature = 0.7
        self._request.max_tokens = 500
        self._request.system_prompt = "You are a fraud detection expert assistant."
        return self
    
    def with_temperature(self, temp: float) -> 'LLMRequestBuilder':
        """Set custom temperature."""
        self._request.temperature = temp
        return self
    
    def with_max_tokens(self, max_tokens: int) -> 'LLMRequestBuilder':
        """Set max tokens."""
        self._request.max_tokens = max_tokens
        return self
    
    def with_request_id(self, request_id: str) -> 'LLMRequestBuilder':
        """Set request ID for logging."""
        self._request.request_id = request_id
        return self
    
    def with_metadata(self, **kwargs) -> 'LLMRequestBuilder':
        """Set provider-specific metadata."""
        self._request.metadata = kwargs
        return self
    
    def build(self) -> LLMRequest:
        """Build and return the LLMRequest."""
        return self._request
