"""
SQL generation utilities.
"""

import logging
from typing import Tuple, Dict, Optional
from llm.providers.base import BaseLLMProvider, LLMRequest
from llm.prompt_builder import PromptBuilder, LLMRequestBuilder

logger = logging.getLogger(__name__)


class SQLGenerator:
    """Handles SQL query generation from natural language."""
    
    def __init__(self, provider: BaseLLMProvider):
        """
        Initialize SQL generator.
        
        Args:
            provider: LLM provider instance
        """
        self.provider = provider
    
    def needs_db_query(self, question: str) -> bool:
        """
        Determine if question requires specific database query beyond general stats.
        
        Args:
            question: User question
            
        Returns:
            True if specific query is needed
        """
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
    
    def generate_sql(
        self,
        question: str,
        schema: str,
        request_id: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Use LLM to generate SQL query from natural language question.
        
        Args:
            question: User question
            schema: Database schema
            request_id: Optional request ID for logging
            
        Returns:
            Tuple of (sql_query, usage_dict)
        """
        log_prefix = f"[{request_id}] " if request_id else ""
        usage_info = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0
        }
        
        try:
            # Build SQL generation prompt
            prompt = PromptBuilder.build_sql_generation_prompt(question, schema)
            
            # Create request with temperature=0 for deterministic SQL generation
            request = (LLMRequestBuilder()
                .with_prompt(prompt)
                .for_sql_generation()
                .with_request_id(request_id)
                .build())
            
            # Generate using provider
            response = self.provider.generate(request)
            
            if not response.is_success:
                logger.error(f"{log_prefix}SQL generation failed: {response.error}")
                return f"SELECT * FROM fraud_transactions LIMIT 100 -- Error: {response.error}", usage_info
            
            query = response.content.strip()
            
            # Extract usage if available
            if response.usage:
                usage_info = response.usage.to_dict()
            
            # Clean up common formatting issues
            query = query.replace('```sql', '').replace('```', '').strip()
            query = query.rstrip(';')
            
            # Smart LIMIT handling: only add if query returns multiple rows
            query = self._add_smart_limit(query)
            
            logger.debug(f"{log_prefix}Generated SQL query: {query}")
            return query, usage_info
            
        except Exception as e:
            logger.error(f"{log_prefix}Error generating SQL query: {e}")
            return f"SELECT * FROM fraud_transactions LIMIT 100 -- Error: {e}", usage_info
    
    def validate_query(self, query: str) -> bool:
        """
        Validate SQL query to prevent SQL injection and dangerous operations.
        
        Args:
            query: SQL query to validate
            
        Returns:
            True if query is safe
        """
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
    
    def _add_smart_limit(self, query: str) -> str:
        """
        Add LIMIT clause intelligently based on query type.
        
        Args:
            query: SQL query
            
        Returns:
            Query with LIMIT added if needed
        """
        query_upper = query.upper()
        
        # Only add if query doesn't already have LIMIT
        if 'LIMIT' in query_upper:
            return query
        
        # Determine if LIMIT is needed
        needs_limit = (
            'GROUP BY' in query_upper or  # Multiple groups need LIMIT
            # Non-aggregation queries need LIMIT (has SELECT but no aggregation functions)
            (query_upper.count('SELECT') == 1 and 
             'COUNT(' not in query_upper and 
             'SUM(' not in query_upper and 
             'AVG(' not in query_upper and
             'MAX(' not in query_upper and
             'MIN(' not in query_upper)
        )
        
        if needs_limit:
            query += ' LIMIT 100'
        
        return query
