"""
Database management utilities.
"""

import sqlite3
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database operations and context retrieval."""
    
    def __init__(self, db_path: str):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
    
    def get_table_stats(self, conn: sqlite3.Connection) -> str:
        """
        Get useful aggregate statistics about the fraud dataset.
        
        Args:
            conn: SQLite connection
            
        Returns:
            Formatted statistics string
        """
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
    
    def get_schema(self, table_name: str = "fraud_transactions") -> Optional[str]:
        """
        Get table schema.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Schema SQL or None if error
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            schema = cursor.fetchone()
            conn.close()
            return schema[0] if schema else None
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return None
    
    def execute_safe_query(self, query: str) -> Optional[pd.DataFrame]:
        """
        Execute a query and return results as DataFrame.
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame with results or None if error
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return None
    
    def get_basic_context(self) -> str:
        """
        Get basic database context (schema + stats).
        
        Returns:
            Formatted context string
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table schema
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='fraud_transactions'")
            schema = cursor.fetchone()
            
            if not schema:
                conn.close()
                return "No fraud data available in database."
            
            # Get statistics
            stats = self.get_table_stats(conn)
            conn.close()
            
            context = f"""Database: fraud_transactions

Schema:
{schema[0]}

Dataset Statistics:
{stats}
"""
            return context
            
        except Exception as e:
            return f"Error accessing database: {str(e)}"
