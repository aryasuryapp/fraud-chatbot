"""
Load fraud transaction data from Kaggle dataset into SQLite database.
"""

import pandas as pd
import sqlite3
from pathlib import Path


def load_fraud_data(csv_path: str, db_path: str = "database.db") -> None:
    """
    Load fraud CSV data into SQLite database.
    
    Args:
        csv_path: Path to fraud.csv file
        db_path: Path to SQLite database file
    """
    print(f"Loading data from {csv_path}...")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    
    # Create table and load data
    df.to_sql('fraud_transactions', conn, if_exists='replace', index=False)
    
    print(f"Data successfully loaded into {db_path}")
    
    # Display basic statistics
    print("\nDataset Info:")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    conn.close()


def query_fraud_data(query: str, db_path: str = "database.db") -> pd.DataFrame:
    """
    Execute SQL query on fraud database.
    
    Args:
        query: SQL query string
        db_path: Path to SQLite database file
        
    Returns:
        Query results as DataFrame
    """
    conn = sqlite3.connect(db_path)
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


if __name__ == "__main__":
    # Example usage
    csv_path = "data/fraud.csv"
    
    if Path(csv_path).exists():
        load_fraud_data(csv_path)
        
        # Example query
        result = query_fraud_data("SELECT * FROM fraud_transactions LIMIT 5")
        print("\nSample records:")
        print(result)
    else:
        print(f"Error: {csv_path} not found. Please download the dataset first.")
