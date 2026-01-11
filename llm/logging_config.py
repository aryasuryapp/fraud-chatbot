"""
Logging configuration utilities.
"""

import logging
import os


def setup_logger(name: str) -> logging.Logger:
    """
    Setup a logger with pretty formatting.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
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
