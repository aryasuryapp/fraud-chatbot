"""
Cost tracking utilities for LLM usage.
"""

from typing import Optional, Dict


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
    """
    Calculate cost for LLM API call based on token usage.
    
    Args:
        model_name: Name of the model used
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        
    Returns:
        Cost in USD
    """
    if model_name not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model_name]
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
    return cost


def get_model_pricing(model_name: str) -> Optional[Dict[str, float]]:
    """
    Get pricing information for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dict with 'input' and 'output' pricing per 1K tokens, or None if not found
    """
    return MODEL_PRICING.get(model_name)
