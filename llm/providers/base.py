"""
Base classes for LLM provider abstraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class TemperatureMode(Enum):
    """Predefined temperature settings for different use cases."""
    SQL_GENERATION = 0.0      # Deterministic for SQL
    ANSWER_GENERATION = 0.7   # Creative for answers
    CLASSIFICATION = 0.3      # Low but not zero for classification
    SUMMARIZATION = 0.5       # Moderate for summaries


@dataclass
class TokenUsage:
    """Standardized token usage tracking across providers."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    model_name: str
    provider: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "model_name": self.model_name,
            "provider": self.provider
        }


@dataclass
class LLMRequest:
    """Standardized request to any LLM provider."""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    usage: Optional[TokenUsage]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        """Check if the response was successful."""
        return self.error is None


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize provider.
        
        Args:
            model_name: Model identifier
            api_key: API key (if required)
            **kwargs: Provider-specific configuration
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
        self._client = None
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider client. Called after __init__."""
        pass
    
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response from LLM.
        
        Args:
            request: Standardized LLM request
            
        Returns:
            Standardized LLM response with usage tracking
        """
        pass
    
    @abstractmethod
    def supports_token_tracking(self) -> bool:
        """Whether this provider supports token usage tracking."""
        pass
    
    @abstractmethod
    def get_pricing(self) -> Optional[Dict[str, float]]:
        """
        Get pricing information for this model.
        
        Returns:
            Dict with 'input' and 'output' pricing per 1K tokens,
            or None if not applicable
        """
        pass
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost based on token usage."""
        pricing = self.get_pricing()
        if not pricing:
            return 0.0
        return (
            input_tokens * pricing["input"] + 
            output_tokens * pricing["output"]
        ) / 1000
    
    @property
    def provider_name(self) -> str:
        """Return provider name (e.g., 'openai', 'anthropic')."""
        return self.__class__.__name__.replace("Provider", "").lower()
    
    def validate_config(self) -> bool:
        """Validate provider configuration. Override if needed."""
        return True
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (e.g., package installed, API accessible)."""
        pass
