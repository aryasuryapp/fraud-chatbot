"""
OpenAI provider implementation.
"""

import logging
from typing import Optional, Dict
from llm.providers.base import BaseLLMProvider, LLMRequest, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation with token tracking."""
    
    # Model pricing (per 1K tokens in USD)
    PRICING = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (defaults to env OPENAI_API_KEY)
            organization: OpenAI organization ID
            base_url: Custom base URL for OpenAI API
            **kwargs: Additional OpenAI client options
        """
        super().__init__(model_name, api_key, **kwargs)
        self.organization = organization
        self.base_url = base_url
    
    def initialize(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.organization:
                client_kwargs["organization"] = self.organization
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            self._client = OpenAI(**client_kwargs)
            logger.info(f"Initialized OpenAI provider with model: {self.model_name}")
            
        except ImportError as e:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            ) from e
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self._client:
            return LLMResponse(
                content="",
                usage=None,
                metadata={},
                error="Provider not initialized. Call initialize() first."
            )
        
        try:
            # Prepare messages
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            # Log request if debug enabled
            if request.request_id:
                logger.debug(f"[{request.request_id}] OpenAI request: {len(request.prompt)} chars")
            
            # Make API call
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                **(request.metadata or {})
            )
            
            # Extract content
            content = response.choices[0].message.content
            
            # Extract token usage
            usage = None
            if hasattr(response, 'usage') and response.usage:
                cost = self.calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
                usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    cost=cost,
                    model_name=self.model_name,
                    provider="openai"
                )
            
            return LLMResponse(
                content=content,
                usage=usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                    "response_id": response.id
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return LLMResponse(
                content="",
                usage=None,
                metadata={},
                error=f"OpenAI API error: {str(e)}"
            )
    
    def supports_token_tracking(self) -> bool:
        """OpenAI supports token tracking."""
        return True
    
    def get_pricing(self) -> Optional[Dict[str, float]]:
        """Get pricing for the current model."""
        return self.PRICING.get(self.model_name)
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        try:
            import openai
            return True
        except ImportError:
            return False
