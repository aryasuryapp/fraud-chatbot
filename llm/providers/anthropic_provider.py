"""
Anthropic provider implementation.
"""

import logging
from typing import Optional, Dict
from llm.providers.base import BaseLLMProvider, LLMRequest, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider implementation with token tracking."""
    
    # Model pricing (per 1K tokens in USD)
    PRICING = {
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    }
    
    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            model_name: Anthropic model name
            api_key: Anthropic API key (defaults to env ANTHROPIC_API_KEY)
            **kwargs: Additional Anthropic client options
        """
        super().__init__(model_name, api_key, **kwargs)
    
    def initialize(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
            
            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            
            self._client = anthropic.Anthropic(**client_kwargs)
            logger.info(f"Initialized Anthropic provider with model: {self.model_name}")
            
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            ) from e
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic API."""
        if not self._client:
            return LLMResponse(
                content="",
                usage=None,
                metadata={},
                error="Provider not initialized. Call initialize() first."
            )
        
        try:
            # Note: Anthropic uses system parameter separately
            create_kwargs = {
                "model": self.model_name,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "messages": [{"role": "user", "content": request.prompt}]
            }
            
            if request.system_prompt:
                create_kwargs["system"] = request.system_prompt
            
            # Add any custom metadata
            if request.metadata:
                create_kwargs.update(request.metadata)
            
            # Log request if debug enabled
            if request.request_id:
                logger.debug(f"[{request.request_id}] Anthropic request: {len(request.prompt)} chars")
            
            # Make API call
            response = self._client.messages.create(**create_kwargs)
            
            # Extract content
            content = response.content[0].text
            
            # Extract token usage
            usage = None
            if hasattr(response, 'usage') and response.usage:
                cost = self.calculate_cost(
                    response.usage.input_tokens,
                    response.usage.output_tokens
                )
                usage = TokenUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                    cost=cost,
                    model_name=self.model_name,
                    provider="anthropic"
                )
            
            return LLMResponse(
                content=content,
                usage=usage,
                metadata={
                    "stop_reason": response.stop_reason,
                    "model": response.model,
                    "response_id": response.id
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            return LLMResponse(
                content="",
                usage=None,
                metadata={},
                error=f"Anthropic API error: {str(e)}"
            )
    
    def supports_token_tracking(self) -> bool:
        """Anthropic supports token tracking."""
        return True
    
    def get_pricing(self) -> Optional[Dict[str, float]]:
        """Get pricing for the current model."""
        return self.PRICING.get(self.model_name)
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        try:
            import anthropic
            return True
        except ImportError:
            return False
