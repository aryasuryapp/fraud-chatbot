"""
Ollama provider implementation.
"""

import logging
from typing import Optional, Dict
from llm.providers.base import BaseLLMProvider, LLMRequest, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local models (no token tracking)."""
    
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
            **kwargs: Additional options
        """
        super().__init__(model_name, api_key=None, **kwargs)
        self.base_url = base_url
    
    def initialize(self) -> None:
        """Initialize Ollama (check availability)."""
        import requests
        
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                logger.info(f"Initialized Ollama provider with model: {self.model_name}")
                logger.info(f"Ollama server: {self.base_url}")
            else:
                logger.warning(f"Ollama server returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Could not connect to Ollama at {self.base_url}: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Ollama API."""
        import requests
        
        try:
            # Prepare prompt (combine system and user)
            full_prompt = request.prompt
            if request.system_prompt:
                full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
            
            # Log request if debug enabled
            if request.request_id:
                logger.debug(f"[{request.request_id}] Ollama request: {len(full_prompt)} chars")
            
            # Make API call
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("response", "")
            
            # Ollama doesn't provide token tracking or costs
            return LLMResponse(
                content=content,
                usage=None,  # No token tracking for local models
                metadata={
                    "model": result.get("model"),
                    "eval_count": result.get("eval_count"),
                    "eval_duration": result.get("eval_duration")
                }
            )
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            return LLMResponse(
                content="",
                usage=None,
                metadata={},
                error=f"Ollama API error: {str(e)}"
            )
    
    def supports_token_tracking(self) -> bool:
        """Ollama does not support token tracking."""
        return False
    
    def get_pricing(self) -> Optional[Dict[str, float]]:
        """Local models have no API cost."""
        return None
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        import requests
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
