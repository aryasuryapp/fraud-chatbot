"""
Factory for creating LLM providers.
"""

import logging
import os
from typing import Optional
from llm.providers.base import BaseLLMProvider
from llm.providers.openai_provider import OpenAIProvider
from llm.providers.anthropic_provider import AnthropicProvider
from llm.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """
        Register a custom provider.
        
        Args:
            name: Provider identifier
            provider_class: Provider class (must inherit from BaseLLMProvider)
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise TypeError("Provider must inherit from BaseLLMProvider")
        cls._providers[name] = provider_class
        logger.info(f"Registered custom provider: {name}")
    
    @classmethod
    def create_provider(
        cls,
        provider: str,
        model_name: str,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create and initialize a provider.
        
        Args:
            provider: Provider name ('openai', 'anthropic', 'ollama')
            model_name: Model identifier
            **kwargs: Provider-specific configuration
            
        Returns:
            Initialized provider instance
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider not in cls._providers:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {', '.join(cls._providers.keys())}"
            )
        
        provider_class = cls._providers[provider]
        instance = provider_class(model_name=model_name, **kwargs)
        instance.initialize()
        
        return instance
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers (installed packages)."""
        available = []
        for name, provider_class in cls._providers.items():
            # Create temporary instance to check availability
            temp_instance = provider_class(model_name="dummy")
            if temp_instance.is_available():
                available.append(name)
        return available
    
    @classmethod
    def create_from_env(cls) -> BaseLLMProvider:
        """
        Create provider from environment variables.
        
        Environment variables:
            LLM_PROVIDER: Provider name (default: openai)
            MODEL_NAME: Model identifier
            
        Returns:
            Initialized provider
        """
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        model_name = os.getenv("MODEL_NAME", cls._get_default_model(provider))
        
        return cls.create_provider(provider, model_name)
    
    @staticmethod
    def _get_default_model(provider: str) -> str:
        """Get default model for a provider."""
        defaults = {
            "openai": "gpt-3.5-turbo",
            "anthropic": "claude-3-sonnet-20240229",
            "ollama": "llama2"
        }
        return defaults.get(provider, "gpt-3.5-turbo")
