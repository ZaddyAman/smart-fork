"""LLM provider abstraction layer (v2).

Supports multiple LLM backends for query decomposition and session summarization:
- Ollama (default): Local Qwen3-0.6B, private, no API key needed
- Anthropic: Claude Haiku for higher quality (opt-in)
- OpenAI: GPT-4o-mini as alternative (opt-in)

The LLMProvider protocol ensures the decomposer and summarizer work
identically regardless of which backend is configured.
"""

from typing import Protocol, Optional
from loguru import logger


class LLMProvider(Protocol):
    """Protocol for LLM completion providers."""
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate a completion from the LLM.
        
        Args:
            prompt: The prompt text
            max_tokens: Maximum tokens in the response
        
        Returns:
            The LLM's response text
        """
        ...


class OllamaLLM:
    """Local LLM via Ollama (default provider).
    
    Runs entirely on the user's machine. No API key needed.
    Recommended model: qwen3:0.6b (fast, good for structured extraction)
    """
    
    def __init__(self, model: str = "qwen3:0.6b"):
        self.model = model
        self._ollama = None
    
    def _get_ollama(self):
        if self._ollama is None:
            try:
                import ollama
                self._ollama = ollama
            except ImportError:
                raise RuntimeError(
                    "Ollama Python client not installed. "
                    "Install with: pip install ollama\n"
                    f"Then run: ollama pull {self.model}"
                )
        return self._ollama
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        ol = self._get_ollama()
        try:
            response = ol.generate(
                model=self.model,
                prompt=prompt,
                options={"num_predict": max_tokens, "temperature": 0.1}
            )
            if isinstance(response, dict):
                return response.get("response", "")
            return getattr(response, "response", "")
        except Exception as e:
            logger.error(f"Ollama completion failed: {e}")
            return ""


class AnthropicLLM:
    """Cloud LLM via Anthropic API (opt-in only).
    
    Requires ANTHROPIC_API_KEY environment variable.
    Uses Claude Haiku for fast, cheap structured extraction.
    """
    
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.model = model
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise RuntimeError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        return self._client
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            return ""


class OpenAILLM:
    """Cloud LLM via OpenAI API (opt-in only).
    
    Requires OPENAI_API_KEY environment variable.
    Uses GPT-4o-mini for fast, cheap structured extraction.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
        return self._client
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            return ""


def get_llm(provider: str = "ollama", model: str = None) -> LLMProvider:
    """Factory: returns the correct LLM provider based on config.
    
    Args:
        provider: "ollama" (default), "anthropic", or "openai"
        model: Model name override (defaults set per provider)
    
    Returns:
        LLMProvider instance
    """
    if provider == "ollama":
        return OllamaLLM(model=model or "qwen3:0.6b")
    elif provider == "anthropic":
        return AnthropicLLM(model=model or "claude-3-haiku-20240307")
    elif provider == "openai":
        return OpenAILLM(model=model or "gpt-4o-mini")
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported: 'ollama', 'anthropic', 'openai'"
        )
