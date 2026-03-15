"""Instruction-aware embedding provider (v2).

Supports multiple embedding backends:
- Ollama (default): Qwen3-Embedding-0.6B, local, private, instruction-aware
- sentence-transformers (fallback): all-MiniLM-L6-v2, local but no instructions
- OpenAI (opt-in): text-embedding-3-small, cloud API

Each document type gets a different instruction prefix at embedding time
to improve retrieval precision (Anthropic's Contextual Retrieval technique).
"""

from typing import Protocol, Optional, List
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION PREFIXES
# ═══════════════════════════════════════════════════════════════════════════════

EMBED_INSTRUCTIONS = {
    "task_doc":      "Represent this developer task description for session retrieval",
    "summary_doc":   "Represent this coding session summary for intelligent search",
    "reasoning_doc": "Represent this technical decision and rationale for retrieval",
    "proposition":   "Represent this factual statement about a coding session for retrieval",
}

QUERY_INSTRUCTION = "Find coding sessions relevant to this developer query"


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING PROVIDER PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def embed(self, text: str, doc_type: str = "task_doc") -> List[float]:
        """Embed a document text with optional instruction based on doc_type."""
        ...
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a search query with query-specific instruction."""
        ...
    
    def embed_batch(self, texts: List[str], doc_type: str = "task_doc") -> List[List[float]]:
        """Embed a batch of texts."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA EMBEDDER (DEFAULT)
# ═══════════════════════════════════════════════════════════════════════════════


class OllamaEmbedder:
    """Embeds text using Qwen3-Embedding via Ollama. Instruction-aware.
    
    This is the default and recommended embedding provider:
    - Runs 100% locally on CPU (no GPU required)
    - User's code never leaves their machine
    - Instruction-aware: different doc types get different instructions
    - Supports 32-1024 dimensions (we use 512)
    - Upgrade path: swap to qwen3-embedding:8b for max quality on GPU
    """
    
    def __init__(self, model: str = "qwen3-embedding:0.6b", dimensions: int = 512):
        self.model = model
        self.dimensions = dimensions
        self._ollama = None
    
    def _get_ollama(self):
        """Lazy-load ollama client."""
        if self._ollama is None:
            try:
                import ollama
                self._ollama = ollama
                logger.debug(f"Ollama initialized with model={self.model}, dims={self.dimensions}")
            except ImportError:
                raise RuntimeError(
                    "Ollama Python client not installed. "
                    "Install with: pip install ollama\n"
                    "Also install Ollama CLI: https://ollama.ai/download\n"
                    f"Then run: ollama pull {self.model}"
                )
        return self._ollama
    
    def embed(self, text: str, doc_type: str = "task_doc") -> List[float]:
        """Embed a document with instruction prefix.
        
        Args:
            text: Document text to embed
            doc_type: One of task_doc, summary_doc, reasoning_doc, proposition
        
        Returns:
            Embedding vector of configured dimensions
        """
        instruction = EMBED_INSTRUCTIONS.get(doc_type, "Represent for retrieval")
        full_text = f"{instruction}: {text}"
        
        ol = self._get_ollama()
        response = ol.embed(
            model=self.model,
            input=full_text,
        )
        
        # New Ollama client returns typed objects, not dicts
        embeddings = response.embeddings if hasattr(response, 'embeddings') else response.get("embeddings", [[]])
        embedding = embeddings[0] if embeddings else []
        
        # Truncate to configured dimensions if model returns more
        if len(embedding) > self.dimensions:
            embedding = embedding[:self.dimensions]
        
        return embedding
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a search query with query-specific instruction.
        
        Args:
            query: User's search query text
        
        Returns:
            Query embedding vector
        """
        full_text = f"{QUERY_INSTRUCTION}: {query}"
        
        ol = self._get_ollama()
        response = ol.embed(
            model=self.model,
            input=full_text,
        )
        
        embeddings = response.embeddings if hasattr(response, 'embeddings') else response.get("embeddings", [[]])
        embedding = embeddings[0] if embeddings else []
        if len(embedding) > self.dimensions:
            embedding = embedding[:self.dimensions]
        
        return embedding
    
    def embed_batch(self, texts: List[str], doc_type: str = "task_doc") -> List[List[float]]:
        """Embed a batch of texts with the same instruction.
        
        Args:
            texts: List of document texts to embed
            doc_type: Document type determining the instruction
        
        Returns:
            List of embedding vectors
        """
        instruction = EMBED_INSTRUCTIONS.get(doc_type, "Represent for retrieval")
        full_texts = [f"{instruction}: {t}" for t in texts]
        
        ol = self._get_ollama()
        response = ol.embed(
            model=self.model,
            input=full_texts,
        )
        
        embeddings = response.embeddings if hasattr(response, 'embeddings') else response.get("embeddings", [])
        # Truncate each to dimensions
        return [e[:self.dimensions] if len(e) > self.dimensions else e for e in embeddings]


# ═══════════════════════════════════════════════════════════════════════════════
# SENTENCE-TRANSFORMERS FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════


class SentenceTransformerEmbedder:
    """Fallback embedder using sentence-transformers library.
    
    Used when Ollama is not installed/available. Provides reasonable
    quality but lacks instruction-awareness.
    """
    
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        self._model = None
    
    def _get_model(self):
        """Lazy-load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"SentenceTransformer fallback loaded: {self.model_name}")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def embed(self, text: str, doc_type: str = "") -> List[float]:
        """Embed text (instruction prefix ignored in fallback mode)."""
        model = self._get_model()
        return model.encode(text).tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query (same as document embedding in fallback mode)."""
        return self.embed(query)
    
    def embed_batch(self, texts: List[str], doc_type: str = "") -> List[List[float]]:
        """Embed batch of texts."""
        model = self._get_model()
        return [e.tolist() for e in model.encode(texts)]


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def get_embedder(provider: str = "ollama", model: str = None,
                 dimensions: int = 512) -> EmbeddingProvider:
    """Factory: returns the correct embedding provider based on config.
    
    Args:
        provider: "ollama" (default), "sentence-transformers", or "openai"
        model: Model name (default depends on provider)
        dimensions: Embedding dimensions (only used for Ollama)
    
    Returns:
        EmbeddingProvider instance
    """
    if provider == "ollama":
        model = model or "qwen3-embedding:0.6b"
        return OllamaEmbedder(model=model, dimensions=dimensions)
    
    elif provider == "sentence-transformers":
        model = model or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(model=model)
    
    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Supported: 'ollama', 'sentence-transformers'"
        )


def check_ollama_available(model: str = "qwen3-embedding:0.6b") -> dict:
    """Check if Ollama is installed and the required model is pulled.
    
    Returns:
        Dict with 'available' (bool), 'ollama_installed' (bool),
        'model_pulled' (bool), 'message' (str)
    """
    result = {
        "available": False,
        "ollama_installed": False,
        "model_pulled": False,
        "message": ""
    }
    
    try:
        import ollama
        result["ollama_installed"] = True
    except ImportError:
        result["message"] = (
            "Ollama Python client not installed.\n"
            "Install with: pip install ollama\n"
            "Also install Ollama CLI: https://ollama.ai/download"
        )
        return result
    
    try:
        # Try to list models to check if Ollama server is running
        response = ollama.list()
        # New Ollama client returns typed objects with .models list
        models_list = response.models if hasattr(response, 'models') else response.get("models", [])
        model_names = []
        for m in models_list:
            # Handle both object (.model) and dict ("name") formats
            name = getattr(m, 'model', None) or (m.get("name", "") if isinstance(m, dict) else "")
            if name:
                model_names.append(name)
        
        if model in model_names or any(model.split(":")[0] in n for n in model_names):
            result["model_pulled"] = True
            result["available"] = True
            result["message"] = f"Ollama ready with model {model}"
        else:
            result["message"] = (
                f"Model '{model}' not found.\n"
                f"Pull it with: ollama pull {model}"
            )
    except Exception as e:
        result["message"] = (
            f"Ollama server not running or error: {e}\n"
            "Start Ollama and run: ollama pull " + model
        )
    
    return result
