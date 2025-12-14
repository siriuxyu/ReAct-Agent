"""
Embedding Service Implementation

Storage team can choose: OpenAI, Cohere, sentence-transformers, etc.
"""

import asyncio
import os
import logging
from typing import List

# Dependency checks
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from agent.interfaces import EmbeddingError

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService:
    """Embedding service using OpenAI"""

    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", dimension: int = 1536,
        cache_size: int = 1024
    ):
        """Initialize OpenAI embedding service"""
        if AsyncOpenAI is None:
            raise ImportError("OpenAI library not installed. Please run `pip install openai`")

        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.client = AsyncOpenAI(api_key=self.api_key)
        # In-process LRU cache: dict preserves insertion order (Python 3.7+)
        self._cache: dict = {}
        self._cache_size = cache_size

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text, with in-process LRU cache and retry."""
        clean = text.replace("\n", " ")
        if clean in self._cache:
            return self._cache[clean]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.embeddings.create(
                    input=[clean],
                    model=self.model
                )
                embedding = response.data[0].embedding
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(f"OpenAI embedding failed (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"OpenAI embedding failed after {max_retries} attempts: {e}")
                    raise EmbeddingError(f"OpenAI embedding error: {str(e)}")

        # Evict oldest entry when cache is full
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[clean] = embedding
        return embedding

    async def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        try:
            # Batch processing
            clean_texts = [t.replace("\n", " ") for t in texts]
            response = await self.client.embeddings.create(
                input=clean_texts,
                model=self.model
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise EmbeddingError(f"OpenAI batch error: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


class SentenceTransformerEmbeddingService:
    """Embedding service using sentence-transformers (local)"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize sentence-transformers embedding service"""
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed. Please run `pip install sentence-transformers`")
            
        self.model_name = model_name
        # Load model
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            text = text.replace("\n", " ")
            # Note: SentenceTransformer is synchronous, but direct call is usually acceptable in async context
            # For high concurrency, consider wrapping with asyncio.to_thread
            return self.model.encode(text).tolist()
        except Exception as e:
            raise EmbeddingError(f"Local embedding error: {str(e)}")

    async def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        try:
            clean_texts = [t.replace("\n", " ") for t in texts]
            return self.model.encode(clean_texts).tolist()
        except Exception as e:
            raise EmbeddingError(f"Local batch embedding error: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


def create_embedding_service(provider: str = "openai", **kwargs):
    """
    Factory function to create embedding service
    """
    if provider == "openai":
        return OpenAIEmbeddingService(**kwargs)
    elif provider == "sentence-transformers":
        return SentenceTransformerEmbeddingService(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
