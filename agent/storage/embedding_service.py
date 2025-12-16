"""
Embedding Service Implementation

Storage team can choose: OpenAI, Cohere, sentence-transformers, etc.
"""

import asyncio
import hashlib
import json
import os
import logging
from typing import List, Optional

# Dependency checks
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

from agent.interfaces import EmbeddingError

logger = logging.getLogger(__name__)

# Redis embedding cache TTL: 7 days (embeddings are deterministic, safe to cache long)
_REDIS_EMB_TTL = 7 * 24 * 3600
_REDIS_KEY_PREFIX = "emb:"


class OpenAIEmbeddingService:
    """Embedding service using OpenAI, with optional Redis cache."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        cache_size: int = 1024,
        redis_url: Optional[str] = None,
    ):
        if AsyncOpenAI is None:
            raise ImportError("OpenAI library not installed. Please run `pip install openai`")

        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Redis cache (shared across processes/instances)
        self._redis: Optional[aioredis.Redis] = None
        _url = redis_url or os.environ.get("REDIS_URL")
        if _url and aioredis is not None:
            try:
                self._redis = aioredis.from_url(_url, decode_responses=False)
                logger.info("Embedding cache: Redis (%s)", _url)
            except Exception as e:
                logger.warning("Redis embedding cache unavailable (%s), falling back to LRU", e)

        # In-process LRU fallback
        self._cache: dict = {}
        self._cache_size = cache_size
        if self._redis is None:
            logger.info("Embedding cache: in-process LRU (size=%d)", cache_size)

    def _cache_key(self, text: str) -> str:
        h = hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
        return f"{_REDIS_KEY_PREFIX}{h}"

    async def _cache_get(self, text: str) -> Optional[List[float]]:
        if self._redis is not None:
            try:
                val = await self._redis.get(self._cache_key(text))
                if val:
                    return json.loads(val)
            except Exception as e:
                logger.debug("Redis cache get failed: %s", e)
        return self._cache.get(text)

    async def _cache_set(self, text: str, embedding: List[float]) -> None:
        if self._redis is not None:
            try:
                await self._redis.set(
                    self._cache_key(text), json.dumps(embedding), ex=_REDIS_EMB_TTL
                )
                return
            except Exception as e:
                logger.debug("Redis cache set failed: %s", e)
        # LRU eviction fallback
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = embedding

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text, with cache and retry."""
        clean = text.replace("\n", " ")

        cached = await self._cache_get(clean)
        if cached is not None:
            return cached

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.embeddings.create(
                    input=[clean],
                    model=self.model,
                )
                embedding = response.data[0].embedding
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"OpenAI embedding failed (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {e}"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"OpenAI embedding failed after {max_retries} attempts: {e}")
                    raise EmbeddingError(f"OpenAI embedding error: {str(e)}")

        await self._cache_set(clean, embedding)
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
