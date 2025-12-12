import asyncio
from unittest.mock import AsyncMock
import pytest


def test_embed_text_cache_hit_skips_api():
    """Second call with same text must not call the OpenAI API."""
    from agent.storage.embedding_service import OpenAIEmbeddingService

    svc = OpenAIEmbeddingService.__new__(OpenAIEmbeddingService)
    svc.api_key = "test"
    svc.model = "text-embedding-3-small"
    svc.dimension = 1536
    svc._cache = {}
    svc._cache_size = 1024

    fake_embedding = [0.1] * 1536
    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = AsyncMock(
        data=[AsyncMock(embedding=fake_embedding)]
    )
    svc.client = mock_client

    result1 = asyncio.get_event_loop().run_until_complete(svc.embed_text("hello world"))
    result2 = asyncio.get_event_loop().run_until_complete(svc.embed_text("hello world"))

    assert result1 == fake_embedding
    assert result2 == fake_embedding
    assert mock_client.embeddings.create.call_count == 1  # only called once


def test_embed_text_cache_miss_calls_api():
    """Different texts each trigger an API call."""
    from agent.storage.embedding_service import OpenAIEmbeddingService

    svc = OpenAIEmbeddingService.__new__(OpenAIEmbeddingService)
    svc.api_key = "test"
    svc.model = "text-embedding-3-small"
    svc.dimension = 1536
    svc._cache = {}
    svc._cache_size = 1024

    fake_a = [0.1] * 1536
    fake_b = [0.2] * 1536
    responses = [
        AsyncMock(data=[AsyncMock(embedding=fake_a)]),
        AsyncMock(data=[AsyncMock(embedding=fake_b)]),
    ]
    mock_client = AsyncMock()
    mock_client.embeddings.create.side_effect = responses
    svc.client = mock_client

    r1 = asyncio.get_event_loop().run_until_complete(svc.embed_text("text A"))
    r2 = asyncio.get_event_loop().run_until_complete(svc.embed_text("text B"))

    assert r1 == fake_a
    assert r2 == fake_b
    assert mock_client.embeddings.create.call_count == 2


def test_embed_text_cache_eviction():
    """When cache is full, the oldest entry is evicted."""
    from agent.storage.embedding_service import OpenAIEmbeddingService

    svc = OpenAIEmbeddingService.__new__(OpenAIEmbeddingService)
    svc.api_key = "test"
    svc.model = "text-embedding-3-small"
    svc.dimension = 1536
    svc._cache = {}
    svc._cache_size = 2  # tiny cache

    def make_response(val):
        return AsyncMock(data=[AsyncMock(embedding=[val] * 1536)])

    mock_client = AsyncMock()
    mock_client.embeddings.create.side_effect = [
        make_response(0.1),  # "a"
        make_response(0.2),  # "b"
        make_response(0.3),  # "c" — evicts "a"
        make_response(0.4),  # "a" again — must call API (was evicted)
    ]
    svc.client = mock_client

    asyncio.get_event_loop().run_until_complete(svc.embed_text("a"))
    asyncio.get_event_loop().run_until_complete(svc.embed_text("b"))
    asyncio.get_event_loop().run_until_complete(svc.embed_text("c"))  # evicts "a"
    asyncio.get_event_loop().run_until_complete(svc.embed_text("a"))  # cache miss

    assert mock_client.embeddings.create.call_count == 4
