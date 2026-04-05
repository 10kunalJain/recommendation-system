"""Tests for FastAPI endpoint contracts.

Uses FastAPI TestClient with mocked pipeline — no training required.
Validates request/response schemas, error handling, and caching behavior.
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from fastapi.testclient import TestClient

from src.serving.api import app, MODEL_REGISTRY, rec_cache, TTLCache


@pytest.fixture(autouse=True)
def clear_registry():
    """Reset model registry and cache between tests."""
    MODEL_REGISTRY.clear()
    rec_cache._cache.clear()
    rec_cache.hits = 0
    rec_cache.misses = 0
    yield
    MODEL_REGISTRY.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_pipeline():
    """Mock pipeline that returns deterministic recommendations."""
    pipeline = MagicMock()
    result = pd.DataFrame({
        "article_id": ["001", "002", "003"],
        "rank_score": [0.9, 0.8, 0.7],
        "source_list": ["als,content", "als", "popularity"],
    })
    result.attrs = {"is_cold_start": False}
    pipeline.recommend.return_value = result
    return pipeline


@pytest.fixture
def mock_content_gen():
    gen = MagicMock()
    gen.get_similar_items.return_value = [("002", 0.95), ("003", 0.88)]
    return gen


@pytest.fixture
def mock_popularity_gen():
    gen = MagicMock()
    gen.generate_candidates.return_value = [("001", 1.0), ("002", 0.9)]
    return gen


def load_mocks(pipeline, content_gen, popularity_gen):
    MODEL_REGISTRY["pipeline"] = pipeline
    MODEL_REGISTRY["content_generator"] = content_gen
    MODEL_REGISTRY["popularity_generator"] = popularity_gen
    MODEL_REGISTRY["n_users"] = 100
    MODEL_REGISTRY["n_items"] = 50


class TestHealthEndpoint:
    def test_healthy_with_models(self, client, mock_pipeline, mock_content_gen, mock_popularity_gen):
        load_mocks(mock_pipeline, mock_content_gen, mock_popularity_gen)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["n_users"] == 100
        assert data["n_items"] == 50
        assert "cache_stats" in data

    def test_no_models_status(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_models"


class TestRecommendEndpoint:
    def test_valid_request(self, client, mock_pipeline, mock_content_gen, mock_popularity_gen):
        load_mocks(mock_pipeline, mock_content_gen, mock_popularity_gen)
        resp = client.post("/recommend", json={
            "customer_id": "user_1",
            "n_recommendations": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["customer_id"] == "user_1"
        assert len(data["recommendations"]) == 3
        assert data["recommendations"][0]["rank"] == 1
        assert data["recommendations"][0]["article_id"] == "001"
        assert "latency_ms" in data
        assert data["cache_hit"] is False

    def test_cache_hit_on_repeat(self, client, mock_pipeline, mock_content_gen, mock_popularity_gen):
        load_mocks(mock_pipeline, mock_content_gen, mock_popularity_gen)

        # First request — cache miss
        resp1 = client.post("/recommend", json={"customer_id": "user_1"})
        assert resp1.json()["cache_hit"] is False

        # Second request — cache hit
        resp2 = client.post("/recommend", json={"customer_id": "user_1"})
        assert resp2.json()["cache_hit"] is True
        assert resp2.json()["latency_ms"] < resp1.json()["latency_ms"]

    def test_503_when_no_pipeline(self, client):
        resp = client.post("/recommend", json={"customer_id": "user_1"})
        assert resp.status_code == 503

    def test_response_schema(self, client, mock_pipeline, mock_content_gen, mock_popularity_gen):
        load_mocks(mock_pipeline, mock_content_gen, mock_popularity_gen)
        resp = client.post("/recommend", json={"customer_id": "user_1"})
        data = resp.json()

        # Validate response structure
        assert "customer_id" in data
        assert "recommendations" in data
        assert "is_cold_start" in data
        assert "latency_ms" in data
        assert "cache_hit" in data
        assert "model_version" in data

        # Validate item structure
        item = data["recommendations"][0]
        assert "article_id" in item
        assert "rank" in item
        assert "score" in item
        assert "sources" in item


class TestBatchEndpoint:
    def test_batch_returns_all_users(self, client, mock_pipeline, mock_content_gen, mock_popularity_gen):
        load_mocks(mock_pipeline, mock_content_gen, mock_popularity_gen)
        resp = client.post("/recommend/batch", json={
            "customer_ids": ["user_1", "user_2", "user_3"],
            "n_recommendations": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) == 3
        assert "total_latency_ms" in data
        assert "avg_latency_ms" in data
        assert "cache_hits" in data

    def test_503_when_no_pipeline(self, client):
        resp = client.post("/recommend/batch", json={"customer_ids": ["u1"]})
        assert resp.status_code == 503


class TestSimilarEndpoint:
    def test_returns_similar_items(self, client, mock_pipeline, mock_content_gen, mock_popularity_gen):
        load_mocks(mock_pipeline, mock_content_gen, mock_popularity_gen)
        resp = client.get("/similar/001?n=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["article_id"] == "001"
        assert len(data["similar_items"]) == 2
        assert "similarity" in data["similar_items"][0]

    def test_503_when_no_model(self, client):
        resp = client.get("/similar/001")
        assert resp.status_code == 503


class TestPopularEndpoint:
    def test_returns_popular_items(self, client, mock_pipeline, mock_content_gen, mock_popularity_gen):
        load_mocks(mock_pipeline, mock_content_gen, mock_popularity_gen)
        resp = client.get("/popular?n=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        assert "popularity_score" in data["items"][0]

    def test_with_age_group(self, client, mock_pipeline, mock_content_gen, mock_popularity_gen):
        load_mocks(mock_pipeline, mock_content_gen, mock_popularity_gen)
        resp = client.get("/popular?n=2&age_group=26-35")
        assert resp.status_code == 200
        assert resp.json()["age_group"] == "26-35"


class TestTTLCache:
    def test_basic_put_get(self):
        cache = TTLCache(max_size=10, ttl_seconds=60)
        cache.put("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_ttl_expiry(self):
        cache = TTLCache(max_size=10, ttl_seconds=0)  # immediate expiry
        cache.put("k1", "v1")
        import time
        time.sleep(0.01)
        assert cache.get("k1") is None

    def test_max_size_eviction(self):
        cache = TTLCache(max_size=2, ttl_seconds=60)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")  # should evict k1
        assert cache.get("k1") is None
        assert cache.get("k3") == "v3"

    def test_stats(self):
        cache = TTLCache(max_size=10, ttl_seconds=60)
        cache.put("k1", "v1")
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
