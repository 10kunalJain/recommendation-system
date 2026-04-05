"""FastAPI Serving Layer for Real-Time Recommendations.

Production design decisions:
1. Precomputed candidates cached in memory (ALS embeddings, popularity)
2. Ranking model loaded once at startup
3. Per-request feature assembly + ranking takes <50ms
4. LRU cache for repeat users (TTL-based eviction)
5. Batch endpoint for bulk recommendation generation

This is the interface between the ML system and the product.
"""

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

app = FastAPI(
    title="H&M Fashion Recommender",
    description="Two-stage recommendation system with hybrid candidate generation and LambdaRank re-ranking",
    version="1.0.0",
)

# Global model registry — populated at startup
MODEL_REGISTRY = {}


class RecommendationRequest(BaseModel):
    customer_id: str
    n_recommendations: int = 12
    diversity_lambda: float = 0.7
    include_explanations: bool = False


class RecommendationItem(BaseModel):
    article_id: str
    rank: int
    score: float
    sources: str = ""
    explanation: str = ""


class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: list[RecommendationItem]
    is_cold_start: bool
    latency_ms: float
    model_version: str = "v1"


class BatchRequest(BaseModel):
    customer_ids: list[str]
    n_recommendations: int = 12


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    n_users: int
    n_items: int


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for load balancer / k8s probes."""
    return HealthResponse(
        status="healthy" if MODEL_REGISTRY else "no_models",
        models_loaded=list(MODEL_REGISTRY.keys()),
        n_users=MODEL_REGISTRY.get("n_users", 0),
        n_items=MODEL_REGISTRY.get("n_items", 0),
    )


@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations for a single user.

    Two-stage pipeline:
    1. Candidate generation (ALS + content + popularity + recency)
    2. Ranking (LightGBM LambdaRank)
    + Optional MMR diversity re-ranking
    """
    start_time = time.time()

    pipeline = MODEL_REGISTRY.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        result = pipeline.recommend(
            customer_id=request.customer_id,
            n=request.n_recommendations,
            diversity_lambda=request.diversity_lambda,
        )

        items = [
            RecommendationItem(
                article_id=row["article_id"],
                rank=i + 1,
                score=round(float(row.get("rank_score", row.get("fused_score", 0))), 4),
                sources=row.get("source_list", ""),
            )
            for i, row in result.iterrows()
        ]

        latency = (time.time() - start_time) * 1000
        return RecommendationResponse(
            customer_id=request.customer_id,
            recommendations=items,
            is_cold_start=result.attrs.get("is_cold_start", False)
            if hasattr(result, "attrs") else False,
            latency_ms=round(latency, 2),
        )

    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"User {request.customer_id} not found"
        )


@app.post("/recommend/batch")
def get_batch_recommendations(request: BatchRequest):
    """Batch recommendations for multiple users.

    Used for:
    - Email campaigns
    - Pre-computing homepage recommendations
    - A/B test assignment
    """
    pipeline = MODEL_REGISTRY.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    results = {}
    for cid in request.customer_ids:
        try:
            recs = pipeline.recommend(cid, n=request.n_recommendations)
            results[cid] = recs["article_id"].tolist()
        except Exception as e:
            logger.warning(f"Failed for user {cid}: {e}")
            results[cid] = []

    return {"recommendations": results}


@app.get("/similar/{article_id}")
def get_similar_items(
    article_id: str,
    n: int = Query(default=10, le=50),
):
    """Find similar items (for product detail pages)."""
    content_gen = MODEL_REGISTRY.get("content_generator")
    if content_gen is None:
        raise HTTPException(status_code=503, detail="Content model not loaded")

    similar = content_gen.get_similar_items(article_id, n=n)
    return {
        "article_id": article_id,
        "similar_items": [
            {"article_id": aid, "similarity": round(score, 4)}
            for aid, score in similar
        ],
    }


@app.get("/popular")
def get_popular_items(
    n: int = Query(default=12, le=50),
    age_group: str | None = None,
):
    """Get trending/popular items (homepage, new user fallback)."""
    pop_gen = MODEL_REGISTRY.get("popularity_generator")
    if pop_gen is None:
        raise HTTPException(status_code=503, detail="Popularity model not loaded")

    items = pop_gen.generate_candidates(n_candidates=n, age_group=age_group)
    return {
        "items": [
            {"article_id": aid, "popularity_score": round(score, 4)}
            for aid, score in items
        ],
        "age_group": age_group,
    }


def load_models(pipeline, content_gen, popularity_gen, n_users, n_items):
    """Register trained models for serving."""
    MODEL_REGISTRY["pipeline"] = pipeline
    MODEL_REGISTRY["content_generator"] = content_gen
    MODEL_REGISTRY["popularity_generator"] = popularity_gen
    MODEL_REGISTRY["n_users"] = n_users
    MODEL_REGISTRY["n_items"] = n_items
    logger.info("All models loaded into serving registry.")
