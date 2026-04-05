"""Diversity & Novelty Optimization.

A common failure mode in recommender systems is the "filter bubble":
showing users the same type of items they already buy. This module
implements post-ranking diversification using Maximal Marginal Relevance
(MMR) to balance relevance with diversity.

Why this matters in fashion:
- Users want to discover new styles, not just see variations of past purchases
- Showing 12 black t-shirts is technically "relevant" but a bad experience
- Diversity correlates with long-term user engagement and retention
"""

import numpy as np
import pandas as pd
from loguru import logger


def maximal_marginal_relevance(
    candidate_scores: list[tuple[str, float]],
    item_features: dict[str, np.ndarray],
    lambda_param: float = 0.7,
    top_k: int = 12,
) -> list[tuple[str, float]]:
    """MMR-based re-ranking for diversity.

    MMR(i) = λ * Relevance(i) - (1-λ) * max_j∈S Similarity(i, j)

    λ controls the relevance-diversity tradeoff:
    - λ=1.0: pure relevance (no diversity)
    - λ=0.5: balanced
    - λ=0.0: maximum diversity

    Args:
        candidate_scores: [(article_id, relevance_score)]
        item_features: {article_id: feature_vector} for similarity
        lambda_param: Relevance vs diversity tradeoff
        top_k: Number of items to select
    """
    if len(candidate_scores) <= top_k:
        return candidate_scores

    # Normalize relevance scores to [0, 1]
    scores = {aid: score for aid, score in candidate_scores}
    max_score = max(scores.values()) if scores else 1.0
    norm_scores = {aid: s / max_score for aid, s in scores.items()}

    selected: list[tuple[str, float]] = []
    remaining = set(scores.keys())

    for _ in range(top_k):
        best_item = None
        best_mmr = -float("inf")

        for item in remaining:
            relevance = norm_scores[item]

            # Max similarity to already selected items
            if selected and item in item_features:
                max_sim = 0.0
                item_vec = item_features[item]
                for sel_item, _ in selected:
                    if sel_item in item_features:
                        sel_vec = item_features[sel_item]
                        sim = _cosine_sim(item_vec, sel_vec)
                        max_sim = max(max_sim, sim)
            else:
                max_sim = 0.0

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_item = item

        if best_item:
            selected.append((best_item, scores[best_item]))
            remaining.discard(best_item)

    return selected


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def category_diversification(
    candidates: list[tuple[str, float]],
    item_categories: dict[str, str],
    max_per_category: int = 3,
    top_k: int = 12,
) -> list[tuple[str, float]]:
    """Simple category-based diversification.

    Ensures no more than max_per_category items from the same
    product type/section appear in the final list.
    """
    selected = []
    category_counts: dict[str, int] = {}

    for item_id, score in candidates:
        cat = item_categories.get(item_id, "unknown")
        if category_counts.get(cat, 0) < max_per_category:
            selected.append((item_id, score))
            category_counts[cat] = category_counts.get(cat, 0) + 1
        if len(selected) >= top_k:
            break

    return selected


def compute_diversity_metrics(
    recommendations: dict[str, list[str]],
    item_categories: dict[str, str],
) -> dict[str, float]:
    """Measure recommendation diversity across users."""
    intra_list_diversity = []
    category_coverages = []

    all_categories = set(item_categories.values())

    for user_id, items in recommendations.items():
        # Intra-list diversity: fraction of unique categories in the list
        categories = [item_categories.get(i, "unk") for i in items]
        if categories:
            ild = len(set(categories)) / len(categories)
            intra_list_diversity.append(ild)
            category_coverages.append(len(set(categories)))

    return {
        "avg_intra_list_diversity": float(np.mean(intra_list_diversity)) if intra_list_diversity else 0.0,
        "avg_unique_categories_per_user": float(np.mean(category_coverages)) if category_coverages else 0.0,
        "total_category_coverage": len(all_categories),
    }
