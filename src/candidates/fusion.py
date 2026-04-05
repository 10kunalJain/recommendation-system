"""Candidate Fusion: Merges candidates from multiple retrieval sources.

In production two-stage systems (Amazon, YouTube, Netflix), the candidate
generation stage typically runs multiple retrieval models in parallel and
fuses their outputs. This ensures:
- Coverage: no single model's blind spots dominate
- Diversity: different models surface different item types
- Robustness: system degrades gracefully if one model fails

Fusion strategy: weighted rank fusion with source tracking.
"""

import pandas as pd
import numpy as np
from loguru import logger
from src.utils.config import CONFIG


class CandidateFusion:
    def __init__(self):
        self.n_candidates = CONFIG["pipeline"]["candidate_generation"]["num_candidates"]

    def fuse(
        self,
        als_candidates: list[tuple[str, float]],
        content_candidates: list[tuple[str, float]],
        popularity_candidates: list[tuple[str, float]],
        recency_candidates: list[tuple[str, float]] | None = None,
        two_tower_candidates: list[tuple[str, float]] | None = None,
        weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Fuse candidates from multiple sources using reciprocal rank fusion.

        Each source provides (article_id, score) pairs. We compute a fused
        score using reciprocal rank: score = sum(weight / (k + rank)) across
        sources. This is robust to score scale differences between models.

        Args:
            als_candidates: From collaborative filtering
            content_candidates: From content-based similarity
            popularity_candidates: From popularity model
            recency_candidates: From user's recent interactions
            two_tower_candidates: From neural two-tower retrieval
            weights: Source weights
        """
        if weights is None:
            weights = {
                "als": 1.0,
                "content": 0.5,
                "popularity": 0.3,
                "recency": 0.8,
                "two_tower": 0.9,
            }

        k = 60  # smoothing constant for reciprocal rank fusion

        all_scores: dict[str, dict] = {}

        sources = {
            "als": als_candidates,
            "content": content_candidates,
            "popularity": popularity_candidates,
        }
        if recency_candidates:
            sources["recency"] = recency_candidates
        if two_tower_candidates:
            sources["two_tower"] = two_tower_candidates

        for source_name, candidates in sources.items():
            weight = weights.get(source_name, 0.5)
            for rank, (article_id, score) in enumerate(candidates):
                if article_id not in all_scores:
                    all_scores[article_id] = {
                        "fused_score": 0.0,
                        "sources": [],
                        "als_score": 0.0,
                        "content_score": 0.0,
                        "popularity_score": 0.0,
                        "recency_score": 0.0,
                        "two_tower_score": 0.0,
                    }
                # Reciprocal rank fusion
                all_scores[article_id]["fused_score"] += weight / (k + rank)
                all_scores[article_id]["sources"].append(source_name)
                all_scores[article_id][f"{source_name}_score"] = score

        # Convert to DataFrame and sort by fused score
        rows = []
        for article_id, info in all_scores.items():
            rows.append({
                "article_id": article_id,
                "fused_score": info["fused_score"],
                "n_sources": len(info["sources"]),
                "source_list": ",".join(sorted(set(info["sources"]))),
                "als_score": info["als_score"],
                "content_score": info["content_score"],
                "popularity_score": info["popularity_score"],
                "recency_score": info.get("recency_score", 0.0),
                "two_tower_score": info.get("two_tower_score", 0.0),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("fused_score", ascending=False).head(self.n_candidates)
        df = df.reset_index(drop=True)

        logger.debug(
            f"Fused {len(all_scores)} unique candidates → top {len(df)}, "
            f"multi-source items: {(df['n_sources'] > 1).sum()}"
        )
        return df
