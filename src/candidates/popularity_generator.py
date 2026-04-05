"""Popularity-Based Candidate Generation with Time Decay.

This is NOT the final recommender — it serves as:
1. Cold-start fallback for new users with no history
2. One signal in the candidate pool (trending items)
3. Baseline for evaluating personalized models

Uses exponential time decay so recent trends outweigh stale popularity.
"""

import numpy as np
import pandas as pd
from loguru import logger
from src.utils.config import CONFIG


class PopularityCandidateGenerator:
    def __init__(self):
        pop_config = CONFIG["candidates"]["popularity"]
        self.decay_days = pop_config["time_decay_days"]
        self.top_k = pop_config["top_k"]
        self.popularity_scores: dict[str, float] = {}
        self.segment_popularity: dict[str, dict[str, float]] = {}

    def fit(
        self,
        transactions: pd.DataFrame,
        customers: pd.DataFrame,
        reference_date: pd.Timestamp,
    ):
        """Compute time-decayed popularity scores.

        Also computes per-demographic-segment popularity for
        better cold-start recommendations.
        """
        logger.info("Computing popularity scores with time decay...")
        txn = transactions.copy()

        # Exponential time decay: recent purchases count more
        days_ago = (reference_date - txn["t_dat"]).dt.days
        txn["decay_weight"] = np.exp(-days_ago / self.decay_days)

        # Global popularity
        global_pop = (
            txn.groupby("article_id")["decay_weight"]
            .sum()
            .sort_values(ascending=False)
        )
        max_score = global_pop.max() if len(global_pop) > 0 else 1.0
        self.popularity_scores = (global_pop / max_score).to_dict()

        # Per-age-group popularity (for cold-start personalization)
        txn_with_age = txn.merge(customers, on="customer_id", how="left")
        txn_with_age["age_group"] = pd.cut(
            txn_with_age["age"].fillna(35),
            bins=[0, 25, 35, 45, 55, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "55+"],
        )

        for group in txn_with_age["age_group"].dropna().unique():
            segment_txn = txn_with_age[txn_with_age["age_group"] == group]
            seg_pop = (
                segment_txn.groupby("article_id")["decay_weight"]
                .sum()
                .sort_values(ascending=False)
            )
            seg_max = seg_pop.max() if len(seg_pop) > 0 else 1.0
            self.segment_popularity[str(group)] = (seg_pop / seg_max).to_dict()

        logger.info(
            f"Popularity: {len(self.popularity_scores)} items scored, "
            f"{len(self.segment_popularity)} segments"
        )

    def generate_candidates(
        self,
        n_candidates: int = 100,
        age_group: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return top popular items, optionally personalized by segment."""
        if age_group and age_group in self.segment_popularity:
            scores = self.segment_popularity[age_group]
        else:
            scores = self.popularity_scores

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_candidates]

    def get_score(self, article_id: str) -> float:
        return self.popularity_scores.get(article_id, 0.0)
