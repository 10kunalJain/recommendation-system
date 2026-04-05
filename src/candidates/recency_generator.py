"""Recency-Based Candidate Generation.

In fashion, recency is king — a user who bought a winter coat yesterday
is more likely to buy winter accessories than summer dresses, regardless
of their overall purchase history.

This generator surfaces items similar to the user's most recent purchases,
weighted by time decay. It captures short-term intent that ALS
(which treats all history equally) misses.
"""

import pandas as pd
import numpy as np
from loguru import logger
from src.utils.config import CONFIG


class RecencyCandidateGenerator:
    def __init__(self):
        recency_config = CONFIG["candidates"]["recency"]
        self.lookback_days = recency_config["lookback_days"]
        self.weight = recency_config["weight"]
        self.item_cooccurrence: dict[str, dict[str, float]] = {}

    def fit(self, transactions: pd.DataFrame, reference_date: pd.Timestamp):
        """Build item co-occurrence from recent transactions.

        Items bought by the same user within the lookback window
        are considered co-occurring. More recent co-occurrences get
        higher weight.
        """
        logger.info(f"Building recency model (lookback={self.lookback_days}d)...")
        cutoff = reference_date - pd.Timedelta(days=self.lookback_days * 2)
        recent = transactions[transactions["t_dat"] >= cutoff].copy()

        days_ago = (reference_date - recent["t_dat"]).dt.days
        recent["recency_weight"] = np.exp(-days_ago / self.lookback_days)

        # Build weighted co-occurrence
        user_items = (
            recent.groupby("customer_id")
            .apply(
                lambda g: list(zip(g["article_id"], g["recency_weight"])),
                include_groups=False,
            )
        )

        cooccur: dict[str, dict[str, float]] = {}
        for items_weights in user_items:
            for i, (item_a, w_a) in enumerate(items_weights):
                if item_a not in cooccur:
                    cooccur[item_a] = {}
                for j, (item_b, w_b) in enumerate(items_weights):
                    if i != j:
                        combined_weight = w_a * w_b
                        cooccur[item_a][item_b] = (
                            cooccur[item_a].get(item_b, 0.0) + combined_weight
                        )

        self.item_cooccurrence = cooccur
        logger.info(f"Recency co-occurrence: {len(cooccur)} source items")

    def generate_candidates(
        self,
        user_recent_items: list[tuple[str, float]],
        n_candidates: int = 50,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Generate candidates from user's recent purchases.

        Args:
            user_recent_items: [(article_id, recency_weight)] sorted by recency
            n_candidates: Number of candidates to return
            exclude: Items to exclude (already purchased)
        """
        if exclude is None:
            exclude = set()

        scores: dict[str, float] = {}
        for item_id, recency_weight in user_recent_items:
            co_items = self.item_cooccurrence.get(item_id, {})
            for co_item, co_score in co_items.items():
                if co_item not in exclude:
                    scores[co_item] = (
                        scores.get(co_item, 0.0)
                        + co_score * recency_weight * self.weight
                    )

        sorted_candidates = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_candidates[:n_candidates]
