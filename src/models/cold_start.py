"""Cold Start Handling.

Cold start is the #1 practical challenge in production recommender systems.
Two variants:
1. New users: no purchase history → can't do collaborative filtering
2. New items: no interaction data → invisible to CF models

This module implements fallback strategies for both cases.
"""

import pandas as pd
import numpy as np
from loguru import logger


class ColdStartHandler:
    def __init__(
        self,
        popularity_generator,
        content_generator,
        min_interactions: int = 3,
    ):
        """
        Args:
            popularity_generator: PopularityCandidateGenerator instance
            content_generator: ContentCandidateGenerator instance
            min_interactions: Minimum purchases to consider a user "warm"
        """
        self.popularity = popularity_generator
        self.content = content_generator
        self.min_interactions = min_interactions
        self.user_interaction_counts: dict[str, int] = {}
        self.item_interaction_counts: dict[str, int] = {}

    def fit(self, transactions: pd.DataFrame):
        """Compute interaction counts for cold-start detection."""
        self.user_interaction_counts = (
            transactions.groupby("customer_id").size().to_dict()
        )
        self.item_interaction_counts = (
            transactions.groupby("article_id").size().to_dict()
        )
        n_cold_users = sum(
            1 for c in self.user_interaction_counts.values()
            if c < self.min_interactions
        )
        logger.info(
            f"Cold start: {n_cold_users} cold users "
            f"(< {self.min_interactions} interactions)"
        )

    def is_cold_user(self, customer_id: str) -> bool:
        count = self.user_interaction_counts.get(customer_id, 0)
        return count < self.min_interactions

    def is_cold_item(self, article_id: str) -> bool:
        return self.item_interaction_counts.get(article_id, 0) == 0

    def get_new_user_recommendations(
        self,
        age: float | None = None,
        n: int = 12,
    ) -> list[tuple[str, float]]:
        """Recommendations for a brand new user.

        Strategy: demographic-segment popularity.
        If age is available, use age-group specific trends.
        Otherwise, fall back to global popularity.
        """
        age_group = None
        if age is not None:
            if age <= 25:
                age_group = "18-25"
            elif age <= 35:
                age_group = "26-35"
            elif age <= 45:
                age_group = "36-45"
            elif age <= 55:
                age_group = "46-55"
            else:
                age_group = "55+"

        return self.popularity.generate_candidates(
            n_candidates=n, age_group=age_group
        )

    def get_cold_user_recommendations(
        self,
        customer_id: str,
        purchase_history: list[str],
        age: float | None = None,
        n: int = 12,
    ) -> list[tuple[str, float]]:
        """Recommendations for users with few interactions.

        Strategy: blend content-based (from sparse history) with popularity.
        As the user accumulates more interactions, content-based signal
        gets stronger and eventually CF takes over.
        """
        n_interactions = len(purchase_history)

        if n_interactions == 0:
            return self.get_new_user_recommendations(age=age, n=n)

        # Blend content-based and popularity based on interaction count
        # More interactions → more content-based weight
        content_weight = min(n_interactions / self.min_interactions, 0.8)
        pop_weight = 1.0 - content_weight

        n_content = max(1, int(n * content_weight))
        n_pop = n - n_content

        content_recs = self.content.generate_candidates_for_user(
            purchase_history, n_candidates=n_content
        )
        pop_recs = self.popularity.generate_candidates(
            n_candidates=n_pop, age_group=self._age_to_group(age)
        )

        # Merge, removing duplicates
        seen = set()
        merged = []
        for item_id, score in content_recs + pop_recs:
            if item_id not in seen:
                merged.append((item_id, score))
                seen.add(item_id)
        return merged[:n]

    def handle_new_items(
        self,
        new_article_ids: list[str],
        articles: pd.DataFrame,
        n_similar: int = 5,
    ) -> dict[str, list[tuple[str, float]]]:
        """Find similar existing items for new catalog items.

        New items have no interaction data, so we rely on content similarity
        to "borrow" collaborative signals from similar existing items.
        """
        new_item_neighbors = {}
        for aid in new_article_ids:
            similar = self.content.get_similar_items(aid, n=n_similar)
            new_item_neighbors[aid] = similar
        return new_item_neighbors

    @staticmethod
    def _age_to_group(age: float | None) -> str | None:
        if age is None:
            return None
        if age <= 25:
            return "18-25"
        elif age <= 35:
            return "26-35"
        elif age <= 45:
            return "36-45"
        elif age <= 55:
            return "46-55"
        return "55+"
