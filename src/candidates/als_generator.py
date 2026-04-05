"""ALS-based Candidate Generation (Collaborative Filtering).

Uses implicit feedback matrix factorization to learn latent user/item
embeddings. This is the primary retrieval mechanism — it captures
"users who bought X also bought Y" patterns at scale.

Why ALS over SVD?
- Designed for implicit feedback (no explicit ratings)
- Scales well to millions of interactions
- Handles sparse matrices efficiently
"""

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from loguru import logger
from src.utils.config import CONFIG


class ALSCandidateGenerator:
    def __init__(self):
        als_config = CONFIG["candidates"]["als"]
        self.model = AlternatingLeastSquares(
            factors=als_config["factors"],
            regularization=als_config["regularization"],
            iterations=als_config["iterations"],
            use_gpu=als_config["use_gpu"],
            random_state=42,
        )
        self.user_item_matrix = None
        self.item_user_matrix = None

    def fit(self, interaction_matrix: csr_matrix):
        """Train ALS on the user-item interaction matrix."""
        logger.info(
            f"Training ALS: {interaction_matrix.shape[0]} users, "
            f"{interaction_matrix.shape[1]} items, "
            f"factors={self.model.factors}"
        )
        self.user_item_matrix = interaction_matrix

        # implicit.fit expects user-item matrix
        self.model.fit(self.user_item_matrix)
        logger.info("ALS training complete.")

    def generate_candidates(
        self,
        user_idx: int,
        n_candidates: int = 100,
        filter_already_purchased: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate top-N candidate items for a user.

        Returns list of (item_idx, score) tuples.
        """
        item_ids, scores = self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=n_candidates,
            filter_already_liked_items=filter_already_purchased,
        )
        return list(zip(item_ids.tolist(), scores.tolist()))

    def generate_candidates_batch(
        self,
        user_indices: list[int],
        n_candidates: int = 100,
    ) -> dict[int, list[tuple[int, float]]]:
        """Batch candidate generation for multiple users."""
        results = {}
        user_items = self.user_item_matrix[user_indices]

        item_ids_batch, scores_batch = self.model.recommend(
            user_indices,
            user_items,
            N=n_candidates,
            filter_already_liked_items=True,
        )

        for i, user_idx in enumerate(user_indices):
            results[user_idx] = list(
                zip(item_ids_batch[i].tolist(), scores_batch[i].tolist())
            )
        return results

    def get_similar_items(
        self, item_idx: int, n: int = 20
    ) -> list[tuple[int, float]]:
        """Find similar items using learned embeddings.

        Useful for "similar items" widget and content-based fallback.
        """
        item_ids, scores = self.model.similar_items(item_idx, N=n)
        return list(zip(item_ids.tolist(), scores.tolist()))

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        return self.model.user_factors[user_idx]

    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        return self.model.item_factors[item_idx]

    @property
    def user_factors(self) -> np.ndarray:
        return self.model.user_factors

    @property
    def item_factors(self) -> np.ndarray:
        return self.model.item_factors
