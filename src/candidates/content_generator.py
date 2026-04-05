"""Content-Based Candidate Generation.

Uses article metadata (product type, color, section) to find similar items.
This serves two critical purposes:
1. Cold-start: recommend to new users based on demographic preferences
2. Diversity: surface items the collaborative filter might miss

Approach: TF-IDF on concatenated metadata → cosine similarity.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from loguru import logger
from src.utils.config import CONFIG


class ContentCandidateGenerator:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.item_vectors: csr_matrix | None = None
        self.article_ids: list[str] = []
        self.aid_to_idx: dict[str, int] = {}

    def fit(self, articles: pd.DataFrame):
        """Build TF-IDF representations from article metadata."""
        logger.info("Building content-based item representations...")

        # Concatenate metadata fields into a single text representation
        articles = articles.copy()
        articles["content_text"] = (
            articles["product_type_name"].fillna("")
            + " " + articles["colour_group_name"].fillna("")
            + " " + articles["section_name"].fillna("")
        )

        self.article_ids = articles["article_id"].tolist()
        self.aid_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}

        self.item_vectors = self.tfidf.fit_transform(articles["content_text"])
        logger.info(
            f"Content vectors: {self.item_vectors.shape}, "
            f"vocab size: {len(self.tfidf.vocabulary_)}"
        )

    def get_similar_items(
        self, article_id: str, n: int = 50
    ) -> list[tuple[str, float]]:
        """Find content-similar items for a given article."""
        if article_id not in self.aid_to_idx:
            return []

        idx = self.aid_to_idx[article_id]
        item_vec = self.item_vectors[idx]
        similarities = cosine_similarity(item_vec, self.item_vectors).flatten()

        # Exclude self
        similarities[idx] = -1
        top_indices = np.argsort(similarities)[-n:][::-1]

        return [
            (self.article_ids[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0
        ]

    def generate_candidates_for_user(
        self,
        user_history: list[str],
        n_candidates: int = 100,
    ) -> list[tuple[str, float]]:
        """Generate candidates based on a user's purchase history.

        Strategy: find items similar to the user's recent purchases,
        weighted by recency (most recent items get higher weight).
        """
        if not user_history:
            return []

        # Weight recent items more heavily (exponential decay)
        n_history = min(len(user_history), 10)  # use last 10 items
        recent_items = user_history[-n_history:]
        weights = np.exp(np.linspace(-1, 0, n_history))

        # Aggregate similarity scores across history
        candidate_scores: dict[str, float] = {}
        for item_id, weight in zip(recent_items, weights):
            similar = self.get_similar_items(item_id, n=30)
            for sim_id, sim_score in similar:
                if sim_id not in set(user_history):  # exclude already purchased
                    candidate_scores[sim_id] = (
                        candidate_scores.get(sim_id, 0.0) + sim_score * weight
                    )

        # Sort and return top candidates
        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_candidates[:n_candidates]

    def get_item_vector(self, article_id: str) -> np.ndarray | None:
        if article_id not in self.aid_to_idx:
            return None
        return self.item_vectors[self.aid_to_idx[article_id]].toarray().flatten()
