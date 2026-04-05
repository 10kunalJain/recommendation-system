"""Tests for cold start handling."""

import pytest
import pandas as pd


class TestColdStartHandler:
    """Cold start must detect and serve cold users."""

    def test_detects_cold_users(self, train_val_test_split, sample_articles, sample_customers):
        from src.candidates.popularity_generator import PopularityCandidateGenerator
        from src.candidates.content_generator import ContentCandidateGenerator
        from src.models.cold_start import ColdStartHandler

        train = train_val_test_split[0]
        ref_date = train["t_dat"].max()

        pop = PopularityCandidateGenerator()
        pop.fit(train, sample_customers, ref_date)
        content = ContentCandidateGenerator()
        content.fit(sample_articles)

        handler = ColdStartHandler(pop, content, min_interactions=3)
        handler.fit(train)

        # A user not in training data should be cold
        assert handler.is_cold_user("BRAND_NEW_USER")

    def test_new_user_gets_recommendations(self, train_val_test_split, sample_articles, sample_customers):
        from src.candidates.popularity_generator import PopularityCandidateGenerator
        from src.candidates.content_generator import ContentCandidateGenerator
        from src.models.cold_start import ColdStartHandler

        train = train_val_test_split[0]
        ref_date = train["t_dat"].max()

        pop = PopularityCandidateGenerator()
        pop.fit(train, sample_customers, ref_date)
        content = ContentCandidateGenerator()
        content.fit(sample_articles)

        handler = ColdStartHandler(pop, content, min_interactions=3)
        handler.fit(train)

        recs = handler.get_new_user_recommendations(age=30, n=5)
        assert len(recs) > 0
        assert len(recs) <= 5


class TestDiversity:
    """Diversity optimization must limit category repetition."""

    def test_category_cap_enforced(self):
        from src.models.diversity import category_diversification

        candidates = [
            ("a1", 0.9), ("a2", 0.85), ("a3", 0.8), ("a4", 0.75),
            ("b1", 0.7), ("c1", 0.65), ("a5", 0.6),
        ]
        categories = {
            "a1": "T-shirt", "a2": "T-shirt", "a3": "T-shirt",
            "a4": "T-shirt", "a5": "T-shirt",
            "b1": "Trousers", "c1": "Dress",
        }

        result = category_diversification(
            candidates, categories, max_per_category=2, top_k=5
        )

        # Count T-shirts in result
        tshirt_count = sum(1 for aid, _ in result if categories.get(aid) == "T-shirt")
        assert tshirt_count <= 2
