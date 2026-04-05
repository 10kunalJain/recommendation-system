"""Tests for candidate generation models."""

import pytest
import numpy as np
import pandas as pd


class TestALSGenerator:
    """ALS candidate generator must produce valid candidates."""

    def test_generates_correct_count(self, train_val_test_split, user_item_mappings):
        from src.data.loader import build_interaction_matrix
        from src.candidates.als_generator import ALSCandidateGenerator

        train = train_val_test_split[0]
        u2i, _, i2i, _ = user_item_mappings
        matrix = build_interaction_matrix(train, u2i, i2i)

        gen = ALSCandidateGenerator()
        gen.fit(matrix)
        candidates = gen.generate_candidates(0, n_candidates=5)

        assert len(candidates) <= 5
        assert all(isinstance(c, tuple) and len(c) == 2 for c in candidates)

    def test_scores_are_finite(self, train_val_test_split, user_item_mappings):
        from src.data.loader import build_interaction_matrix
        from src.candidates.als_generator import ALSCandidateGenerator

        train = train_val_test_split[0]
        u2i, _, i2i, _ = user_item_mappings
        matrix = build_interaction_matrix(train, u2i, i2i)

        gen = ALSCandidateGenerator()
        gen.fit(matrix)
        candidates = gen.generate_candidates(0, n_candidates=5)

        for idx, score in candidates:
            assert np.isfinite(score)

    def test_similar_items_returns_results(self, train_val_test_split, user_item_mappings):
        from src.data.loader import build_interaction_matrix
        from src.candidates.als_generator import ALSCandidateGenerator

        train = train_val_test_split[0]
        u2i, _, i2i, _ = user_item_mappings
        matrix = build_interaction_matrix(train, u2i, i2i)

        gen = ALSCandidateGenerator()
        gen.fit(matrix)
        similar = gen.get_similar_items(0, n=3)

        assert len(similar) <= 3


class TestContentGenerator:
    """Content-based generator must work with article metadata."""

    def test_fit_and_retrieve(self, sample_articles):
        from src.candidates.content_generator import ContentCandidateGenerator

        gen = ContentCandidateGenerator()
        gen.fit(sample_articles)

        similar = gen.get_similar_items("001", n=3)
        assert len(similar) <= 3
        assert all(aid != "001" for aid, _ in similar)

    def test_user_candidates_exclude_history(self, sample_articles):
        from src.candidates.content_generator import ContentCandidateGenerator

        gen = ContentCandidateGenerator()
        gen.fit(sample_articles)

        history = ["001", "002"]
        candidates = gen.generate_candidates_for_user(history, n_candidates=5)

        recommended_ids = {aid for aid, _ in candidates}
        assert "001" not in recommended_ids
        assert "002" not in recommended_ids

    def test_unknown_article_returns_empty(self, sample_articles):
        from src.candidates.content_generator import ContentCandidateGenerator

        gen = ContentCandidateGenerator()
        gen.fit(sample_articles)

        similar = gen.get_similar_items("NONEXISTENT")
        assert similar == []


class TestPopularityGenerator:
    """Popularity generator must handle time decay and segments."""

    def test_returns_ranked_items(self, train_val_test_split, sample_customers):
        from src.candidates.popularity_generator import PopularityCandidateGenerator

        train = train_val_test_split[0]
        ref_date = train["t_dat"].max()

        gen = PopularityCandidateGenerator()
        gen.fit(train, sample_customers, ref_date)

        candidates = gen.generate_candidates(n_candidates=5)
        assert len(candidates) == 5

        # Scores should be descending
        scores = [s for _, s in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_segment_popularity(self, train_val_test_split, sample_customers):
        from src.candidates.popularity_generator import PopularityCandidateGenerator

        train = train_val_test_split[0]
        ref_date = train["t_dat"].max()

        gen = PopularityCandidateGenerator()
        gen.fit(train, sample_customers, ref_date)

        # Should return results even with age group filter
        candidates = gen.generate_candidates(n_candidates=3, age_group="26-35")
        assert len(candidates) > 0


class TestCandidateFusion:
    """Fusion must correctly merge multiple sources."""

    def test_fused_output_has_required_columns(self):
        from src.candidates.fusion import CandidateFusion

        fusion = CandidateFusion()
        result = fusion.fuse(
            als_candidates=[("001", 0.9), ("002", 0.8)],
            content_candidates=[("003", 0.7), ("001", 0.6)],
            popularity_candidates=[("004", 0.5)],
        )

        assert "article_id" in result.columns
        assert "fused_score" in result.columns
        assert "n_sources" in result.columns

    def test_multi_source_items_scored_higher(self):
        from src.candidates.fusion import CandidateFusion

        fusion = CandidateFusion()
        result = fusion.fuse(
            als_candidates=[("001", 0.9)],
            content_candidates=[("001", 0.8)],
            popularity_candidates=[("002", 0.5)],
        )

        score_001 = result[result["article_id"] == "001"]["fused_score"].values[0]
        score_002 = result[result["article_id"] == "002"]["fused_score"].values[0]
        assert score_001 > score_002  # item from 2 sources beats 1 source
