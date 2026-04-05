"""Tests for the ranking model."""

import pytest
import numpy as np
import pandas as pd


class TestLGBMRanker:
    """LambdaRank model must train and predict correctly."""

    def test_train_and_predict(self):
        from src.ranking.ranker import LGBMRanker, FEATURE_COLUMNS

        ranker = LGBMRanker()
        n_samples = 200
        n_groups = 10
        group_size = n_samples // n_groups

        # Synthetic training data
        df = pd.DataFrame(
            np.random.randn(n_samples, len(FEATURE_COLUMNS)),
            columns=FEATURE_COLUMNS,
        )
        labels = np.random.randint(0, 2, size=n_samples)
        groups = np.array([group_size] * n_groups)

        ranker.fit(df, labels, groups)
        assert ranker.is_fitted

        scores = ranker.predict(df)
        assert len(scores) == n_samples
        assert all(np.isfinite(scores))

    def test_rank_candidates_returns_top_k(self):
        from src.ranking.ranker import LGBMRanker, FEATURE_COLUMNS

        ranker = LGBMRanker()
        n_samples = 200
        n_groups = 10
        group_size = n_samples // n_groups

        df = pd.DataFrame(
            np.random.randn(n_samples, len(FEATURE_COLUMNS)),
            columns=FEATURE_COLUMNS,
        )
        df["customer_id"] = np.repeat([f"u{i}" for i in range(n_groups)], group_size)
        df["article_id"] = [f"a{i}" for i in range(n_samples)]
        labels = np.random.randint(0, 2, size=n_samples)
        groups = np.array([group_size] * n_groups)

        ranker.fit(df, labels, groups)
        result = ranker.rank_candidates(df, top_k=5)

        # Each user should have at most 5 recommendations
        for uid in result["customer_id"].unique():
            user_recs = result[result["customer_id"] == uid]
            assert len(user_recs) <= 5

    def test_feature_importance_has_all_features(self):
        from src.ranking.ranker import LGBMRanker, FEATURE_COLUMNS

        ranker = LGBMRanker()
        n_samples = 200
        df = pd.DataFrame(
            np.random.randn(n_samples, len(FEATURE_COLUMNS)),
            columns=FEATURE_COLUMNS,
        )
        labels = np.random.randint(0, 2, size=n_samples)
        groups = np.array([20] * 10)

        ranker.fit(df, labels, groups)
        importance = ranker.get_feature_importance()

        assert len(importance) == len(FEATURE_COLUMNS)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
