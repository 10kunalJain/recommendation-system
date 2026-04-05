"""Tests for feature engineering."""

import pytest
import pandas as pd
import numpy as np


class TestUserFeatures:
    """User features must be complete and valid."""

    def test_all_users_present(self, train_val_test_split, sample_customers, sample_articles):
        from src.features.engineer import build_user_features
        train = train_val_test_split[0]
        ref_date = train["t_dat"].max()
        uf = build_user_features(train, sample_customers, sample_articles, ref_date)

        train_users = set(train["customer_id"].unique())
        feature_users = set(uf["customer_id"].unique())
        assert train_users == feature_users

    def test_no_negative_purchase_count(self, train_val_test_split, sample_customers, sample_articles):
        from src.features.engineer import build_user_features
        train = train_val_test_split[0]
        ref_date = train["t_dat"].max()
        uf = build_user_features(train, sample_customers, sample_articles, ref_date)

        assert (uf["purchase_count"] > 0).all()

    def test_diversity_bounded_zero_one(self, train_val_test_split, sample_customers, sample_articles):
        from src.features.engineer import build_user_features
        train = train_val_test_split[0]
        ref_date = train["t_dat"].max()
        uf = build_user_features(train, sample_customers, sample_articles, ref_date)

        assert (uf["color_diversity"] >= 0).all()
        assert (uf["color_diversity"] <= 1).all()

    def test_recency_is_non_negative(self, train_val_test_split, sample_customers, sample_articles):
        from src.features.engineer import build_user_features
        train = train_val_test_split[0]
        ref_date = train["t_dat"].max()
        uf = build_user_features(train, sample_customers, sample_articles, ref_date)

        assert (uf["purchase_recency_days"] >= 0).all()


class TestItemFeatures:
    """Item features must be valid."""

    def test_repurchase_rate_bounded(self, train_val_test_split, sample_articles):
        from src.features.engineer import build_item_features
        train = train_val_test_split[0]
        ref_date = train["t_dat"].max()
        itf = build_item_features(train, sample_articles, ref_date)

        assert (itf["repurchase_rate"] >= 0).all()
        assert (itf["repurchase_rate"] <= 1).all()


class TestInteractionFeatures:
    """Interaction features must be valid."""

    def test_affinity_bounded(self, train_val_test_split, sample_articles, sample_customers):
        from src.features.engineer import build_user_features, build_interaction_features
        train = train_val_test_split[0]
        ref_date = train["t_dat"].max()
        uf = build_user_features(train, sample_customers, sample_articles, ref_date)
        interaction = build_interaction_features(train, sample_articles, uf, ref_date)

        section_aff = interaction["user_section_affinity"]
        assert (section_aff["user_section_affinity"] >= 0).all()
        assert (section_aff["user_section_affinity"] <= 1).all()

    def test_covisitation_scores_normalized(self, train_val_test_split, sample_articles, sample_customers):
        from src.features.engineer import build_user_features, build_interaction_features
        train = train_val_test_split[0]
        ref_date = train["t_dat"].max()
        uf = build_user_features(train, sample_customers, sample_articles, ref_date)
        interaction = build_interaction_features(train, sample_articles, uf, ref_date)

        covisit = interaction["covisitation"]
        if covisit:
            max_score = max(covisit.values())
            assert max_score <= 1.0
