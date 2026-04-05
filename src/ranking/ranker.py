"""LightGBM Learning-to-Rank Model.

The ranking stage takes ~100 candidates per user from the retrieval stage
and re-ranks them using rich features. This is where personalization
really happens — the retrieval stage casts a wide net, the ranker sharpens it.

Why LambdaRank (LightGBM)?
- Directly optimizes ranking metrics (NDCG), not pointwise loss
- Handles mixed feature types (categorical + numerical) natively
- Fast inference (<10ms per user) — critical for real-time serving
- Interpretable via feature importance (debugging poor recs)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger
from src.utils.config import CONFIG


FEATURE_COLUMNS = [
    # User features
    "purchase_count", "unique_items", "purchase_recency_days",
    "purchase_frequency", "color_diversity", "age",
    # Item features
    "total_purchases", "unique_buyers", "repurchase_rate",
    "days_since_first_purchase", "section_popularity_rank",
    # Interaction features
    "user_section_affinity", "user_color_affinity",
    "user_product_type_affinity",
    # Candidate generation scores (model stacking)
    "als_score", "content_score", "popularity_score",
    "two_tower_score", "fused_score", "n_sources",
]


class LGBMRanker:
    def __init__(self):
        lgbm_config = CONFIG["ranking"]["lgbm"]
        self.model = lgb.LGBMRanker(
            objective=lgbm_config["objective"],
            metric=lgbm_config["metric"],
            n_estimators=lgbm_config["n_estimators"],
            num_leaves=lgbm_config["num_leaves"],
            learning_rate=lgbm_config["learning_rate"],
            feature_fraction=lgbm_config["feature_fraction"],
            min_child_samples=lgbm_config["min_child_samples"],
            label_gain=list(range(50)),  # for NDCG calculation
            random_state=42,
            verbose=-1,
        )
        self.feature_columns = FEATURE_COLUMNS
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and validate feature columns."""
        available = [c for c in self.feature_columns if c in df.columns]
        missing = set(self.feature_columns) - set(available)
        if missing:
            logger.warning(f"Missing features (will be 0): {missing}")
            for col in missing:
                df[col] = 0.0
        return df[self.feature_columns].copy()

    def fit(
        self,
        train_df: pd.DataFrame,
        train_labels: np.ndarray,
        train_groups: np.ndarray,
        val_df: pd.DataFrame | None = None,
        val_labels: np.ndarray | None = None,
        val_groups: np.ndarray | None = None,
    ):
        """Train the ranking model.

        Args:
            train_df: Feature matrix with candidate pairs
            train_labels: Relevance labels (1=purchased, 0=not)
            train_groups: Group sizes (number of candidates per user query)
            val_df/val_labels/val_groups: Validation set for early stopping
        """
        X_train = self._prepare_features(train_df)
        logger.info(
            f"Training ranker: {len(X_train)} samples, "
            f"{len(train_groups)} queries, {len(self.feature_columns)} features"
        )

        callbacks = [lgb.log_evaluation(50)]
        eval_set = []
        eval_group = []
        eval_names = []

        if val_df is not None:
            X_val = self._prepare_features(val_df)
            eval_set = [(X_val, val_labels)]
            eval_group = [val_groups]
            eval_names = ["val"]
            callbacks.append(lgb.early_stopping(30))

        self.model.fit(
            X_train,
            train_labels,
            group=train_groups,
            eval_set=eval_set,
            eval_group=eval_group,
            eval_names=eval_names,
            callbacks=callbacks,
        )
        self.is_fitted = True
        logger.info("Ranker training complete.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Score candidate items for ranking."""
        X = self._prepare_features(df)
        return self.model.predict(X)

    def rank_candidates(
        self, candidates_df: pd.DataFrame, top_k: int = 12
    ) -> pd.DataFrame:
        """Re-rank candidates and return top-K per user.

        This is the main inference method — takes fused candidates
        with features and returns final recommendations.
        """
        df = candidates_df.copy()
        df["rank_score"] = self.predict(df)

        # Rank within each user's candidate set
        df["rank"] = df.groupby("customer_id")["rank_score"].rank(
            ascending=False, method="first"
        )
        result = df[df["rank"] <= top_k].sort_values(
            ["customer_id", "rank"]
        )
        return result

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance for model interpretation."""
        importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return importance

    def save(self, path: str):
        self.model.booster_.save_model(path)
        logger.info(f"Ranker saved to {path}")

    def load(self, path: str):
        self.model = lgb.Booster(model_file=path)
        self.is_fitted = True
        logger.info(f"Ranker loaded from {path}")
