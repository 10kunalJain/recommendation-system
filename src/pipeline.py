"""End-to-End Recommendation Pipeline.

Orchestrates the full two-stage recommendation flow:
  Data → Features → Candidate Generation → Ranking → Diversity → Output

This is the core class that ties all components together.
Separates offline training from online inference.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from src.data.loader import (
    load_raw_data,
    temporal_split,
    build_user_item_mappings,
    build_interaction_matrix,
)
from src.features.engineer import (
    build_user_features,
    build_item_features,
    build_interaction_features,
    assemble_ranking_features,
)
from src.candidates.als_generator import ALSCandidateGenerator
from src.candidates.content_generator import ContentCandidateGenerator
from src.candidates.popularity_generator import PopularityCandidateGenerator
from src.candidates.recency_generator import RecencyCandidateGenerator
from src.candidates.two_tower_generator import TwoTowerCandidateGenerator
from src.candidates.fusion import CandidateFusion
from src.ranking.ranker import LGBMRanker
from src.models.cold_start import ColdStartHandler
from src.models.diversity import maximal_marginal_relevance, category_diversification
from src.models.user_segmentation import UserSegmentation
from src.evaluation.metrics import evaluate_recommendations
from src.utils.config import CONFIG


class RecommendationPipeline:
    def __init__(self):
        # Data
        self.articles: pd.DataFrame | None = None
        self.customers: pd.DataFrame | None = None
        self.train_txn: pd.DataFrame | None = None
        self.val_txn: pd.DataFrame | None = None
        self.test_txn: pd.DataFrame | None = None

        # Mappings
        self.user_to_idx: dict = {}
        self.idx_to_user: dict = {}
        self.item_to_idx: dict = {}
        self.idx_to_item: dict = {}

        # Models
        self.als_generator = ALSCandidateGenerator()
        self.content_generator = ContentCandidateGenerator()
        self.popularity_generator = PopularityCandidateGenerator()
        self.recency_generator = RecencyCandidateGenerator()
        self.two_tower_generator = TwoTowerCandidateGenerator(
            embedding_dim=64, n_epochs=10, batch_size=512
        )
        self.candidate_fusion = CandidateFusion()
        self.ranker = LGBMRanker()
        self.cold_start: ColdStartHandler | None = None
        self.segmentation = UserSegmentation(n_segments=5)

        # Features (cached for inference)
        self.user_features: pd.DataFrame | None = None
        self.item_features: pd.DataFrame | None = None
        self.interaction_features: dict | None = None

        # User history cache for fast lookup
        self._user_history: dict[str, list[str]] = {}
        self._user_recent: dict[str, list[tuple[str, float]]] = {}

    def load_data(self):
        """Step 1: Load and split data."""
        logger.info("=" * 60)
        logger.info("STEP 1: Loading data")
        logger.info("=" * 60)

        self.articles, transactions, self.customers = load_raw_data()
        self.train_txn, self.val_txn, self.test_txn = temporal_split(transactions)

        # Build mappings from training data only (no data leakage)
        self.user_to_idx, self.idx_to_user, self.item_to_idx, self.idx_to_item = (
            build_user_item_mappings(self.train_txn)
        )

        # Cache user histories
        self._build_user_history_cache(self.train_txn)

    def _build_user_history_cache(self, transactions: pd.DataFrame):
        """Cache user purchase histories sorted by date."""
        sorted_txn = transactions.sort_values("t_dat")
        for cid, group in sorted_txn.groupby("customer_id"):
            self._user_history[cid] = group["article_id"].tolist()

        # Recent items with recency weights
        reference_date = transactions["t_dat"].max()
        for cid, group in sorted_txn.groupby("customer_id"):
            days_ago = (reference_date - group["t_dat"]).dt.days
            weights = np.exp(-days_ago / 14).values
            items_weights = list(zip(group["article_id"].tolist(), weights.tolist()))
            self._user_recent[cid] = items_weights

    def build_features(self):
        """Step 2: Feature engineering."""
        logger.info("=" * 60)
        logger.info("STEP 2: Feature Engineering")
        logger.info("=" * 60)

        ref_date = self.train_txn["t_dat"].max()

        self.user_features = build_user_features(
            self.train_txn, self.customers, self.articles, ref_date
        )
        self.item_features = build_item_features(
            self.train_txn, self.articles, ref_date
        )
        self.interaction_features = build_interaction_features(
            self.train_txn, self.articles, self.user_features, ref_date
        )

        # User segmentation
        self.segmentation.fit(self.user_features)

    def train_candidate_generators(self):
        """Step 3: Train all candidate generation models."""
        logger.info("=" * 60)
        logger.info("STEP 3: Training Candidate Generators")
        logger.info("=" * 60)

        # ALS (collaborative filtering)
        interaction_matrix = build_interaction_matrix(
            self.train_txn, self.user_to_idx, self.item_to_idx
        )
        self.als_generator.fit(interaction_matrix)

        # Content-based
        self.content_generator.fit(self.articles)

        # Popularity with time decay
        ref_date = self.train_txn["t_dat"].max()
        self.popularity_generator.fit(self.train_txn, self.customers, ref_date)

        # Recency-based
        self.recency_generator.fit(self.train_txn, ref_date)

        # Two-Tower neural retrieval
        self.two_tower_generator.fit(
            self.train_txn,
            self.user_features,
            self.item_features,
            self.user_to_idx,
            self.item_to_idx,
        )

        # Cold start handler
        self.cold_start = ColdStartHandler(
            self.popularity_generator,
            self.content_generator,
            min_interactions=CONFIG["cold_start"]["min_interactions_warm"],
        )
        self.cold_start.fit(self.train_txn)

    def _generate_candidates_for_user(self, customer_id: str) -> pd.DataFrame:
        """Generate fused candidates for a single user."""
        history = self._user_history.get(customer_id, [])

        # ALS candidates
        als_candidates = []
        if customer_id in self.user_to_idx:
            user_idx = self.user_to_idx[customer_id]
            raw_als = self.als_generator.generate_candidates(user_idx, n_candidates=80)
            als_candidates = [
                (self.idx_to_item[idx], score) for idx, score in raw_als
                if idx in self.idx_to_item
            ]

        # Content-based candidates
        content_candidates = self.content_generator.generate_candidates_for_user(
            history, n_candidates=50
        )

        # Popularity candidates
        age = None
        if self.user_features is not None:
            user_row = self.user_features[
                self.user_features["customer_id"] == customer_id
            ]
            if len(user_row) > 0:
                age_group = user_row.iloc[0].get("age_group")
                age_group = str(age_group) if pd.notna(age_group) else None
            else:
                age_group = None
        else:
            age_group = None

        pop_candidates = self.popularity_generator.generate_candidates(
            n_candidates=50, age_group=age_group
        )

        # Recency candidates
        recent_items = self._user_recent.get(customer_id, [])
        recency_candidates = self.recency_generator.generate_candidates(
            recent_items, n_candidates=40, exclude=set(history)
        )

        # Two-Tower neural candidates
        two_tower_candidates = []
        if customer_id in self.user_to_idx:
            user_idx = self.user_to_idx[customer_id]
            user_feat_cols = TwoTowerCandidateGenerator.USER_FEATURE_COLS
            user_row = self.user_features[
                self.user_features["customer_id"] == customer_id
            ]
            if len(user_row) > 0:
                user_feats = np.array([
                    float(user_row.iloc[0].get(c, 0) or 0) for c in user_feat_cols
                ], dtype=np.float32)
            else:
                user_feats = np.zeros(len(user_feat_cols), dtype=np.float32)

            purchased_idx = {
                self.item_to_idx[aid] for aid in history if aid in self.item_to_idx
            }
            two_tower_candidates = self.two_tower_generator.generate_candidates(
                user_idx, user_feats, n_candidates=50, exclude_items=purchased_idx
            )

        # Fuse all sources (5 retrieval models)
        fused = self.candidate_fusion.fuse(
            als_candidates=als_candidates,
            content_candidates=content_candidates,
            popularity_candidates=pop_candidates,
            recency_candidates=recency_candidates,
            two_tower_candidates=two_tower_candidates,
        )
        fused["customer_id"] = customer_id
        return fused

    def build_training_data(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Build labeled training data for the ranking model.

        For each user in validation set:
        1. Generate candidates using train data
        2. Label: 1 if user actually purchased in val period, 0 otherwise
        3. Attach features for ranking model
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Building Ranking Training Data")
        logger.info("=" * 60)

        val_purchases = (
            self.val_txn.groupby("customer_id")["article_id"]
            .apply(set)
            .to_dict()
        )

        all_candidates = []
        all_labels = []
        group_sizes = []

        users_with_val = [
            uid for uid in val_purchases.keys()
            if uid in self.user_to_idx
        ]
        logger.info(f"Building ranking data for {len(users_with_val)} users...")

        for i, customer_id in enumerate(users_with_val):
            if i % 500 == 0:
                logger.info(f"  Processing user {i}/{len(users_with_val)}...")

            candidates = self._generate_candidates_for_user(customer_id)
            if len(candidates) == 0:
                continue

            # Label: did the user actually buy this item in val period?
            purchased = val_purchases.get(customer_id, set())
            labels = candidates["article_id"].isin(purchased).astype(int).values

            all_candidates.append(candidates)
            all_labels.append(labels)
            group_sizes.append(len(candidates))

        if not all_candidates:
            raise ValueError("No training data generated. Check data splits.")

        candidates_df = pd.concat(all_candidates, ignore_index=True)
        labels = np.concatenate(all_labels)
        groups = np.array(group_sizes)

        # Attach features
        candidates_df = assemble_ranking_features(
            candidates_df,
            self.user_features,
            self.item_features,
            self.interaction_features,
            self.articles,
        )

        positive_rate = labels.mean()
        logger.info(
            f"Ranking data: {len(candidates_df)} samples, "
            f"{len(groups)} queries, positive rate={positive_rate:.4f}"
        )
        return candidates_df, labels, groups

    def train_ranker(self):
        """Step 5: Train the LambdaRank ranking model."""
        logger.info("=" * 60)
        logger.info("STEP 5: Training Ranking Model")
        logger.info("=" * 60)

        candidates_df, labels, groups = self.build_training_data()
        self.ranker.fit(candidates_df, labels, groups)

        # Feature importance
        importance = self.ranker.get_feature_importance()
        logger.info("Top features:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']}")

    def recommend(
        self,
        customer_id: str,
        n: int = 12,
        diversity_lambda: float = 0.7,
    ) -> pd.DataFrame:
        """Generate final recommendations for a user (online inference).

        This is the real-time serving path:
        1. Check cold-start status
        2. Generate candidates (from cache or on-the-fly)
        3. Rank with LightGBM
        4. Apply diversity optimization
        """
        # Cold start check
        if self.cold_start and self.cold_start.is_cold_user(customer_id):
            history = self._user_history.get(customer_id, [])
            age = None
            cust_row = self.customers[self.customers["customer_id"] == customer_id]
            if len(cust_row) > 0:
                age = cust_row.iloc[0].get("age")

            recs = self.cold_start.get_cold_user_recommendations(
                customer_id, history, age=age, n=n
            )
            result = pd.DataFrame(recs, columns=["article_id", "fused_score"])
            result.attrs["is_cold_start"] = True
            return result

        # Stage 1: Candidate generation
        candidates = self._generate_candidates_for_user(customer_id)
        if len(candidates) == 0:
            # Fallback to popularity
            pop = self.popularity_generator.generate_candidates(n)
            return pd.DataFrame(pop, columns=["article_id", "fused_score"])

        # Attach features for ranking
        candidates = assemble_ranking_features(
            candidates,
            self.user_features,
            self.item_features,
            self.interaction_features,
            self.articles,
        )

        # Stage 2: Ranking
        if self.ranker.is_fitted:
            result = self.ranker.rank_candidates(candidates, top_k=n * 2)
        else:
            # If ranker not trained, use fused score
            result = candidates.nlargest(n * 2, "fused_score")

        # Stage 3: Diversity optimization
        item_categories = dict(
            zip(self.articles["article_id"], self.articles["product_type_name"])
        )
        diverse_recs = category_diversification(
            list(zip(
                result["article_id"].tolist(),
                result.get("rank_score", result["fused_score"]).tolist(),
            )),
            item_categories,
            max_per_category=3,
            top_k=n,
        )

        final_ids = [aid for aid, _ in diverse_recs]
        result = result[result["article_id"].isin(final_ids)].head(n)
        result.attrs["is_cold_start"] = False
        return result

    def evaluate(self, split: str = "test") -> dict[str, float]:
        """Evaluate the full pipeline on val or test set."""
        logger.info("=" * 60)
        logger.info(f"EVALUATION on {split} set")
        logger.info("=" * 60)

        eval_txn = self.test_txn if split == "test" else self.val_txn
        ground_truth = (
            eval_txn.groupby("customer_id")["article_id"]
            .apply(set)
            .to_dict()
        )

        predictions = {}
        users = list(ground_truth.keys())
        logger.info(f"Generating recommendations for {len(users)} users...")

        for i, customer_id in enumerate(users):
            if i % 200 == 0:
                logger.info(f"  User {i}/{len(users)}...")
            try:
                recs = self.recommend(customer_id, n=12)
                predictions[customer_id] = recs["article_id"].tolist()
            except Exception as e:
                logger.warning(f"Failed for {customer_id}: {e}")
                predictions[customer_id] = []

        k_values = CONFIG["evaluation"]["k_values"]
        metrics = evaluate_recommendations(predictions, ground_truth, k_values)
        return metrics

    def train_full(self):
        """Run the complete offline training pipeline."""
        start = time.time()

        self.load_data()
        self.build_features()
        self.train_candidate_generators()
        self.train_ranker()

        elapsed = time.time() - start
        logger.info(f"Full training completed in {elapsed:.1f}s")

    def save_artifacts(self, output_dir: str = "artifacts"):
        """Save all trained models and artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.ranker.is_fitted:
            self.ranker.save(str(output_path / "models" / "ranker.lgbm"))

        import joblib
        joblib.dump(self.user_to_idx, output_path / "models" / "user_to_idx.pkl")
        joblib.dump(self.item_to_idx, output_path / "models" / "item_to_idx.pkl")
        joblib.dump(self.idx_to_user, output_path / "models" / "idx_to_user.pkl")
        joblib.dump(self.idx_to_item, output_path / "models" / "idx_to_item.pkl")

        if self.user_features is not None:
            self.user_features.to_parquet(output_path / "caches" / "user_features.parquet")
        if self.item_features is not None:
            self.item_features.to_parquet(output_path / "caches" / "item_features.parquet")

        logger.info(f"Artifacts saved to {output_path}")
