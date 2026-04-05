"""User Segmentation via Behavioral Clustering.

Instead of treating all users identically, we segment them by
purchasing behavior. This enables:
1. Segment-specific model tuning
2. Better cold-start handling (assign new users to nearest segment)
3. Business insights ("high-value explorers" vs "budget loyalists")

Segments are built from behavioral features, not demographics —
behavior is a better predictor of future purchases.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from loguru import logger


class UserSegmentation:
    def __init__(self, n_segments: int = 5):
        self.n_segments = n_segments
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        self.segment_profiles: dict[int, dict] = {}
        self.user_segments: dict[str, int] = {}

    def fit(self, user_features: pd.DataFrame):
        """Cluster users into behavioral segments."""
        logger.info(f"Segmenting users into {self.n_segments} clusters...")

        feature_cols = [
            "purchase_count", "unique_items", "purchase_recency_days",
            "purchase_frequency", "color_diversity",
        ]
        available = [c for c in feature_cols if c in user_features.columns]

        X = user_features[available].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        labels = self.kmeans.fit_predict(X_scaled)
        user_features = user_features.copy()
        user_features["segment"] = labels

        # Build segment profiles
        for seg in range(self.n_segments):
            seg_users = user_features[user_features["segment"] == seg]
            profile = {}
            for col in available:
                profile[f"avg_{col}"] = float(seg_users[col].mean())
                profile[f"std_{col}"] = float(seg_users[col].std())
            profile["size"] = len(seg_users)
            profile["pct"] = len(seg_users) / len(user_features)
            self.segment_profiles[seg] = profile

        # Store user-segment mapping
        self.user_segments = dict(
            zip(user_features["customer_id"], user_features["segment"])
        )

        self._name_segments(user_features, available)
        logger.info(f"Segmentation complete. Profiles: {self.segment_profiles}")

    def _name_segments(self, df: pd.DataFrame, feature_cols: list[str]):
        """Auto-name segments based on distinguishing characteristics."""
        segment_names = {
            "high_frequency_high_diversity": "Style Explorer",
            "high_frequency_low_diversity": "Brand Loyalist",
            "low_frequency_recent": "Casual Shopper",
            "low_frequency_old": "Dormant User",
            "high_items": "Power Buyer",
        }
        # Assign names based on cluster centers
        centers = self.kmeans.cluster_centers_
        for seg in range(self.n_segments):
            center = centers[seg]
            self.segment_profiles[seg]["name"] = f"Segment_{seg}"

    def get_segment(self, customer_id: str) -> int:
        return self.user_segments.get(customer_id, -1)

    def predict_segment(self, features: np.ndarray) -> int:
        """Assign a new user to the nearest segment."""
        scaled = self.scaler.transform(features.reshape(1, -1))
        return int(self.kmeans.predict(scaled)[0])

    def get_segment_summary(self) -> pd.DataFrame:
        rows = []
        for seg, profile in self.segment_profiles.items():
            row = {"segment": seg}
            row.update(profile)
            rows.append(row)
        return pd.DataFrame(rows)
