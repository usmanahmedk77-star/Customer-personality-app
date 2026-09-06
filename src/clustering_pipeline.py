"""
clustering_pipeline.py
========================
Production-ready clustering pipeline — Module 6, Task 17.

Wraps feature selection, scaling, and the final K-Means segmentation model
into a single reusable object so a future batch of customers (already run
through the Module 5 feature-engineering pipeline) can be scored / assigned
to a segment with one call.

Usage
-----
    from clustering_pipeline import CustomerSegmentationModel

    model = CustomerSegmentationModel()
    model.fit(engineered_df)                       # engineered_df = Module 5 output
    labels = model.predict(engineered_df)           # cluster assignment per row
    profile = model.get_cluster_profile()            # cluster-level summary stats

    model.save("clustering_model.joblib")
    model2 = CustomerSegmentationModel.load("clustering_model.joblib")
    new_labels = model2.predict(new_engineered_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Features selected in Task 2, after removing highly-correlated redundant
# columns (Total_Purchases and Total_Children — see Feature Selection Report)
CLUSTERING_FEATURES = [
    "Customer_Age", "Income", "Total_Spending", "Recency", "Customer_Tenure",
    "Family_Size", "Total_Campaign_Acceptance",
    "NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases",
]

DEFAULT_K = 4

# Data-driven segment names derived from Task 13 cluster profiling.
# Re-map cluster index -> business name after inspecting a new fit's
# cluster_centers_ order (KMeans cluster indices are not guaranteed stable
# across refits) using `label_segments()` below.
SEGMENT_NAME_HINTS = {
    "low_value": "Budget Shoppers",
    "mid_value": "Emerging / Mid-Tier Customers",
    "high_value": "High-Value Customers",
    "premium_responsive": "Premium & Campaign-Responsive Customers",
}


class CustomerSegmentationModel:
    """End-to-end, reusable customer segmentation (clustering) model."""

    def __init__(self, k: int = DEFAULT_K, features=None, random_state: int = 42):
        self.k = k
        self.features = features or list(CLUSTERING_FEATURES)
        self.random_state = random_state

        self.scaler_: StandardScaler | None = None
        self.model_: KMeans | None = None
        self.cluster_profile_: pd.DataFrame | None = None
        self.segment_names_: dict | None = None
        self.is_fitted_ = False

    # ------------------------------------------------------------------ #
    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        return df[self.features].copy()

    def fit(self, df: pd.DataFrame) -> "CustomerSegmentationModel":
        X_raw = self._select(df)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_raw)

        self.model_ = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
        labels = self.model_.fit_predict(X_scaled)

        profile = X_raw.copy()
        profile["Cluster"] = labels
        self.cluster_profile_ = profile.groupby("Cluster").mean().round(1)
        self.cluster_profile_["n_customers"] = profile.groupby("Cluster").size()
        self.cluster_profile_["pct_of_base"] = (
            self.cluster_profile_["n_customers"] / len(profile) * 100
        ).round(1)

        self.segment_names_ = self._auto_label_segments()
        self.is_fitted_ = True
        return self

    def _auto_label_segments(self) -> dict:
        """Assign a human-readable business name to each cluster index based
        on its relative spending / income / campaign-acceptance profile."""
        p = self.cluster_profile_
        k = len(p)
        names = {}

        # Order clusters from highest to lowest average spend, then split
        # into up to 4 value tiers (highest spend first).
        order = p["Total_Spending"].sort_values(ascending=False).index.tolist()
        top_campaign_idx = p["Total_Campaign_Acceptance"].idxmax()
        runner_up_campaign = p["Total_Campaign_Acceptance"].drop(top_campaign_idx).max()
        is_standout_campaign = p.loc[top_campaign_idx, "Total_Campaign_Acceptance"] > 1.5 * max(runner_up_campaign, 0.1)

        for position, idx in enumerate(order):
            tier = min(int(position / max(k, 1) * 4), 3)
            if tier == 0:
                if idx == top_campaign_idx and is_standout_campaign:
                    names[idx] = SEGMENT_NAME_HINTS["premium_responsive"]
                else:
                    names[idx] = SEGMENT_NAME_HINTS["high_value"]
            elif tier == 1:
                names[idx] = SEGMENT_NAME_HINTS["high_value"]
            elif tier == 2:
                names[idx] = SEGMENT_NAME_HINTS["mid_value"]
            else:
                names[idx] = SEGMENT_NAME_HINTS["low_value"]
        return names

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")
        X_raw = self._select(df)
        X_scaled = self.scaler_.transform(X_raw)
        return self.model_.predict(X_scaled)

    def predict_segment_names(self, df: pd.DataFrame) -> list[str]:
        labels = self.predict(df)
        return [self.segment_names_[int(l)] for l in labels]

    def get_cluster_profile(self) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")
        out = self.cluster_profile_.copy()
        out["Segment_Name"] = [self.segment_names_[i] for i in out.index]
        return out

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "CustomerSegmentationModel":
        return joblib.load(path)


if __name__ == "__main__":
    import sys
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/customer_personality_feature_engineered.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "customer_segments.csv"

    df = pd.read_csv(in_path)
    model = CustomerSegmentationModel()
    model.fit(df)

    df_out = df.copy()
    df_out["Cluster"] = model.predict(df)
    df_out["Segment_Name"] = model.predict_segment_names(df)
    df_out.to_csv(out_path, index=False)

    print("Cluster profile:")
    print(model.get_cluster_profile())
    print(f"\nSaved segmented dataset -> {out_path}")
