# Unsupervised clustering of motion segments.
# Groups telemetry windows by similarity in feature space (e.g., k-means, DBSCAN)
# to discover recurring behaviours without labelled training data.
#
# Context.txt Section 6 ("Classification Architecture") calls this the *Secondary,
# Exploratory* approach: rules.py sorts a run into a tier using fixed thresholds,
# while this module instead lets the engineered features (RMSE, stability variance,
# Cost of Transport, latency, torque stress -- see Section 4) group themselves. That
# is useful when comparing a batch of RL checkpoints/policies where you don't yet
# know how many distinct performance tiers actually show up in the data.

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Best-guess column names for the feature vector this module expects, mapped
# directly to Context.txt Section 4 ("Feature Engineering"):
#   rmse               -> Control Precision (commanded vs. actual joint position)
#   imu_variance        -> Dynamic Stability (roll/pitch variance during gait)
#   cost_of_transport  -> Cost of Transport (power drawn per distance travelled)
#   control_latency    -> Control Latency (policy output -> mechanical actuation delay)
#   torque_spikes      -> Hardware Stress (count/magnitude of extreme torque events)
# `PolicyClusterer` only requires *some* of these to be present (see
# `_select_feature_columns`), so it keeps working even if upstream feature
# modules add/rename columns.
DEFAULT_FEATURE_COLUMNS = [
    "rmse",
    "imu_variance",
    "cost_of_transport",
    "control_latency",
    "torque_spikes",
]


@dataclass
class ClusteringResult:
    """Everything a caller needs to interpret one clustering run."""

    labels: pd.Series          # cluster id per input row, indexed like the input df
                                # (DBSCAN uses -1 for "noise" / outlier points)
    algorithm: str              # "kmeans" or "dbscan"
    n_clusters: int             # number of real clusters found (excludes DBSCAN noise)
    silhouette: Optional[float] # separation quality in [-1, 1]; None if not computable
    feature_columns: List[str]  # which columns were actually used for clustering


class PolicyClusterer:
    """
    Groups control-policy telemetry runs by similarity in engineered-feature space.

    Typical usage:
        clusterer = PolicyClusterer(algorithm="kmeans", n_clusters=3)
        result = clusterer.fit_predict(features_df)
        features_df["cluster"] = result.labels  # aligns back on the DataFrame index
    """

    def __init__(
        self,
        algorithm: Literal["kmeans", "dbscan"] = "kmeans",
        n_clusters: int = 3,
        eps: float = 0.5,
        min_samples: int = 3,
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        """
        Args:
            algorithm: "kmeans" (needs a known cluster count) or "dbscan" (infers
                cluster count, but needs a density radius). K-Means is the sane
                default for "sort N policies into K tiers"; DBSCAN is better when
                you expect outlier/failed runs that shouldn't be forced into a group.
            n_clusters: number of clusters for K-Means. Ignored by DBSCAN.
            eps: DBSCAN neighbourhood radius, in *standardised* units (see
                `_prepare_features`) since features are scaled before clustering.
                Ignored by K-Means.
            min_samples: DBSCAN's minimum neighbourhood size to form a dense
                region. Ignored by K-Means.
            feature_columns: explicit list of feature columns to cluster on.
                Defaults to `DEFAULT_FEATURE_COLUMNS`, filtered down to whichever
                of those are actually present in the input DataFrame.
            random_state: seed for K-Means' centroid initialisation, so repeated
                runs on the same data reproduce the same cluster assignment.
        """
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.feature_columns = feature_columns
        self.random_state = random_state
        # A fresh scaler per instance -- fit_predict() re-fits it on whatever data
        # comes in, so scaling stats never leak between unrelated calls.
        self._scaler = StandardScaler()

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Resolve which requested feature columns actually exist in `df`."""
        requested = self.feature_columns or DEFAULT_FEATURE_COLUMNS
        available = [c for c in requested if c in df.columns]
        missing = [c for c in requested if c not in df.columns]

        if missing:
            logger.warning(f"Feature column(s) not found in input data, skipping: {missing}")
        if not available:
            raise ValueError(
                f"None of the requested feature columns {requested} are present in the "
                f"input DataFrame (available columns: {list(df.columns)})."
            )
        return available

    def _prepare_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Drop rows with missing feature values before they reach the clustering model."""
        # KMeans/DBSCAN can't handle NaNs, and silently imputing a performance metric
        # (e.g. a missing Cost of Transport because a run fell over early) could hide
        # a real failure inside an average. Dropping is the honest default here.
        clean_df = df.dropna(subset=feature_columns)
        dropped = len(df) - len(clean_df)
        if dropped:
            logger.warning(f"Dropping {dropped} row(s) with missing feature values before clustering.")
        if clean_df.empty:
            raise ValueError("All rows were dropped due to missing feature values; nothing to cluster.")
        return clean_df

    def fit_predict(self, df: pd.DataFrame) -> ClusteringResult:
        """
        Cluster the rows of `df` by their feature-space similarity.

        Args:
            df: one row per policy run/telemetry window, with the engineered
                feature columns (see `DEFAULT_FEATURE_COLUMNS`) as columns.

        Returns:
            A `ClusteringResult` whose `.labels` is indexed like `df` (after
            dropping rows with missing feature values), so it can be reattached
            with `df["cluster"] = result.labels`.
        """
        if df.empty:
            raise ValueError("Cannot cluster an empty DataFrame.")

        feature_columns = self._select_feature_columns(df)
        clean_df = self._prepare_features(df, feature_columns)

        # Standardise to zero mean / unit variance so no single feature dominates
        # the distance metric purely because of its units -- e.g. Cost of Transport
        # (tens/hundreds of Watts per m/s) would otherwise swamp an RMSE measured
        # in radians (fractions of 1), regardless of which feature is actually more
        # discriminative between policies.
        X = self._scaler.fit_transform(clean_df[feature_columns].to_numpy())

        if self.algorithm == "kmeans":
            # n_init is set explicitly (rather than left to the sklearn default,
            # which has changed across versions) so K-Means always restarts from
            # multiple random centroid seeds and keeps the best-inertia result.
            model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        elif self.algorithm == "dbscan":
            model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        else:
            raise ValueError(f"Unknown algorithm '{self.algorithm}', expected 'kmeans' or 'dbscan'.")

        raw_labels = model.fit_predict(X)
        labels = pd.Series(raw_labels, index=clean_df.index, name="cluster")

        # -1 is DBSCAN's "noise" label (a point too far from any dense region to
        # belong to a cluster) -- it isn't a real cluster, so exclude it from the count.
        unique_clusters = set(raw_labels) - {-1}

        silhouette = self._safe_silhouette(X, raw_labels, unique_clusters)

        return ClusteringResult(
            labels=labels,
            algorithm=self.algorithm,
            n_clusters=len(unique_clusters),
            silhouette=silhouette,
            feature_columns=feature_columns,
        )

    @staticmethod
    def _safe_silhouette(X: np.ndarray, raw_labels: np.ndarray, unique_clusters: set) -> Optional[float]:
        """
        Compute the silhouette score (how well-separated the clusters are, in
        [-1, 1], higher is better) without letting a bad hyperparameter choice
        crash the whole pipeline.

        silhouette_score() requires at least 2 clusters and cannot score points
        DBSCAN marked as noise, so both conditions are guarded here rather than
        left to raise inside the caller.
        """
        if len(unique_clusters) < 2:
            logger.warning(
                f"Only {len(unique_clusters)} cluster(s) found; silhouette score needs at least 2 "
                "and will be reported as None. For DBSCAN, try a larger eps or smaller min_samples."
            )
            return None

        non_noise = raw_labels != -1
        try:
            return float(silhouette_score(X[non_noise], raw_labels[non_noise]))
        except ValueError as e:
            logger.warning(f"Could not compute silhouette score: {e}")
            return None

    def find_optimal_k(self, df: pd.DataFrame, k_range=range(2, 8)) -> pd.DataFrame:
        """
        Sweep K-Means over a range of cluster counts and score each by silhouette,
        for when there's no a-priori answer to "how many performance tiers are
        actually in this batch of policies?" (the exploratory use case called out
        in Context.txt Section 6).

        Returns a DataFrame with one row per candidate k (columns: k, inertia,
        silhouette) so the caller can pick a peak/elbow rather than trusting a
        single auto-selected number.
        """
        feature_columns = self._select_feature_columns(df)
        clean_df = self._prepare_features(df, feature_columns)
        X = self._scaler.fit_transform(clean_df[feature_columns].to_numpy())

        rows = []
        for k in k_range:
            if k < 1 or k >= len(clean_df):
                logger.warning(
                    f"Skipping k={k}: need 1 <= k < number of samples ({len(clean_df)})."
                )
                continue

            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(X)
            # A single cluster has nothing to separate itself from, so silhouette
            # is undefined at k=1 -- record NaN rather than calling the sklearn
            # function and letting it raise.
            score = silhouette_score(X, labels) if k > 1 else float("nan")
            rows.append({"k": k, "inertia": model.inertia_, "silhouette": score})

        return pd.DataFrame(rows)
