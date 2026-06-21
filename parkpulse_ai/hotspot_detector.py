"""
Hotspot_Detector: applies DBSCAN clustering to parking violation GPS coordinates
and aggregates per-zone statistics to identify illegal parking hotspots.

DBSCAN configuration:
- Metric: Haversine (coordinates converted to radians)
- eps_rad = eps_km / 6371
- algorithm = "ball_tree"
- cluster_id = -1 indicates noise (not part of any dense cluster)
"""

from __future__ import annotations

import warnings

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN


def detect_hotspots(
    df: pd.DataFrame,
    eps_km: float = 0.1,
    min_samples: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply DBSCAN to GPS coordinates and aggregate hotspot statistics per zone.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: latitude, longitude, zone, violation_type,
        vehicle_type.
    eps_km : float
        DBSCAN neighbourhood radius in kilometres. Converted to radians
        internally as ``eps_rad = eps_km / 6371``.
    min_samples : int
        Minimum number of samples in a neighbourhood for DBSCAN to form a
        core point.

    Returns
    -------
    df_with_cluster_col : pd.DataFrame
        A copy of *df* with an added ``cluster_id`` column.  Values are
        integers ≥ 0 for clustered records and −1 for noise.
    hotspot_summary_df : pd.DataFrame
        Zone-level summary with columns:
        ``zone``, ``violation_count``, ``cluster_count``,
        ``top_violation_type``, ``top_vehicle_type``.
        Zones with fewer than 5 records are excluded.
        Sorted by ``violation_count`` descending.
    """
    # ------------------------------------------------------------------
    # 1. Run DBSCAN on radians-converted coordinates.
    # ------------------------------------------------------------------
    eps_rad = eps_km / 6371.0

    coords = df[["latitude", "longitude"]].values.astype(float)
    coords_rad = np.radians(coords)

    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = db.fit_predict(coords_rad)

    # ------------------------------------------------------------------
    # 2. Attach cluster_id to a copy of the input DataFrame.
    # ------------------------------------------------------------------
    df_out = df.copy()
    df_out["cluster_id"] = labels

    # ------------------------------------------------------------------
    # 3. Aggregate per-zone statistics.
    # ------------------------------------------------------------------
    def _top_value(series: pd.Series) -> object:
        """Return the most frequent value in *series*, or None if empty."""
        if series.empty:
            return None
        return series.value_counts().idxmax()

    def _cluster_count(cluster_ids: pd.Series) -> int:
        """Count distinct non-noise cluster labels (≥ 0) in *cluster_ids*."""
        return int((cluster_ids >= 0).sum() > 0) if False else \
            int(cluster_ids[cluster_ids >= 0].nunique())

    # Build per-zone aggregation.
    agg = (
        df_out.groupby("zone", sort=False)
        .agg(
            violation_count=("zone", "size"),
            cluster_count=("cluster_id", _cluster_count),
            top_violation_type=("violation_type", _top_value),
            top_vehicle_type=("vehicle_type", _top_value),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 4. Filter zones with fewer than 5 records (Requirement 3.7).
    # ------------------------------------------------------------------
    agg = agg[agg["violation_count"] >= 5].copy()

    # ------------------------------------------------------------------
    # 5. Sort by violation_count descending (Requirement 3.4).
    # ------------------------------------------------------------------
    agg = agg.sort_values("violation_count", ascending=False).reset_index(drop=True)

    # Enforce column dtypes for clarity.
    agg["violation_count"] = agg["violation_count"].astype("int64")
    agg["cluster_count"] = agg["cluster_count"].astype("int64")

    hotspot_summary_df = agg[
        ["zone", "violation_count", "cluster_count", "top_violation_type", "top_vehicle_type"]
    ]

    return df_out, hotspot_summary_df


def generate_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """
    Generate a Folium heatmap from violation GPS points and save as HTML.

    Each GPS point is weighted by the violation count of its zone so that
    denser zones appear more prominently on the heatmap (Requirement 3.6).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: latitude, longitude, and optionally zone.
        If a ``zone`` column is present, points are weighted by their
        zone's violation count.
    output_path : str
        File system path where the HTML file will be written.
    """
    lats = df["latitude"].values.astype(float)
    lons = df["longitude"].values.astype(float)

    # Compute zone-level violation counts for weighting when possible.
    if "zone" in df.columns:
        zone_counts = df.groupby("zone")["zone"].transform("count").values.astype(float)
        # Normalise weights to [0, 1] to keep Folium HeatMap happy.
        max_count = zone_counts.max()
        weights = (zone_counts / max_count).tolist() if max_count > 0 else None
    else:
        weights = None

    # Centre map on the median coordinate of the data.
    center_lat = float(np.median(lats[np.isfinite(lats)])) if len(lats) > 0 else 12.97
    center_lon = float(np.median(lons[np.isfinite(lons)])) if len(lons) > 0 else 77.59

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Build heat data: list of [lat, lon, weight] or [lat, lon].
    valid_mask = np.isfinite(lats) & np.isfinite(lons)
    if weights is not None:
        heat_data = [
            [float(lats[i]), float(lons[i]), float(weights[i])]
            for i in range(len(lats))
            if valid_mask[i]
        ]
    else:
        heat_data = [
            [float(lats[i]), float(lons[i])]
            for i in range(len(lats))
            if valid_mask[i]
        ]

    if heat_data:
        HeatMap(heat_data, radius=10, blur=15, max_zoom=1).add_to(m)
    else:
        warnings.warn(
            "generate_heatmap: no valid GPS points found; heatmap will be empty.",
            stacklevel=2,
        )

    m.save(output_path)
