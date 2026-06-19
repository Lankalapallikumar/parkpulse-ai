"""
Impact_Scorer: computes a composite Traffic Impact Score (0–100) for each
parking-violation zone and assigns a Risk Level category.

Sub-score weights
-----------------
- Violation frequency : 0.40
- Vehicle severity    : 0.30
- Violation severity  : 0.20
- Peak hour           : 0.10

Peak hours: {0, 1, 2, 3, 4, 5, 19, 20, 21}

Vehicle severity weights
------------------------
TRUCK / MAXI-CAB          → 3
CAR / PASSENGER AUTO      → 2
SCOOTER / MOTOR CYCLE     → 1
unknown / anything else   → 1

Violation severity weights
--------------------------
violation_type contains "JUNCTION" → 3
violation_type contains "MAIN"     → 2
else                               → 1

Risk level thresholds
---------------------
≥ 80  → Critical
≥ 60  → High
≥ 40  → Medium
< 40  → Low

Single-zone edge case: traffic_impact_score = 50.0 (min-max undefined with one zone).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PEAK_HOURS: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5, 19, 20, 21})

_VEHICLE_WEIGHT_MAP: dict[str, float] = {
    "TRUCK": 3.0,
    "MAXI-CAB": 3.0,
    "CAR": 2.0,
    "PASSENGER AUTO": 2.0,
    "SCOOTER": 1.0,
    "MOTOR CYCLE": 1.0,
}

_DEFAULT_VEHICLE_WEIGHT: float = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _minmax_scale(series: pd.Series) -> pd.Series:
    """
    Min-max scale *series* to [0, 100].

    If all values are identical (range == 0) returns a series of 0.0 for
    all elements, which is later overridden to 50.0 at the top level for
    the single-zone edge case.
    """
    min_val = series.min()
    max_val = series.max()
    rng = max_val - min_val
    if rng == 0:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / rng * 100.0


def _vehicle_weight(vehicle_type: str) -> float:
    """Return the severity weight for a vehicle type string."""
    if not isinstance(vehicle_type, str):
        return _DEFAULT_VEHICLE_WEIGHT
    return _VEHICLE_WEIGHT_MAP.get(vehicle_type.strip().upper(), _DEFAULT_VEHICLE_WEIGHT)


def _violation_weight(violation_type: str) -> float:
    """Return the severity weight for a violation type string."""
    if not isinstance(violation_type, str):
        return 1.0
    vt_upper = violation_type.upper()
    if "JUNCTION" in vt_upper:
        return 3.0
    if "MAIN" in vt_upper:
        return 2.0
    return 1.0


def _is_peak_hour(dt) -> bool:
    """Return True if *dt* is a datetime whose hour falls in peak hours."""
    try:
        return int(dt.hour) in _PEAK_HOURS
    except (AttributeError, TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Traffic Impact Score for each zone in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``zone``, ``vehicle_type``, ``violation_type``,
        ``created_datetime``.

    Returns
    -------
    pd.DataFrame
        Zone-level DataFrame with columns:
        ``zone``, ``traffic_impact_score``, ``risk_level``,
        ``violation_count``, ``vehicle_severity_score``,
        ``violation_severity_score``, ``peak_hour_score``.
    """
    # ------------------------------------------------------------------
    # 1. Per-record feature columns (added to a working copy).
    # ------------------------------------------------------------------
    work = df.copy()

    work["_vehicle_weight"] = work["vehicle_type"].apply(_vehicle_weight)
    work["_violation_weight"] = work["violation_type"].apply(_violation_weight)
    work["_is_peak"] = work["created_datetime"].apply(_is_peak_hour)

    # ------------------------------------------------------------------
    # 2. Aggregate per zone.
    # ------------------------------------------------------------------
    grp = work.groupby("zone", sort=False)

    zone_stats = pd.DataFrame(
        {
            "violation_count": grp.size(),
            "_mean_vehicle_weight": grp["_vehicle_weight"].mean(),
            "_mean_violation_weight": grp["_violation_weight"].mean(),
            # Peak-hour percentage: mean of boolean * 100 gives the % directly.
            "_peak_pct": grp["_is_peak"].mean() * 100.0,
        }
    ).reset_index()

    # ------------------------------------------------------------------
    # 3. Handle single-zone edge case (min-max undefined).
    # ------------------------------------------------------------------
    single_zone = len(zone_stats) == 1

    # ------------------------------------------------------------------
    # 4. Min-max scale each sub-score to [0, 100].
    # ------------------------------------------------------------------
    zone_stats["freq_score"] = _minmax_scale(zone_stats["violation_count"].astype(float))
    zone_stats["vehicle_severity_score"] = _minmax_scale(zone_stats["_mean_vehicle_weight"])
    zone_stats["violation_severity_score"] = _minmax_scale(zone_stats["_mean_violation_weight"])
    # Peak percentage is already in [0, 100]; scale relative to min/max across zones.
    zone_stats["peak_hour_score"] = _minmax_scale(zone_stats["_peak_pct"])

    # ------------------------------------------------------------------
    # 5. Weighted raw score and final min-max normalisation.
    # ------------------------------------------------------------------
    zone_stats["_raw"] = (
        zone_stats["freq_score"] * 0.4
        + zone_stats["vehicle_severity_score"] * 0.3
        + zone_stats["violation_severity_score"] * 0.2
        + zone_stats["peak_hour_score"] * 0.1
    )

    if single_zone:
        # Single-zone: all scaled values are 0 → raw is 0; override with 50.0.
        zone_stats["traffic_impact_score"] = 50.0
    else:
        zone_stats["traffic_impact_score"] = _minmax_scale(zone_stats["_raw"])

    # ------------------------------------------------------------------
    # 6. Risk level assignment.
    # ------------------------------------------------------------------
    def _risk_level(score: float) -> str:
        if score >= 80.0:
            return "Critical"
        if score >= 60.0:
            return "High"
        if score >= 40.0:
            return "Medium"
        return "Low"

    zone_stats["risk_level"] = zone_stats["traffic_impact_score"].apply(_risk_level)

    # ------------------------------------------------------------------
    # 7. Build and return final output DataFrame.
    # ------------------------------------------------------------------
    result = zone_stats[
        [
            "zone",
            "traffic_impact_score",
            "risk_level",
            "violation_count",
            "vehicle_severity_score",
            "violation_severity_score",
            "peak_hour_score",
        ]
    ].copy()

    # Cast violation_count to int64 for consistency with schema.
    result["violation_count"] = result["violation_count"].astype("int64")

    return result.reset_index(drop=True)
