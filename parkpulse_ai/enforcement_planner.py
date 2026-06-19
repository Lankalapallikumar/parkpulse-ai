"""
Enforcement_Planner: converts Traffic Impact Scores into Enforcement Plans
and runs the What-If resource simulator.

Resource allocation lookup table
---------------------------------
Risk Level | Officers | Tow Trucks | Patrol Freq (hrs)
-----------+----------+------------+------------------
Critical   |    5     |     2      |        1
High       |    3     |     1      |        2
Medium     |    2     |     1      |        4
Low        |    1     |     0      |        8

What-if simulation algorithm
------------------------------
1. Sort zones by traffic_impact_score descending.
2. Greedily allocate officers until n_officers exhausted.
3. Per zone: reduction = min(allocated / recommended, 1.0) × base_rate
   - base_rates: Critical=0.40, High=0.35, Medium=0.30, Low=0.25
4. Overall = violation-count-weighted mean of per-zone reductions.
5. If n_officers > total recommended, cap at total recommended and report surplus.

Edge cases
----------
- n_officers = 0  → return empty allocation DataFrame and 0.0 overall reduction.
- n_officers > total recommended → cap allocation, report surplus.
- Input scored_df must contain: zone, risk_level, traffic_impact_score, violation_count.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# (recommended_officers, recommended_tow_trucks, patrol_frequency_hours)
_RESOURCE_TABLE: dict[str, tuple[int, int, int]] = {
    "Critical": (5, 2, 1),
    "High":     (3, 1, 2),
    "Medium":   (2, 1, 4),
    "Low":      (1, 0, 8),
}

_BASE_REDUCTION_RATES: dict[str, float] = {
    "Critical": 0.40,
    "High":     0.35,
    "Medium":   0.30,
    "Low":      0.25,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_enforcement_plan(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a resource allocation plan from a scored zone DataFrame.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Must contain columns: ``zone``, ``risk_level``,
        ``traffic_impact_score``, ``violation_count``.

    Returns
    -------
    pd.DataFrame
        Columns: ``zone``, ``risk_level``, ``traffic_impact_score``,
        ``recommended_officers``, ``recommended_tow_trucks``,
        ``patrol_frequency_hours``.
        Sorted by ``traffic_impact_score`` descending.
    """
    required_cols = {"zone", "risk_level", "traffic_impact_score", "violation_count"}
    missing = required_cols - set(scored_df.columns)
    if missing:
        raise ValueError(f"scored_df is missing required columns: {missing}")

    rows = []
    for _, row in scored_df.iterrows():
        risk = row["risk_level"]
        officers, tow_trucks, patrol_freq = _RESOURCE_TABLE.get(risk, (1, 0, 8))
        rows.append(
            {
                "zone": row["zone"],
                "risk_level": risk,
                "traffic_impact_score": row["traffic_impact_score"],
                "recommended_officers": officers,
                "recommended_tow_trucks": tow_trucks,
                "patrol_frequency_hours": patrol_freq,
                "violation_count": int(row["violation_count"]),
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "zone",
            "risk_level",
            "traffic_impact_score",
            "recommended_officers",
            "recommended_tow_trucks",
            "patrol_frequency_hours",
            "violation_count",
        ],
    )

    # Sort by traffic_impact_score descending, then reset index.
    result = result.sort_values("traffic_impact_score", ascending=False).reset_index(drop=True)

    # Enforce integer dtypes for resource and count columns.
    result["recommended_officers"] = result["recommended_officers"].astype("int64")
    result["recommended_tow_trucks"] = result["recommended_tow_trucks"].astype("int64")
    result["patrol_frequency_hours"] = result["patrol_frequency_hours"].astype("int64")
    result["violation_count"] = result["violation_count"].astype("int64")

    return result


def simulate_what_if(
    enforcement_df: pd.DataFrame,
    n_officers: int,
) -> tuple[pd.DataFrame, float]:
    """
    Simulate officer deployment across zones and estimate impact reduction.

    Parameters
    ----------
    enforcement_df : pd.DataFrame
        Output of :func:`build_enforcement_plan` (or equivalent).  Must
        contain columns: ``zone``, ``risk_level``, ``traffic_impact_score``,
        ``recommended_officers``.  Also requires ``violation_count`` if
        present (used for weighted mean); if absent, equal weighting is used.
    n_officers : int
        Total number of officers to deploy.  0 returns empty results.

    Returns
    -------
    what_if_df : pd.DataFrame
        Columns: ``zone``, ``officers_allocated``, ``expected_reduction_pct``,
        ``risk_level``.  Only zones that received at least 1 officer are
        included.
    overall_expected_reduction_pct : float
        Violation-count-weighted mean of per-zone ``expected_reduction_pct``
        across all covered zones.  0.0 when n_officers=0 or no zones covered.
    """
    # ------------------------------------------------------------------
    # Edge case: no officers to deploy.
    # ------------------------------------------------------------------
    if n_officers == 0:
        empty_df = pd.DataFrame(
            columns=["zone", "officers_allocated", "expected_reduction_pct", "risk_level"]
        )
        empty_df["officers_allocated"] = empty_df["officers_allocated"].astype("int64")
        empty_df["expected_reduction_pct"] = empty_df["expected_reduction_pct"].astype("float64")
        return empty_df, 0.0

    required_cols = {"zone", "risk_level", "traffic_impact_score", "recommended_officers"}
    missing = required_cols - set(enforcement_df.columns)
    if missing:
        raise ValueError(f"enforcement_df is missing required columns: {missing}")

    # ------------------------------------------------------------------
    # 1. Sort zones by traffic_impact_score descending.
    # ------------------------------------------------------------------
    sorted_df = enforcement_df.sort_values("traffic_impact_score", ascending=False).reset_index(
        drop=True
    )

    total_recommended = int(sorted_df["recommended_officers"].sum())

    # ------------------------------------------------------------------
    # 5. Cap if n_officers exceeds total recommended; track surplus.
    # ------------------------------------------------------------------
    surplus = 0
    budget = n_officers
    if budget > total_recommended:
        surplus = budget - total_recommended
        budget = total_recommended

    # ------------------------------------------------------------------
    # 2. Greedy allocation.
    # ------------------------------------------------------------------
    allocation_rows = []
    remaining = budget

    for _, row in sorted_df.iterrows():
        if remaining <= 0:
            break

        recommended = int(row["recommended_officers"])
        risk = row["risk_level"]

        # Allocate as many as needed (up to recommended) from the budget.
        allocated = min(recommended, remaining)
        remaining -= allocated

        base_rate = _BASE_REDUCTION_RATES.get(risk, 0.25)
        reduction = min(allocated / recommended, 1.0) * base_rate * 100.0  # as percentage

        # Grab violation_count if available for weighted mean later.
        violation_count = int(row["violation_count"]) if "violation_count" in row.index else 1

        allocation_rows.append(
            {
                "zone": row["zone"],
                "officers_allocated": allocated,
                "expected_reduction_pct": reduction,
                "risk_level": risk,
                "_violation_count": violation_count,
            }
        )

    # ------------------------------------------------------------------
    # Build what_if_df (exclude internal helper column).
    # ------------------------------------------------------------------
    if not allocation_rows:
        what_if_df = pd.DataFrame(
            columns=["zone", "officers_allocated", "expected_reduction_pct", "risk_level"]
        )
        what_if_df["officers_allocated"] = what_if_df["officers_allocated"].astype("int64")
        what_if_df["expected_reduction_pct"] = what_if_df["expected_reduction_pct"].astype(
            "float64"
        )
        return what_if_df, 0.0

    alloc_df = pd.DataFrame(allocation_rows)

    # ------------------------------------------------------------------
    # 3 & 4. Overall = violation-count-weighted mean of per-zone reductions.
    # ------------------------------------------------------------------
    total_violations = alloc_df["_violation_count"].sum()
    if total_violations > 0:
        overall = float(
            (alloc_df["expected_reduction_pct"] * alloc_df["_violation_count"]).sum()
            / total_violations
        )
    else:
        overall = float(alloc_df["expected_reduction_pct"].mean())

    # Attach surplus info as a module-level attribute for callers that want it
    # (not part of the public return signature, but accessible if needed).
    simulate_what_if._last_surplus = surplus  # type: ignore[attr-defined]

    what_if_df = alloc_df[
        ["zone", "officers_allocated", "expected_reduction_pct", "risk_level"]
    ].copy()
    what_if_df["officers_allocated"] = what_if_df["officers_allocated"].astype("int64")
    what_if_df["expected_reduction_pct"] = what_if_df["expected_reduction_pct"].astype("float64")
    what_if_df = what_if_df.reset_index(drop=True)

    return what_if_df, overall
