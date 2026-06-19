"""
Unit tests for Enforcement_Planner (parkpulse_ai/enforcement_planner.py).

Covers (Req 5.1, 6.1, 6.4):
- n_officers=0 returns 0.0 overall reduction and empty DataFrame
- n_officers > total recommended officers completes without crashing
- Critical zone gets exactly 5 recommended_officers in the plan
- What-if simulator allocates officers to the highest-scored zones first
"""

import pandas as pd
import pytest

from parkpulse_ai.enforcement_planner import build_enforcement_plan, simulate_what_if


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scored_df(zones: list[dict]) -> pd.DataFrame:
    """
    Build a minimal scored-zone DataFrame suitable for build_enforcement_plan.

    Each dict in *zones* should supply:
        zone (str), risk_level (str), traffic_impact_score (float),
        violation_count (int).
    """
    defaults = {
        "zone": "Zone A",
        "risk_level": "Medium",
        "traffic_impact_score": 50.0,
        "violation_count": 10,
    }
    return pd.DataFrame([{**defaults, **z} for z in zones])


def _make_enforcement_df(zones: list[dict]) -> pd.DataFrame:
    """
    Convenience wrapper: build scored_df then run build_enforcement_plan.
    """
    scored = _make_scored_df(zones)
    return build_enforcement_plan(scored)


# ---------------------------------------------------------------------------
# Test class: n_officers=0
# ---------------------------------------------------------------------------

class TestZeroOfficers:
    """
    When n_officers=0 the simulator must return 0.0 overall reduction and an
    empty what-if DataFrame.  (Req 6.1, error-handling spec)
    """

    def test_zero_officers_returns_0_overall_reduction(self):
        """simulate_what_if(n=0) must return overall_expected_reduction_pct = 0.0."""
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "Critical", "traffic_impact_score": 90.0, "violation_count": 50},
            {"zone": "Zone B", "risk_level": "High",     "traffic_impact_score": 70.0, "violation_count": 30},
        ])
        _, overall = simulate_what_if(enforcement_df, n_officers=0)
        assert overall == 0.0

    def test_zero_officers_returns_empty_dataframe(self):
        """simulate_what_if(n=0) must return an empty what-if DataFrame."""
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "High", "traffic_impact_score": 70.0, "violation_count": 20},
        ])
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=0)
        assert len(what_if_df) == 0

    def test_zero_officers_empty_df_has_correct_columns(self):
        """The empty DataFrame returned for n=0 must have the correct schema."""
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "Low", "traffic_impact_score": 20.0, "violation_count": 5},
        ])
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=0)
        expected_cols = {"zone", "officers_allocated", "expected_reduction_pct", "risk_level"}
        assert expected_cols.issubset(set(what_if_df.columns))

    def test_zero_officers_single_zone_returns_0(self):
        """Edge case: single-zone plan with n=0 still returns 0.0."""
        enforcement_df = _make_enforcement_df([
            {"zone": "Z", "risk_level": "Critical", "traffic_impact_score": 95.0, "violation_count": 100},
        ])
        _, overall = simulate_what_if(enforcement_df, n_officers=0)
        assert overall == 0.0


# ---------------------------------------------------------------------------
# Test class: surplus handling (n_officers > total recommended)
# ---------------------------------------------------------------------------

class TestSurplusOfficers:
    """
    When n_officers exceeds the total recommended officers across all zones
    the simulator must not crash and must cap allocation at total recommended.
    (Req 6.4)
    """

    def _total_recommended(self, enforcement_df: pd.DataFrame) -> int:
        return int(enforcement_df["recommended_officers"].sum())

    def test_surplus_does_not_crash(self):
        """
        Passing n_officers > total recommended must not raise any exception.
        (Req 6.4)
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "Critical", "traffic_impact_score": 90.0, "violation_count": 50},
            {"zone": "Zone B", "risk_level": "Medium",   "traffic_impact_score": 50.0, "violation_count": 20},
        ])
        total = self._total_recommended(enforcement_df)
        # Pass way more officers than needed
        what_if_df, overall = simulate_what_if(enforcement_df, n_officers=total + 100)
        # Just verify it returns valid values without exception
        assert isinstance(overall, float)
        assert len(what_if_df) > 0

    def test_surplus_total_allocated_does_not_exceed_recommended(self):
        """
        With surplus, sum(officers_allocated) must equal the total recommended
        (not the inflated n_officers value).  (Req 6.4)
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "High",   "traffic_impact_score": 70.0, "violation_count": 30},
            {"zone": "Zone B", "risk_level": "Low",    "traffic_impact_score": 20.0, "violation_count": 10},
        ])
        total = self._total_recommended(enforcement_df)
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=total + 50)
        assert what_if_df["officers_allocated"].sum() <= total

    def test_surplus_all_zones_are_covered(self):
        """
        When officers exceed total recommended, all zones should be allocated
        at least their recommended count (i.e., all zones appear in results).
        (Req 6.1, 6.4)
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "Critical", "traffic_impact_score": 90.0, "violation_count": 40},
            {"zone": "Zone B", "risk_level": "High",     "traffic_impact_score": 65.0, "violation_count": 20},
            {"zone": "Zone C", "risk_level": "Medium",   "traffic_impact_score": 50.0, "violation_count": 10},
        ])
        total = self._total_recommended(enforcement_df)
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=total + 20)
        assert set(what_if_df["zone"]) == {"Zone A", "Zone B", "Zone C"}

    def test_surplus_reduction_is_capped_at_base_rate(self):
        """
        Even with surplus officers, per-zone reduction must not exceed the
        base reduction rate × 100 for that risk level.  (Req 6.2)
        """
        # Critical base rate = 0.40, so max = 40% for a Critical zone
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "Critical", "traffic_impact_score": 90.0, "violation_count": 50},
        ])
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=9999)
        critical_row = what_if_df[what_if_df["zone"] == "Zone A"].iloc[0]
        assert critical_row["expected_reduction_pct"] <= 40.0

    def test_surplus_stores_correct_surplus_attribute(self):
        """
        simulate_what_if attaches _last_surplus to track how many officers
        were unused.  With surplus, this must be > 0.
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "Low", "traffic_impact_score": 30.0, "violation_count": 5},
        ])
        total = self._total_recommended(enforcement_df)
        simulate_what_if(enforcement_df, n_officers=total + 10)
        assert simulate_what_if._last_surplus == 10  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test class: Critical zone recommended_officers
# ---------------------------------------------------------------------------

class TestCriticalZoneOfficers:
    """
    A Critical-risk zone must receive exactly 5 recommended officers in the
    enforcement plan.  (Req 5.1)
    """

    def test_critical_zone_gets_5_officers(self):
        """
        build_enforcement_plan must assign recommended_officers = 5 to a
        Critical zone.  (Req 5.1)
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Hot Zone", "risk_level": "Critical", "traffic_impact_score": 95.0, "violation_count": 100},
        ])
        row = enforcement_df[enforcement_df["zone"] == "Hot Zone"].iloc[0]
        assert row["recommended_officers"] == 5

    def test_all_risk_levels_get_correct_officers(self):
        """
        All risk levels must map to the correct officer count per lookup table:
        Critical→5, High→3, Medium→2, Low→1.  (Req 5.1)
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "C", "risk_level": "Critical", "traffic_impact_score": 90.0, "violation_count": 40},
            {"zone": "H", "risk_level": "High",     "traffic_impact_score": 70.0, "violation_count": 30},
            {"zone": "M", "risk_level": "Medium",   "traffic_impact_score": 50.0, "violation_count": 20},
            {"zone": "L", "risk_level": "Low",      "traffic_impact_score": 20.0, "violation_count": 10},
        ])
        indexed = enforcement_df.set_index("zone")
        assert indexed.loc["C", "recommended_officers"] == 5
        assert indexed.loc["H", "recommended_officers"] == 3
        assert indexed.loc["M", "recommended_officers"] == 2
        assert indexed.loc["L", "recommended_officers"] == 1

    def test_critical_zone_output_type_is_int64(self):
        """recommended_officers must be int64."""
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "Critical", "traffic_impact_score": 85.0, "violation_count": 50},
        ])
        assert enforcement_df["recommended_officers"].dtype == "int64"

    def test_critical_zone_full_row_values(self):
        """
        A Critical zone row must have officers=5, tow_trucks=2, patrol=1 hr.
        (Req 5.1, 5.2, 5.3)
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Critical Zone", "risk_level": "Critical", "traffic_impact_score": 88.0, "violation_count": 60},
        ])
        row = enforcement_df[enforcement_df["zone"] == "Critical Zone"].iloc[0]
        assert row["recommended_officers"] == 5
        assert row["recommended_tow_trucks"] == 2
        assert row["patrol_frequency_hours"] == 1

    def test_critical_zone_what_if_gets_full_5_officers(self):
        """
        With enough budget, a Critical zone should be allocated all 5 officers.
        (Req 6.1)
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Critical Zone", "risk_level": "Critical", "traffic_impact_score": 90.0, "violation_count": 50},
        ])
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=5)
        row = what_if_df[what_if_df["zone"] == "Critical Zone"].iloc[0]
        assert row["officers_allocated"] == 5


# ---------------------------------------------------------------------------
# Test class: What-if allocates to highest-scored zones first
# ---------------------------------------------------------------------------

class TestWhatIfAllocationOrder:
    """
    The what-if simulator must allocate officers in descending order of
    traffic_impact_score (greedy, highest priority first).  (Req 6.1)
    """

    def test_highest_score_zone_gets_officers_first(self):
        """
        With only enough officers for one zone, the zone with the highest
        traffic_impact_score should receive the allocation.  (Req 6.1)
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Low Zone",  "risk_level": "Low",      "traffic_impact_score": 15.0, "violation_count": 5},
            {"zone": "High Zone", "risk_level": "Critical", "traffic_impact_score": 95.0, "violation_count": 50},
            {"zone": "Med Zone",  "risk_level": "Medium",   "traffic_impact_score": 45.0, "violation_count": 20},
        ])
        # Only 5 officers — enough for just the Critical zone (which needs 5)
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=5)
        assert "High Zone" in what_if_df["zone"].values
        assert "Low Zone" not in what_if_df["zone"].values
        assert "Med Zone" not in what_if_df["zone"].values

    def test_allocation_order_with_three_zones(self):
        """
        Officers fill zones from highest to lowest score.
        Give enough budget for top-2 zones only.  (Req 6.1)
        """
        # Critical needs 5, High needs 3, Low needs 1 → total top-2 = 8
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "Critical", "traffic_impact_score": 90.0, "violation_count": 40},
            {"zone": "Zone B", "risk_level": "High",     "traffic_impact_score": 70.0, "violation_count": 25},
            {"zone": "Zone C", "risk_level": "Low",      "traffic_impact_score": 20.0, "violation_count": 10},
        ])
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=8)
        assert "Zone A" in what_if_df["zone"].values
        assert "Zone B" in what_if_df["zone"].values
        assert "Zone C" not in what_if_df["zone"].values

    def test_partial_allocation_goes_to_highest_zone_first(self):
        """
        When budget is less than a zone's full requirement, the partial
        allocation still goes to the highest-scored zone.  (Req 6.1)
        """
        # Critical zone needs 5, but we only provide 3
        enforcement_df = _make_enforcement_df([
            {"zone": "Top Zone",    "risk_level": "Critical", "traffic_impact_score": 95.0, "violation_count": 50},
            {"zone": "Bottom Zone", "risk_level": "Low",      "traffic_impact_score": 10.0, "violation_count": 5},
        ])
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=3)
        top_row = what_if_df[what_if_df["zone"] == "Top Zone"].iloc[0]
        assert top_row["officers_allocated"] == 3
        # Bottom zone should not be touched since all 3 officers went to Top Zone
        assert "Bottom Zone" not in what_if_df["zone"].values

    def test_allocation_sum_equals_budget_when_budget_is_small(self):
        """
        sum(officers_allocated) must equal n_officers when budget is below
        total recommended.  (Req 6.1)
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "Critical", "traffic_impact_score": 90.0, "violation_count": 40},
            {"zone": "Zone B", "risk_level": "High",     "traffic_impact_score": 65.0, "violation_count": 20},
            {"zone": "Zone C", "risk_level": "Medium",   "traffic_impact_score": 50.0, "violation_count": 10},
        ])
        budget = 6  # less than total (5+3+2=10)
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=budget)
        assert what_if_df["officers_allocated"].sum() == budget

    def test_tied_scores_are_stable_and_complete(self):
        """
        Zones with identical scores must still all be considered; both must
        appear in results when the budget covers them.
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Tie A", "risk_level": "High", "traffic_impact_score": 70.0, "violation_count": 20},
            {"zone": "Tie B", "risk_level": "High", "traffic_impact_score": 70.0, "violation_count": 20},
        ])
        # High needs 3 each → total 6
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=6)
        assert set(what_if_df["zone"]) == {"Tie A", "Tie B"}

    def test_reduction_formula_for_full_allocation(self):
        """
        A zone fully allocated (allocated == recommended) should get
        reduction = base_rate × 100.  For Critical: 40.0%.  (Req 6.2)
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Critical Zone", "risk_level": "Critical", "traffic_impact_score": 90.0, "violation_count": 50},
        ])
        what_if_df, _ = simulate_what_if(enforcement_df, n_officers=5)
        row = what_if_df[what_if_df["zone"] == "Critical Zone"].iloc[0]
        assert row["expected_reduction_pct"] == pytest.approx(40.0)

    def test_overall_reduction_weighted_by_violations(self):
        """
        overall_expected_reduction_pct must be the violation-count-weighted
        mean of per-zone reductions.  (Req 6.3)
        """
        enforcement_df = _make_enforcement_df([
            {"zone": "Zone A", "risk_level": "Critical", "traffic_impact_score": 90.0, "violation_count": 100},
            {"zone": "Zone B", "risk_level": "Low",      "traffic_impact_score": 20.0, "violation_count": 100},
        ])
        # Fully allocate both zones (5 + 1 = 6 officers)
        what_if_df, overall = simulate_what_if(enforcement_df, n_officers=6)

        # Critical full reduction = 40%, Low full reduction = 25%
        # Weighted mean (equal violation counts) = (40 + 25) / 2 = 32.5
        assert overall == pytest.approx(32.5, abs=1e-6)


# ===========================================================================
# Property-Based Tests (Hypothesis)
# ===========================================================================

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_RISK_LEVELS = ["Critical", "High", "Medium", "Low"]

# Resource lookup mirrored from implementation for property assertions.
_RESOURCE_LOOKUP = {
    "Critical": (5, 2, 1),
    "High":     (3, 1, 2),
    "Medium":   (2, 1, 4),
    "Low":      (1, 0, 8),
}

_BASE_RATES = {
    "Critical": 0.40,
    "High":     0.35,
    "Medium":   0.30,
    "Low":      0.25,
}


@st.composite
def _scored_df_strategy(draw, min_zones: int = 1, max_zones: int = 20):
    """
    Strategy that produces a valid scored-zone DataFrame with at least
    *min_zones* rows, suitable for build_enforcement_plan.
    """
    n = draw(st.integers(min_value=min_zones, max_value=max_zones))
    zones = draw(st.lists(st.text(min_size=1, max_size=20), min_size=n, max_size=n))
    risk_levels = draw(st.lists(st.sampled_from(_RISK_LEVELS), min_size=n, max_size=n))
    scores = draw(st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    violation_counts = draw(st.lists(st.integers(min_value=1, max_value=10_000), min_size=n, max_size=n))
    return pd.DataFrame({
        "zone": zones,
        "risk_level": risk_levels,
        "traffic_impact_score": scores,
        "violation_count": violation_counts,
    })


# ---------------------------------------------------------------------------
# Property 8: Enforcement plan resource rules
# Feature: parkpulse-ai, Property 8: resource columns match risk_level lookup
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(scored_df=_scored_df_strategy())
def test_enforcement_resource_rules(scored_df):
    """
    For every row in the enforcement plan, recommended_officers,
    recommended_tow_trucks, and patrol_frequency_hours must match the
    lookup table for the row's risk_level.

    **Validates: Requirements 5.1, 5.2, 5.3**
    """
    # Feature: parkpulse-ai, Property 8: resource columns match risk_level lookup
    plan = build_enforcement_plan(scored_df)

    for _, row in plan.iterrows():
        risk = row["risk_level"]
        expected_officers, expected_tow_trucks, expected_patrol = _RESOURCE_LOOKUP[risk]

        assert row["recommended_officers"] == expected_officers, (
            f"Zone '{row['zone']}' (risk={risk}): "
            f"expected {expected_officers} officers, got {row['recommended_officers']}"
        )
        assert row["recommended_tow_trucks"] == expected_tow_trucks, (
            f"Zone '{row['zone']}' (risk={risk}): "
            f"expected {expected_tow_trucks} tow trucks, got {row['recommended_tow_trucks']}"
        )
        assert row["patrol_frequency_hours"] == expected_patrol, (
            f"Zone '{row['zone']}' (risk={risk}): "
            f"expected patrol freq {expected_patrol} hrs, got {row['patrol_frequency_hours']}"
        )


# ---------------------------------------------------------------------------
# Property 9: Enforcement plan sort order
# Feature: parkpulse-ai, Property 9: enforcement plan sorted descending by traffic_impact_score
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(scored_df=_scored_df_strategy())
def test_enforcement_sort_order(scored_df):
    """
    traffic_impact_score must be non-increasing from the first row to the
    last row of the enforcement plan.

    **Validates: Requirements 5.4**
    """
    # Feature: parkpulse-ai, Property 9: enforcement plan sorted descending by traffic_impact_score
    plan = build_enforcement_plan(scored_df)

    scores = plan["traffic_impact_score"].tolist()
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Sort order violated at position {i}: "
            f"score[{i}]={scores[i]} < score[{i+1}]={scores[i+1]}"
        )


# ---------------------------------------------------------------------------
# Property 10: What-if officer allocation does not exceed N
# Feature: parkpulse-ai, Property 10: sum(officers_allocated) <= n_officers
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    scored_df=_scored_df_strategy(),
    n_officers=st.integers(min_value=0, max_value=500),
)
def test_what_if_officer_budget(scored_df, n_officers):
    """
    The sum of officers_allocated across all zones must never exceed n_officers,
    regardless of the input plan or officer count.

    **Validates: Requirements 6.1, 6.4**
    """
    # Feature: parkpulse-ai, Property 10: sum(officers_allocated) <= n_officers
    plan = build_enforcement_plan(scored_df)
    what_if_df, _overall = simulate_what_if(plan, n_officers=n_officers)

    total_allocated = int(what_if_df["officers_allocated"].sum())
    assert total_allocated <= n_officers, (
        f"Allocated {total_allocated} officers but budget was {n_officers}"
    )


# ---------------------------------------------------------------------------
# Property 11: What-if per-zone reduction bounded
# Feature: parkpulse-ai, Property 11: expected_reduction_pct in [0, 40]
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    scored_df=_scored_df_strategy(),
    n_officers=st.integers(min_value=1, max_value=500),
)
def test_what_if_reduction_bounded(scored_df, n_officers):
    """
    For every row in the what-if results, expected_reduction_pct must be
    in [0, 40] because the maximum base reduction rate is 40 % (Critical)
    and the allocation ratio is capped at 1.0.

    **Validates: Requirements 6.2**
    """
    # Feature: parkpulse-ai, Property 11: expected_reduction_pct in [0, 40]
    plan = build_enforcement_plan(scored_df)
    what_if_df, _overall = simulate_what_if(plan, n_officers=n_officers)

    for _, row in what_if_df.iterrows():
        pct = row["expected_reduction_pct"]
        assert 0.0 <= pct <= 40.0, (
            f"Zone '{row['zone']}' (risk={row['risk_level']}): "
            f"expected_reduction_pct={pct} is outside [0, 40]"
        )


# ---------------------------------------------------------------------------
# Property 14: What-if overall reduction is violation-count-weighted
# Feature: parkpulse-ai, Property 14: overall_expected_reduction_pct == weighted mean of per-zone reductions
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    scored_df=_scored_df_strategy(),
    n_officers=st.integers(min_value=1, max_value=500),
)
def test_what_if_weighted_mean(scored_df, n_officers):
    """
    overall_expected_reduction_pct must equal the violation-count-weighted
    mean of the per-zone expected_reduction_pct values for all covered zones,
    within floating-point tolerance.

    **Validates: Requirements 6.3**
    """
    # Feature: parkpulse-ai, Property 14: overall_expected_reduction_pct == weighted mean of per-zone reductions
    plan = build_enforcement_plan(scored_df)
    what_if_df, overall = simulate_what_if(plan, n_officers=n_officers)

    # No zones covered → overall must be 0.0
    if what_if_df.empty:
        assert overall == 0.0
        return

    # Replicate the greedy allocation order the simulator uses: sort plan
    # descending by traffic_impact_score (stable sort, same as implementation).
    # Then match covered rows positionally — this avoids ambiguity with
    # duplicate zone names.
    sorted_plan = plan.sort_values("traffic_impact_score", ascending=False).reset_index(drop=True)

    reductions = []
    weights = []
    remaining = min(n_officers, int(sorted_plan["recommended_officers"].sum()))

    for _, plan_row in sorted_plan.iterrows():
        if remaining <= 0:
            break
        recommended = int(plan_row["recommended_officers"])
        allocated = min(recommended, remaining)
        remaining -= allocated

        risk = plan_row["risk_level"]
        base_rate = {"Critical": 0.40, "High": 0.35, "Medium": 0.30, "Low": 0.25}.get(risk, 0.25)
        reduction = min(allocated / recommended, 1.0) * base_rate * 100.0
        vc = int(plan_row["violation_count"])

        reductions.append(reduction)
        weights.append(vc)

    reductions_arr = np.array(reductions, dtype=float)
    weights_arr = np.array(weights, dtype=float)

    total_weight = weights_arr.sum()
    if total_weight > 0:
        expected_overall = float((reductions_arr * weights_arr).sum() / total_weight)
    else:
        expected_overall = float(reductions_arr.mean())

    assert abs(overall - expected_overall) < 1e-6, (
        f"overall_expected_reduction_pct={overall} but weighted mean={expected_overall}"
    )
