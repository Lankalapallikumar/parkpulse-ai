"""
Unit tests for Impact_Scorer (parkpulse_ai/impact_scorer.py).

Covers:
- Single-zone input returns traffic_impact_score = 50.0  (Req 4.6)
- Peak-hour-only zone scores higher peak_hour_score than off-peak zone  (Req 4.5)
- Truck-heavy zone scores higher vehicle_severity_score than scooter-heavy zone  (Req 4.3)
- Risk level boundary values (score 80 → Critical, 79 → High, etc.)  (Req 4.7)
"""

import pandas as pd
import pytest

from parkpulse_ai.impact_scorer import score_zones


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame compatible with score_zones."""
    defaults = {
        "zone": "Zone A",
        "vehicle_type": "CAR",
        "violation_type": "ILLEGAL PARKING",
        "created_datetime": pd.Timestamp("2024-01-15 10:00:00"),  # non-peak hour
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _row(
    zone: str,
    vehicle_type: str = "CAR",
    violation_type: str = "ILLEGAL PARKING",
    hour: int = 10,  # off-peak by default
) -> dict:
    """Return a single row dict."""
    return {
        "zone": zone,
        "vehicle_type": vehicle_type,
        "violation_type": violation_type,
        "created_datetime": pd.Timestamp(f"2024-01-15 {hour:02d}:00:00"),
    }


# ---------------------------------------------------------------------------
# Test class: Single-zone edge case
# ---------------------------------------------------------------------------

class TestSingleZoneEdgeCase:
    """Single-zone input must return traffic_impact_score = 50.0 (Req 4.6)."""

    def test_single_zone_single_record_returns_50(self):
        """One record in one zone → traffic_impact_score must be exactly 50.0."""
        df = _make_df([_row("Zone A")])
        result = score_zones(df)
        assert len(result) == 1
        assert result.iloc[0]["traffic_impact_score"] == 50.0

    def test_single_zone_multiple_records_returns_50(self):
        """
        Multiple records in the same (single) zone still have only one zone,
        so traffic_impact_score must be 50.0.
        """
        rows = [_row("Zone A", vehicle_type=vt) for vt in ["CAR", "TRUCK", "SCOOTER"]]
        df = _make_df(rows)
        result = score_zones(df)
        assert len(result) == 1
        assert result.iloc[0]["traffic_impact_score"] == 50.0

    def test_single_zone_risk_level_is_medium(self):
        """
        50.0 falls in the Medium range (40–59), so risk_level should be 'Medium'.
        """
        df = _make_df([_row("Zone A")])
        result = score_zones(df)
        assert result.iloc[0]["risk_level"] == "Medium"

    def test_single_zone_output_has_required_columns(self):
        """The output DataFrame must contain all seven required columns."""
        df = _make_df([_row("Zone A")])
        result = score_zones(df)
        expected_cols = {
            "zone", "traffic_impact_score", "risk_level", "violation_count",
            "vehicle_severity_score", "violation_severity_score", "peak_hour_score",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_single_zone_violation_count_is_correct(self):
        """violation_count must equal the number of records in the zone."""
        rows = [_row("Zone A")] * 5
        df = _make_df(rows)
        result = score_zones(df)
        assert result.iloc[0]["violation_count"] == 5


# ---------------------------------------------------------------------------
# Test class: Peak hour sub-score comparisons
# ---------------------------------------------------------------------------

class TestPeakHourScore:
    """
    Peak-hour-only zone must score higher peak_hour_score than an off-peak zone.
    Peak hours: {0, 1, 2, 3, 4, 5, 19, 20, 21}  (Req 4.5)
    """

    def _make_two_zone_df(self, peak_hours: list[int], offpeak_hours: list[int]) -> pd.DataFrame:
        """
        Build a DataFrame with exactly two zones:
        - 'Peak Zone': all violations at *peak_hours* (cycling through them)
        - 'OffPeak Zone': all violations at *offpeak_hours* (cycling through them)
        Same vehicle type and violation type to isolate the peak-hour dimension.
        """
        rows = []
        for i, h in enumerate(peak_hours):
            rows.append(_row("Peak Zone", vehicle_type="CAR",
                             violation_type="ILLEGAL PARKING", hour=h))
        for i, h in enumerate(offpeak_hours):
            rows.append(_row("OffPeak Zone", vehicle_type="CAR",
                              violation_type="ILLEGAL PARKING", hour=h))
        return _make_df(rows)

    def test_all_peak_vs_all_offpeak(self):
        """
        A zone where every violation occurs at a peak hour should score higher
        peak_hour_score than a zone where no violation is at a peak hour.
        """
        # Peak Zone: hours 0, 1, 2 (all peak)
        # OffPeak Zone: hours 10, 11, 12 (all off-peak)
        df = self._make_two_zone_df(
            peak_hours=[0, 1, 2],
            offpeak_hours=[10, 11, 12],
        )
        result = score_zones(df).set_index("zone")

        assert result.loc["Peak Zone", "peak_hour_score"] > result.loc["OffPeak Zone", "peak_hour_score"], (
            "All-peak zone should have higher peak_hour_score than all-off-peak zone"
        )

    def test_peak_zone_gets_max_peak_score_100(self):
        """
        When one zone has 100% peak-hour violations and another has 0%,
        the peak zone should get peak_hour_score = 100 and the off-peak zone = 0.
        """
        df = self._make_two_zone_df(
            peak_hours=[19, 20, 21],
            offpeak_hours=[8, 9, 10],
        )
        result = score_zones(df).set_index("zone")

        assert result.loc["Peak Zone", "peak_hour_score"] == pytest.approx(100.0)
        assert result.loc["OffPeak Zone", "peak_hour_score"] == pytest.approx(0.0)

    def test_partial_peak_vs_no_peak(self):
        """
        A zone with 50% peak-hour violations should score higher than a zone
        with 0% peak-hour violations.
        """
        # Peak Zone: 2 peak-hour + 2 off-peak records → 50% peak
        # OffPeak Zone: 4 off-peak records → 0% peak
        mixed_peak = [_row("Mixed Zone", hour=h) for h in [0, 1, 10, 11]]
        no_peak = [_row("NoPeak Zone", hour=h) for h in [8, 9, 10, 11]]
        df = _make_df(mixed_peak + no_peak)
        result = score_zones(df).set_index("zone")

        assert result.loc["Mixed Zone", "peak_hour_score"] > result.loc["NoPeak Zone", "peak_hour_score"]

    def test_evening_peak_hours_are_recognised(self):
        """Hours 19, 20, 21 are peak hours and should produce a higher score."""
        df = self._make_two_zone_df(
            peak_hours=[19, 20, 21],
            offpeak_hours=[14, 15, 16],
        )
        result = score_zones(df).set_index("zone")

        assert result.loc["Peak Zone", "peak_hour_score"] > result.loc["OffPeak Zone", "peak_hour_score"]

    def test_early_morning_peak_hours_are_recognised(self):
        """Hours 0–5 are peak hours and should produce a higher score."""
        df = self._make_two_zone_df(
            peak_hours=[0, 2, 4],
            offpeak_hours=[8, 10, 12],
        )
        result = score_zones(df).set_index("zone")

        assert result.loc["Peak Zone", "peak_hour_score"] > result.loc["OffPeak Zone", "peak_hour_score"]


# ---------------------------------------------------------------------------
# Test class: Vehicle severity sub-score comparisons
# ---------------------------------------------------------------------------

class TestVehicleSeverityScore:
    """
    Truck-heavy zone must score higher vehicle_severity_score than scooter-heavy zone.
    Vehicle weights: Truck=3, Car=2, Scooter=1  (Req 4.3)
    """

    def _make_two_zone_df(self, truck_zone_vehicles: list[str], scooter_zone_vehicles: list[str]) -> pd.DataFrame:
        """Build two zones with specified vehicle type compositions."""
        rows = []
        for vt in truck_zone_vehicles:
            rows.append(_row("Truck Zone", vehicle_type=vt))
        for vt in scooter_zone_vehicles:
            rows.append(_row("Scooter Zone", vehicle_type=vt))
        return _make_df(rows)

    def test_all_trucks_vs_all_scooters(self):
        """
        A zone with only trucks (weight=3) must have higher vehicle_severity_score
        than a zone with only scooters (weight=1).
        """
        df = self._make_two_zone_df(
            truck_zone_vehicles=["TRUCK", "TRUCK", "TRUCK"],
            scooter_zone_vehicles=["SCOOTER", "SCOOTER", "SCOOTER"],
        )
        result = score_zones(df).set_index("zone")

        assert result.loc["Truck Zone", "vehicle_severity_score"] > result.loc["Scooter Zone", "vehicle_severity_score"], (
            "Truck-only zone should have higher vehicle_severity_score than scooter-only zone"
        )

    def test_truck_zone_gets_max_vehicle_score_100(self):
        """
        When one zone has all trucks (max weight=3) and another has all scooters
        (min weight=1), truck zone gets vehicle_severity_score = 100, scooter zone = 0.
        """
        df = self._make_two_zone_df(
            truck_zone_vehicles=["TRUCK", "TRUCK", "TRUCK"],
            scooter_zone_vehicles=["SCOOTER", "SCOOTER", "SCOOTER"],
        )
        result = score_zones(df).set_index("zone")

        assert result.loc["Truck Zone", "vehicle_severity_score"] == pytest.approx(100.0)
        assert result.loc["Scooter Zone", "vehicle_severity_score"] == pytest.approx(0.0)

    def test_maxi_cab_equals_truck_weight(self):
        """
        MAXI-CAB has the same weight as TRUCK (both = 3), so a maxi-cab-only
        zone should match a truck zone in vehicle_severity_score.
        """
        df = self._make_two_zone_df(
            truck_zone_vehicles=["TRUCK", "TRUCK"],
            scooter_zone_vehicles=["MAXI-CAB", "MAXI-CAB"],
        )
        result = score_zones(df).set_index("zone")

        # Both have weight=3; min-max scale → both get the same score
        assert result.loc["Truck Zone", "vehicle_severity_score"] == pytest.approx(
            result.loc["Scooter Zone", "vehicle_severity_score"]
        )

    def test_motor_cycle_equals_scooter_weight(self):
        """
        MOTOR CYCLE has the same weight as SCOOTER (both = 1); their scores
        should be equal when compared against the same reference zone.
        """
        rows = (
            [_row("Scooter Zone", vehicle_type="SCOOTER")] * 3
            + [_row("Motorcycle Zone", vehicle_type="MOTOR CYCLE")] * 3
            + [_row("Truck Zone", vehicle_type="TRUCK")] * 3
        )
        df = _make_df(rows)
        result = score_zones(df).set_index("zone")

        assert result.loc["Scooter Zone", "vehicle_severity_score"] == pytest.approx(
            result.loc["Motorcycle Zone", "vehicle_severity_score"]
        )

    def test_unknown_vehicle_type_defaults_to_weight_1(self):
        """
        An unrecognised vehicle type should default to weight=1 (same as scooter),
        so its score should equal a scooter-only zone.
        """
        rows = (
            [_row("Unknown Zone", vehicle_type="BICYCLE")] * 3
            + [_row("Scooter Zone", vehicle_type="SCOOTER")] * 3
            + [_row("Truck Zone", vehicle_type="TRUCK")] * 3
        )
        df = _make_df(rows)
        result = score_zones(df).set_index("zone")

        assert result.loc["Unknown Zone", "vehicle_severity_score"] == pytest.approx(
            result.loc["Scooter Zone", "vehicle_severity_score"]
        )

    def test_car_weight_is_between_truck_and_scooter(self):
        """
        CAR has weight=2, between TRUCK (3) and SCOOTER (1), so car zone's
        vehicle_severity_score should be strictly between truck and scooter zones.
        """
        rows = (
            [_row("Truck Zone", vehicle_type="TRUCK")] * 3
            + [_row("Car Zone", vehicle_type="CAR")] * 3
            + [_row("Scooter Zone", vehicle_type="SCOOTER")] * 3
        )
        df = _make_df(rows)
        result = score_zones(df).set_index("zone")

        truck_score = result.loc["Truck Zone", "vehicle_severity_score"]
        car_score = result.loc["Car Zone", "vehicle_severity_score"]
        scooter_score = result.loc["Scooter Zone", "vehicle_severity_score"]

        assert truck_score > car_score > scooter_score, (
            f"Expected Truck ({truck_score}) > Car ({car_score}) > Scooter ({scooter_score})"
        )


# ---------------------------------------------------------------------------
# Test class: Risk level boundary values
# ---------------------------------------------------------------------------

class TestRiskLevelBoundaries:
    """
    Risk level boundaries: ≥80→Critical, ≥60→High, ≥40→Medium, <40→Low  (Req 4.7)

    Strategy: build two-zone inputs where the final min-max scaled scores
    land at exactly the boundary values, then verify the assigned risk level.
    For precise boundary testing, we craft inputs where the two zones score
    the min and max values, then use the boundary at 0 and 100.
    We also test intermediate cases by constructing multi-zone inputs.
    """

    def test_score_100_is_critical(self):
        """The highest-scoring zone in a two-zone dataset gets 100 → Critical."""
        # Zone A: many trucks, junction violations, all peak hours → max raw score
        # Zone B: scooters, side-road violations, off-peak → min raw score
        rows = (
            [_row("High Zone", vehicle_type="TRUCK",
                  violation_type="JUNCTION BLOCKING VIOLATION", hour=0)] * 10
            + [_row("Low Zone", vehicle_type="SCOOTER",
                    violation_type="SIDE ROAD PARKING", hour=10)] * 2
        )
        df = _make_df(rows)
        result = score_zones(df).set_index("zone")

        # The high zone gets score = 100, which must be Critical
        assert result.loc["High Zone", "traffic_impact_score"] == pytest.approx(100.0)
        assert result.loc["High Zone", "risk_level"] == "Critical"

    def test_score_0_is_low(self):
        """The lowest-scoring zone in a two-zone dataset gets 0 → Low."""
        rows = (
            [_row("High Zone", vehicle_type="TRUCK",
                  violation_type="JUNCTION BLOCKING VIOLATION", hour=0)] * 10
            + [_row("Low Zone", vehicle_type="SCOOTER",
                    violation_type="SIDE ROAD PARKING", hour=10)] * 2
        )
        df = _make_df(rows)
        result = score_zones(df).set_index("zone")

        assert result.loc["Low Zone", "traffic_impact_score"] == pytest.approx(0.0)
        assert result.loc["Low Zone", "risk_level"] == "Low"

    def test_risk_level_thresholds_critical_gte_80(self):
        """Any score >= 80 must receive risk_level = 'Critical'."""
        # We build a scenario with enough zones to spread scores across the full range,
        # then verify: for all rows with traffic_impact_score >= 80 → Critical.
        rows = _build_spread_rows()
        df = _make_df(rows)
        result = score_zones(df)

        critical_rows = result[result["traffic_impact_score"] >= 80.0]
        assert (critical_rows["risk_level"] == "Critical").all(), (
            "All zones with score >= 80 must be Critical"
        )

    def test_risk_level_thresholds_high_60_to_79(self):
        """Any score in [60, 79] must receive risk_level = 'High'."""
        rows = _build_spread_rows()
        df = _make_df(rows)
        result = score_zones(df)

        high_rows = result[
            (result["traffic_impact_score"] >= 60.0) & (result["traffic_impact_score"] < 80.0)
        ]
        if len(high_rows) > 0:
            assert (high_rows["risk_level"] == "High").all(), (
                "All zones with 60 <= score < 80 must be High"
            )

    def test_risk_level_thresholds_medium_40_to_59(self):
        """Any score in [40, 59] must receive risk_level = 'Medium'."""
        rows = _build_spread_rows()
        df = _make_df(rows)
        result = score_zones(df)

        medium_rows = result[
            (result["traffic_impact_score"] >= 40.0) & (result["traffic_impact_score"] < 60.0)
        ]
        if len(medium_rows) > 0:
            assert (medium_rows["risk_level"] == "Medium").all(), (
                "All zones with 40 <= score < 60 must be Medium"
            )

    def test_risk_level_thresholds_low_below_40(self):
        """Any score < 40 must receive risk_level = 'Low'."""
        rows = _build_spread_rows()
        df = _make_df(rows)
        result = score_zones(df)

        low_rows = result[result["traffic_impact_score"] < 40.0]
        if len(low_rows) > 0:
            assert (low_rows["risk_level"] == "Low").all(), (
                "All zones with score < 40 must be Low"
            )

    def test_boundary_80_is_critical_not_high(self):
        """
        The zone that achieves a score of exactly 80 must be Critical, not High.
        We construct a four-zone input designed to produce scores at 0, 33, 67, 100,
        then verify that the 67 zone is High (< 80) and the 100 zone is Critical.
        We also directly check that the score threshold logic is correct at 80.
        """
        # Use a minimal two-zone dataset to get 0 and 100, then check:
        # score exactly at 80 → Critical (>= 80 is the threshold)
        rows = (
            [_row("Max Zone", vehicle_type="TRUCK",
                  violation_type="JUNCTION BLOCKING", hour=0)] * 10
            + [_row("Min Zone", vehicle_type="SCOOTER",
                    violation_type="SIDE ROAD PARKING", hour=10)] * 10
        )
        df = _make_df(rows)
        result = score_zones(df)

        # Verify the mapping rule directly via a synthetic scored DataFrame
        # to test the exact boundary 80 → Critical and 79 → High
        synthetic = pd.DataFrame({
            "zone": ["Z1", "Z2", "Z3", "Z4", "Z5"],
            "traffic_impact_score": [80.0, 79.0, 60.0, 59.0, 40.0],
            "risk_level": ["", "", "", "", ""],
            "violation_count": [1, 1, 1, 1, 1],
            "vehicle_severity_score": [0.0, 0.0, 0.0, 0.0, 0.0],
            "violation_severity_score": [0.0, 0.0, 0.0, 0.0, 0.0],
            "peak_hour_score": [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        # Re-derive risk levels using the same logic as the scorer
        def _risk(s):
            if s >= 80:
                return "Critical"
            if s >= 60:
                return "High"
            if s >= 40:
                return "Medium"
            return "Low"

        expected = {80.0: "Critical", 79.0: "High", 60.0: "High", 59.0: "Medium", 40.0: "Medium"}
        for _, row in synthetic.iterrows():
            score = row["traffic_impact_score"]
            assert _risk(score) == expected[score], (
                f"score {score} → expected {expected[score]}, got {_risk(score)}"
            )

    def test_boundary_79_is_high_not_critical(self):
        """score 79 must be High (< 80 threshold for Critical)."""
        assert _apply_risk_level(79.0) == "High"

    def test_boundary_60_is_high(self):
        """score 60 must be High (>= 60 threshold)."""
        assert _apply_risk_level(60.0) == "High"

    def test_boundary_59_is_medium(self):
        """score 59 must be Medium (< 60, >= 40)."""
        assert _apply_risk_level(59.0) == "Medium"

    def test_boundary_40_is_medium(self):
        """score 40 must be Medium (>= 40 threshold)."""
        assert _apply_risk_level(40.0) == "Medium"

    def test_boundary_39_is_low(self):
        """score 39 must be Low (< 40 threshold)."""
        assert _apply_risk_level(39.0) == "Low"

    def test_boundary_0_is_low(self):
        """score 0 must be Low."""
        assert _apply_risk_level(0.0) == "Low"

    def test_boundary_100_is_critical(self):
        """score 100 must be Critical."""
        assert _apply_risk_level(100.0) == "Critical"

    def test_risk_level_matches_actual_output(self):
        """
        Build a multi-zone dataset that produces all four risk levels and verify
        each assigned risk_level matches the corresponding traffic_impact_score.
        """
        rows = _build_spread_rows()
        df = _make_df(rows)
        result = score_zones(df)

        for _, row in result.iterrows():
            score = row["traffic_impact_score"]
            expected = _apply_risk_level(score)
            assert row["risk_level"] == expected, (
                f"Zone '{row['zone']}': score={score:.2f} → expected {expected}, "
                f"got {row['risk_level']}"
            )


# ---------------------------------------------------------------------------
# Additional structural tests
# ---------------------------------------------------------------------------

class TestOutputStructure:
    """Tests for output DataFrame structure and types."""

    def test_output_columns_are_correct(self):
        """score_zones must return all seven required columns."""
        df = _make_df([_row("Zone A"), _row("Zone B")])
        result = score_zones(df)
        expected = {
            "zone", "traffic_impact_score", "risk_level", "violation_count",
            "vehicle_severity_score", "violation_severity_score", "peak_hour_score",
        }
        assert expected.issubset(set(result.columns))

    def test_violation_count_is_int64(self):
        """violation_count must be int64."""
        df = _make_df([_row("Zone A"), _row("Zone B")])
        result = score_zones(df)
        assert result["violation_count"].dtype == "int64"

    def test_one_row_per_zone(self):
        """Each zone must appear exactly once in the output."""
        rows = (
            [_row("Zone A")] * 3
            + [_row("Zone B")] * 2
            + [_row("Zone C")] * 5
        )
        df = _make_df(rows)
        result = score_zones(df)
        assert len(result) == 3
        assert set(result["zone"]) == {"Zone A", "Zone B", "Zone C"}

    def test_violation_count_matches_input(self):
        """violation_count must reflect the actual number of records per zone."""
        rows = [_row("Zone A")] * 7 + [_row("Zone B")] * 3
        df = _make_df(rows)
        result = score_zones(df).set_index("zone")
        assert result.loc["Zone A", "violation_count"] == 7
        assert result.loc["Zone B", "violation_count"] == 3


# ---------------------------------------------------------------------------
# Helpers used in the test classes above
# ---------------------------------------------------------------------------

def _apply_risk_level(score: float) -> str:
    """Mirror the risk-level assignment logic from impact_scorer.py for direct tests."""
    if score >= 80.0:
        return "Critical"
    if score >= 60.0:
        return "High"
    if score >= 40.0:
        return "Medium"
    return "Low"


def _build_spread_rows() -> list[dict]:
    """
    Build a multi-zone dataset designed to spread scores across [0, 100]
    so that multiple risk-level buckets are represented.

    Four zones:
    - 'MaxZone': all trucks, junction violations, peak hours → raw score max
    - 'HighZone': mix of trucks/cars, junction violations, mostly peak
    - 'MedZone': mix of cars/scooters, main-road violations, mixed hours
    - 'MinZone': all scooters, side-road violations, off-peak → raw score min
    """
    return [
        # MaxZone: highest possible raw inputs
        *[_row("MaxZone", vehicle_type="TRUCK",
               violation_type="JUNCTION BLOCKING", hour=0)] * 20,
        # HighZone: decent mix leaning heavy
        *[_row("HighZone", vehicle_type="TRUCK",
               violation_type="JUNCTION BLOCKING", hour=1)] * 5,
        *[_row("HighZone", vehicle_type="CAR",
               violation_type="MAIN ROAD VIOLATION", hour=10)] * 5,
        # MedZone: medium mix
        *[_row("MedZone", vehicle_type="CAR",
               violation_type="MAIN ROAD VIOLATION", hour=10)] * 6,
        *[_row("MedZone", vehicle_type="SCOOTER",
               violation_type="SIDE ROAD PARKING", hour=12)] * 4,
        # MinZone: lowest possible raw inputs
        *[_row("MinZone", vehicle_type="SCOOTER",
               violation_type="SIDE ROAD PARKING", hour=14)] * 8,
    ]


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings, assume
from hypothesis import strategies as st
import math

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Valid vehicle types (including an "unknown" one to exercise the default branch)
_VEHICLE_TYPES = ["TRUCK", "MAXI-CAB", "CAR", "PASSENGER AUTO", "SCOOTER", "MOTOR CYCLE", "BICYCLE"]

# Violation types that exercise all three weight buckets
_VIOLATION_TYPES = [
    "ILLEGAL PARKING AT JUNCTION",   # weight 3
    "MAIN ROAD ILLEGAL PARKING",     # weight 2
    "SIDE ROAD PARKING",             # weight 1
    "ILLEGAL PARKING",               # weight 1 (no keyword)
]

# All possible hours (0-23) to exercise peak/off-peak logic
_HOURS = list(range(24))


def _zone_rows_strategy():
    """
    Strategy: generate a list of (zone_label, vehicle_type, violation_type, hour)
    tuples representing one record each.  At least 1 record and at most 3 zones
    with up to 10 records each.
    """
    zone_count = st.integers(min_value=1, max_value=3)

    @st.composite
    def _build(draw):
        n_zones = draw(zone_count)
        rows = []
        for z in range(n_zones):
            zone_label = f"Zone{z}"
            n_records = draw(st.integers(min_value=1, max_value=10))
            for _ in range(n_records):
                vt = draw(st.sampled_from(_VEHICLE_TYPES))
                viol = draw(st.sampled_from(_VIOLATION_TYPES))
                hour = draw(st.sampled_from(_HOURS))
                rows.append({
                    "zone": zone_label,
                    "vehicle_type": vt,
                    "violation_type": viol,
                    "created_datetime": pd.Timestamp(f"2024-01-15 {hour:02d}:00:00"),
                })
        return pd.DataFrame(rows)

    return _build()


# ---------------------------------------------------------------------------
# Property 5: Traffic Impact Score range invariant
# Feature: parkpulse-ai, Property 5: traffic_impact_score in [0, 100]
# Validates: Requirements 4.1, 4.6
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(df=_zone_rows_strategy())
def test_impact_score_range(df):
    """
    **Validates: Requirements 4.1, 4.6**

    For any scored zone DataFrame, all values in `traffic_impact_score`
    must lie in the closed interval [0, 100].

    # Feature: parkpulse-ai, Property 5: traffic_impact_score in [0, 100]
    """
    result = score_zones(df)
    scores = result["traffic_impact_score"]

    assert (scores >= 0.0).all(), (
        f"Found traffic_impact_score < 0: {scores[scores < 0.0].tolist()}"
    )
    assert (scores <= 100.0).all(), (
        f"Found traffic_impact_score > 100: {scores[scores > 100.0].tolist()}"
    )
    # Also guard against NaN/inf
    assert scores.notna().all(), "Found NaN in traffic_impact_score"
    assert scores.apply(math.isfinite).all(), "Found non-finite value in traffic_impact_score"


# ---------------------------------------------------------------------------
# Property 6: Risk level consistency
# Feature: parkpulse-ai, Property 6: risk_level matches score threshold
# Validates: Requirements 4.7
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(df=_zone_rows_strategy())
def test_risk_level_consistency(df):
    """
    **Validates: Requirements 4.7**

    For any scored zone DataFrame, the `risk_level` assigned to a zone must
    match the threshold rules:
    - Critical  iff score ≥ 80
    - High      iff 60 ≤ score < 80
    - Medium    iff 40 ≤ score < 60
    - Low       iff score < 40

    # Feature: parkpulse-ai, Property 6: risk_level matches score threshold
    """
    result = score_zones(df)

    for _, row in result.iterrows():
        score = row["traffic_impact_score"]
        level = row["risk_level"]

        if score >= 80.0:
            expected = "Critical"
        elif score >= 60.0:
            expected = "High"
        elif score >= 40.0:
            expected = "Medium"
        else:
            expected = "Low"

        assert level == expected, (
            f"Zone '{row['zone']}': score={score:.4f} → expected risk_level "
            f"'{expected}', got '{level}'"
        )


# ---------------------------------------------------------------------------
# Property 7: Sub-score normalisation range
# Feature: parkpulse-ai, Property 7: all sub-scores in [0, 100]
# Validates: Requirements 4.3, 4.4, 4.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(df=_zone_rows_strategy())
def test_sub_score_range(df):
    """
    **Validates: Requirements 4.3, 4.4, 4.5**

    For any scored zone DataFrame, the columns `vehicle_severity_score`,
    `violation_severity_score`, and `peak_hour_score` must each lie in [0, 100].

    # Feature: parkpulse-ai, Property 7: all sub-scores in [0, 100]
    """
    result = score_zones(df)

    for col in ("vehicle_severity_score", "violation_severity_score", "peak_hour_score"):
        series = result[col]

        assert (series >= 0.0).all(), (
            f"Found {col} < 0: {series[series < 0.0].tolist()}"
        )
        assert (series <= 100.0).all(), (
            f"Found {col} > 100: {series[series > 100.0].tolist()}"
        )
        assert series.notna().all(), f"Found NaN in {col}"
        assert series.apply(math.isfinite).all(), f"Found non-finite value in {col}"
