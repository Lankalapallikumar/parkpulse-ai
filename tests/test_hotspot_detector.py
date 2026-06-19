"""
Unit tests for Hotspot_Detector.

Covers:
- DBSCAN assigns cluster IDs to tight synthetic GPS clusters (Requirements 3.1, 3.2)
- Folium heatmap HTML file is created at the specified output path (Requirement 3.6)
- All-noise scenario (spread points) returns empty hotspot summary without crashing (Requirement 3.7)
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from parkpulse_ai.hotspot_detector import detect_hotspots, generate_heatmap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame compatible with detect_hotspots."""
    defaults = {
        "zone": "Zone A",
        "violation_type": "PARKING VIOLATION",
        "vehicle_type": "CAR",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _cluster_points(
    center_lat: float,
    center_lon: float,
    n: int,
    jitter: float = 0.0002,
    zone: str = "Zone A",
    violation_type: str = "PARKING VIOLATION",
    vehicle_type: str = "CAR",
) -> list[dict]:
    """
    Return *n* points tightly packed around (center_lat, center_lon).
    jitter of ~0.0002 degrees is about 20 m — well within eps=0.1 km.
    """
    rng = np.random.default_rng(seed=42)
    return [
        {
            "latitude": center_lat + rng.uniform(-jitter, jitter),
            "longitude": center_lon + rng.uniform(-jitter, jitter),
            "zone": zone,
            "violation_type": violation_type,
            "vehicle_type": vehicle_type,
        }
        for _ in range(n)
    ]


def _spread_points(n: int) -> list[dict]:
    """
    Return *n* points spread far apart (> 1 degree apart each) so every
    point is classified as noise by DBSCAN.
    """
    return [
        {
            "latitude": 10.0 + i * 2.0,
            "longitude": 77.0 + i * 2.0,
            "zone": f"Zone_{i}",
            "violation_type": "PARKING VIOLATION",
            "vehicle_type": "CAR",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test class: DBSCAN cluster assignment
# ---------------------------------------------------------------------------

class TestClusterAssignment:
    """Tests for correct cluster_id assignment via DBSCAN (Requirements 3.1, 3.2)."""

    def test_tight_cluster_gets_non_noise_cluster_id(self):
        """Points tightly packed around one location should form a cluster (cluster_id >= 0)."""
        rows = _cluster_points(12.97, 77.59, n=15)
        df = _make_df(rows)

        df_out, _ = detect_hotspots(df, eps_km=0.1, min_samples=5)

        assert "cluster_id" in df_out.columns
        # At least some points should be in a cluster (non-noise)
        assert (df_out["cluster_id"] >= 0).any(), (
            "Expected at least one point in a cluster for tight GPS points"
        )

    def test_two_separated_clusters_get_distinct_ids(self):
        """
        Two geographically separated clusters should receive different cluster IDs.
        ~1 degree of latitude apart is ~111 km — far beyond eps=0.1 km.
        """
        cluster_a = _cluster_points(12.97, 77.59, n=15, zone="Zone A")
        cluster_b = _cluster_points(13.97, 77.59, n=15, zone="Zone B")
        df = _make_df(cluster_a + cluster_b)

        df_out, _ = detect_hotspots(df, eps_km=0.1, min_samples=5)

        clustered = df_out[df_out["cluster_id"] >= 0]
        distinct_ids = clustered["cluster_id"].unique()
        assert len(distinct_ids) == 2, (
            f"Expected 2 distinct cluster IDs for two separated groups, got {distinct_ids}"
        )

    def test_cluster_id_column_present_in_output(self):
        """detect_hotspots must always add a cluster_id column to the returned DataFrame."""
        rows = _cluster_points(12.97, 77.59, n=20, zone="Zone A")
        df = _make_df(rows)

        df_out, _ = detect_hotspots(df, eps_km=0.1, min_samples=5)

        assert "cluster_id" in df_out.columns

    def test_noise_points_get_minus_one(self):
        """Points isolated far from others should be labelled noise (cluster_id = -1)."""
        # One tight cluster plus one lone isolated point far away
        tight = _cluster_points(12.97, 77.59, n=15, zone="Zone A")
        isolated = [
            {
                "latitude": 20.0,
                "longitude": 80.0,
                "zone": "Isolated Zone",
                "violation_type": "PARKING VIOLATION",
                "vehicle_type": "CAR",
            }
        ]
        df = _make_df(tight + isolated)

        df_out, _ = detect_hotspots(df, eps_km=0.1, min_samples=5)

        # The isolated point should be noise
        isolated_row = df_out[df_out["zone"] == "Isolated Zone"]
        assert len(isolated_row) == 1
        assert isolated_row.iloc[0]["cluster_id"] == -1, (
            "Isolated point far from all others should get cluster_id = -1"
        )

    def test_hotspot_summary_has_required_columns(self):
        """Hotspot summary must contain all five required columns."""
        rows = _cluster_points(12.97, 77.59, n=15, zone="Zone A")
        df = _make_df(rows)

        _, summary = detect_hotspots(df, eps_km=0.1, min_samples=5)

        expected_cols = {
            "zone", "violation_count", "cluster_count",
            "top_violation_type", "top_vehicle_type",
        }
        assert expected_cols.issubset(set(summary.columns)), (
            f"Missing columns: {expected_cols - set(summary.columns)}"
        )

    def test_zone_with_enough_records_appears_in_summary(self):
        """A zone with >= 5 records must appear in the hotspot summary."""
        rows = _cluster_points(12.97, 77.59, n=15, zone="Dense Zone")
        df = _make_df(rows)

        _, summary = detect_hotspots(df, eps_km=0.1, min_samples=5)

        assert "Dense Zone" in summary["zone"].values, (
            "Zone with 15 records should appear in hotspot summary"
        )


# ---------------------------------------------------------------------------
# Test class: Heatmap HTML generation
# ---------------------------------------------------------------------------

class TestHeatmapGeneration:
    """Tests for generate_heatmap output (Requirement 3.6)."""

    def test_heatmap_file_is_created(self):
        """generate_heatmap must write an HTML file at the specified output path."""
        rows = _cluster_points(12.97, 77.59, n=10)
        df = _make_df(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "heatmap.html")
            generate_heatmap(df, output_path)
            assert os.path.exists(output_path), (
                f"Expected heatmap HTML file at {output_path}"
            )

    def test_heatmap_file_is_non_empty(self):
        """The generated HTML file must contain content (not empty)."""
        rows = _cluster_points(12.97, 77.59, n=10)
        df = _make_df(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "heatmap.html")
            generate_heatmap(df, output_path)
            assert os.path.getsize(output_path) > 0, "Heatmap HTML file should not be empty"

    def test_heatmap_file_contains_html(self):
        """The generated file must be valid HTML (contains '<html' tag)."""
        rows = _cluster_points(12.97, 77.59, n=10)
        df = _make_df(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "heatmap.html")
            generate_heatmap(df, output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "<html" in content.lower(), "Generated file should contain HTML content"

    def test_heatmap_created_at_custom_path(self):
        """generate_heatmap should respect the specified output path, including subdirectories."""
        rows = _cluster_points(12.97, 77.59, n=10)
        df = _make_df(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "output", "maps")
            os.makedirs(subdir, exist_ok=True)
            output_path = os.path.join(subdir, "my_heatmap.html")
            generate_heatmap(df, output_path)
            assert os.path.exists(output_path), (
                f"Expected heatmap at custom path {output_path}"
            )

    def test_heatmap_works_without_zone_column(self):
        """generate_heatmap should not crash when the DataFrame has no 'zone' column."""
        df = pd.DataFrame({
            "latitude": [12.97, 12.971, 12.972],
            "longitude": [77.59, 77.591, 77.592],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "heatmap_no_zone.html")
            generate_heatmap(df, output_path)
            assert os.path.exists(output_path)


# ---------------------------------------------------------------------------
# Test class: All-noise scenario
# ---------------------------------------------------------------------------

class TestAllNoiseScenario:
    """
    Tests for the edge case where all GPS points are spread far apart so
    DBSCAN classifies every point as noise (Requirement 3.7).
    """

    def test_all_noise_does_not_crash(self):
        """detect_hotspots must return without raising an exception when all points are noise."""
        rows = _spread_points(n=5)
        df = _make_df(rows)

        # Should not raise
        df_out, summary = detect_hotspots(df, eps_km=0.1, min_samples=10)

    def test_all_noise_returns_dataframe_pair(self):
        """detect_hotspots must return a 2-tuple of DataFrames even in all-noise case."""
        rows = _spread_points(n=5)
        df = _make_df(rows)

        result = detect_hotspots(df, eps_km=0.1, min_samples=10)

        assert isinstance(result, tuple) and len(result) == 2
        df_out, summary = result
        assert isinstance(df_out, pd.DataFrame)
        assert isinstance(summary, pd.DataFrame)

    def test_all_noise_cluster_ids_are_minus_one(self):
        """In the all-noise case every cluster_id must be -1."""
        rows = _spread_points(n=5)
        df = _make_df(rows)

        df_out, _ = detect_hotspots(df, eps_km=0.1, min_samples=10)

        assert (df_out["cluster_id"] == -1).all(), (
            "All spread-out points should be labelled noise (cluster_id = -1)"
        )

    def test_all_noise_with_few_records_per_zone_returns_empty_summary(self):
        """
        When every zone has fewer than 5 records (and all are noise),
        the hotspot summary must be empty (no rows).
        Validates Requirement 3.7: zones with < 5 records are excluded.
        """
        # 5 isolated points, each in a different zone — each zone has 1 record
        rows = _spread_points(n=5)
        df = _make_df(rows)

        _, summary = detect_hotspots(df, eps_km=0.1, min_samples=10)

        assert len(summary) == 0, (
            "Hotspot summary should be empty when all zones have fewer than 5 records"
        )

    def test_all_noise_empty_summary_has_correct_columns(self):
        """Even an empty hotspot summary must contain the five required columns."""
        rows = _spread_points(n=5)
        df = _make_df(rows)

        _, summary = detect_hotspots(df, eps_km=0.1, min_samples=10)

        expected_cols = {
            "zone", "violation_count", "cluster_count",
            "top_violation_type", "top_vehicle_type",
        }
        assert expected_cols.issubset(set(summary.columns)), (
            f"Empty summary is missing columns: {expected_cols - set(summary.columns)}"
        )

    def test_all_noise_spread_points_each_zone_one_record(self):
        """
        If we spread points so each zone only has 1 record,
        no zone should appear in the summary (all filtered by the < 5 threshold).
        """
        # 20 widely-spread points, each with a unique zone name
        rows = [
            {
                "latitude": 10.0 + i * 1.5,
                "longitude": 77.0 + i * 1.5,
                "zone": f"Unique_Zone_{i}",
                "violation_type": "PARKING VIOLATION",
                "vehicle_type": "CAR",
            }
            for i in range(20)
        ]
        df = _make_df(rows)

        _, summary = detect_hotspots(df, eps_km=0.1, min_samples=5)

        assert len(summary) == 0, (
            "No zone should appear in summary when each zone has only 1 record"
        )


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Shared strategy: generate a minimal DataFrame for detect_hotspots
# ---------------------------------------------------------------------------

# Bangalore-ish lat/lon bounds to keep coordinates plausible
_LAT = st.floats(min_value=12.0, max_value=14.0, allow_nan=False, allow_infinity=False)
_LON = st.floats(min_value=77.0, max_value=79.0, allow_nan=False, allow_infinity=False)
_ZONE = st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" _-"), min_size=1, max_size=20)
_VTYPE = st.sampled_from(["PARKING VIOLATION", "JUNCTION BLOCKING", "MAIN ROAD PARKING", "SIDE ROAD PARKING"])
_VEHICLE = st.sampled_from(["CAR", "SCOOTER", "TRUCK", "MOTOR CYCLE"])


def _row_strategy():
    """Strategy for a single row dict."""
    return st.fixed_dictionaries({
        "latitude": _LAT,
        "longitude": _LON,
        "zone": _ZONE,
        "violation_type": _VTYPE,
        "vehicle_type": _VEHICLE,
    })


@st.composite
def _hotspot_df(draw, min_rows=1, max_rows=60):
    """Generate a DataFrame with 1–60 rows for hotspot detection."""
    rows = draw(st.lists(_row_strategy(), min_size=min_rows, max_size=max_rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Property 3: Hotspot summary excludes sparse zones
# Feature: parkpulse-ai, Property 3: hotspot summary only contains zones with violation_count >= 5
# Validates: Requirements 3.7
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(df=_hotspot_df())
def test_hotspot_excludes_sparse_zones(df):
    """
    Property 3: For any hotspot summary produced by detect_hotspots, every row
    must have violation_count >= 5.

    Validates: Requirements 3.7
    """
    _, summary = detect_hotspots(df, eps_km=0.1, min_samples=5)

    if len(summary) == 0:
        return  # empty summary trivially satisfies the property

    assert (summary["violation_count"] >= 5).all(), (
        f"Hotspot summary contains zones with fewer than 5 violations:\n"
        f"{summary[summary['violation_count'] < 5]}"
    )


# ---------------------------------------------------------------------------
# Property 4: Hotspot summary sort order
# Feature: parkpulse-ai, Property 4: hotspot summary sorted descending by violation_count
# Validates: Requirements 3.4
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(df=_hotspot_df())
def test_hotspot_sort_order(df):
    """
    Property 4: For any hotspot summary produced by detect_hotspots,
    violation_count must be non-increasing (sorted descending).

    Validates: Requirements 3.4
    """
    _, summary = detect_hotspots(df, eps_km=0.1, min_samples=5)

    if len(summary) <= 1:
        return  # trivially sorted

    counts = summary["violation_count"].tolist()
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1], (
            f"violation_count not non-increasing at index {i}: "
            f"{counts[i]} < {counts[i + 1]}\nFull counts: {counts}"
        )
