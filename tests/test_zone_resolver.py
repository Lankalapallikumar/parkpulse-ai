"""
Unit tests for parkpulse_ai/zone_resolver.py

Covers:
- Named junction records get zone_source = "junction" and zone = junction_name  (Req 2.1)
- "No Junction" record within 400m of a centroid gets zone_source = "proximity"  (Req 2.2)
- Record > 600m from all centroids with a usable location string gets
  zone_source = "location_text"  (Req 2.3)
- Empty location string fallback produces "Zone {lat_bin}_{lon_bin}" format  (Req 2.4)
"""

import math
import pandas as pd
import pytest

from parkpulse_ai.zone_resolver import (
    resolve_zones,
    _haversine_distance,
    _compute_junction_centroids,
    _parse_location_token,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_row(**overrides) -> dict:
    """Return a dict with all columns required by resolve_zones."""
    row = {
        "id": "1",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "location": "MG Road, Bangalore",
        "vehicle_number": "KA01AB1234",
        "vehicle_type": "CAR",
        "violation_type": "ILLEGAL PARKING",
        "offence_code": "MV-001",
        "created_datetime": "2024-01-15 09:30:00",
        "junction_name": "MG Road Junction",
    }
    row.update(overrides)
    return row


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# Centroid anchor for geometry tests — a well-known Bangalore location.
_CENTROID_LAT = 12.9716
_CENTROID_LON = 77.5946

# A record ~400 m north of the centroid (well within 500 m).
# 1 degree latitude ≈ 111 km  →  400 m ≈ 0.0036°
_NEAR_LAT = _CENTROID_LAT + 0.0036   # ≈ 400 m away
_NEAR_LON = _CENTROID_LON

# A record ~670 m north of the centroid (beyond 500 m).
# 670 m ≈ 0.0060°
_FAR_LAT = _CENTROID_LAT + 0.0060    # ≈ 670 m away
_FAR_LON = _CENTROID_LON


# ---------------------------------------------------------------------------
# 1. Named junction → zone_source = "junction"  (Requirement 2.1)
# ---------------------------------------------------------------------------

class TestNamedJunction:
    def test_named_junction_sets_zone_source_to_junction(self):
        """A record with a real junction_name must get zone_source='junction'."""
        df = _make_df([_base_row(junction_name="Silk Board Junction")])
        result = resolve_zones(df)
        assert result.iloc[0]["zone_source"] == "junction"

    def test_named_junction_sets_zone_to_junction_name(self):
        """The zone must equal the original junction_name value."""
        jname = "Silk Board Junction"
        df = _make_df([_base_row(junction_name=jname)])
        result = resolve_zones(df)
        assert result.iloc[0]["zone"] == jname

    def test_multiple_named_junctions_each_resolved_correctly(self):
        """Every row with a distinct junction_name must carry its own name."""
        rows = [
            _base_row(id="1", junction_name="Hebbal Flyover"),
            _base_row(id="2", junction_name="Marathahalli Bridge"),
        ]
        result = resolve_zones(_make_df(rows))
        assert list(result["zone_source"]) == ["junction", "junction"]
        assert list(result["zone"]) == ["Hebbal Flyover", "Marathahalli Bridge"]

    def test_no_junction_string_is_not_treated_as_named(self):
        """'No Junction' must NOT receive zone_source='junction'."""
        df = _make_df([_base_row(junction_name="No Junction")])
        result = resolve_zones(df)
        assert result.iloc[0]["zone_source"] != "junction"

    def test_null_junction_name_is_not_treated_as_named(self):
        """A null junction_name must NOT receive zone_source='junction'."""
        df = _make_df([_base_row(junction_name=None)])
        result = resolve_zones(df)
        assert result.iloc[0]["zone_source"] != "junction"


# ---------------------------------------------------------------------------
# 2. Proximity match within 400m → zone_source = "proximity"  (Requirement 2.2)
# ---------------------------------------------------------------------------

class TestProximityMatch:
    def _build_df_with_near_record(self, near_location: str = "Some Road, Area") -> pd.DataFrame:
        """
        One named-junction anchor record establishes a centroid at (_CENTROID_LAT, _CENTROID_LON).
        One "No Junction" record sits ~400 m away.
        """
        rows = [
            # The anchor: establishes the centroid
            _base_row(
                id="anchor",
                latitude=_CENTROID_LAT,
                longitude=_CENTROID_LON,
                junction_name="Centroid Junction",
            ),
            # The target: No Junction, physically close
            _base_row(
                id="near",
                latitude=_NEAR_LAT,
                longitude=_NEAR_LON,
                junction_name="No Junction",
                location=near_location,
            ),
        ]
        return _make_df(rows)

    def test_no_junction_within_400m_gets_proximity_source(self):
        """A 'No Junction' record ~400 m from a named centroid must use proximity."""
        # Sanity-check the distance really is within 500 m.
        dist = _haversine_distance(_NEAR_LAT, _NEAR_LON, _CENTROID_LAT, _CENTROID_LON)
        assert dist < 0.5, f"Test setup error: expected dist < 0.5 km, got {dist:.4f} km"

        result = resolve_zones(self._build_df_with_near_record())
        near_row = result[result["id"] == "near"].iloc[0]
        assert near_row["zone_source"] == "proximity"

    def test_proximity_zone_name_uses_near_prefix(self):
        """The zone value for a proximity match must be 'Near <junction_name>'."""
        result = resolve_zones(self._build_df_with_near_record())
        near_row = result[result["id"] == "near"].iloc[0]
        assert near_row["zone"] == "Near Centroid Junction"

    def test_anchor_itself_retains_junction_source(self):
        """The named-junction anchor row must keep zone_source='junction'."""
        result = resolve_zones(self._build_df_with_near_record())
        anchor_row = result[result["id"] == "anchor"].iloc[0]
        assert anchor_row["zone_source"] == "junction"

    def test_closest_centroid_wins_for_proximity(self):
        """When multiple centroids exist, the nearest one is used."""
        rows = [
            # Two anchors at different distances
            _base_row(id="a1", latitude=_CENTROID_LAT, longitude=_CENTROID_LON,
                      junction_name="Close Junction"),
            _base_row(id="a2", latitude=_CENTROID_LAT + 0.05, longitude=_CENTROID_LON,
                      junction_name="Far Junction"),
            # No Junction record is near "Close Junction"
            _base_row(id="target", latitude=_NEAR_LAT, longitude=_NEAR_LON,
                      junction_name="No Junction", location="Road, Area"),
        ]
        result = resolve_zones(_make_df(rows))
        target = result[result["id"] == "target"].iloc[0]
        assert target["zone"] == "Near Close Junction"


# ---------------------------------------------------------------------------
# 3. > 600m from all centroids + usable location → zone_source = "location_text"
#    (Requirement 2.3)
# ---------------------------------------------------------------------------

class TestLocationTextFallback:
    def _build_df_with_far_record(self, location: str) -> pd.DataFrame:
        """
        One named-junction anchor at the centroid.
        One 'No Junction' record ~670 m away (beyond the 500 m threshold).
        """
        rows = [
            _base_row(
                id="anchor",
                latitude=_CENTROID_LAT,
                longitude=_CENTROID_LON,
                junction_name="Centroid Junction",
            ),
            _base_row(
                id="far",
                latitude=_FAR_LAT,
                longitude=_FAR_LON,
                junction_name="No Junction",
                location=location,
            ),
        ]
        return _make_df(rows)

    def test_far_record_with_location_gets_location_text_source(self):
        """A record > 600 m from all centroids with a valid location must use location_text."""
        dist = _haversine_distance(_FAR_LAT, _FAR_LON, _CENTROID_LAT, _CENTROID_LON)
        assert dist > 0.5, f"Test setup error: expected dist > 0.5 km, got {dist:.4f} km"

        result = resolve_zones(self._build_df_with_far_record("Koramangala, Bangalore"))
        far_row = result[result["id"] == "far"].iloc[0]
        assert far_row["zone_source"] == "location_text"

    def test_location_text_zone_uses_first_useful_token(self):
        """The zone value must be the first comma/slash token with ≥ 4 chars."""
        result = resolve_zones(self._build_df_with_far_record("Koramangala, Bangalore"))
        far_row = result[result["id"] == "far"].iloc[0]
        # "Koramangala" is the first token (len > 4, not a number)
        assert far_row["zone"] == "Koramangala"

    def test_location_text_with_slash_delimiter(self):
        """Slash-delimited location strings must also be parsed correctly."""
        result = resolve_zones(self._build_df_with_far_record("Indiranagar/100 Feet Road"))
        far_row = result[result["id"] == "far"].iloc[0]
        assert far_row["zone_source"] == "location_text"
        assert far_row["zone"] == "Indiranagar"

    def test_location_text_skips_short_tokens(self):
        """Tokens shorter than 4 characters must be skipped; next valid token is used."""
        # "MG" has 2 chars, so it should be skipped; "Road" has 4 chars and is next.
        result = resolve_zones(self._build_df_with_far_record("MG, Road, Bangalore"))
        far_row = result[result["id"] == "far"].iloc[0]
        assert far_row["zone_source"] == "location_text"
        assert far_row["zone"] == "Road"


# ---------------------------------------------------------------------------
# 4. Empty / unusable location string → coordinate_bin fallback  (Requirement 2.4)
# ---------------------------------------------------------------------------

class TestCoordinateBinFallback:
    def _build_df_isolated_no_junction(self, location: str, lat: float, lon: float) -> pd.DataFrame:
        """A single 'No Junction' record with no named junctions anywhere — forces bin."""
        return _make_df([
            _base_row(
                id="target",
                latitude=lat,
                longitude=lon,
                junction_name="No Junction",
                location=location,
            )
        ])

    def test_empty_location_produces_coordinate_bin_source(self):
        """An empty location string must fall back to coordinate_bin."""
        df = self._build_df_isolated_no_junction("", 12.97, 77.59)
        result = resolve_zones(df)
        assert result.iloc[0]["zone_source"] == "coordinate_bin"

    def test_coordinate_bin_zone_format(self):
        """The zone must match 'Zone {round(lat,2)}_{round(lon,2)}'."""
        lat, lon = 12.9716, 77.5946
        expected_lat_bin = round(lat, 2)   # 12.97
        expected_lon_bin = round(lon, 2)   # 77.59
        expected_zone = f"Zone {expected_lat_bin}_{expected_lon_bin}"

        df = self._build_df_isolated_no_junction("", lat, lon)
        result = resolve_zones(df)
        assert result.iloc[0]["zone"] == expected_zone

    def test_whitespace_only_location_produces_coordinate_bin(self):
        """A location that is only whitespace must also fall through to coordinate_bin."""
        df = self._build_df_isolated_no_junction("   ", 12.97, 77.59)
        result = resolve_zones(df)
        assert result.iloc[0]["zone_source"] == "coordinate_bin"

    def test_numeric_only_location_produces_coordinate_bin(self):
        """A location containing only numbers (no useful text) must fall back to bin."""
        # "1234" is 4 chars but is a pure number — should be skipped.
        df = self._build_df_isolated_no_junction("1234, 5678", 12.97, 77.59)
        result = resolve_zones(df)
        assert result.iloc[0]["zone_source"] == "coordinate_bin"

    def test_coordinate_bin_uses_two_decimal_places(self):
        """lat_bin and lon_bin must be rounded to exactly two decimal places."""
        lat, lon = 12.9716123, 77.5946789
        df = self._build_df_isolated_no_junction("", lat, lon)
        result = resolve_zones(df)
        zone = result.iloc[0]["zone"]

        # Zone format: "Zone {lat_bin}_{lon_bin}"
        assert zone.startswith("Zone ")
        parts = zone[5:].split("_")
        assert len(parts) == 2
        lat_bin_str, lon_bin_str = parts
        # Verify the values match round(..., 2)
        assert float(lat_bin_str) == round(lat, 2)
        assert float(lon_bin_str) == round(lon, 2)

    def test_far_record_with_no_usable_location_falls_back_to_bin(self):
        """
        A record > 500 m from all centroids with an unusable location must
        end up with zone_source = 'coordinate_bin', not 'location_text'.
        """
        rows = [
            _base_row(id="anchor", latitude=_CENTROID_LAT, longitude=_CENTROID_LON,
                      junction_name="Centroid Junction"),
            _base_row(id="far", latitude=_FAR_LAT, longitude=_FAR_LON,
                      junction_name="No Junction", location=""),
        ]
        result = resolve_zones(_make_df(rows))
        far_row = result[result["id"] == "far"].iloc[0]
        assert far_row["zone_source"] == "coordinate_bin"


# ---------------------------------------------------------------------------
# 5. zone and zone_source columns are always present  (Requirement 2.5, 2.6)
# ---------------------------------------------------------------------------

class TestOutputColumns:
    def test_zone_column_added(self):
        """resolve_zones must always add a 'zone' column."""
        result = resolve_zones(_make_df([_base_row()]))
        assert "zone" in result.columns

    def test_zone_source_column_added(self):
        """resolve_zones must always add a 'zone_source' column."""
        result = resolve_zones(_make_df([_base_row()]))
        assert "zone_source" in result.columns

    def test_no_null_zones(self):
        """Every row must have a non-null zone value."""
        rows = [
            _base_row(id="1", junction_name="Some Junction"),
            _base_row(id="2", junction_name="No Junction", location="Whitefield, Bangalore"),
            _base_row(id="3", junction_name=None, location=""),
        ]
        result = resolve_zones(_make_df(rows))
        assert result["zone"].notna().all()
        assert (result["zone"] != "").all()


# ---------------------------------------------------------------------------
# 6. Empty DataFrame raises ValueError
# ---------------------------------------------------------------------------

class TestEmptyDataFrameRaises:
    def test_empty_dataframe_raises_value_error(self):
        """resolve_zones must raise ValueError when given an empty DataFrame."""
        empty_df = pd.DataFrame(
            columns=["id", "latitude", "longitude", "location", "junction_name"]
        )
        with pytest.raises(ValueError):
            resolve_zones(empty_df)


# ===========================================================================
# Property-Based Tests (Hypothesis)
# ===========================================================================

from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Valid latitude range: -90 to 90; longitude: -180 to 180
_lat_strategy = st.floats(min_value=-89.9, max_value=89.9, allow_nan=False, allow_infinity=False)
_lon_strategy = st.floats(min_value=-179.9, max_value=179.9, allow_nan=False, allow_infinity=False)

# A strategy for junction names that are neither null nor "No Junction"
_junction_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
    min_size=1,
    max_size=40,
).filter(lambda s: s.strip() and s != "No Junction")

# A strategy for the location string — can be anything, including empty
_location_strategy = st.one_of(
    st.just(""),
    st.just("   "),
    st.just("1234"),
    st.text(min_size=0, max_size=80),
)


def _make_row_dict(
    junction_name,
    lat: float,
    lon: float,
    location: str,
    row_id: str = "1",
) -> dict:
    return {
        "id": row_id,
        "latitude": lat,
        "longitude": lon,
        "location": location,
        "vehicle_number": "KA01AB0001",
        "vehicle_type": "CAR",
        "violation_type": "ILLEGAL PARKING",
        "offence_code": "MV-001",
        "created_datetime": "2024-01-15 09:30:00",
        "junction_name": junction_name,
    }


# ---------------------------------------------------------------------------
# Property 1: Zone coverage completeness
# Feature: parkpulse-ai, Property 1: every row has non-null zone after Zone_Resolver
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    # Build a list of 1-10 rows mixing named junctions and "No Junction" rows
    rows=st.lists(
        st.one_of(
            # Named junction row
            st.builds(
                lambda jname, lat, lon, loc, rid: _make_row_dict(jname, lat, lon, loc, rid),
                jname=_junction_name_strategy,
                lat=_lat_strategy,
                lon=_lon_strategy,
                loc=_location_strategy,
                rid=st.uuids().map(str),
            ),
            # "No Junction" row
            st.builds(
                lambda lat, lon, loc, rid: _make_row_dict("No Junction", lat, lon, loc, rid),
                lat=_lat_strategy,
                lon=_lon_strategy,
                loc=_location_strategy,
                rid=st.uuids().map(str),
            ),
        ),
        min_size=1,
        max_size=10,
    )
)
def test_zone_coverage_completeness(rows):
    """
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

    For any input DataFrame with valid lat/lon, every row must have a
    non-null, non-empty `zone` value after Zone_Resolver runs.
    """
    df = pd.DataFrame(rows)
    result = resolve_zones(df)

    assert "zone" in result.columns, "zone column must be present"
    assert result["zone"].notna().all(), "Every zone must be non-null"
    assert (result["zone"] != "").all(), "Every zone must be non-empty"


# ---------------------------------------------------------------------------
# Property 2: Zone source exhaustiveness
# Feature: parkpulse-ai, Property 2: zone_source is one of four allowed values
# ---------------------------------------------------------------------------

_ALLOWED_SOURCES = {"junction", "proximity", "location_text", "coordinate_bin"}


@settings(max_examples=100)
@given(
    rows=st.lists(
        st.one_of(
            st.builds(
                lambda jname, lat, lon, loc, rid: _make_row_dict(jname, lat, lon, loc, rid),
                jname=_junction_name_strategy,
                lat=_lat_strategy,
                lon=_lon_strategy,
                loc=_location_strategy,
                rid=st.uuids().map(str),
            ),
            st.builds(
                lambda lat, lon, loc, rid: _make_row_dict("No Junction", lat, lon, loc, rid),
                lat=_lat_strategy,
                lon=_lon_strategy,
                loc=_location_strategy,
                rid=st.uuids().map(str),
            ),
        ),
        min_size=1,
        max_size=10,
    )
)
def test_zone_source_exhaustiveness(rows):
    """
    **Validates: Requirements 2.6**

    For any resolved row, `zone_source` must be one of exactly
    {"junction", "proximity", "location_text", "coordinate_bin"}.
    """
    df = pd.DataFrame(rows)
    result = resolve_zones(df)

    assert "zone_source" in result.columns, "zone_source column must be present"
    invalid = result[~result["zone_source"].isin(_ALLOWED_SOURCES)]
    assert invalid.empty, (
        f"Unexpected zone_source values found: {invalid['zone_source'].unique().tolist()}"
    )


# ---------------------------------------------------------------------------
# Property 13: Haversine distance symmetry
# Feature: parkpulse-ai, Property 13: haversine(A,B) == haversine(B,A)
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    lat1=_lat_strategy,
    lon1=_lon_strategy,
    lat2=_lat_strategy,
    lon2=_lon_strategy,
)
def test_haversine_symmetry(lat1, lon1, lat2, lon2):
    """
    **Validates: Requirements 2.2**

    For any two GPS coordinates A and B,
    haversine(A, B) must equal haversine(B, A) within floating-point tolerance.
    """
    d_ab = _haversine_distance(lat1, lon1, lat2, lon2)
    d_ba = _haversine_distance(lat2, lon2, lat1, lon1)

    assert math.isclose(d_ab, d_ba, rel_tol=1e-9, abs_tol=1e-12), (
        f"Haversine asymmetry: d({lat1},{lon1} → {lat2},{lon2}) = {d_ab}, "
        f"d({lat2},{lon2} → {lat1},{lon1}) = {d_ba}"
    )
