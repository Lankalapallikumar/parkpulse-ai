"""
Unit tests for parkpulse_ai/loader.py

Covers:
- FileNotFoundError for missing file path (Requirement 1.7)
- ValueError listing missing columns for incomplete CSV (Requirement 1.8)
- NaT assignment for unparseable datetime strings (Requirement 1.3)
- Non-float lat/lon rows are dropped without crash (Requirement 1.5)
"""

import io
import os
import textwrap
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from parkpulse_ai.loader import load_dataset, _REQUIRED_COLUMNS, _DATETIME_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_path: Path, content: str) -> str:
    """Write a CSV string to a temp file and return its path."""
    f = tmp_path / "test_data.csv"
    f.write_text(textwrap.dedent(content))
    return str(f)


def _minimal_row(**overrides) -> dict:
    """Return a dict with all required columns populated with valid defaults."""
    base = {
        "id": "1",
        "latitude": "12.9716",
        "longitude": "77.5946",
        "location": "MG Road, Bangalore",
        "vehicle_number": "KA01AB1234",
        "vehicle_type": "CAR",
        "violation_type": "ILLEGAL PARKING",
        "offence_code": "MV-001",
        "created_datetime": "2024-01-15 09:30:00",
        "junction_name": "MG Road Junction",
    }
    base.update(overrides)
    return base


def _make_csv_from_rows(rows: list[dict]) -> str:
    """Build a CSV string from a list of row dicts (all must share the same keys)."""
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


# ---------------------------------------------------------------------------
# 1. FileNotFoundError for missing file path  (Requirement 1.7)
# ---------------------------------------------------------------------------

class TestFileNotFound:
    def test_raises_file_not_found_error_for_nonexistent_path(self):
        """load_dataset must raise FileNotFoundError when the path doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/to/data.csv")

    def test_error_message_contains_path(self):
        """The FileNotFoundError message should include the bad path for debuggability."""
        bad_path = "/tmp/this_file_does_not_exist_parkpulse.csv"
        with pytest.raises(FileNotFoundError, match=bad_path):
            load_dataset(bad_path)


# ---------------------------------------------------------------------------
# 2. ValueError listing missing columns  (Requirement 1.8)
# ---------------------------------------------------------------------------

class TestMissingColumns:
    def test_raises_value_error_when_required_column_absent(self, tmp_path):
        """A CSV missing one required column must raise ValueError."""
        row = _minimal_row()
        del row["latitude"]           # remove a required column
        csv_content = _make_csv_from_rows([row])
        csv_path = _write_csv(tmp_path, csv_content)

        with pytest.raises(ValueError):
            load_dataset(csv_path)

    def test_error_message_lists_missing_column(self, tmp_path):
        """ValueError message must name the missing column(s)."""
        row = _minimal_row()
        del row["junction_name"]
        csv_content = _make_csv_from_rows([row])
        csv_path = _write_csv(tmp_path, csv_content)

        with pytest.raises(ValueError, match="junction_name"):
            load_dataset(csv_path)

    def test_error_message_lists_multiple_missing_columns(self, tmp_path):
        """ValueError message must list every missing column when several are absent."""
        row = _minimal_row()
        del row["latitude"]
        del row["longitude"]
        del row["vehicle_type"]
        csv_content = _make_csv_from_rows([row])
        csv_path = _write_csv(tmp_path, csv_content)

        with pytest.raises(ValueError) as exc_info:
            load_dataset(csv_path)

        error_msg = str(exc_info.value)
        assert "latitude" in error_msg
        assert "longitude" in error_msg
        assert "vehicle_type" in error_msg

    def test_valid_csv_does_not_raise(self, tmp_path):
        """A CSV with all required columns must load without raising."""
        csv_content = _make_csv_from_rows([_minimal_row()])
        csv_path = _write_csv(tmp_path, csv_content)

        df, dropped = load_dataset(csv_path)
        assert len(df) == 1
        assert dropped == 0


# ---------------------------------------------------------------------------
# 3. NaT for unparseable datetime strings  (Requirement 1.3)
# ---------------------------------------------------------------------------

class TestDatetimeParsing:
    def test_unparseable_datetime_becomes_nat(self, tmp_path):
        """An unparseable created_datetime should be NaT, not crash the loader."""
        row = _minimal_row(created_datetime="not-a-date")
        csv_content = _make_csv_from_rows([row])
        csv_path = _write_csv(tmp_path, csv_content)

        df, dropped = load_dataset(csv_path)

        assert len(df) == 1, "Row must not be dropped for a bad datetime"
        assert dropped == 0
        assert pd.isna(df["created_datetime"].iloc[0])

    def test_valid_datetime_is_parsed_correctly(self, tmp_path):
        """A well-formed datetime string must be parsed to a proper datetime64."""
        row = _minimal_row(created_datetime="2024-03-20 14:45:00")
        csv_content = _make_csv_from_rows([row])
        csv_path = _write_csv(tmp_path, csv_content)

        df, _ = load_dataset(csv_path)

        assert pd.api.types.is_datetime64_any_dtype(df["created_datetime"])
        assert df["created_datetime"].iloc[0] == pd.Timestamp("2024-03-20 14:45:00")

    def test_all_datetime_columns_coerced(self, tmp_path):
        """Every datetime column should be NaT (not raise) when given garbage input."""
        # Build a row that includes all datetime columns with bad values
        row = _minimal_row(
            created_datetime="bad",
            closed_datetime="bad",
            modified_datetime="bad",
            action_taken_timestamp="bad",
            data_sent_to_scita_timestamp="bad",
            validation_timestamp="bad",
        )
        csv_content = _make_csv_from_rows([row])
        csv_path = _write_csv(tmp_path, csv_content)

        df, dropped = load_dataset(csv_path)

        assert dropped == 0, "Bad datetimes must not cause row drops"
        for col in _DATETIME_COLUMNS:
            assert pd.isna(df[col].iloc[0]), f"Expected NaT for column {col}"

    def test_mixed_datetime_rows_handled_gracefully(self, tmp_path):
        """A mix of valid and invalid datetimes across rows must both survive."""
        rows = [
            _minimal_row(created_datetime="2024-01-01 08:00:00"),
            _minimal_row(id="2", created_datetime="INVALID"),
            _minimal_row(id="3", created_datetime="2024-02-15 17:30:00"),
        ]
        csv_content = _make_csv_from_rows(rows)
        csv_path = _write_csv(tmp_path, csv_content)

        df, dropped = load_dataset(csv_path)

        assert len(df) == 3
        assert dropped == 0
        assert pd.notna(df["created_datetime"].iloc[0])
        assert pd.isna(df["created_datetime"].iloc[1])
        assert pd.notna(df["created_datetime"].iloc[2])


# ---------------------------------------------------------------------------
# 4. Non-float lat/lon rows dropped without crash  (Requirement 1.5)
# ---------------------------------------------------------------------------

class TestLatLonDropping:
    def test_non_float_latitude_drops_row(self, tmp_path):
        """A row with a non-numeric latitude must be silently dropped."""
        rows = [
            _minimal_row(id="1"),                          # valid
            _minimal_row(id="2", latitude="N/A"),          # bad lat
        ]
        csv_content = _make_csv_from_rows(rows)
        csv_path = _write_csv(tmp_path, csv_content)

        df, dropped = load_dataset(csv_path)

        assert len(df) == 1
        assert dropped == 1

    def test_non_float_longitude_drops_row(self, tmp_path):
        """A row with a non-numeric longitude must be silently dropped."""
        rows = [
            _minimal_row(id="1"),
            _minimal_row(id="2", longitude="unknown"),
        ]
        csv_content = _make_csv_from_rows(rows)
        csv_path = _write_csv(tmp_path, csv_content)

        df, dropped = load_dataset(csv_path)

        assert len(df) == 1
        assert dropped == 1

    def test_both_bad_lat_and_lon_counts_as_one_drop(self, tmp_path):
        """A row bad in both lat and lon counts as a single dropped row."""
        rows = [
            _minimal_row(id="1"),
            _minimal_row(id="2", latitude="abc", longitude="xyz"),
        ]
        csv_content = _make_csv_from_rows(rows)
        csv_path = _write_csv(tmp_path, csv_content)

        df, dropped = load_dataset(csv_path)

        assert len(df) == 1
        assert dropped == 1

    def test_multiple_bad_rows_all_dropped(self, tmp_path):
        """Multiple invalid lat/lon rows must all be dropped and counted."""
        rows = [
            _minimal_row(id="1"),
            _minimal_row(id="2", latitude="bad"),
            _minimal_row(id="3", longitude="bad"),
            _minimal_row(id="4", latitude="bad", longitude="bad"),
            _minimal_row(id="5"),
        ]
        csv_content = _make_csv_from_rows(rows)
        csv_path = _write_csv(tmp_path, csv_content)

        df, dropped = load_dataset(csv_path)

        assert len(df) == 2
        assert dropped == 3

    def test_valid_float_strings_are_kept(self, tmp_path):
        """String representations of floats must be cast and the row retained."""
        row = _minimal_row(latitude="12.971600", longitude="77.594600")
        csv_content = _make_csv_from_rows([row])
        csv_path = _write_csv(tmp_path, csv_content)

        df, dropped = load_dataset(csv_path)

        assert dropped == 0
        assert df["latitude"].dtype == "float64"
        assert df["longitude"].dtype == "float64"
        assert abs(df["latitude"].iloc[0] - 12.9716) < 1e-6

    def test_dropped_count_matches_row_difference(self, tmp_path):
        """dropped_row_count must equal original row count minus returned row count."""
        rows = [
            _minimal_row(id=str(i), latitude="bad" if i % 2 == 0 else "12.9716")
            for i in range(10)
        ]
        csv_content = _make_csv_from_rows(rows)
        csv_path = _write_csv(tmp_path, csv_content)

        df, dropped = load_dataset(csv_path)

        assert dropped == 5
        assert len(df) == 5
        assert dropped + len(df) == 10

    def test_no_crash_when_all_rows_invalid(self, tmp_path):
        """Loader must return an empty DataFrame (not crash) when all lat/lon are bad."""
        rows = [
            _minimal_row(id="1", latitude="bad"),
            _minimal_row(id="2", longitude="bad"),
        ]
        csv_content = _make_csv_from_rows(rows)
        csv_path = _write_csv(tmp_path, csv_content)

        df, dropped = load_dataset(csv_path)

        assert len(df) == 0
        assert dropped == 2


# ---------------------------------------------------------------------------
# 5. description column is dropped  (Requirement 1.1)
# ---------------------------------------------------------------------------

class TestDescriptionColumnDropped:
    def test_description_column_removed_when_present(self, tmp_path):
        """The `description` column must be dropped from the output DataFrame."""
        row = _minimal_row()
        row["description"] = "Some descriptive text"
        csv_content = _make_csv_from_rows([row])
        csv_path = _write_csv(tmp_path, csv_content)

        df, _ = load_dataset(csv_path)

        assert "description" not in df.columns

    def test_no_error_when_description_absent(self, tmp_path):
        """Loader must not crash if `description` is already absent."""
        csv_content = _make_csv_from_rows([_minimal_row()])
        csv_path = _write_csv(tmp_path, csv_content)

        df, _ = load_dataset(csv_path)
        assert "description" not in df.columns


# ---------------------------------------------------------------------------
# Property-Based Tests  (Hypothesis)
# ---------------------------------------------------------------------------

# Feature: parkpulse-ai, Property 12: dropped_row_count == original_len - len(returned_df)

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames


def _valid_lat() -> st.SearchStrategy:
    """Floats in a realistic latitude range."""
    return st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False)


def _valid_lon() -> st.SearchStrategy:
    """Floats in a realistic longitude range."""
    return st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)


def _invalid_coord() -> st.SearchStrategy:
    """Strings that cannot be parsed as float."""
    return st.one_of(
        st.just("N/A"),
        st.just("unknown"),
        st.just("bad"),
        st.just("--"),
        st.just(""),
        st.text(min_size=1, max_size=8).filter(
            lambda s: s.strip() and not _is_float(s)
        ),
    )


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


@settings(max_examples=100)
@given(
    rows=st.lists(
        st.fixed_dictionaries(
            {
                # Required columns with valid defaults
                "id": st.integers(min_value=1, max_value=10_000).map(str),
                "location": st.just("MG Road, Bangalore"),
                "vehicle_number": st.just("KA01AB1234"),
                "vehicle_type": st.just("CAR"),
                "violation_type": st.just("ILLEGAL PARKING"),
                "offence_code": st.just("MV-001"),
                "created_datetime": st.just("2024-01-15 09:30:00"),
                "junction_name": st.just("MG Road Junction"),
                # lat/lon may be valid floats or invalid strings
                "latitude": st.one_of(
                    _valid_lat().map(str),
                    _invalid_coord(),
                ),
                "longitude": st.one_of(
                    _valid_lon().map(str),
                    _invalid_coord(),
                ),
            }
        ),
        min_size=1,
        max_size=30,
    )
)
def test_dropped_row_count(rows):
    """
    Property 12: dropped_row_count is non-negative and equals
    original_len - len(returned_df) for any combination of valid/invalid lat/lon.

    Validates: Requirements 1.5, 1.6
    """
    import csv
    import tempfile

    # Write rows to a temp CSV using a standard tempfile (not pytest's tmp_path)
    # so Hypothesis can manage multiple examples without fixture issues.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as fh:
        csv_path = fh.name
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    try:
        original_len = len(rows)
        df, dropped = load_dataset(csv_path)

        # Property: non-negative
        assert dropped >= 0, f"dropped_row_count must be >= 0, got {dropped}"

        # Property: consistent with actual output
        assert dropped == original_len - len(df), (
            f"Expected dropped={original_len - len(df)}, got {dropped}. "
            f"original_len={original_len}, returned_len={len(df)}"
        )
    finally:
        import os
        os.unlink(csv_path)
