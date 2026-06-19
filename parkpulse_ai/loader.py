"""
Loader module for ParkPulse AI.

Reads and validates the Bangalore parking violation CSV into a clean DataFrame.
"""

import os
import logging
from typing import Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Datetime columns to parse
_DATETIME_COLUMNS = [
    "created_datetime",
    "closed_datetime",
    "modified_datetime",
    "action_taken_timestamp",
    "data_sent_to_scita_timestamp",
    "validation_timestamp",
]

# Columns that must be present after loading
_REQUIRED_COLUMNS = [
    "id",
    "latitude",
    "longitude",
    "location",
    "vehicle_number",
    "vehicle_type",
    "violation_type",
    "offence_code",
    "created_datetime",
    "junction_name",
]


def load_dataset(file_path: str) -> Tuple[pd.DataFrame, int]:
    """
    Load and validate the parking violation CSV dataset.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    tuple[pd.DataFrame, int]
        A tuple of (cleaned DataFrame, number of rows dropped during validation).

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If any required columns are missing from the CSV.
    """
    # --- 1. File existence check (Requirement 1.7) ---
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset CSV not found at path: '{file_path}'"
        )

    # --- 2. Read CSV (Requirement 1.1) ---
    df = pd.read_csv(file_path, low_memory=False)

    # Drop description column if present (Requirement 1.1)
    if "description" in df.columns:
        df = df.drop(columns=["description"])

    # --- 3. Validate required columns (Requirement 1.8) ---
    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset CSV is missing required column(s): {missing}"
        )

    # --- 4. Parse datetime columns (Requirements 1.2, 1.3) ---
    for col in _DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- 5. Cast lat/lon to float64, drop invalid rows (Requirements 1.4, 1.5, 1.6) ---
    original_len = len(df)

    # Coerce to numeric — non-castable values become NaN
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").astype("float64")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").astype("float64")

    # Identify rows where either coordinate failed to parse
    bad_mask = df["latitude"].isna() | df["longitude"].isna()
    bad_indices = df.index[bad_mask].tolist()

    if bad_indices:
        logger.info(
            "Dropping %d row(s) with non-float latitude/longitude at indices: %s",
            len(bad_indices),
            bad_indices[:20],  # log first 20 to keep output manageable
        )

    df = df[~bad_mask].reset_index(drop=True)

    dropped_row_count = original_len - len(df)

    return df, dropped_row_count
