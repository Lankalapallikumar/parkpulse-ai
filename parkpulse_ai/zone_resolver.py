"""
Zone_Resolver: assigns a meaningful zone name to every violation record.

Resolution priority per record:
1. junction_name not "No Junction" and not null → zone_source = "junction"
2. Nearest named centroid ≤ 0.5 km → zone_source = "proximity", zone = "Near {junction}"
3. Parse first useful token from location string → zone_source = "location_text"
4. "Zone {round(lat,2)}_{round(lon,2)}" → zone_source = "coordinate_bin"
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return the great-circle distance in kilometres between two GPS coordinates.

    Uses the Haversine formula:
        a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
        d = 2R · arcsin(√a)
    where R = 6371 km.
    """
    R = 6371.0  # Earth radius in km

    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2.0 * math.asin(math.sqrt(a))
    return R * c


def _compute_junction_centroids(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    Compute the mean GPS centroid for every named junction.

    A "named junction" is any row whose `junction_name` is not null
    and not equal to "No Junction".

    Returns
    -------
    dict mapping junction_name → (mean_lat, mean_lon)
    """
    mask = df["junction_name"].notna() & (df["junction_name"] != "No Junction")
    named = df.loc[mask, ["junction_name", "latitude", "longitude"]]

    centroids: dict[str, tuple[float, float]] = {}
    for junction, group in named.groupby("junction_name"):
        mean_lat = float(group["latitude"].mean())
        mean_lon = float(group["longitude"].mean())
        centroids[junction] = (mean_lat, mean_lon)

    return centroids


def _parse_location_token(location: object) -> str | None:
    """
    Extract the first useful token from a location string.

    Splitting strategy:
    1. Split on ',' to get comma-separated segments.
    2. For each segment, split further on '/' to get sub-tokens.
    3. Strip whitespace from each token.
    4. Return the first token that:
       - has ≥ 4 characters, AND
       - is not a pure number (int or float).

    Returns None if no usable token is found.
    """
    if not isinstance(location, str) or not location.strip():
        return None

    # Split first by comma, then by slash within each comma segment.
    tokens: list[str] = []
    for comma_part in location.split(","):
        for slash_part in comma_part.split("/"):
            token = slash_part.strip()
            if token:
                tokens.append(token)

    for token in tokens:
        if len(token) < 4:
            continue
        # Skip pure numbers (integer or decimal).
        try:
            float(token)
            continue  # it's a number — skip
        except ValueError:
            pass
        return token

    return None


def resolve_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `zone` and `zone_source` columns to *df* according to the four-level
    resolution priority.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: junction_name, latitude, longitude, location.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with `zone` and `zone_source` columns appended
        (existing columns are overwritten if already present).

    Raises
    ------
    ValueError
        If *df* is empty.
    """
    if df.empty:
        raise ValueError("resolve_zones received an empty DataFrame.")

    # ------------------------------------------------------------------ #
    # Step 1: Classify records that have a named junction directly.       #
    # ------------------------------------------------------------------ #
    junction_mask = df["junction_name"].notna() & (df["junction_name"] != "No Junction")

    # Initialise output arrays with empty strings; we'll fill progressively.
    zones = np.full(len(df), "", dtype=object)
    sources = np.full(len(df), "", dtype=object)

    # Apply priority 1: named junction.
    zones[junction_mask.values] = df.loc[junction_mask, "junction_name"].values
    sources[junction_mask.values] = "junction"

    # ------------------------------------------------------------------ #
    # Step 2: Compute centroids from the records resolved in step 1.     #
    # ------------------------------------------------------------------ #
    # We need a temporary column to compute centroids correctly (Req 2.7).
    temp_df = df.copy()
    temp_df["zone"] = zones
    temp_df["zone_source"] = sources

    centroids = _compute_junction_centroids(df)  # uses original junction_name column

    # ------------------------------------------------------------------ #
    # Step 3: Resolve remaining records (those NOT resolved in step 1).  #
    # ------------------------------------------------------------------ #
    remaining_mask = ~junction_mask

    if remaining_mask.any() and centroids:
        # Chunked vectorised proximity search — processes CHUNK_SIZE rows at a
        # time to keep peak RAM under ~200 MB on Streamlit Cloud free tier.
        CHUNK_SIZE = 5000

        centroid_names = list(centroids.keys())
        centroid_lats = np.radians(np.array([centroids[n][0] for n in centroid_names]))
        centroid_lons = np.radians(np.array([centroids[n][1] for n in centroid_names]))

        remaining_indices = df.index[remaining_mask].tolist()
        rem_lats_deg = df.loc[remaining_indices, "latitude"].values.astype(float)
        rem_lons_deg = df.loc[remaining_indices, "longitude"].values.astype(float)
        loc_values   = df.loc[remaining_indices, "location"].values

        n = len(remaining_indices)
        nearest_idx_all = np.empty(n, dtype=np.intp)
        min_dists_all   = np.empty(n, dtype=np.float64)

        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            rlat = np.radians(rem_lats_deg[start:end])
            rlon = np.radians(rem_lons_deg[start:end])
            dlat = rlat[:, None] - centroid_lats[None, :]
            dlon = rlon[:, None] - centroid_lons[None, :]
            a = np.sin(dlat / 2) ** 2 + (
                np.cos(rlat[:, None]) * np.cos(centroid_lats[None, :]) * np.sin(dlon / 2) ** 2
            )
            dist_chunk = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            nearest_idx_all[start:end] = np.argmin(dist_chunk, axis=1)
            min_dists_all[start:end]   = dist_chunk[np.arange(end - start), nearest_idx_all[start:end]]

        pos_lookup = {orig_idx: i for i, orig_idx in enumerate(remaining_indices)}
        for i, (orig_idx, row_lat, row_lon, min_dist, c_idx, loc_val) in enumerate(
            zip(remaining_indices, rem_lats_deg, rem_lons_deg,
                min_dists_all, nearest_idx_all, loc_values)
        ):
            pos = df.index.get_loc(orig_idx)
            if min_dist <= 0.5:
                zones[pos] = f"Near {centroid_names[c_idx]}"
                sources[pos] = "proximity"
            else:
                token = _parse_location_token(loc_val)
                if token is not None:
                    zones[pos] = token
                    sources[pos] = "location_text"
                else:
                    zones[pos] = f"Zone {round(float(row_lat), 2)}_{round(float(row_lon), 2)}"
                    sources[pos] = "coordinate_bin"

    elif remaining_mask.any():
        # No centroids at all — skip proximity, go straight to location/bin.
        remaining_indices = df.index[remaining_mask]
        loc_values = df.loc[remaining_indices, "location"].values
        rem_lats = df.loc[remaining_indices, "latitude"].values.astype(float)
        rem_lons = df.loc[remaining_indices, "longitude"].values.astype(float)

        for orig_idx, row_lat, row_lon, loc_val in zip(
            remaining_indices, rem_lats, rem_lons, loc_values
        ):
            pos = df.index.get_loc(orig_idx)
            token = _parse_location_token(loc_val)
            if token is not None:
                zones[pos] = token
                sources[pos] = "location_text"
            else:
                lat_bin = round(row_lat, 2)
                lon_bin = round(row_lon, 2)
                zones[pos] = f"Zone {lat_bin}_{lon_bin}"
                sources[pos] = "coordinate_bin"

    # ------------------------------------------------------------------ #
    # Step 4: Write results back to the DataFrame.                       #
    # ------------------------------------------------------------------ #
    df = df.copy()
    df["zone"] = zones
    df["zone_source"] = sources

    return df
