"""
ParkPulse AI — Streamlit Dashboard
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from parkpulse_ai.enforcement_planner import build_enforcement_plan, simulate_what_if
from parkpulse_ai.hotspot_detector import detect_hotspots, generate_heatmap
from parkpulse_ai.impact_scorer import score_zones
from parkpulse_ai.loader import load_dataset
from parkpulse_ai.zone_resolver import resolve_zones

st.set_page_config(page_title="ParkPulse AI", page_icon="🚗", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_hash(uploaded_file) -> str:
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return digest

def _colour_risk_level(val: str) -> str:
    colours = {"Critical": "#d62728", "High": "#ff7f0e", "Medium": "#f7c948", "Low": "#2ca02c"}
    colour = colours.get(val, "")
    return f"background-color: {colour}; color: white; font-weight: bold;" if colour else ""

# ---------------------------------------------------------------------------
# Cached pipeline stages
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset…")
def _cached_load(file_bytes: bytes, file_hash: str):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        df, dropped = load_dataset(tmp_path)
    finally:
        os.unlink(tmp_path)
    # Cap at 50k rows on Streamlit Cloud free tier (1 GB RAM)
    CLOUD_ROW_LIMIT = 50_000
    is_cloud = bool(os.environ.get("STREAMLIT_SHARING_MODE") or
                    os.environ.get("HOME", "").startswith("/home/appuser"))
    if is_cloud and len(df) > CLOUD_ROW_LIMIT:
        df = df.sample(n=CLOUD_ROW_LIMIT, random_state=42).reset_index(drop=True)
    return df, dropped

@st.cache_data(show_spinner="Resolving zones…")
def _cached_resolve(df: pd.DataFrame, file_hash: str):
    return resolve_zones(df)

@st.cache_data(show_spinner="Detecting hotspots…")
def _cached_hotspots(df: pd.DataFrame, file_hash: str, eps_km: float, min_samples: int):
    return detect_hotspots(df, eps_km=eps_km, min_samples=min_samples)

@st.cache_data(show_spinner="Generating heatmap…")
def _cached_heatmap_html(df: pd.DataFrame, file_hash: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        generate_heatmap(df, tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as fh:
            html_content = fh.read()
    finally:
        os.unlink(tmp_path)
    return html_content

@st.cache_data(show_spinner="Scoring zones…")
def _cached_score(df: pd.DataFrame, file_hash: str):
    return score_zones(df)

@st.cache_data(show_spinner="Building enforcement plan…")
def _cached_enforcement(scored_df: pd.DataFrame, file_hash: str):
    return build_enforcement_plan(scored_df)

# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _render_summary(df: pd.DataFrame, dropped: int) -> None:
    st.subheader("Dataset Summary")
    if len(df) <= 50_000:
        st.info("ℹ️ Running on Streamlit Cloud (1 GB RAM): analysis uses a 50,000-row sample. All features work identically.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Rows Dropped (invalid GPS)", f"{dropped:,}")
    if "created_datetime" in df.columns:
        valid_dates = df["created_datetime"].dropna()
        if not valid_dates.empty:
            col3.metric("Date Range", f"{valid_dates.min().date()} → {valid_dates.max().date()}")
        else:
            col3.metric("Date Range", "N/A")
    else:
        col3.metric("Date Range", "N/A")
    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Top 5 Violation Types")
        if "violation_type" in df.columns:
            top_v = df["violation_type"].value_counts().head(5).reset_index()
            top_v.columns = ["Violation Type", "Count"]
            st.dataframe(top_v, use_container_width=True, hide_index=True)
        else:
            st.info("No violation_type column found.")
    with col_right:
        st.markdown("#### Top 5 Zones by Violation Count")
        if "zone" in df.columns:
            top_z = df["zone"].value_counts().head(5).reset_index()
            top_z.columns = ["Zone", "Count"]
            st.dataframe(top_z, use_container_width=True, hide_index=True)
        else:
            st.info("Run the analysis to populate zone data.")


def _render_hotspot_map(df_clustered: pd.DataFrame, hotspot_df: pd.DataFrame, heatmap_html: str) -> None:
    st.subheader("Hotspot Heatmap")
    if heatmap_html:
        components.html(heatmap_html, height=500, scrolling=False)
    else:
        st.warning("Heatmap could not be generated.")
    st.divider()
    st.subheader("Ranked Hotspot Summary")
    if hotspot_df is not None and not hotspot_df.empty:
        st.dataframe(hotspot_df, use_container_width=True, hide_index=True)
    else:
        st.info("No hotspot zones found (all zones have fewer than 5 violations).")


def _render_impact_scoring(scored_df: pd.DataFrame) -> None:
    st.subheader("Traffic Impact Scoring")
    if scored_df is None or scored_df.empty:
        st.info("No scored data available.")
        return
    top20 = scored_df.nlargest(20, "traffic_impact_score")
    fig = px.bar(
        top20.sort_values("traffic_impact_score"),
        x="traffic_impact_score", y="zone", orientation="h",
        color="risk_level",
        color_discrete_map={"Critical": "#d62728", "High": "#ff7f0e", "Medium": "#f7c948", "Low": "#2ca02c"},
        title="Top 20 Zones by Traffic Impact Score",
        labels={"traffic_impact_score": "Traffic Impact Score", "zone": "Zone"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("Risk Level Table")
    display_df = scored_df[["zone","traffic_impact_score","risk_level","violation_count",
                             "vehicle_severity_score","violation_severity_score","peak_hour_score"]].copy()
    for col in ["traffic_impact_score","vehicle_severity_score","violation_severity_score","peak_hour_score"]:
        display_df[col] = display_df[col].round(2)
    styled = display_df.style.map(_colour_risk_level, subset=["risk_level"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_enforcement_plan(enforcement_df: pd.DataFrame) -> None:
    st.subheader("Enforcement Resource Allocation")
    if enforcement_df is None or enforcement_df.empty:
        st.info("No enforcement plan available.")
        return
    all_levels = ["Critical", "High", "Medium", "Low"]
    available = [l for l in all_levels if l in enforcement_df["risk_level"].values]
    selected = st.multiselect("Filter by Risk Level", options=available, default=available)
    filtered = enforcement_df[enforcement_df["risk_level"].isin(selected)]
    display_cols = ["zone","risk_level","traffic_impact_score","recommended_officers","recommended_tow_trucks","patrol_frequency_hours"]
    display_df = filtered[display_cols].copy()
    display_df["traffic_impact_score"] = display_df["traffic_impact_score"].round(2)
    styled = display_df.style.map(_colour_risk_level, subset=["risk_level"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(filtered):,} of {len(enforcement_df):,} zones.")


def _render_what_if(enforcement_df: pd.DataFrame) -> None:
    st.subheader("What-If Officer Deployment Simulator")
    if enforcement_df is None or enforcement_df.empty:
        st.info("No enforcement plan available. Run the analysis first.")
        return
    n_officers = st.slider("Total officers to deploy", min_value=1, max_value=200, value=20, step=1)
    if st.button("Simulate", type="primary"):
        try:
            what_if_df, overall_reduction = simulate_what_if(enforcement_df, n_officers)
            st.metric(label="Overall Expected Impact Reduction", value=f"{overall_reduction:.1f}%")
            st.divider()
            st.subheader("Per-Zone Allocation Results")
            if what_if_df.empty:
                st.info("No zones covered with the specified officer count.")
            else:
                display_df = what_if_df.copy()
                display_df["expected_reduction_pct"] = display_df["expected_reduction_pct"].round(2)
                styled = display_df.style.map(_colour_risk_level, subset=["risk_level"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(str(e))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🚗 ParkPulse AI — Parking Enforcement Analytics")
    st.markdown("Upload the Bangalore parking violation CSV, tune DBSCAN parameters, then click **Run Analysis**.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("Upload violation CSV", type=["csv"])
        st.markdown("#### DBSCAN Parameters")
        eps_km = st.slider("Neighbourhood radius (eps, km)", min_value=0.05, max_value=0.50, value=0.10, step=0.05)
        min_samples = st.slider("Minimum samples", min_value=5, max_value=50, value=10, step=1)
        run_button = st.button("🔍 Run Analysis", type="primary", disabled=(uploaded_file is None))
        if uploaded_file is None:
            st.info("Upload a CSV file to begin.")

    tab_summary, tab_hotspot, tab_impact, tab_enforce, tab_whatif = st.tabs([
        "📊 Summary", "🗺️ Hotspot Map", "📈 Impact Scoring", "🚔 Enforcement Plan", "🔮 What-If Simulator"
    ])

    _KEYS = ["df_raw","dropped","df_resolved","df_clustered","hotspot_df","heatmap_html",
             "scored_df","enforcement_df","file_hash","pipeline_ran"]
    for key in _KEYS:
        if key not in st.session_state:
            st.session_state[key] = None

    if run_button and uploaded_file is not None:
        file_hash = _file_hash(uploaded_file)
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        st.session_state["file_hash"] = file_hash
        st.session_state["pipeline_ran"] = False

        try:
            df_raw, dropped = _cached_load(file_bytes, file_hash)
            st.session_state["df_raw"] = df_raw
            st.session_state["dropped"] = dropped
        except Exception as e:
            st.error(str(e)); st.stop()

        try:
            df_resolved = _cached_resolve(df_raw, file_hash)
            st.session_state["df_resolved"] = df_resolved
        except Exception as e:
            st.error(str(e)); st.stop()

        try:
            df_clustered, hotspot_df = _cached_hotspots(df_resolved, file_hash, eps_km, min_samples)
            st.session_state["df_clustered"] = df_clustered
            st.session_state["hotspot_df"] = hotspot_df
        except Exception as e:
            st.error(str(e)); st.stop()

        try:
            heatmap_html = _cached_heatmap_html(df_resolved, file_hash)
            st.session_state["heatmap_html"] = heatmap_html
        except Exception as e:
            st.error(str(e)); st.stop()

        try:
            scored_df = _cached_score(df_resolved, file_hash)
            st.session_state["scored_df"] = scored_df
        except Exception as e:
            st.error(str(e)); st.stop()

        try:
            enforcement_df = _cached_enforcement(scored_df, file_hash)
            st.session_state["enforcement_df"] = enforcement_df
        except Exception as e:
            st.error(str(e)); st.stop()

        st.session_state["pipeline_ran"] = True

    with tab_summary:
        if st.session_state["df_raw"] is not None:
            _render_summary(
                st.session_state["df_resolved"] if st.session_state["df_resolved"] is not None else st.session_state["df_raw"],
                st.session_state["dropped"] if st.session_state["dropped"] is not None else 0,
            )
        else:
            st.info("Upload a CSV and click **Run Analysis** to see the summary.")

    with tab_hotspot:
        if st.session_state["df_clustered"] is not None:
            _render_hotspot_map(
                st.session_state["df_clustered"],
                st.session_state["hotspot_df"],
                st.session_state["heatmap_html"] if st.session_state["heatmap_html"] is not None else "",
            )
        else:
            st.info("Upload a CSV and click **Run Analysis** to view the hotspot map.")

    with tab_impact:
        if st.session_state["scored_df"] is not None:
            _render_impact_scoring(st.session_state["scored_df"])
        else:
            st.info("Upload a CSV and click **Run Analysis** to view impact scores.")

    with tab_enforce:
        if st.session_state["enforcement_df"] is not None:
            _render_enforcement_plan(st.session_state["enforcement_df"])
        else:
            st.info("Upload a CSV and click **Run Analysis** to view the enforcement plan.")

    with tab_whatif:
        if st.session_state["enforcement_df"] is not None:
            _render_what_if(st.session_state["enforcement_df"])
        else:
            st.info("Upload a CSV and click **Run Analysis** to use the What-If Simulator.")


try:
    main()
except Exception as _e:
    import traceback
    st.error(f"App startup error: {_e}")
    st.code(traceback.format_exc())
