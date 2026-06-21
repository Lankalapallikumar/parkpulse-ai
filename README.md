================================================================================
                        ParkPulse AI — Complete Project Report
================================================================================

------------------------------------------------------------------------
WHAT IS THIS PROJECT?
------------------------------------------------------------------------
ParkPulse AI is an intelligent illegal parking analytics and enforcement
planning system built for Bangalore traffic data. It takes raw CSV violation
records and turns them into actionable intelligence — hotspot maps, risk
scores, resource allocation plans, and what-if simulations — all through
an interactive offline dashboard.

------------------------------------------------------------------------
THE PROBLEM IT SOLVES
------------------------------------------------------------------------
Traffic enforcement teams in Bangalore deal with ~300,000 parking violation
records (Jan–May) but have no easy way to answer:
  - WHERE are violations clustered geographically?
  - WHICH zones cause the most traffic disruption?
  - HOW MANY officers should be deployed where?
  - WHAT REDUCTION can I expect if I send 20 officers today?

ParkPulse AI answers all of these automatically.

------------------------------------------------------------------------
THE DATASET
------------------------------------------------------------------------
File    : jan to may police violation_anonymized791b166.csv
Size    : ~298,450 records
Period  : January to May

Key columns used:
  latitude, longitude    — GPS location of violation
  junction_name          — Named junction (or "No Junction")
  location               — Free-text location string
  vehicle_type           — Car, Truck, Scooter, etc.
  violation_type         — Type of parking violation
  created_datetime       — When the violation was recorded

------------------------------------------------------------------------
HOW IT WORKS — PIPELINE
------------------------------------------------------------------------

  CSV File
     |
  [1] Loader              -> Cleans & validates data
     |
  [2] Zone_Resolver       -> Assigns a zone name to every record
     |
  [3] Hotspot_Detector    -> Clusters GPS points, ranks hotspots
     |
  [4] Impact_Scorer       -> Scores each zone 0-100 for traffic disruption
     |
  [5] Enforcement_Planner -> Recommends resources + what-if simulation
     |
  Streamlit Dashboard     -> Interactive views for all results

------------------------------------------------------------------------
STAGE 1 — LOADER (loader.py)
------------------------------------------------------------------------
What it does:
  - Reads the CSV, drops the description column
  - Parses 6 datetime columns (bad values -> NaT, not dropped)
  - Casts latitude/longitude to float64 — rows with non-numeric GPS are
    dropped and counted
  - Validates all 10 required columns are present

What you get:
  A clean DataFrame ready for analysis, plus a count of dropped rows.

------------------------------------------------------------------------
STAGE 2 — ZONE RESOLVER (zone_resolver.py)
------------------------------------------------------------------------
What it does:
  Every record needs a meaningful zone name for aggregation.
  Priority order per record:
    1. If junction_name is set (not "No Junction") -> use it directly
    2. If GPS is within 500m of a named junction centroid -> "Near [Junction]"
    3. Parse first useful token from the location string
    4. Fallback: "Zone 12.97_77.59" (coordinate bin)

What you get:
  Two new columns — zone (the name) and zone_source (how it was resolved).

------------------------------------------------------------------------
STAGE 3 — HOTSPOT DETECTOR (hotspot_detector.py)
------------------------------------------------------------------------
What it does:
  - Runs DBSCAN (Density-Based Spatial Clustering) on all GPS points
    using the Haversine distance metric
  - Default: radius = 0.1 km, minimum 10 points to form a cluster
  - Each record gets a cluster_id (-1 = noise/isolated)
  - Aggregates per zone: violation count, cluster count, top violation
    type, top vehicle type
  - Filters out zones with fewer than 5 violations
  - Generates a Folium heatmap (HTML file) showing violation density

What you get:
  A ranked hotspot summary table + an interactive heatmap.

------------------------------------------------------------------------
STAGE 4 — IMPACT SCORER (impact_scorer.py)
------------------------------------------------------------------------
What it does:
  Computes a Traffic Impact Score (0-100) for each zone using a weighted
  formula:

    raw_score = (violation_frequency x 0.40)
              + (vehicle_severity    x 0.30)
              + (violation_severity  x 0.20)
              + (peak_hour_activity  x 0.10)

  Sub-score details:
    Violation frequency — how many violations in this zone vs others
    Vehicle severity    — Truck/Maxi-Cab=3, Car/Auto=2, Scooter/Motorcycle=1
    Violation severity  — "JUNCTION" in type=3, "MAIN"=2, else=1
    Peak hours          — midnight-5am and 7pm-9pm violations score higher

  All sub-scores are min-max scaled to [0, 100].
  Final score also normalised to [0, 100].

  Risk levels assigned:
    Score 80-100  ->  Critical
    Score 60-79   ->  High
    Score 40-59   ->  Medium
    Score  0-39   ->  Low

What you get:
  A per-zone DataFrame with scores, risk levels, and all sub-scores.

------------------------------------------------------------------------
STAGE 5 — ENFORCEMENT PLANNER (enforcement_planner.py)
------------------------------------------------------------------------
Build Enforcement Plan:
  Maps each zone's risk level to recommended resources:

    Risk Level | Officers | Tow Trucks | Patrol Frequency
    -----------+----------+------------+-----------------
    Critical   |    5     |     2      | Every 1 hour
    High       |    3     |     1      | Every 2 hours
    Medium     |    2     |     1      | Every 4 hours
    Low        |    1     |     0      | Every 8 hours

What-If Simulator:
  - You specify a total officer count N
  - Officers are allocated greedily, highest-risk zones first
  - Per-zone expected reduction:
      min(allocated/recommended, 1.0) x base_rate
      Critical: 40%, High: 35%, Medium: 30%, Low: 25%
  - Overall reduction = violation-count-weighted average across all
    covered zones

What you get:
  A resource allocation table + expected city-wide impact reduction %.

------------------------------------------------------------------------
THE DASHBOARD — 5 TABS
------------------------------------------------------------------------

Tab 1 — Summary
  - Total records, date range (Jan-May)
  - Top 5 violation types
  - Top 5 zones by violation count

Tab 2 — Hotspot Map
  - Interactive Folium heatmap — brighter = more violations
  - Ranked hotspot table below the map

Tab 3 — Impact Scoring
  - Plotly horizontal bar chart: top 20 zones by score,
    colour-coded by risk level
  - Full risk level table (red/orange/yellow/green)

Tab 4 — Enforcement Plan
  - Full resource allocation table
  - Filterable by risk level (multiselect)

Tab 5 — What-If Simulator
  - Slider: 1-200 officers
  - Click Simulate -> see per-zone allocation + overall expected
    reduction %

------------------------------------------------------------------------
PROJECT FILE STRUCTURE
------------------------------------------------------------------------

  parkpulse_ai\
      __init__.py
      loader.py               Stage 1: Data ingestion
      zone_resolver.py        Stage 2: Zone naming
      hotspot_detector.py     Stage 3: DBSCAN clustering
      impact_scorer.py        Stage 4: Traffic impact scoring
      enforcement_planner.py  Stage 5: Resource planning + what-if
      app.py                  Streamlit dashboard

  tests\
      __init__.py
      test_loader.py
      test_zone_resolver.py
      test_hotspot_detector.py
      test_impact_scorer.py
      test_enforcement_planner.py

  requirements.txt
  jan to may police violation_anonymized791b166.csv

------------------------------------------------------------------------
TEST COVERAGE
------------------------------------------------------------------------
129 tests total — all passing.

  Unit tests:
    Specific examples, edge cases, error paths for every module.

  Property-based tests (Hypothesis) — 14 correctness properties:
    Property  1: Every record always gets a zone name
    Property  2: zone_source is always one of 4 valid values
    Property  3: Hotspot summary only contains zones with >= 5 violations
    Property  4: Hotspot summary always sorted by violation count desc
    Property  5: All traffic_impact_score values in [0, 100]
    Property  6: Risk levels always match score thresholds
    Property  7: All sub-scores (vehicle, violation, peak) in [0, 100]
    Property  8: Enforcement plan resources match risk level lookup
    Property  9: Enforcement plan always sorted by score desc
    Property 10: Officer allocation never exceeds the budget N
    Property 11: Per-zone reduction always in [0, 40]%
    Property 12: Dropped row count = original rows - returned rows
    Property 13: Haversine distance is symmetric (A->B == B->A)
    Property 14: Overall reduction = violation-count-weighted mean

------------------------------------------------------------------------
DEPENDENCIES
------------------------------------------------------------------------

  Package        Version   Purpose
  -------------- --------- ----------------------------------
  pandas         2.2.2     Data manipulation
  numpy          1.26.4    Numeric operations
  scikit-learn   1.5.0     DBSCAN clustering
  folium         0.16.0    Interactive heatmap
  plotly         5.22.0    Bar charts
  streamlit      1.35.0    Dashboard UI
  hypothesis     6.103.0   Property-based testing
  pytest         8.2.0     Test runner

------------------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------------------

Step 1 — Install dependencies
  pip install -r requirements.txt

Step 2 — Start the dashboard
  streamlit run parkpulse_ai/app.py

  Opens at: http://localhost:8501

Step 3 — Use the dashboard
  1. Click "Browse files" in the sidebar
  2. Upload: jan to may police violation_anonymized791b166.csv
  3. Adjust DBSCAN sliders if desired (defaults work well)
  4. Click "Run Analysis"
  5. Explore the 5 tabs

Step 4 — Run tests
  python -m pytest tests/ -v

------------------------------------------------------------------------
WHAT YOU GET FROM THIS PROJECT
------------------------------------------------------------------------

  Output               | Value
  ---------------------|--------------------------------------------------
  Hotspot map          | Visual proof of where violations cluster
  Impact scores        | Objective 0-100 ranking by traffic disruption
  Risk levels          | Critical/High/Medium/Low per zone
  Resource plan        | Exact officer/truck counts + patrol schedules
  What-if simulation   | Predict impact reduction before committing resources
  Test suite           | 129 tests proving analytics are correct

================================================================================
                                  END OF REPORT
================================================================================
