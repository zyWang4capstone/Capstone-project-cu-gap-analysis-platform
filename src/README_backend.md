# Backend README

## Capstone Backend — 3D Gap Analysis

### 1) Purpose
The backend ingests public geoscience files, standardises schemas, produces fixed “cleaned” CSVs (Task-1), builds unified point tables and overlap/difference splits (Task-2), and exposes light compute helpers for horizontal/vertical slicing used by the Streamlit UI. It is deterministic, config-driven, and safe for headless servers.

### 2) High-level flow (data contracts)
- **Input**: exactly one ZIP under `data/` containing four CSVs (DRILLHOLE/SURFACE × ORIGINAL/DL).  
  *Optionally*, `tools/run_all_test.py` can stage Shapefile archives to CSV before cleaning.
- **Task-1 (clean)** → `reports/task1/cleaned/`
  - `drillhole_original_clean.csv`, `drillhole_dnn_clean.csv`
  - `surface_original_clean.csv`, `surface_dnn_clean.csv`  
  Canonical columns: `LONGITUDE, LATITUDE, VALUE` (+ `FROM, TO, DEPTH` for drillhole). Depth units auto-detected (`m/cm/mm`) and normalised to **meters**.
- **Task-2 (points & splits)** → `reports/task2/difference/`
  - `{drillhole|surface}_points_{overlap|origonly|dlonly|all}.csv`
  - Each row carries `LONGITUDE, LATITUDE, DEPTH, VALUE, SOURCE` with `SOURCE ∈ {DH_ORIG, DH_DL, SURF_ORIG, SURF_DL}`.
- **Sections/Slices**: compute-only helpers return gridded tables for Plotly (no GUI dependency).

### 3) Code layout
```
src/
  cap_common/      # config (YAML loader), IO helpers, schema standardisation, light cache
  cap_eda/         # Task-1 ingest/clean (value alias/regex, min/max clamps)
  cap_original/    # points unification & section wrappers
  cap_task2/       # overlap logic & diff columns
  pyvista_ops.py   # compute-only slicing backend (horizontal/vertical grids)
tools/
  run_eda.py               # run Task-1 clean with controllable aliases/regex/min-max
  run_all.py               # end-to-end: EDA → points → Task-2 (with format/compression/sort)
  run_all_test.py          # as above + SHP→CSV staging for ZIPs of shapefiles
  build_points_all.py      # produce a consolidated points_all (CSV/Parquet) and optional alias
  build_points_all_v2.py   # variant with order/label normalisation & parquet options
  recompute_task2.py       # recompute Task-2 splits only
  run_eda_old.py           # legacy wrapper (kept for reference)
reports/
  task1/cleaned/
  task2/difference/
data/
```

### 4) Configuration (shared with the UI)
A single YAML (e.g., `app.yaml`) loaded by `cap_common.config.load_cfg()`:
- `data`: `task1_clean_dir`, `task2_diff_dir`, optional `original_points_all`, `cache_dir`
- `schema`: `aliases` and `value_regex` (e.g., `^cu_?ppm(_pred)?$`)
- `grid`: `xy_cell_km`, `z_cell_m`
- `diff`: `min_overlap_m`
- `viz_defaults`: section corridor half-width, resolutions, aggregation mode, etc.

### 5) Design notes & non-goals
- **Deterministic outputs** with **fixed filenames** to keep the front-end simple.
- **Human-in-the-loop** analytics: no automated decisions.
- **Out of scope**: authentication, job scheduling, long-running queues; keep this repo computation-focused and file-oriented.
