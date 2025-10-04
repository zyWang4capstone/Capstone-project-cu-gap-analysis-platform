# Front-end README

## 1) Purpose
This Streamlit multipage UI presents 3D/2D explorations of Western Australian geoscience data. It reads **fixed-name artifacts** produced by the backend (Task-1 cleaned CSVs; Task-2 points splits) and provides:

- Raw points and section views for the **Original** datasets.
- **Record-level** and **Aggregated** views for Drillhole and Surface (**All / Overlap / Orig-only / DL-only**).
- A unified **Insights** page with voxel and DBSCAN exploration.

Heavy computation remains in the backend; the UI performs light gridding/slicing via compute-only helpers.

## 2) Layout (pages)
```
pages/
  01_Home.py                 # Hero, dataset readiness, auto-detected element, quick links
  02_Original.py             # Original dataset: raw points 3D viewer + horizontal/vertical sections
  04_Diff_Home.py            # Analysis hub; global grid config; can trigger Task-2 recomputation
  05_Drillhole_Record.py     # Record-level drillhole visuals (tabs)
  06_Drillhole_Aggregated.py # 3D differences (All/Overlap/Orig-only/DL-only)
  07_Surface_Record.py       # Record-level surface visuals (2D)
  08_Surface_Aggregated.py   # 2D differences (All/Overlap/Orig-only/DL-only)
  09_Insights.py             # Unified insights (voxel & DBSCAN exploration)
_ui_common.py                 # Shared theme, grid config helpers, session bootstrap
```

### Page responsibilities (what each expects/produces)

#### 01_Home
- Detects the working **element** (e.g., Cu) by scanning columns in `reports/task1/cleaned/*.csv` for `ppm` suffixes; shows dataset status; provides CTA to analysis pages.
- Uses `st.session_state["data_src"]` and `["element"]` as global context.
- Offers lightweight download of summary previews.

#### 02_Original
- Consumes unified “points” and renders 3D scatter + **vertical** and **horizontal** sections.
- Improvements: VALUE shown in hover reliably (Plotly hovertemplate), default section grid = **1.0 km** (minimum **0.1 km**), overlay points removed for clarity.
- Uses cached data reads to keep navigation responsive.

#### 04_Diff_Home
- Global entry for difference analysis; exposes grid configuration (reads/writes via `_ui_common.load_grid_cfg()/save_grid_cfg()`).
- Integrates with backend difference logic (`cap_task2.overlap.Params`) so the UI can trigger Task-2 recomputation if allowed.

#### 05_Drillhole_Record / 06_Drillhole_Aggregated
- Record-level tabs and a **3D Aggregated** view respectively.
- Read from `reports/task2/difference/drillhole_points_{all|overlap|origonly|dlonly}.csv`.

#### 07_Surface_Record / 08_Surface_Aggregated
- 2D record-level and aggregated views for Surface.
- Read from `reports/task2/difference/surface_points_{all|overlap|origonly|dlonly}.csv`.
- `08_Surface_Aggregated` handles a `value_mode` session key where relevant.

#### 09_Insights
- Unified (Drillhole + Surface) heuristics:
  - (A) DL-only hotspots (drillhole)
  - (B) Overlap uplift (DL − ORIG) on drillhole
  - (C) Surface vs Drill discrepancy at a Z slice
  - (D) DBSCAN clustering in normalized lon/lat/dep space; **Voxel** mode regards any occupied voxel as a hotspot
- Session keys include: `vox_nx/vox_ny/vox_nz` (voxel grid), `vox_min_cnt` (min occupancy), `vox_opacity` (display).

## 3) Data contracts (assumptions)
- **Task-1 cleaned files** (fixed names) must exist under `reports/task1/cleaned/`:
  - `drillhole_original_clean.csv`, `drillhole_dnn_clean.csv`, `surface_original_clean.csv`, `surface_dnn_clean.csv`
- **Task-2 splits** must exist under `reports/task2/difference/`:
  - `{drillhole|surface}_points_{all|overlap|origonly|dlonly}.csv`
- Canonical columns are expected: `LONGITUDE, LATITUDE, VALUE` (plus `FROM, TO, DEPTH` for drillhole).
- Depth units are already normalized to **meters** by the backend.
- Element auto-detection on Home relies on columns that end with `ppm`.

## 4) Configuration & shared utilities
- **Config**: the UI loads the same YAML (e.g., `app.yaml`) via `cap_common.config.load_cfg()`.
- **Grid defaults**: `_ui_common.load_grid_cfg()` and `save_grid_cfg()` centralize the global grid size and section parameters so that pages remain consistent.
- **Session bootstrap**: `_ui_common.ensure_session_bootstrap()` seeds keys like `data_src`, `element`, and grid defaults.

## 5) Session state & caching
- Common session keys:
  - Global context: `data_src`, `element`
  - Insights / voxel: `vox_nx`, `vox_ny`, `vox_nz`, `vox_min_cnt`, `vox_opacity`
  - Surface aggregated: `value_mode`
- Heavy readers cache with `@st.cache_data` where safe; ensure cache keys include element and grid settings to avoid stale visuals.

## 6) Errors, safeguards, UX notes
- If a page reports **no cleaned** or **no split** files, ensure backend tools were run with the correct VALUE column and ranges.
- The UI treats outputs as **decision support**; all insights require geological validation.
- Hover/value formatting uses stable Plotly templates to avoid version quirks. CSS is scoped to Streamlit containers to keep a clean theme.

## 7) Extensibility (how to add a page)
- Start from the `_ui_common` scaffolding (`inject_theme`, `ensure_session_bootstrap`).
- Read artifacts via `cap_task2.io.read_points()` or the cleaned CSVs contract.
- Reuse session keys for consistency; add page-specific keys with clear names.
- Keep visual defaults (grid size, section res) in `_ui_common` to remain coherent across pages.
