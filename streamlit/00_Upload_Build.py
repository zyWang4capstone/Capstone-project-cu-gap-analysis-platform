# 00_Upload_Build.py — Upload & Build (UI only; no backend yet)
import json
from dataclasses import asdict, dataclass
from typing import Optional

import streamlit as st

st.set_page_config(page_title="Upload & Build — Ingest Options", layout="wide")

# -----------------------------
# Simple config dataclass (UI-only)
# -----------------------------
@dataclass
class IngestConfig:
    # CRS / projection
    crs_mode: str = "WGS84 (EPSG:4326)"
    epsg_code: Optional[int] = None  # only used when crs_mode != WGS84

    # Aggregation
    agg_method: str = "mean"  # mean | median | max

    # Gridding / resolution (meters)
    surf_xy_m: int = 10       # surface XY resolution
    dh_xy_m: int = 10         # drillhole XY resolution
    dh_z_m: int = 1           # drillhole Z resolution

    # Placeholders for future toggles
    keep_raw_records: bool = True  # keep original records for validation
    make_tiles: bool = True        # also build gridded/heatmap tiles


# -----------------------------
# Page title
# -----------------------------
st.title("Upload & Build — Data Ingest Options")
st.caption(
    "Upload your source data (e.g., a ZIP of shapefiles) and choose how to build "
    "the processed datasets. This page only collects options — no backend is executed yet."
)

# -----------------------------
# Left: Upload area
# -----------------------------
st.subheader("1) Upload source data")
col1, col2 = st.columns([1.2, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload a ZIP (shapefiles) or CSV files",
        type=["zip", "csv"],
        accept_multiple_files=True,
        help="You can upload a single ZIP that contains .shp/.shx/.dbf (+.prj/.cpg), "
             "or multiple CSVs. Backend processing is not wired yet."
    )
    if uploaded:
        st.success(f"{len(uploaded)} file(s) selected.")
        with st.expander("Preview uploaded filenames"):
            for f in uploaded:
                st.write("•", f.name)
    else:
        st.info("No files selected yet.")

with col2:
    st.markdown("**Guidance**")
    st.write(
        "- For shapefiles, put each layer (.shp/.shx/.dbf) into **one ZIP**.\n"
        "- Optional files like `.prj` (CRS) and `.cpg` (encoding) are supported.\n"
        "- CSV is also allowed (must contain LONGITUDE/LATITUDE and value columns)."
    )

st.divider()

# -----------------------------
# Middle: Options
# -----------------------------
st.subheader("2) Build options (UI only)")

cfg = IngestConfig()

# --- CRS / Projection
st.markdown("### Coordinate Reference System (CRS)")
crs_mode = st.radio(
    "Choose CRS",
    ["WGS84 (EPSG:4326)", "Other — reproject to WGS84"],
    horizontal=True
)
cfg.crs_mode = crs_mode

if crs_mode != "WGS84 (EPSG:4326)":
    st.info("Data will be reprojected to WGS84 for mapping. Please provide the source EPSG.")
    epsg = st.number_input("Source EPSG code", min_value=2000, max_value=900000, value=28350, step=1)
    cfg.epsg_code = int(epsg)

st.caption(
    "Maps and web visualization use WGS84 (EPSG:4326). If your data is in a projected CRS (e.g., GDA94 / MGA Zone 50), "
    "we'll reproject to WGS84 during ingest (backend to be added later)."
)

st.markdown("### Aggregation method")
cfg.agg_method = st.selectbox("How to aggregate values inside each grid cell",
                              ["mean", "median", "max"], index=0,
                              help="This applies when multiple records fall into the same grid cell.")

st.markdown("### Grid resolution (meters)")
cxy1, cxy2, cz = st.columns(3)
with cxy1:
    cfg.surf_xy_m = st.slider("Surface XY", min_value=5, max_value=200, value=10, step=5,
                              help="Surface grid size in meters (X and Y).")
with cxy2:
    cfg.dh_xy_m = st.slider("Drillhole XY", min_value=5, max_value=200, value=10, step=5,
                            help="Drillhole grid size in meters (X and Y).")
with cz:
    cfg.dh_z_m = st.slider("Drillhole Z", min_value=1, max_value=20, value=1, step=1,
                           help="Drillhole vertical (Z) bin size in meters.")

# Optional toggles
st.markdown("### Extras")
copt1, copt2 = st.columns(2)
with copt1:
    cfg.keep_raw_records = st.checkbox(
        "Keep raw record-level copies for validation",
        value=True,
        help="Store original normalized records so you can switch to raw points at high zoom."
    )
with copt2:
    cfg.make_tiles = st.checkbox(
        "Also build gridded/heatmap tiles",
        value=True,
        help="Prepare pre-aggregated tiles for fast maps (e.g., pydeck ScreenGrid/Hex)."
    )

st.divider()

# -----------------------------
# Right: Config preview & actions
# -----------------------------
st.subheader("3) Preview & actions")

preview = asdict(cfg)
# Keep it human-readable
st.code(json.dumps(preview, indent=2), language="json")

left, right = st.columns([1, 1])
with left:
    st.button(
        "Save options (no backend)",
        type="secondary",
        help="In a later step, this will persist your config and trigger a build.",
        on_click=lambda: st.toast("Options saved (UI only). Backend is not connected yet.")
    )
with right:
    st.button(
        "Build processed datasets",
        type="primary",
        help="UI only for now. In the next iteration, this will run the full ingest pipeline.",
        on_click=lambda: st.warning("Backend not implemented yet. This button is a placeholder.")
    )

st.caption(
    "Once the backend is wired, clicking **Build** will: normalize data → reproject (if needed) → "
    "split into All/Overlap/Orig-only/DL-only → aggregate using your grid/resolution → write processed files "
    "for all visualization pages."
)