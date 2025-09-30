# 01_Home.py — Hero + CTAs + current status + dataset overview (Summary + Preview only)

import streamlit as st
import pandas as pd
import re
from pathlib import Path
from _ui_common import (
    inject_theme, render_top_right_logo, render_cta_row,
    load_grid_cfg, ensure_session_bootstrap
)
from cap_common.config import load_cfg

st.set_page_config(layout="wide", page_title="Home • Modeling Dashboard")
inject_theme()
ensure_session_bootstrap()

# ----------------------------- Config ---------------------------------------
cfg = load_grid_cfg()
ds = st.session_state.get("data_src", "default")

# Ensure we have the cleaned dataset directory path (safe getattr fallback)
clean_dir = getattr(cfg, "task1_clean_dir", Path("reports/task1/cleaned"))

# ----------------------------- Element Detection ----------------------------
def detect_element_from_cleaned(clean_dir: Path) -> str:
    """Detect which element (e.g. Cu, Te) from cleaned CSVs by scanning columns."""
    try:
        files = list(clean_dir.glob("*.csv"))
        for f in files:
            df = pd.read_csv(f, nrows=50)  # peek only first rows for speed
            candidates = [c for c in df.columns if re.search(r"ppm$", c.lower()) or c.upper() == "VALUE"]
            if candidates:
                col = candidates[0]
                return col.split("_")[0].title() if "_" in col else col.title()
    except Exception:
        pass
    return "Element"

element = detect_element_from_cleaned(clean_dir)
st.session_state["element"] = element

# ----------------------------- HERO -----------------------------------------
left, right = st.columns([0.78, 0.22])
with left:
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.2rem; font-size:2.4rem;">
          Data-Driven Discovery: 3D Gap Analysis of Imputed vs Original {element} Assays
        </h1>
        <p style="font-size:1.05rem; color:#444; margin-top:0;">
        Explore, compare, and discover {element.lower()} anomalies across drillhole and surface datasets.  
        This dashboard brings together record-level data, voxel-based aggregation, and deep-learning 
        imputation results, so you can identify overlaps, differences, and potential anomalies.  
        Switch between multiple levels of granularity, overlay drillhole with surface values, 
        and generate insights into where imputed predictions diverge from original assays.
        </p>
        """,
        unsafe_allow_html=True,
    )
    render_cta_row()
with right:
    render_top_right_logo()

st.divider()

# ----------------------------- Current Status -------------------------------
st.subheader("Current status")

import math

# Helper: convert degrees to approximate meters at a given latitude (default Perth ≈ -32°)
def deg_to_m_split(deg: float, lat_deg: float = -32.0) -> str:
    lat_m = deg * 111_000
    lon_m = deg * 111_000 * math.cos(math.radians(lat_deg))
    return f"approx. {lon_m:.1f} m (lon), {lat_m:.1f} m (lat)"

# Safely fetch config values (dict or object attributes)
def _cfg_fetch(root, candidates: list[str], default: float) -> float:
    for path in candidates:
        obj = root
        ok = True
        for part in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(part) if part in obj else None
            else:
                obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok:
            try:
                return float(obj)
            except Exception:
                pass
    return float(default)

# fetch current config
cfg = load_cfg()
cur_xy = _cfg_fetch(cfg, ["grid_step_deg", "task2.grid_step_deg", "difference.grid_step_deg"], 1e-4)
cur_z  = _cfg_fetch(cfg, ["z_step_m", "task2.z_step_m", "difference.z_step_m"], 1.0)

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Data source**")
    st.write(f"{element} dataset")
    st.caption("To use another dataset, please put your shapefile `.zip` inside the `data/` folder.")

with c2:
    st.markdown("**Global grid**")
    st.markdown(
        f"<p style='margin:0.2rem 0; font-size:0.95rem;'>"
        f"XY grid step = {cur_xy:.6f} degree ({deg_to_m_split(cur_xy)})<br>"
        f"Z step = {cur_z:.1f} m"
        f"</p>",
        unsafe_allow_html=True
    )
# ----------------------------- Dataset overview -----------------------------
st.subheader("Dataset overview")

# Only two tabs: Summary + Preview
tabs = st.tabs(["Summary", "Preview"])

# Collect dataset paths
datasets = {
    "Drillhole Original": clean_dir / "drillhole_original_clean.csv",
    "Drillhole DL": clean_dir / "drillhole_dnn_clean.csv",
    "Surface Original": clean_dir / "surface_original_clean.csv",
    "Surface DL": clean_dir / "surface_dnn_clean.csv",
}

# ---------- Tab 1: Summary ----------
with tabs[0]:
    rows = []
    for label, path in datasets.items():
        if path.exists():
            # get total rows fast
            n_rows = sum(1 for _ in open(path)) - 1  
            
            # read first few rows to detect columns
            df_head = pd.read_csv(path, nrows=50)
            n_cols = df_head.shape[1]

            # detect value column
            candidates = [c for c in df_head.columns if re.search(r"ppm$", c.lower()) or c.upper() == "VALUE"]
            value_range = "-"
            if candidates:
                col = candidates[0]
                # only load this column to compute full min/max
                series = pd.read_csv(path, usecols=[col])[col]
                vmin, vmax = series.min(), series.max()
                value_range = f"{vmin:.2f} – {vmax:.2f}"

            rows.append((label, n_rows, n_cols, value_range))
    summary_df = pd.DataFrame(rows, columns=["Dataset", "Rows", "Cols", "Value Range"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ---------- Tab 2: Preview ----------
with tabs[1]:
    choice = st.selectbox("Select dataset to preview", list(datasets.keys()))
    path = datasets[choice]
    if path.exists():
        # Only load first 100 rows for preview
        df_preview = pd.read_csv(path, nrows=100)
        st.dataframe(df_preview, use_container_width=True, hide_index=True, height=600)

        # Provide a download option for the full dataset
        df_full = pd.read_csv(path)  # load full for download only
        st.download_button(
            label=f"Download full {choice} CSV",
            data=df_full.to_csv(index=False).encode("utf-8"),
            file_name=f"{choice.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )

