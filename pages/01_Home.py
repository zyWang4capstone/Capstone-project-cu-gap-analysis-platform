# 01_Home.py — Hero + CTAs + current status + dataset overview (Summary + Preview only)

import streamlit as st
import pandas as pd
import re
from pathlib import Path
from _ui_common import (
    inject_theme, render_top_right_logo, render_cta_row,
    load_grid_cfg, ensure_session_bootstrap
)

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
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Data source**")
    st.write(f"{element} dataset (default)" if ds == "default" else f"{element} dataset (uploaded)")
    if "uploaded_file" in st.session_state:
        st.caption(f"File: {st.session_state['uploaded_file']}")

with c2:
    st.markdown("**Global grid**")
    st.write(f"{cfg['cell_x_m']} × {cfg['cell_y_m']} × {cfg['cell_z_m']} m, {cfg['agg']}, min_count={cfg['min_count']}")

st.caption(
    "Use **Original** to start exploring the default data, or go to **Data Manager** to upload and validate your own files."
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

