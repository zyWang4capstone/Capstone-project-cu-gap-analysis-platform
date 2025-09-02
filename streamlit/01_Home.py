# 01_Home.py — Hero + CTAs + current status

import streamlit as st
from _ui_common import inject_theme, render_top_right_logo, render_cta_row, load_grid_cfg, ensure_session_bootstrap

st.set_page_config(layout="wide", page_title="Home • Copper Modeling Dashboard")
inject_theme()
ensure_session_bootstrap()

# -------------------------------- HERO --------------------------------------
left, right = st.columns([0.78, 0.22])

with left:
    st.markdown(
        """
        <h1 style="margin-bottom:0.2rem; font-size:2.4rem;">
          Data-Driven Discovery: 3D Gap Analysis of Imputed vs Original Cu Assays
        </h1>
        <p style="font-size:1.05rem; color:#444; margin-top:0;">
        Explore, compare, and discover copper anomalies across drillhole and surface datasets.  
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

# ----------------------------- Current status -------------------------------
cfg = load_grid_cfg()
ds = st.session_state.get("data_src", "default")

st.subheader("Current status")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Data source**")
    st.write("Default dataset" if ds == "default" else "Uploaded dataset (set in Data Manager)")
with c2:
    st.markdown("**Global grid**")
    st.write(f"{cfg['cell_x_m']} × {cfg['cell_y_m']} × {cfg['cell_z_m']} m, {cfg['agg']}, min_count={cfg['min_count']}")

st.caption(
    "Use **Original** to start exploring the default data, or go to **Data Manager** to upload and validate your own files."
)