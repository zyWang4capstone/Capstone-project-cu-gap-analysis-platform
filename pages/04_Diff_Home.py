# 04_Diff_Home.py — Analysis Hub (entry) + global grid

from __future__ import annotations
import io
from pathlib import Path
import pandas as pd
import streamlit as st
from _ui_common import inject_theme, ensure_session_bootstrap, load_grid_cfg, save_grid_cfg, DEFAULT_GRID
from cap_common.config import load_cfg
from cap_task2.overlap import recompute_all, Params

# ---------------- Page meta / theme ----------------
st.set_page_config(layout="wide", page_title="Analysis Hub")
inject_theme()

# ---------------- Local CSS (cards & clean layout) ----------------
st.markdown(
    """
    <style>
      .hub-subtle { color:#6b7280; font-size:0.95rem; margin-top:-0.25rem; }
      .hub-section { margin: 1.25rem 0 0.25rem 0; }
      .hub-grid { display:grid; gap:18px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
      .hub-card {
        border:1px solid #e5e7eb; border-radius:16px; padding:18px 18px 14px 18px;
        background:#fff;
      }
      .hub-card h3 { margin:0 0 0.25rem 0; font-size:1.1rem; }
      .hub-card p  { margin:0 0 0.75rem 0; color:#6b7280; font-size:0.95rem; }
      .hub-actions { display:flex; gap:10px; flex-wrap:wrap; }

      /* Promote st.page_link to button-like */
      .hub-actions [data-testid="stPageLink"] > a,
      .hub-cta {
        display:inline-block; padding:12px 16px; border-radius:12px;
        background:#111827; color:#fff !important; text-decoration:none !important;
        font-weight:700; font-size:0.98rem; border:none;
      }
      .hub-actions [data-testid="stPageLink"] > a:hover,
      .hub-cta:hover { filter:brightness(.92); }

      .btn-secondary {
        background:#f3f4f6; color:#111827 !important; border:1px solid #e5e7eb;
      }

      .disabled {
        opacity:0.45; cursor:not-allowed; pointer-events:none;
      }

      /* Tighten the global-grid section a bit */
      .grid-wrap .stNumberInput > div > div { width:100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Hero ----------------
st.markdown(
    """
    <h1 style="margin-bottom:0; font-size:2.4rem;">
      Analysis Hub
    </h1>
    <p style="font-size:1.05rem; color:#444; margin-top:0.5rem;">
      Explore how <b>Deep Learning–imputed</b> assays diverge from <b>Original</b> samples.  
      This hub gives you a <b>global difference summary</b> and lets you drill down into  
      <b>Drillhole</b> or <b>Surface</b> datasets for detailed record-level or aggregated comparisons.  
      Use the <b>Global Grid</b> settings below to configure how aggregated views are calculated.
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------- Navigation cards ----------------
st.markdown("### Choose a starting point", help="Pick by data domain and analysis granularity.")

st.markdown("<div class='hub-grid'>", unsafe_allow_html=True)

# Drillhole card 
with st.container(border=True):
    st.subheader("Drillhole analysis")
    st.caption("Work with drillhole samples at fine resolution, or use voxel aggregation to observe medium/coarse patterns.")
    a, b = st.columns([1, 1])
    with a:
        st.page_link("pages/05_Drillhole_Record.py",
                     label="Finer level · Record-level (samples)")
    with b:
        st.page_link("pages/06_Drillhole_Aggregated.py",
                     label="Medium/Coarse · Aggregated (voxels)")

# Surface card
with st.container(border=True):
    st.subheader("Surface analysis")
    st.caption("Work with surface grid points at fine resolution, or aggregated grids/contours for broader patterns.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.page_link("pages/07_Surface_Record.py",
                     label="Finer level · Record-level (grid points)")
    with c2:
        st.page_link("pages/08_Surface_Aggregated.py",
                     label="Medium/Coarse · Aggregated (grids/contours)")

st.markdown("</div>", unsafe_allow_html=True)  # end grid

# -----------------------------------------------------------------------------
# Task2 — Recompute difference tables
# -----------------------------------------------------------------------------
st.subheader("Change global grid settings")

cfg = load_cfg()
import math

# Helper: convert degrees to approximate meters at Perth latitude (~ -32°)
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

cur_xy = _cfg_fetch(cfg, ["grid_step_deg", "task2.grid_step_deg", "difference.grid_step_deg"], 1e-4)
cur_z  = _cfg_fetch(cfg, ["z_step_m", "task2.z_step_m", "difference.z_step_m"], 1.0)

# ---- Card: show current grid configuration ----
st.markdown(
    f"""
    <div class='hub-card'>
      <h3>Current grid configuration</h3>
      <p style='margin:0; font-size:0.9rem;'>XY grid step = {cur_xy:.6f} degree ({deg_to_m_split(cur_xy)})</p>
      <p style='margin:0; font-size:0.9rem;'>Z step = {cur_z:.1f} m</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Update form ----
st.markdown(
    "<p style='margin-top:1.2rem; font-size:1rem; font-weight:500;'>"
    "Update grid configuration for this run:"
    "</p>",
    unsafe_allow_html=True
)
col1, col2 = st.columns(2)
grid_step = col1.number_input("XY grid step (degree)", value=cur_xy, step=1e-4, format="%.6f")
z_step    = col2.number_input("Z step (m)", value=cur_z, step=0.1, format="%.1f")

if st.button("Update difference tables", type="primary"):
    with st.spinner("Updating tables, please wait..."):
        try:
            params = Params(grid_step_deg=float(grid_step), z_step_m=float(z_step))
            out = recompute_all(cfg, params)

            # success message styled as a hub-card
            st.markdown(
                f"""
                <div class='hub-card' style='border-left:4px solid #16a34a; background:#f0fdf4;'>
                  <h3 style='color:#166534;'>Tables successfully updated</h3>
                  <p style='margin:0; font-size:0.9rem;'>Files written to <code>reports/task2/difference/</code></p>
                  <p style='margin:0.5rem 0 0 0; font-size:0.9rem;'><b>Run used:</b><br>
                  XY grid step = {params.grid_step_deg:.6f} degree ({deg_to_m_split(params.grid_step_deg)})<br>
                  Z step = {params.z_step_m:.1f} m</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # show output file list
            for kind, mp in out.items():
                with st.expander(f"{kind.capitalize()} output files", expanded=False):
                    for name, p in mp.items():
                        st.caption(f"- {name}: `{p}`")

        except Exception as e:
            st.error("Failed to update tables")