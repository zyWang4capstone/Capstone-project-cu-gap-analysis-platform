# 04_Diff_Home.py — Analysis Hub (entry) + global grid

from __future__ import annotations
import streamlit as st
from _ui_common import inject_theme, load_grid_cfg, save_grid_cfg, DEFAULT_GRID

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
st.markdown("# Analysis Hub")
st.markdown(
    "<div class='hub-subtle'>Explore differences between "
    "<b>Original</b> and <b>Deep Learning</b> datasets. Choose a path below."
    " You can always return to adjust <b>Global Grid</b> parameters for aggregated views.</div>",
    unsafe_allow_html=True,
)
st.markdown("")

# ---------------- Navigation cards ----------------
st.markdown("### Choose a starting point", help="Pick by data domain and analysis granularity.")

st.markdown("<div class='hub-grid'>", unsafe_allow_html=True)

# Drillhole card 
with st.container(border=True):
    st.subheader("Drillhole analysis")
    st.caption("Work with drillhole records directly, or switch to aggregated points/voxels for large-scale patterns.")
    a, b = st.columns([1, 1])
    with a:
        st.page_link("pages/05_Drillhole_Record.py",
                     label="Finer level · Record-level")
    with b:
        st.page_link("pages/06_Drillhole_Aggregated.py",
                     label="Medium/Coarse · Aggregated (points/voxels)")

# Surface card
with st.container(border=True):
    st.subheader("Surface analysis")
    st.caption("Work with surface grids—record-level or aggregated contours/grids.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.page_link("pages/07_Surface_Record.py",
                     label="Finer level · Record-level")
    with c2:
        st.page_link("pages/08_Surface_Aggregated.py",
                     label="Medium/Coarse · Aggregated (grid/contours)")

st.markdown("</div>", unsafe_allow_html=True)  # end grid

st.divider()

# -----------------------------------------------------------------------------
# Global grid (applies to all analysis pages)
# -----------------------------------------------------------------------------
st.subheader("Global grid (applies to all analysis pages)")

cfg = load_grid_cfg()

c1, c2, c3 = st.columns(3)
with c1:
    cx = st.number_input("Cell X (m)", min_value=0.1, value=float(cfg["cell_x_m"]), step=0.5)
with c2:
    cy = st.number_input("Cell Y (m)", min_value=0.1, value=float(cfg["cell_y_m"]), step=0.5)
with c3:
    cz = st.number_input("Cell Z (m)", min_value=0.1, value=float(cfg["cell_z_m"]), step=0.1)

c4, c5 = st.columns(2)
with c4:
    agg = st.selectbox(
        "Aggregator", ["mean", "median", "max"],
        index=["mean", "median", "max"].index(str(cfg["agg"]))
    )
with c5:
    mc = st.number_input("Min samples per voxel", min_value=1, value=int(cfg["min_count"]), step=1)

b1, b2 = st.columns(2)
with b1:
    if st.button("Apply grid settings", use_container_width=True):
        save_grid_cfg(dict(
            cell_x_m=float(cx), cell_y_m=float(cy), cell_z_m=float(cz),
            agg=str(agg), min_count=int(mc)
        ))
        st.success("Grid settings saved. Aggregated pages will use the new grid.")
with b2:
    if st.button("Reset to defaults", use_container_width=True):
        save_grid_cfg(DEFAULT_GRID.copy())
        st.info("Grid settings reset to defaults (≈ 10×10×1 m, mean, min_count=3).")