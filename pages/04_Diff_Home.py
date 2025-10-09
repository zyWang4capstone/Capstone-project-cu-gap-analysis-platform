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
      .stAppDeployButton { display: none !important; }
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

# -----------------------------------------------------------------------------
# Global Difference Summary (Aggregated-level only)
# -----------------------------------------------------------------------------

st.subheader("Difference Summary")

# ----------------- Paths -----------------
DIFF_DIR = Path("reports/task2/difference")

diff_files = {
    "drillhole_all": DIFF_DIR / "drillhole_points_all.csv",
    "drillhole_overlap": DIFF_DIR / "drillhole_points_overlap.csv",
    "drillhole_dlonly": DIFF_DIR / "drillhole_points_dlonly.csv",
    "drillhole_origonly": DIFF_DIR / "drillhole_points_origonly.csv",
    "surface_all": DIFF_DIR / "surface_points_all.csv",
    "surface_overlap": DIFF_DIR / "surface_points_overlap.csv",
    "surface_dlonly": DIFF_DIR / "surface_points_dlonly.csv",
    "surface_origonly": DIFF_DIR / "surface_points_origonly.csv",
}

tabs = st.tabs([
    "Coverage & Value Summary",
    "Difference Metrics"
])

# -----------------------------------------------------------------------------
# TAB 1: Coverage (with counts + value distributions)
# -----------------------------------------------------------------------------
with tabs[0]:
    st.caption("Aggregated Coverage Summary (cell counts with percentages)")

    # ---------- Coverage Summary ----------
    def get_agg_coverage(prefix: str, label: str):
        """Compute overlap and unique coverage stats with counts and percentages."""
        try:
            df_all = pd.read_csv(DIFF_DIR / f"{prefix}_points_all.csv")
            df_overlap = pd.read_csv(DIFF_DIR / f"{prefix}_points_overlap.csv")
            df_dlonly = pd.read_csv(DIFF_DIR / f"{prefix}_points_dlonly.csv")
            df_origonly = pd.read_csv(DIFF_DIR / f"{prefix}_points_origonly.csv")

            n_total = len(df_all)
            n_overlap = len(df_overlap)
            n_dlonly = len(df_dlonly)
            n_origonly = len(df_origonly)

            overlap_pct = n_overlap / n_total * 100 if n_total else 0
            dlonly_pct = n_dlonly / n_total * 100 if n_total else 0
            origonly_pct = n_origonly / n_total * 100 if n_total else 0

            def fmt(n, pct):
                return f"{n:,} ({pct:.2f}%)"

            return {
                "Dataset": label,
                "Total Cells": f"{n_total:,}",
                "Overlap": fmt(n_overlap, overlap_pct),
                "DL-only": fmt(n_dlonly, dlonly_pct),
                "Original-only": fmt(n_origonly, origonly_pct),
            }
        except Exception as e:
            return {"Dataset": label, "Status": f"Error: {e}"}

    coverage_df = pd.DataFrame([
        get_agg_coverage("drillhole", "Drillhole"),
        get_agg_coverage("surface", "Surface")
    ])
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)

    # ---------- Helper for value summary ----------
    def get_value_summary(df: pd.DataFrame, label: str):
        """Return descriptive statistics for value columns."""
        val_cols = [c for c in df.columns if "value" in c.lower()]
        if not val_cols:
            return None
        col = val_cols[0]
        s = df[col].dropna()
        return {
            "Subset": label,
            "Count": len(s),
            "Mean": s.mean(),
            "Std": s.std(),
            "Q25": s.quantile(0.25),
            "Median": s.median(),
            "Q75": s.quantile(0.75),
            "Q95": s.quantile(0.95),
            "Q99": s.quantile(0.99),
            "Min": s.min(),
            "Max": s.max(),
        }

    # ---------- Domain-level summaries ----------
    def show_value_summary(prefix: str, domain_label: str):
        """Show value distributions for overlap, DL-only, and Original-only subsets."""

        try:
            df_overlap = pd.read_csv(DIFF_DIR / f"{prefix}_points_overlap.csv")
            df_dlonly = pd.read_csv(DIFF_DIR / f"{prefix}_points_dlonly.csv")
            df_origonly = pd.read_csv(DIFF_DIR / f"{prefix}_points_origonly.csv")

            stats = [
                get_value_summary(df_overlap, "Overlap"),
                get_value_summary(df_dlonly, "DL-only"),
                get_value_summary(df_origonly, "Original-only"),
            ]
            stats = [s for s in stats if s]
            summary_df = pd.DataFrame(stats).round(3)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load {domain_label} datasets: {e}")

    # Drillhole & Surface summaries
    st.caption("Value Distributions by Subset")
    show_value_summary("drillhole", "Drillhole")
    show_value_summary("surface", "Surface")

# -----------------------------------------------------------------------------
# TAB 2: Aggregated Difference Metrics (DL – Original)
# -----------------------------------------------------------------------------
with tabs[1]:
    st.caption(
        "All metrics below are computed on overlapping grid cells where both DL and Original values exist."
    )

    def get_overlap_pct(prefix: str) -> float:
        df_all = pd.read_csv(DIFF_DIR / f"{prefix}_points_all.csv")
        df_overlap = pd.read_csv(DIFF_DIR / f"{prefix}_points_overlap.csv")
        n_total, n_overlap = len(df_all), len(df_overlap)
        return n_overlap / n_total * 100 if n_total else 0

    def compute_detailed_diff(path: Path, label: str, overlap_pct: float, col_name: str = "diff"):
        """Compute summary statistics for overlap region."""
        if not path.exists():
            return None

        df = pd.read_csv(path)
        diff_cols = [c for c in df.columns if c.lower() == col_name.lower()]
        if not diff_cols:
            return None

        diff = df[diff_cols[0]].dropna()
        n_total = len(diff)
        n_higher = (diff > 0).sum()
        n_lower = (diff < 0).sum()
        pct_higher = n_higher / n_total * 100 if n_total else 0
        pct_lower = n_lower / n_total * 100 if n_total else 0

        overview = {
            "Dataset": label,
            "Overlap Cells": f"{n_total:,} ({overlap_pct:.2f}%)",
            "DL Higher": f"{n_higher:,} ({pct_higher:.2f}%)",
            "DL Lower": f"{n_lower:,} ({pct_lower:.2f}%)",
        }

        summary = {
            "Dataset": label,
            "Mean Diff": diff.mean(),
            "Std Diff": diff.std(),
            "MAE": diff.abs().mean(),
            "RMSE": (diff ** 2).mean() ** 0.5,
            "Q25": diff.quantile(0.25),
            "Median": diff.median(),
            "Q75": diff.quantile(0.75),
            "Q95": diff.quantile(0.95),
            "Q99": diff.quantile(0.99),
            "Min": diff.min(),
            "Max": diff.max(),
        }

        return overview, summary

    # Run for both datasets
    overlap_dh = get_overlap_pct("drillhole")
    overlap_sf = get_overlap_pct("surface")

    drillhole = compute_detailed_diff(diff_files["drillhole_overlap"], "Drillhole", overlap_dh)
    surface = compute_detailed_diff(diff_files["surface_overlap"], "Surface", overlap_sf)

    if drillhole and surface:
        overview_df = pd.DataFrame([drillhole[0], surface[0]])
        summary_df = pd.DataFrame([drillhole[1], surface[1]]).round(3)

        st.caption("A. Directional Overview")
        st.dataframe(overview_df, use_container_width=True, hide_index=True)

        st.caption("B. Summary Statistics")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Download option
        csv_buf = io.StringIO()
        summary_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download Difference Metrics CSV",
            data=csv_buf.getvalue(),
            file_name="aggregated_diff_metrics.csv",
            mime="text/csv",
        )
    else:
        st.warning("Missing overlap files or 'diff' column.")
