# pages/09_Insights.py — Unified Insights (Drillhole + Surface)
# Rules:
#   A) DL-only high-value hotspots (drillhole)
#   B) Overlap uplift hotspots (DL - ORIG) on drillhole
#   C) Surface vs Drill discrepancy at a Z slice
#   D) DL-only clustering (Union Drillhole + Surface, DBSCAN Exploration)
# Note:
#   We cluster in degrees for lon/lat (no meter conversion), and meters for depth.
#   Voxel mode: each occupied voxel is a hotspot (no adjacency merge).
#   DBSCAN mode: clusters are discovered in normalized (lon/lat/dep) using voxel sizes.

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import io
import plotly.graph_objects as go
from cap_common.config import load_cfg
from cap_task2.io import read_points

from _ui_common import inject_theme
inject_theme()

# Optional sklearn (DBSCAN)
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ---------------------------- Page & theme ----------------------------
st.set_page_config(layout="wide", page_title="Insights (Unified)")
inject_theme()

# Match tab style used in 05/06/07/08
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] {
        background: #f8fafc !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
        border-bottom: none !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 0.45rem 1rem !important;
        margin-right: 6px !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    .stTabs [data-baseweb="tab"] * { color: inherit !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #111827 !important;
        color: #ffffff !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        border-color: #111827 !important;
        transform: translateY(1px);
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
    .stTabs + div [data-testid="stVerticalBlock"] > div:first-child {
        border-top: 1px solid #11182710;
        margin-top: -1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- Paths ----------------------------
DIFF_DIR = Path("reports/task2/difference")

# Drillhole points (3D)
DH_SPLITS = {
    "all":      DIFF_DIR / "drillhole_points_all.csv",
    "overlap":  DIFF_DIR / "drillhole_points_overlap.csv",
    "origonly": DIFF_DIR / "drillhole_points_origonly.csv",
    "dlonly":   DIFF_DIR / "drillhole_points_dlonly.csv",
}

# Surface points (2D)
SF_SPLITS = {
    "all":      DIFF_DIR / "surface_points_all.csv",
    "overlap":  DIFF_DIR / "surface_points_overlap.csv",
    "origonly": DIFF_DIR / "surface_points_origonly.csv",
    "dlonly":   DIFF_DIR / "surface_points_dlonly.csv",
}

# ---------------------------- I/O helpers ----------------------------
@st.cache_data
def get_cfg():
    return load_cfg()
cfg = get_cfg()

def _fix_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Standardise columns for drillhole or surface data."""
    df = df.loc[:, ~df.columns.duplicated()].rename(columns=lambda c: str(c).upper())
    if kind == "surface":
        if "DLAT" in df.columns:  df = df.rename(columns={"DLAT":"LATITUDE"})
        if "DLONG" in df.columns: df = df.rename(columns={"DLONG":"LONGITUDE"})
    # numeric conversion
    for c in ["LONGITUDE","LATITUDE","DEPTH","VALUE_ORIG","VALUE_DL","DIFF","DIFF_PCT"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # DIFF/DIFF_PCT backfill
    if "DIFF" not in df.columns and {"VALUE_ORIG","VALUE_DL"}.issubset(df.columns):
        df["DIFF"] = df["VALUE_DL"] - df["VALUE_ORIG"]
    if "DIFF_PCT" not in df.columns and {"DIFF","VALUE_ORIG"}.issubset(df.columns):
        eps = 1e-9
        denom = df["VALUE_ORIG"].where(df["VALUE_ORIG"].abs() >= eps, np.nan)
        df["DIFF_PCT"] = 100.0 * df["DIFF"] / denom
    return df.dropna(subset=["LONGITUDE","LATITUDE"]).copy()

@st.cache_data
def read_drill_split(split: str) -> pd.DataFrame:
    df = read_points("drillhole", split, cfg)
    return _fix_columns(df, "drillhole")

@st.cache_data
def read_surface_split(split: str) -> pd.DataFrame:
    df = read_points("surface", split, cfg)
    return _fix_columns(df, "surface")

# Load once
dh_all      = read_drill_split("all")
dh_overlap  = read_drill_split("overlap")
dh_dlonly   = read_drill_split("dlonly")

sf_all      = read_surface_split("all")
sf_overlap  = read_surface_split("overlap")
sf_dlonly   = read_surface_split("dlonly")

# ---------------------------- Page header ----------------------------
h1, h2, h3 = st.columns([0.6, 0.2, 0.2])
with h1:
    st.markdown(
        """
        <h1 style="margin-bottom:0.25rem; font-size:2.0rem;">
        Insights • Unified (Drillhole + Surface)
        </h1>
        <p style="color:#555; margin-top:0;">
        Cluster-level findings only — hotspots, summary tables, and downloads.
        </p>
        """,
        unsafe_allow_html=True,
    )
with h2:
    st.page_link("pages/04_Diff_Home.py", label="Back to Diff • Home")
with h3:
    st.page_link("pages/05_Drillhole_Record.py", label="Open Record-level")


# ---------------------------- Sidebar: global filters ----------------------------
union_for_ranges = pd.concat(
    [df for df in [dh_all, dh_overlap, dh_dlonly, sf_all, sf_overlap, sf_dlonly] if not df.empty],
    ignore_index=True
)
if union_for_ranges.empty:
    st.warning("No drillhole/surface data found in `reports/task2/difference/`. Check paths.")
    st.stop()

# --- Longitude / Latitude always available ---
lon_min, lon_max = float(union_for_ranges["LONGITUDE"].min()), float(union_for_ranges["LONGITUDE"].max())
lat_min, lat_max = float(union_for_ranges["LATITUDE"].min()),  float(union_for_ranges["LATITUDE"].max())

# --- Depth only for drillhole ---
if "DEPTH" in union_for_ranges.columns:
    dep_min, dep_max = float(union_for_ranges["DEPTH"].min()), float(union_for_ranges["DEPTH"].max())
else:
    dep_min, dep_max = 0.0, 2500.0  # fallback for surface-only case

# --- Sliders (same style as other pages) ---
xr = st.sidebar.slider("Longitude", lon_min, lon_max, (lon_min, lon_max))
yr = st.sidebar.slider("Latitude",  lat_min, lat_max, (lat_min, lat_max))
zr = st.sidebar.slider("Depth",     max(0.0, dep_min), min(2500.0, dep_max),
                       (max(0.0, dep_min), min(2500.0, dep_max)))

# --- Clustering parameters ---
st.sidebar.markdown("---")
st.sidebar.subheader("Clustering defaults (degrees for lon/lat; meters for depth)")

# Default 0.01° ≈ ~1 km at equator, ~100 m at mine scale if zoomed in
grid_lon = st.sidebar.number_input("Voxel size • lon (°)", 0.0001, 1.0, 0.01, 0.01, format="%.4f")
grid_lat = st.sidebar.number_input("Voxel size • lat (°)", 0.0001, 1.0, 0.01, 0.01, format="%.4f")

# Depth in meters, only relevant for drillhole
grid_dep = st.sidebar.number_input("Voxel size • depth (m)", 0.1, 1000.0, 5.0, 1.0)

# ---------------------------- Shared utilities ----------------------------
def apply_window_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe by current lon/lat/depth sliders."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df[
        df["LONGITUDE"].between(*xr) &
        df["LATITUDE"].between(*yr) &
        (df["DEPTH"].between(*zr) if "DEPTH" in df.columns else True)
    ].copy()
    return out


def as_1d_numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    """Return numeric numpy array for a single column, safe against dup/NA."""
    if col not in df.columns:
        return np.asarray([], dtype=float)
    s = df[col]
    if isinstance(s, pd.DataFrame):  # handle duplicate column names
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").to_numpy()


def voxelize_deg(df: pd.DataFrame, gx_deg: float, gy_deg: float, gz_m: float) -> pd.DataFrame:
    """
    Add voxel indices (_vx,_vy,_vz) in degree/degree/meter space.
    Origin is local (per-dataset min values) to keep indices small.
    Works for both drillhole (3D) and surface (2D with depth=0).
    """
    if df is None or df.empty:
        return df.copy()

    df = df.loc[:, ~df.columns.duplicated()].copy()

    lon = as_1d_numeric(df, "LONGITUDE")
    lat = as_1d_numeric(df, "LATITUDE")
    dep = as_1d_numeric(df, "DEPTH") if "DEPTH" in df.columns else np.zeros(len(lon))

    # reference point to keep voxel indices small
    lon0, lat0, z0 = np.nanmin(lon), np.nanmin(lat), np.nanmin(dep)

    vx = np.floor((lon - lon0) / float(gx_deg)).astype(int)
    vy = np.floor((lat - lat0) / float(gy_deg)).astype(int)
    vz = np.floor((dep - z0)   / float(gz_m)).astype(int)

    df["_vx"], df["_vy"], df["_vz"] = vx, vy, vz
    df["_voxel"] = list(zip(vx, vy, vz))
    return df

def dbscan_on_deg(df: pd.DataFrame, gx_deg: float, gy_deg: float, gz_m: float,
                  min_count: int) -> pd.DataFrame:
    """
    Run DBSCAN in normalized lon/lat/dep space.
    - Normalize by voxel sizes so eps≈1 means ~one voxel.
    - Handles both drillhole (3D) and surface (2D, depth=0).
    - Returns cluster summary (size, score stats, center).
    """
    if not SKLEARN_OK or df is None or df.empty:
        return pd.DataFrame()

    df = df.loc[:, ~df.columns.duplicated()].copy()

    lon = as_1d_numeric(df, "LONGITUDE")
    lat = as_1d_numeric(df, "LATITUDE")
    dep = as_1d_numeric(df, "DEPTH") if "DEPTH" in df.columns else np.zeros(len(lon))

    # Normalize
    lon0, lat0, z0 = np.nanmin(lon), np.nanmin(lat), np.nanmin(dep)
    X = np.c_[
        (lon - lon0) / float(gx_deg),
        (lat - lat0) / float(gy_deg),
        (dep - z0)  / float(gz_m)
    ]

    # Run DBSCAN
    model = DBSCAN(eps=1.05, min_samples=int(min_count)).fit(X)
    df["_cid"] = model.labels_

    # Summarize clusters
    rows = []
    for cid, sub in df.groupby("_cid"):
        if cid < 0 or len(sub) < min_count:
            continue

        # Optional weight: SCORE if exists, else uniform
        if "SCORE" in sub.columns:
            vals = pd.to_numeric(sub["SCORE"], errors="coerce").to_numpy()
            w = np.nan_to_num(vals, nan=0.0) + 1e-9
        else:
            vals = np.zeros(len(sub), dtype=float)
            w = np.ones(len(sub), dtype=float)

        cx = float(np.average(as_1d_numeric(sub, "LONGITUDE"), weights=w))
        cy = float(np.average(as_1d_numeric(sub, "LATITUDE"),  weights=w))
        cz = float(np.average(as_1d_numeric(sub, "DEPTH"),     weights=w)) if "DEPTH" in sub.columns else 0.0

        rows.append(dict(
            cluster_id=int(cid),
            size=len(sub),
            score_mean=float(np.nanmean(vals)) if len(vals) else 0.0,
            score_median=float(np.nanmedian(vals)) if len(vals) else 0.0,
            score_p95=float(np.nanpercentile(vals, 95)) if len(vals) else 0.0,
            lon_center=cx, lat_center=cy, z_center=cz
        ))

    out = pd.DataFrame(rows)

    # Re-index cluster IDs by size/importance
    if not out.empty:
        out = out.sort_values(["size", "score_mean"], ascending=[False, False]).reset_index(drop=True)
        out["cluster_id"] = out.index + 1

    return out

def cluster_voxel(df: pd.DataFrame, score_col: str = "SCORE", min_count: int = 1) -> pd.DataFrame:
    """
    Voxel-based clustering:
    - Each occupied voxel is one cluster.
    - Compatible with both drillhole (3D) and surface (2D, depth=0).
    - Returns cluster summary (size, score stats, center).
    """
    if df is None or df.empty or "_voxel" not in df.columns:
        return pd.DataFrame()

    rows = []
    cid = 0
    for vox, sub in df.groupby("_voxel"):
        if len(sub) < min_count:
            continue

        # If score_col not present, use zeros (uniform weight)
        if score_col not in sub.columns:
            vals = np.zeros(len(sub), dtype=float)
        else:
            vals = pd.to_numeric(sub[score_col], errors="coerce").to_numpy()

        w = np.nan_to_num(vals, nan=0.0) + 1e-9

        # Weighted centers
        cx = float(np.average(sub["LONGITUDE"].to_numpy(), weights=w))
        cy = float(np.average(sub["LATITUDE"].to_numpy(),  weights=w))
        cz = float(np.average(sub["DEPTH"].to_numpy(),     weights=w)) if "DEPTH" in sub.columns else 0.0

        # Record cluster summary
        cid += 1
        rows.append(dict(
            cluster_id=cid,
            size=len(sub),
            score_mean=float(np.nanmean(vals)) if len(vals) else 0.0,
            score_median=float(np.nanmedian(vals)) if len(vals) else 0.0,
            score_p95=float(np.nanpercentile(vals, 95)) if len(vals) else 0.0,
            lon_center=cx, lat_center=cy, z_center=cz
        ))

    return pd.DataFrame(rows)

# ---------- Ranking / legend helpers ----------
TOP_COLORS = ["#ef4444", "#f59e0b", "#10b981"]  # red, orange, green
OTHER_COLOR = "#9ca3af"                         # gray

# ---------- Ranking / legend helpers ----------
TOP_COLORS = ["#ef4444", "#f59e0b", "#10b981"]  # red, orange, green
OTHER_COLOR = "#9ca3af"                         # gray

def rank_and_color(df_clusters: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """
    Rank clusters and assign labels/colors for plotting.

    Ranking criteria:
      1. size (descending)
      2. score_mean (descending)

    Output: same DataFrame with extra columns:
      - rank (int, 1-based)
      - legend ("Top N" or "Others")
      - color (hex string)
    """
    if df_clusters is None or df_clusters.empty:
        return pd.DataFrame()

    df = df_clusters.copy()
    df = df.sort_values(["size", "score_mean"], ascending=[False, False]).reset_index(drop=True)

    # Assign ranks
    df["rank"] = np.arange(1, len(df) + 1)

    # Assign legend & colors
    def assign_label_color(r: int) -> tuple[str, str]:
        if r <= top_k:
            color = TOP_COLORS[r - 1] if (r - 1) < len(TOP_COLORS) else TOP_COLORS[-1]
            return (f"Top {r}", color)
        else:
            return ("Others", OTHER_COLOR)

    legends, colors = zip(*[assign_label_color(r) for r in df["rank"]])
    df["legend"] = legends
    df["color"] = colors

    return df

def plot_star_clusters_ranked(df_clusters: pd.DataFrame, title: str,
                              text_size: int = 16, anchor_size: int = 6):
    """
    Plot 3D clusters with Top1/Top2/Top3 marked as stars (★).
    Others shown as gray anchors.
    """

    if df_clusters is None or df_clusters.empty:
        st.info("No clusters under current parameters.")
        return

    fig = go.Figure()
    order = ["Top 1", "Top 2", "Top 3", "Others"]

    for label in order:
        sub = df_clusters[df_clusters["legend"] == label]
        if sub.empty:
            continue

        color_this = sub["color"].iloc[0]

        # --- "Others" just show points ---
        if label == "Others":
            fig.add_trace(go.Scatter3d(
                x=sub["lon_center"], y=sub["lat_center"], z=sub["z_center"],
                mode="markers",
                marker=dict(symbol="diamond-open", size=anchor_size,
                            color=color_this, opacity=0.6),
                name="Others",
                customdata=np.c_[
                    sub["size"].to_numpy(),
                    sub["score_mean"].to_numpy(),
                    sub["score_median"].to_numpy(),
                    sub["score_p95"].to_numpy(),
                ],
                hovertemplate=(
                    "Center: (%{x:.5f}, %{y:.5f}, %{z:.1f})<br>"
                    "Size: %{customdata[0]}<br>"
                    "Mean: %{customdata[1]:.3f} · "
                    "Median: %{customdata[2]:.3f} · "
                    "P95: %{customdata[3]:.3f}"
                    "<extra></extra>"
                ),
                showlegend=True,
            ))

        # --- Top 1/2/3: add ★ text labels ---
        else:
            labels = [f"★{r}" for r in sub["rank"]]   # 用 rank，而不是 cluster_id
            fig.add_trace(go.Scatter3d(
                x=sub["lon_center"], y=sub["lat_center"], z=sub["z_center"],
                mode="markers+text",
                marker=dict(symbol="diamond", size=anchor_size + 2,
                            color=color_this, opacity=0.95, line=dict(width=1, color="black")),
                text=labels,
                textposition="top center",
                textfont=dict(size=text_size, color=color_this),
                name=label,
                customdata=np.c_[
                    sub["size"].to_numpy(),
                    sub["score_mean"].to_numpy(),
                    sub["score_median"].to_numpy(),
                    sub["score_p95"].to_numpy(),
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Center: (%{x:.5f}, %{y:.5f}, %{z:.1f})<br>"
                    "Size: %{customdata[0]}<br>"
                    "Mean: %{customdata[1]:.3f} · "
                    "Median: %{customdata[2]:.3f} · "
                    "P95: %{customdata[3]:.3f}"
                    "<extra></extra>"
                ),
                showlegend=True,
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis=dict(title="Depth", autorange="reversed"),
        ),
        height=720,
        margin=dict(l=0, r=0, t=40, b=10),
        title=dict(text=title, x=0.5, font=dict(size=20)),
        legend=dict(title="Hotspot rank", itemsizing="constant", font=dict(size=14))
    )

    st.plotly_chart(fig, use_container_width=True)

def export_table(df: pd.DataFrame, filename: str):
    """Render a table and a CSV download button."""
    if df is None or df.empty:
        st.info("No table available for export.")
        return
    st.dataframe(df, use_container_width=True, height=420, hide_index=True)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        "Download CSV",
        data=buf.getvalue().encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        type="primary"
    )

def percentile_threshold(arr: np.ndarray, p: float) -> float:
    """Safe wrapper for np.nanpercentile (returns nan if array empty)."""
    return float(np.nanpercentile(arr, p)) if len(arr) else float("nan")


def summarize_clusters_topk(df: pd.DataFrame, metric_name: str, top_k: int = 3) -> str:
    """Plain-language summary focusing on Top-K hotspots."""
    if df is None or df.empty:
        return "No hotspots were found under the current filters."

    if "rank" not in df.columns:
        df = rank_and_color(df, top_k=top_k)

    n = int(df["cluster_id"].nunique())
    lines = [f"Found **{n}** hotspots in total."]

    top = df[df["legend"].str.startswith("Top")].sort_values("rank").head(top_k)
    if not top.empty:
        lines.append("Top hotspots:")
        for _, r in top.iterrows():
            lines.append(
                f"- **{r['legend']}** — size **{int(r['size'])}**, "
                f"center (lon **{r['lon_center']:.5f}**, lat **{r['lat_center']:.5f}**, depth **{r['z_center']:.0f} m**), "
                f"mean {metric_name} **{r['score_mean']:.3f}**."
            )

    mean_val = float(df["score_mean"].mean())
    max_p95 = float(df["score_p95"].max())
    lines.append(
        f"Overall average {metric_name}: **{mean_val:.3f}** "
        f"(max P95 **{max_p95:.3f}**)."
    )
    return "\n".join(lines)

def voxel_coverage_stats_deg(base_df: pd.DataFrame, sub_df_vox: pd.DataFrame,
                             gx_deg: float, gy_deg: float, gz_m: float) -> tuple[int, int]:
    """
    Estimate total voxels within the current bbox (in degrees/meters),
    and actually occupied voxels after thresholding.
    """
    if base_df is None or base_df.empty:
        return (0, 0)

    base_df = base_df.loc[:, ~base_df.columns.duplicated()]
    lon = as_1d_numeric(base_df, "LONGITUDE")
    lat = as_1d_numeric(base_df, "LATITUDE")
    dep = as_1d_numeric(base_df, "DEPTH")

    if len(lon) == 0 or len(lat) == 0 or len(dep) == 0:
        return (0, 0)

    nx = int(np.ceil((np.nanmax(lon) - np.nanmin(lon)) / float(gx_deg)))
    ny = int(np.ceil((np.nanmax(lat) - np.nanmin(lat)) / float(gy_deg)))
    nz = int(np.ceil((np.nanmax(dep) - np.nanmin(dep)) / float(gz_m)))
    total = max(0, nx * ny * nz)

    occ = 0
    if sub_df_vox is not None and not sub_df_vox.empty and "_voxel" in sub_df_vox.columns:
        occ = int(pd.Series(sub_df_vox["_voxel"]).nunique())

    return (total, occ)

def render_hotspot_metrics(total_vox: int, occ_vox: int, n_hot: int):
    """Display three small metric cards."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Total voxels (in window)", f"{total_vox:,}")
    c2.metric("Occupied voxels (selected)", f"{occ_vox:,}")
    c3.metric("Hotspots (clusters)", f"{n_hot:,}")

# ---------------------------- Tabs ----------------------------
tabA, tabB, tabC, tabD, tabE = st.tabs([
    "DL-only peaks (drillhole)",
    "DL uplift hotspots (overlap)",
    "Surface vs Drill discrepancy",
     "DL-only Clustering (Exploration)",
     "Overlay: Surface + Drillhole"
])

# ==============================================================
# A) DL-only high-value hotspots (score = VALUE_DL)
# ==============================================================

with tabA:
    st.caption(
        "These areas only exist in the DL dataset, "
        "with no records in the original data. "
        "We highlight new high-value regions discovered by DL, showing the additional insights it brings."
    )

    lc, rc = st.columns([0.36, 0.64])

    # --- Left column: clustering & threshold parameters ---
    with lc:
        method = st.selectbox(
            "Clustering", ["Voxel (fast)", "DBSCAN (sklearn)"], index=0,
            help="Voxel: space is divided into fixed cubes; "
                 "DBSCAN: groups points based on density without predefined cubes."
        )
        min_count = st.number_input("Min samples/cluster", 1, 1000, 5, 1)
        thr_mode = st.radio("Threshold mode", ["Percentile", "Absolute"], index=0, horizontal=True)
        pctl = st.slider(
            "Percentile (on DL values)", 50.0, 99.9, 90.0, 0.1,
            disabled=(thr_mode == "Absolute")
        )
        thr_abs = st.number_input(
            "Absolute threshold (DL values)", 0.0, 1e12, 0.0, 1.0,
            disabled=(thr_mode == "Percentile")
        )
        star_size = st.slider("Marker size for clusters", 6, 24, 14)
        st.button("Recalculate", type="primary")

    # --- Right column: clustering results & visualization ---
    with rc:
        base = apply_window_3d(dh_dlonly)
        element = st.session_state.get("element", "Element")

        score_col = "VALUE_DL"          # Standard DL value column
        display_label = f"{element} DL" # Display name for summary

        if base.empty or score_col not in base.columns:
            st.info(f"No DL-only drillhole rows for {element} in current window.")
        else:
            # Prepare SCORE column
            base["SCORE"] = pd.to_numeric(base[score_col], errors="coerce")

            # Thresholding
            thr = percentile_threshold(base["SCORE"].to_numpy(), pctl) \
                  if thr_mode == "Percentile" else float(thr_abs)
            sub = base[base["SCORE"] >= thr].copy()

            if sub.empty:
                st.info("No points meet the current threshold.")
            else:
                # --- Run clustering ---
                if method.startswith("Voxel"):
                    sub_vox = voxelize_deg(sub, grid_lon, grid_lat, grid_dep)
                    clusters = cluster_voxel(sub_vox, "SCORE", min_count)
                else:
                    sub_vox = None
                    clusters = dbscan_on_deg(sub, grid_lon, grid_lat, grid_dep, min_count)

                # --- Stats ---
                total_vox, occ_vox = voxel_coverage_stats_deg(base, sub_vox, grid_lon, grid_lat, grid_dep)
                n_hot = 0 if clusters is None or clusters.empty else int(clusters.shape[0])
                render_hotspot_metrics(total_vox, occ_vox, n_hot)

                # --- Plot ---
                if clusters is not None and not clusters.empty:
                    clusters = rank_and_color(clusters, top_k=3)
                    plot_star_clusters_ranked(
                        clusters, f"DL-only peaks ({display_label})",
                        text_size=star_size, anchor_size=max(5, star_size // 2)
                    )

                    # Scatter plot with cluster colors
                    if "cluster_id" in sub.columns:
                        fig = px.scatter(
                            sub,
                            x="LONGITUDE", y="LATITUDE", color="cluster_id",
                            hover_data={
                                "LONGITUDE": True,
                                "LATITUDE": True,
                                "DEPTH": "DEPTH" in sub.columns,
                                "SCORE": True,
                                "VALUE_DL": True
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Export + summary
                    export_table(clusters, f"insights_A_dl_only_clusters_{element}.csv")
                    st.markdown(summarize_clusters_topk(clusters, display_label, top_k=3))
                else:
                    st.info("No valid clusters found under current parameters.")

# ==============================================================
# B) Overlap uplift hotspots (score = DIFF = DL - ORIG)
# ==============================================================
with tabB:
    st.caption(
        "These areas have both DL and ORIG data, but DL values are much higher. "
        "Such uplift indicates DL captures stronger anomalies even in regions already covered by ORIG."
    )

    lc, rc = st.columns([0.36, 0.64])

    # --- Left column: clustering & threshold parameters ---
    with lc:
        method = st.selectbox(
            "Clustering", ["Voxel (fast)", "DBSCAN (sklearn)"],
            index=0, key="b_method",
            help="Voxel: space is divided into fixed cubes; "
                 "DBSCAN: groups points based on density without predefined cubes."
        )
        min_count = st.number_input("Min samples/cluster", 1, 1000, 5, 1, key="b_min")
        thr_mode = st.radio("Threshold mode", ["Percentile", "Absolute"], index=0, horizontal=True, key="b_thr_mode")
        pctl = st.slider("Percentile (on DIFF)", 50.0, 99.9, 90.0, 0.1,
                         disabled=(thr_mode == "Absolute"), key="b_pctl")
        thr_abs = st.number_input("Absolute threshold (DIFF)", 0.0, 1e12, 0.0, 1.0,
                                  disabled=(thr_mode == "Percentile"), key="b_thr_abs")
        star_size = st.slider("Star size", 6, 24, 14, key="b_star")
        st.button("Recalculate", type="primary", key="b_go")

    # --- Right column: clustering results & visualization ---
    with rc:
        base = apply_window_3d(dh_overlap)

        if base.empty or "DIFF" not in base.columns:
            st.info("No overlap rows with DIFF in current window.")
        else:
            base["SCORE"] = pd.to_numeric(base["DIFF"], errors="coerce")

            thr = percentile_threshold(base["SCORE"].to_numpy(), pctl) \
                  if thr_mode == "Percentile" else float(thr_abs)
            sub = base[base["SCORE"] >= thr].copy()

            if sub.empty:
                st.info("No points meet the threshold.")
            else:
                # --- Run clustering ---
                if method.startswith("Voxel"):
                    sub_vox = voxelize_deg(sub, grid_lon, grid_lat, grid_dep)
                    clusters = cluster_voxel(sub_vox, "SCORE", min_count)
                    if not clusters.empty:
                        # Merge cluster_id back to sub
                        sub = sub.merge(sub_vox[["_voxel"]], left_index=True, right_index=True, how="left")
                        sub = sub.merge(clusters[["cluster_id"]], left_index=True, right_index=True, how="left")
                else:
                    sub_vox = None
                    clusters = dbscan_on_deg(sub, grid_lon, grid_lat, grid_dep, min_count)
                    if not clusters.empty:
                        # Assign cluster_id from DBSCAN labels
                        sub["_cid"] = DBSCAN(eps=1.05, min_samples=int(min_count)).fit_predict(
                            np.c_[
                                (sub["LONGITUDE"] - sub["LONGITUDE"].min()) / grid_lon,
                                (sub["LATITUDE"] - sub["LATITUDE"].min()) / grid_lat,
                                (sub["DEPTH"] - sub["DEPTH"].min()) / grid_dep
                            ]
                        )
                        sub = sub.rename(columns={"_cid": "cluster_id"})

                # --- Stats ---
                total_vox, occ_vox = voxel_coverage_stats_deg(base, sub_vox, grid_lon, grid_lat, grid_dep)
                n_hot = 0 if clusters is None or clusters.empty else int(clusters.shape[0])
                render_hotspot_metrics(total_vox, occ_vox, n_hot)

                # --- Plot ---
                if clusters is not None and not clusters.empty:
                    clusters = rank_and_color(clusters, top_k=3)
                    plot_star_clusters_ranked(
                        clusters, "DL uplift hotspots",
                        text_size=star_size, anchor_size=max(5, star_size // 2)
                    )

                    # Add scatter plot for individual points
                    fig = px.scatter(
                        sub,
                        x="LONGITUDE", y="LATITUDE", color="cluster_id",
                        hover_data={
                            "LONGITUDE": True,
                            "LATITUDE": True,
                            "DEPTH": "DEPTH" in sub.columns,
                            "VALUE_ORIG": True,
                            "VALUE_DL": True,
                            "DIFF": True,
                            "SCORE": True
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Export + summary
                    export_table(clusters, "insights_B_uplift_clusters.csv")
                    st.markdown(summarize_clusters_topk(clusters, "uplift (DL − ORIG)", top_k=3))
                else:
                    st.info("No valid clusters found under current parameters.")

# ==============================================================
# C) Surface vs Drill discrepancy at a Z slice
# ==============================================================
with tabC:
    st.caption(
        "At a chosen depth, we compare surface predictions against drillhole measurements. "
        "Significant differences reveal where surface models and drill samples do not align."
    )

    lc, rc = st.columns([0.36, 0.64])

    # --- Left column: parameters ---
    with lc:
        z_center = st.slider("Target Z (depth)", float(zr[0]), float(zr[1]), 100.0, 1.0)
        z_halfwin = st.number_input("±Z window (drill)", 0.0, 500.0, 25.0, 1.0)
        nx = st.slider("Surface NX (lon bins)", 20, 400, 120, 10)
        ny = st.slider("Surface NY (lat bins)", 20, 400, 120, 10)
        min_tile_cnt = st.number_input("Min samples/tile (surface)", 1, 500, 3, 1)

        # new: filter by discrepancy direction
        direction = st.selectbox(
            "Discrepancy direction",
            ["Surface higher", "Drill higher"],
            index=0,
            key="c_dir"
        )

        method = st.selectbox("Clustering", ["Voxel (fast)", "DBSCAN (sklearn)"], index=0, key="c_method")
        min_count = st.number_input("Min samples/cluster", 1, 1000, 5, 1, key="c_min")

        thr_mode = st.radio("Threshold mode", ["Percentile", "Absolute"], index=0, horizontal=True, key="c_thr_mode")
        pctl = st.slider("Percentile", 50.0, 99.9, 90.0, 0.1,
                         disabled=(thr_mode == "Absolute"), key="c_p")
        thr_abs = st.number_input("Absolute threshold", 0.0, 1e12, 0.0, 1.0,
                                  disabled=(thr_mode == "Percentile"), key="c_abs")

        star_size = st.slider("Star size", 6, 24, 14, key="c_star")
        st.button("Recalculate", type="primary", key="c_go")

    # --- Right column: results ---
    with rc:
        # 1) Surface tiles
        sbase = sf_all.copy()
        sbase = sbase[sbase["LONGITUDE"].between(*xr) & sbase["LATITUDE"].between(*yr)]
        if sbase.empty or "VALUE_DL" not in sbase.columns:
            st.info("No usable surface rows in current window.")
        else:
            x = sbase["LONGITUDE"].to_numpy()
            y = sbase["LATITUDE"].to_numpy()
            v = pd.to_numeric(sbase["VALUE_DL"], errors="coerce").to_numpy()

            x_edges = np.linspace(x.min(), x.max(), nx + 1)
            y_edges = np.linspace(y.min(), y.max(), ny + 1)
            ix = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1)
            iy = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1)
            flat = ix + nx * iy

            dfb = pd.DataFrame({"flat": flat, "val": v, "lon": x, "lat": y})
            g = (
                dfb.groupby("flat", as_index=False)
                   .agg(val=("val", "mean"),
                        cnt=("val", "size"),
                        lon_center=("lon", "mean"),
                        lat_center=("lat", "mean"))
            )
            g = g[g["cnt"] >= int(min_tile_cnt)]

            if g.empty:
                st.info("No surface tiles under these parameters.")
            else:
                # 2) Drill slice near Z
                z0, z1 = float(z_center - z_halfwin), float(z_center + z_halfwin)
                dbase = apply_window_3d(dh_all)
                dbase = dbase[dbase["DEPTH"].between(z0, z1)]
                if dbase.empty or "VALUE_ORIG" not in dbase.columns:
                    st.info("No drillhole rows near the Z slice.")
                else:
                    ix_d = np.clip(np.searchsorted(x_edges, dbase["LONGITUDE"].to_numpy(), side="right") - 1, 0, nx - 1)
                    iy_d = np.clip(np.searchsorted(y_edges, dbase["LATITUDE"].to_numpy(), side="right") - 1, 0, ny - 1)
                    flat_d = ix_d + nx * iy_d

                    dtmp = pd.DataFrame({
                        "flat": flat_d,
                        "LONGITUDE": dbase["LONGITUDE"].to_numpy(),
                        "LATITUDE": dbase["LATITUDE"].to_numpy(),
                        "DEPTH": dbase["DEPTH"].to_numpy(),
                        "drill_val": pd.to_numeric(dbase["VALUE_ORIG"], errors="coerce").to_numpy(),
                    })
                    merged = dtmp.merge(g[["flat", "val", "lon_center", "lat_center"]],
                                        on="flat", how="inner")

                    if merged.empty:
                        st.info("No drill points matched to surface tiles.")
                    else:
                        merged["delta"] = merged["val"] - merged["drill_val"]  # surface - drill

                        # 3) Apply direction filter
                        # --- Apply direction filter into SCORE ---
                        if direction == "Surface higher":
                            arr = merged["delta"].to_numpy()
                            sub = merged[merged["delta"] >= thr].copy()
                        else:  # Drill higher
                            arr = -merged["delta"].to_numpy()
                            sub = merged[-merged["delta"] >= thr].copy()

                        if sub.empty:
                            st.info("No pairs meet the discrepancy threshold.")
                        else:
                            sub = sub.rename(columns={"lon_center": "LONGITUDE", "lat_center": "LATITUDE"})
                            sub["SCORE"] = np.abs(sub["delta"])

                            if method.startswith("Voxel"):
                                sub_vox = voxelize_deg(sub, grid_lon, grid_lat, grid_dep)
                                clusters = cluster_voxel(sub_vox, "SCORE", min_count)
                            else:
                                sub_vox = None
                                clusters = dbscan_on_deg(sub, grid_lon, grid_lat, grid_dep, min_count)

                            total_vox, occ_vox = voxel_coverage_stats_deg(sub, sub_vox, grid_lon, grid_lat, grid_dep)
                            n_hot = 0 if clusters is None or clusters.empty else int(clusters.shape[0])
                            render_hotspot_metrics(total_vox, occ_vox, n_hot)

                            clusters = rank_and_color(clusters, top_k=3)
                            plot_star_clusters_ranked(
                                clusters,
                                f"Surface vs Drill discrepancy",
                                text_size=star_size, anchor_size=max(5, star_size // 2)
                            )

                            export_table(clusters, "insights_C_surface_vs_drill_clusters.csv")
                            st.markdown(summarize_clusters_topk(clusters, f"discrepancy {direction}", top_k=3))

# ==============================================================
# D) DL-only clustering (Union Drillhole + Surface, DBSCAN Exploration)
# ==============================================================
with tabD:
    st.caption("Experimental clustering of DL-only regions (drillhole + surface) using DBSCAN.")

    PATH_DH_DLONLY  = Path("reports/task2/difference/drillhole_points_dlonly.csv")
    PATH_SURF_DLONLY= Path("reports/task2/difference/surface_points_dlonly.csv")

    @st.cache_data
    def safe_read_csv(p: Path, with_depth: bool = False) -> pd.DataFrame:
        """Read DL-only CSV and standardize columns to VALUE_* schema."""
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        # unify to VALUE_* naming to match A/B/C
        cols = ["LONGITUDE","LATITUDE","VALUE_ORIG","VALUE_DL","DIFF","DIFF_PCT","SOURCE"]
        if with_depth:
            cols.insert(2, "DEPTH")
        # keep only existing columns; cast numerics
        keep = [c for c in cols if c in df.columns]
        df = df[keep].copy()
        for c in keep:
            if c not in ("SOURCE",):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["LONGITUDE","LATITUDE"]).copy()

    surf_dlonly = safe_read_csv(PATH_SURF_DLONLY, with_depth=False)
    dh_dlonly   = safe_read_csv(PATH_DH_DLONLY,   with_depth=True)
    df_union    = pd.concat([surf_dlonly, dh_dlonly], ignore_index=True)

    if df_union.empty:
        st.info("No DL-only data available.")
        st.stop()

    # Apply global bbox filters
    df = df_union[
        df_union["LONGITUDE"].between(*xr) &
        df_union["LATITUDE"].between(*yr)
    ].copy()
    if "DEPTH" in df.columns:
        df = df[df["DEPTH"].between(*zr)]
    if df.empty:
        st.info("No DL-only data under current filters.")
        st.stop()

    # ---------------- Controls ----------------
    c1, c2, c3, c4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with c1:
        eps = st.slider("DBSCAN eps", 0.005, 1.0, 0.10, 0.005, key="d_eps")
    with c2:
        min_samples = st.slider("Min samples", 2, 100, 10, 1, key="d_min_samples")
    with c3:
        cluster_space = st.radio("Space", ["2D (Lon-Lat)", "3D (Lon-Lat-Depth)"],
                                 index=0, horizontal=True, key="d_space")
    with c4:
        color_mode = st.radio("Color by", ["Cluster ID (discrete)", "Mean VALUE_DL (continuous)"],
                              index=0, horizontal=True, key="d_color_mode")

    # point reduction options
    r1, r2, r3 = st.columns([0.33, 0.33, 0.34])
    with r1:
        reduce_mode = st.radio("Reduce points", ["Random sample", "Voxel centers"],
                               index=1, horizontal=True, key="d_reduce_mode")
    with r2:
        max_points = st.number_input("Max points for DBSCAN",
                                     min_value=5_000, max_value=200_000, value=20_000, step=5_000,
                                     help="Applied only when 'Random sample' is selected.", key="d_max_pts")
    with r3:
        st.caption("Voxel size uses the sidebar grid (lon/lat/dep). "
                   "Voxel centers greatly shrink points before clustering.")

    # ---------------- Reduction ----------------
    df_work = df.copy()

    if reduce_mode == "Random sample":
        if len(df_work) > max_points:
            st.info(f"Sampling {max_points:,} out of {len(df_work):,} points for DBSCAN & plotting.")
            df_work = df_work.sample(max_points, random_state=42)
    else:
        # Use voxel centers to aggregate points before DBSCAN
        # Depth is optional; voxelize_deg handles missing DEPTH by using zeros.
        df_vox = voxelize_deg(df_work, grid_lon, grid_lat, grid_dep)
        # aggregate per voxel: center coords + mean VALUE_DL + count
        agg = (
            df_vox.groupby("_voxel", as_index=False)
                  .agg(LONGITUDE=("LONGITUDE", "mean"),
                       LATITUDE=("LATITUDE",  "mean"),
                       DEPTH=("DEPTH", "mean") if "DEPTH" in df_vox.columns else ("_vx","size"),
                       VALUE_DL=("VALUE_DL", "mean"),
                       COUNT=("LONGITUDE", "size"))
        ).rename(columns={"DEPTH": "DEPTH"})
        # if there is no DEPTH column at all (pure surface), create it as zeros
        if "DEPTH" not in agg.columns:
            agg["DEPTH"] = 0.0
        df_work = agg

        st.info(f"Voxel reduction: {len(df):,} → {len(df_work):,} voxel centers for DBSCAN.")

    # ---------------- DBSCAN ----------------
    # Choose feature space (2D vs 3D). If 3D, drop rows with missing depth.
    if cluster_space.startswith("2D"):
        X = df_work[["LONGITUDE", "LATITUDE"]].to_numpy()
    else:
        # ensure DEPTH exists
        if "DEPTH" not in df_work.columns:
            df_work["DEPTH"] = 0.0
        df_work = df_work.dropna(subset=["DEPTH"]).copy()
        X = df_work[["LONGITUDE", "LATITUDE", "DEPTH"]].to_numpy()

    if len(df_work) == 0:
        st.info("No points available after reduction; please widen filters or change voxel size.")
        st.stop()

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    df_work["CLUSTER_ID"] = db.labels_

    # drop noise for visualization
    df_work = df_work[df_work["CLUSTER_ID"] >= 0].copy()
    if df_work.empty:
        st.info("DBSCAN found only noise under current parameters. Try larger eps or smaller min_samples.")
        st.stop()

    # cluster-level mean VALUE_DL
    cluster_means = df_work.groupby("CLUSTER_ID")["VALUE_DL"].mean().rename("CLUSTER_MEAN")
    df_work = df_work.merge(cluster_means, on="CLUSTER_ID", how="left")

    # For discrete legend by ID, cast to string
    df_work["CLUSTER_ID_STR"] = df_work["CLUSTER_ID"].astype(str)

    # ---------------- Plot ----------------
    st.subheader("DL-only Clusters")
    # build safe hover columns depending on availability
    hover_cols = [c for c in ["VALUE_DL", "VALUE_ORIG", "DIFF", "COUNT", "DEPTH"] if c in df_work.columns]

    if cluster_space.startswith("2D"):
        fig = px.scatter(
            df_work, x="LONGITUDE", y="LATITUDE",
            # color: either discrete by ID or continuous by cluster mean
            color=("CLUSTER_ID_STR" if color_mode.startswith("Cluster ID")
                   else "CLUSTER_MEAN"),
            color_continuous_scale="Turbo",
            hover_data=hover_cols,
            opacity=0.85, height=780
        )
        fig.update_layout(xaxis_title="Longitude", yaxis_title="Latitude")
    else:
        fig = px.scatter_3d(
            df_work, x="LONGITUDE", y="LATITUDE", z="DEPTH",
            color=("CLUSTER_ID_STR" if color_mode.startswith("Cluster ID")
                   else "CLUSTER_MEAN"),
            color_continuous_scale="Turbo",
            hover_data=[c for c in hover_cols if c != "DEPTH"],
            opacity=0.85, height=820
        )
        fig.update_layout(scene=dict(zaxis=dict(autorange="reversed", title="Depth (m)")))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Summary ----------------
    st.subheader("Cluster Summary")
    # compute centers on the reduced data (matches what DBSCAN ran on)
    grp = (df_work.groupby("CLUSTER_ID")
            .agg(points=("CLUSTER_ID", "size"),
                 mean_value=("VALUE_DL", "mean"),
                 lon_center=("LONGITUDE", "mean"),
                 lat_center=("LATITUDE",  "mean"),
                 z_center=("DEPTH", "mean"))
            .reset_index()
            .sort_values("points", ascending=False))
    st.dataframe(grp, use_container_width=True, hide_index=True)

    # ---------------- Export ----------------
    buf = io.StringIO()
    df_work.to_csv(buf, index=False)
    st.download_button(
        "Download clustered CSV (reduced set used by DBSCAN)",
        data=buf.getvalue().encode("utf-8"),
        file_name="insights_D_dlonly_union_clusters.csv",
        mime="text/csv",
        type="primary",
        key="d_download"
    )

# ==============================================================
# E) Overlay: Surface + Drillhole slices
# ==============================================================
with tabE:
    st.caption("Overlay surface grids with drillhole slices, "
               "to directly compare spatial coverage (drillhole = cross, surface = circle).")

    # ----------------- Controls in one row (3 columns) -----------------
    c1, c2, c3 = st.columns([0.3, 0.3, 0.4])
    with c1:
        source = st.radio("Source", ["All","Overlap","Orig-only","DL-only"],
                          index=0, horizontal=True, key="e_source")
    with c2:
        slice_mode = st.radio("Drillhole slice orientation",
                              ["XY (map)","XZ (cross-section)","YZ (cross-section)"],
                              index=0, key="e_slice")
    with c3:
        max_points = st.number_input("Max points to render", 1_000, 300_000,
                                     80_000, 1_000, key="e_max")

    # ----------------- Helpers -----------------
    def pick_df(split: str, surf: bool) -> pd.DataFrame:
        if surf:
            return {
                "All": sf_all,
                "Overlap": sf_overlap,
                "Orig-only": read_surface_split("origonly"),
                "DL-only": sf_dlonly,
            }[split]
        else:
            return {
                "All": dh_all,
                "Overlap": dh_overlap,
                "Orig-only": read_drill_split("origonly"),
                "DL-only": dh_dlonly,
            }[split]

    symbol_map_all = {
        "orig_only_surface": "circle",
        "dl_only_surface":   "circle",
        "overlap_surface":   "circle",
        "orig_only_drillhole": "x",
        "dl_only_drillhole":   "x",
        "overlap_drillhole":   "x",
        "surface": "circle",
        "drillhole": "x",
    }
    cmap = "Turbo"

    # ----------------- Build filtered frames -----------------
    surf_df = pick_df(source, surf=True)
    dh_df   = pick_df(source, surf=False)

    surf_filt = surf_df[surf_df["LONGITUDE"].between(*xr) & surf_df["LATITUDE"].between(*yr)].copy()
    dh_filt   = dh_df[
        dh_df["LONGITUDE"].between(*xr) &
        dh_df["LATITUDE"].between(*yr) &
        dh_df["DEPTH"].between(*zr)
    ].copy()

    if len(surf_filt) > max_points:
        surf_filt = surf_filt.sample(max_points, random_state=42)
    if len(dh_filt) > max_points:
        dh_filt = dh_filt.sample(max_points, random_state=42)

    # ----------------- Rendering logic -----------------
    def make_overlay(xcol, ycol, xlabel, ylabel, yreverse=False, include_surface=True):
        if source == "All":
            # --- keep the original All logic ---
            sf_o = read_surface_split("origonly").copy()
            sf_d = sf_dlonly.copy()
            sf_v = sf_overlap.copy()
            dh_o = read_drill_split("origonly").copy()
            dh_d = dh_dlonly.copy()
            dh_v = dh_overlap.copy()

            def w2d(df): return df[df["LONGITUDE"].between(*xr) & df["LATITUDE"].between(*yr)]
            def w3d(df): return df[df["LONGITUDE"].between(*xr) & df["LATITUDE"].between(*yr) & df["DEPTH"].between(*zr)]

            sf_o, sf_d, sf_v = w2d(sf_o), w2d(sf_d), w2d(sf_v)
            dh_o, dh_d, dh_v = w3d(dh_o), w3d(dh_d), w3d(dh_v)

            def cap(df): return df.sample(min(len(df), max_points//3), random_state=42) if len(df) > 0 else df
            sf_o, sf_d, sf_v = cap(sf_o), cap(sf_d), cap(sf_v)
            dh_o, dh_d, dh_v = cap(dh_o), cap(dh_d), cap(dh_v)

            if not sf_o.empty: sf_o["LAYER"] = "orig_only_surface"
            if not sf_d.empty: sf_d["LAYER"] = "dl_only_surface"
            if not sf_v.empty: sf_v["LAYER"] = "overlap_surface"
            if not dh_o.empty: dh_o["LAYER"] = "orig_only_drillhole"
            if not dh_d.empty: dh_d["LAYER"] = "dl_only_drillhole"
            if not dh_v.empty: dh_v["LAYER"] = "overlap_drillhole"

            if ycol == "DEPTH":
                df_plot = pd.concat([dh_o, dh_d, dh_v], ignore_index=True)
            else:
                df_plot = pd.concat([dh_o, dh_d, dh_v, sf_o, sf_d, sf_v], ignore_index=True)

            if df_plot.empty:
                st.info("No data available under current filters."); return

            hov = [c for c in ["DEPTH","VALUE_ORIG","VALUE_DL","DIFF","DIFF_PCT"] if c in df_plot.columns]
            fig = px.scatter(
                df_plot, x=xcol, y=ycol,
                color="LAYER", symbol="LAYER", symbol_map=symbol_map_all,
                opacity=0.85, hover_data=hov
            )
        else:
            # --- create local copies instead of overwriting surf_filt/dh_filt ---
            surf_local = surf_filt.copy() if not surf_filt.empty else pd.DataFrame()
            dh_local   = dh_filt.copy() if not dh_filt.empty else pd.DataFrame()

            if not surf_local.empty:
                surf_local["LAYER"] = "surface"
            if not dh_local.empty:
                dh_local["LAYER"] = "drillhole"

            value_map = {"Overlap": "DIFF", "Orig-only": "VALUE_ORIG", "DL-only": "VALUE_DL"}
            value_col = value_map.get(source, None)

            # fallback check
            if value_col not in dh_local.columns and value_col not in surf_local.columns:
                fallback_cols = [c for c in ["DIFF","VALUE_DL","VALUE_ORIG"] if c in dh_local.columns or c in surf_local.columns]
                value_col = fallback_cols[0] if fallback_cols else None

            hover_dh = [c for c in ["DEPTH","VALUE_ORIG","VALUE_DL","DIFF","DIFF_PCT"] if c in dh_local.columns]
            hover_sf = [c for c in ["VALUE_ORIG","VALUE_DL","DIFF","DIFF_PCT"] if c in surf_local.columns]

            fig = px.scatter(
                dh_local, x=xcol, y=ycol,
                color=value_col if value_col else None,
                color_continuous_scale=cmap if value_col else None,
                symbol="LAYER", symbol_map=symbol_map_all,
                opacity=0.9, hover_data=hover_dh
            )
            if include_surface and not surf_local.empty and ycol != "DEPTH":
                fig2 = px.scatter(
                    surf_local, x=xcol, y=ycol,
                    color=value_col if value_col else None,
                    color_continuous_scale=cmap if value_col else None,
                    symbol="LAYER", symbol_map=symbol_map_all,
                    opacity=0.6, hover_data=hover_sf
                )
                for tr in fig2.data: fig.add_trace(tr)

            if value_col:
                fig.update_layout(coloraxis_colorbar=dict(title=value_col))

        if yreverse: fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=800, margin=dict(l=0, r=0, t=50, b=10),
                        xaxis_title=xlabel, yaxis_title=ylabel)
        st.plotly_chart(fig, use_container_width=True)
    # ----------------- Slices -----------------
    if slice_mode == "XY (map)":
        st.subheader(f"Overlay XY Map slice (Depth {zr[0]}–{zr[1]} m)")
        make_overlay("LONGITUDE","LATITUDE","Longitude","Latitude",yreverse=False, include_surface=True)
    elif slice_mode == "XZ (cross-section)":
        st.subheader(f"Overlay XZ Cross-section (Latitude {yr[0]}–{yr[1]})")
        make_overlay("LONGITUDE","DEPTH","Longitude","Depth (m)",yreverse=True, include_surface=False)
    else:
        st.subheader(f"Overlay YZ Cross-section (Longitude {xr[0]}–{xr[1]})")
        make_overlay("LATITUDE","DEPTH","Latitude","Depth (m)",yreverse=True, include_surface=False)

# ---------------------------- Footer note ----------------------------
st.markdown(
    """
    <div style="color:#6b7280; font-size:0.95rem; padding-top:0.5rem;">
    Notes:
    • Stars (★) mark cluster centers; raw points are hidden to avoid duplication with 05/06/07/08.<br>
    • Voxel mode: each occupied voxel (lon/lat in degrees, depth in meters) = one hotspot.<br>
    • DBSCAN uses normalized lon/lat/dep by the voxel sizes (eps≈1 ≈ within ~one voxel).<br>
    </div>
    """,
    unsafe_allow_html=True,
)

