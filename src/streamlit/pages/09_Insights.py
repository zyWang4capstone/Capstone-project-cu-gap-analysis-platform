# pages/09_Insights.py — Unified Insights (Drillhole + Surface)
# Rules:
#   A) DL-only high-value hotspots (drillhole)
#   B) Overlap uplift hotspots (DL - ORIG) on drillhole
#   C) Surface vs Drill discrepancy at a Z slice
#
# Note:
#   We cluster in degrees for lon/lat (no meter conversion), and meters for depth.
#   Voxel mode: each occupied voxel is a hotspot (no adjacency merge).
#   DBSCAN mode: clusters are discovered in normalized (lon/lat/dep) using voxel sizes.

from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from cap_common.config import load_cfg
from cap_task2.io import read_points

from _ui_common import inject_theme

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

# ---------------------------- Paths (consistent with 06/08) ----------------------------
DIFF_DIR = Path("reports/task2/difference")

# Drillhole points (3D)
DH_ALL       = DIFF_DIR / "drillhole_points_all.csv"
DH_OVERLAP   = DIFF_DIR / "drillhole_points_overlap.csv"
DH_ORIGONLY  = DIFF_DIR / "drillhole_points_origonly.csv"
DH_DLONLY    = DIFF_DIR / "drillhole_points_dlonly.csv"

# Surface points (2D)
SF_ALL       = DIFF_DIR / "surface_points_all.csv"
SF_OVERLAP   = DIFF_DIR / "surface_points_overlap.csv"
SF_ORIGONLY  = DIFF_DIR / "surface_points_origonly.csv"
SF_DLONLY    = DIFF_DIR / "surface_points_dlonly.csv"

# ---------------------------- I/O helpers ----------------------------
@st.cache_data
cfg = load_cfg()

@st.cache_data
def read_drill_split(split: str) -> pd.DataFrame:
    df = read_points("drillhole", split, cfg)
    df = df.loc[:, ~df.columns.duplicated()]
    for c in ["LONGITUDE","LATITUDE","DEPTH","CU_ORIG","CU_DL","DIFF","DIFF_PCT"]:
        if c in df.columns and c != "SOURCE":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["LONGITUDE","LATITUDE"])

@st.cache_data
def read_surface_split(split: str) -> pd.DataFrame:
    df = read_points("surface", split, cfg)
    df = df.loc[:, ~df.columns.duplicated()]
    if "DLAT" in df.columns:  df = df.rename(columns={"DLAT":"LATITUDE"})
    if "DLONG" in df.columns: df = df.rename(columns={"DLONG":"LONGITUDE"})
    for c in ["LONGITUDE","LATITUDE","CU_ORIG","CU_DL","DIFF","DIFF_PCT"]:
        if c in df.columns and c != "SOURCE":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["LONGITUDE","LATITUDE"]).copy()

# Load once
dh_overlap  = read_drill_split("overlap")
dh_dlonly   = read_drill_split("dlonly")
dh_all      = read_drill_split("all")
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
        Cluster-level findings only — stars (★), summary tables, and downloads.
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
    [df for df in [dh_all, dh_overlap, dh_dlonly] if not df.empty],
    ignore_index=True
)
if union_for_ranges.empty:
    st.warning("No drillhole data found in `reports/task2/difference/`. Check paths.")
    st.stop()

lon_min, lon_max = float(union_for_ranges["LONGITUDE"].min()), float(union_for_ranges["LONGITUDE"].max())
lat_min, lat_max = float(union_for_ranges["LATITUDE"].min()),  float(union_for_ranges["LATITUDE"].max())
dep_min, dep_max = float(union_for_ranges["DEPTH"].min()),     float(union_for_ranges["DEPTH"].max()) if "DEPTH" in union_for_ranges.columns else (0.0, 2500.0)

# Window sliders (same style as other pages)
xr = st.sidebar.slider("Longitude", lon_min, lon_max, (lon_min, lon_max))
yr = st.sidebar.slider("Latitude",  lat_min, lat_max, (lat_min, lat_max))
zr = st.sidebar.slider("Depth",     max(0.0, dep_min), min(2500.0, dep_max),
                       (max(0.0, dep_min), min(2500.0, dep_max)))

st.sidebar.markdown("---")
st.sidebar.subheader("Clustering defaults (degrees for lon/lat; meters for depth)")
# Default 0.001° ≈ 100 m; we don't convert by latitude here.
grid_lon = st.sidebar.number_input("Voxel size • lon (°)", 0.0001, 1.0, 0.01, 0.01, format="%.4f")
grid_lat = st.sidebar.number_input("Voxel size • lat (°)", 0.0001, 1.0, 0.01, 0.01, format="%.4f")
grid_dep = st.sidebar.number_input("Voxel size • depth (m)", 0.1, 1000.0, 5.0, 1.0)

# ---------------------------- Shared utilities ----------------------------
def apply_window_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Filter by lon/lat/depth (3D)."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df[
        df["LONGITUDE"].between(*xr) &
        df["LATITUDE"].between(*yr) &
        df["DEPTH"].between(*zr)
    ].copy()
    return out

def as_1d_numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    """Return a single numeric column even if duplicated names exist."""
    if col not in df.columns:
        return np.asarray([], dtype=float)
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").to_numpy()

def voxelize_deg(df: pd.DataFrame, gx_deg: float, gy_deg: float, gz_m: float) -> pd.DataFrame:
    """
    Add integer voxel indices (_vx,_vy,_vz) in degree/degree/meter space.
    Local origin uses per-chunk min values to keep indices small.
    """
    if df.empty: 
        return df
    df = df.loc[:, ~df.columns.duplicated()].copy()

    lon = as_1d_numeric(df, "LONGITUDE")
    lat = as_1d_numeric(df, "LATITUDE")
    dep = as_1d_numeric(df, "DEPTH") if "DEPTH" in df.columns else np.zeros(len(lon))

    lon0, lat0, z0 = np.nanmin(lon), np.nanmin(lat), np.nanmin(dep)

    vx = np.floor((lon - lon0) / float(gx_deg)).astype(int)
    vy = np.floor((lat - lat0) / float(gy_deg)).astype(int)
    vz = np.floor((dep - z0)   / float(gz_m)).astype(int)

    df["_vx"] = vx
    df["_vy"] = vy
    df["_vz"] = vz
    df["_voxel"] = list(zip(vx, vy, vz))
    return df

def dbscan_on_deg(df: pd.DataFrame, gx_deg: float, gy_deg: float, gz_m: float,
                  min_count: int) -> pd.DataFrame:
    """
    DBSCAN on normalized degree/degree/meter space.
    Normalize by voxel sizes so eps≈1 means within ~one voxel.
    """
    if not SKLEARN_OK or df.empty:
        return pd.DataFrame()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    lon = as_1d_numeric(df, "LONGITUDE")
    lat = as_1d_numeric(df, "LATITUDE")
    dep = as_1d_numeric(df, "DEPTH") if "DEPTH" in df.columns else np.zeros(len(lon))

    lon0, lat0, z0 = np.nanmin(lon), np.nanmin(lat), np.nanmin(dep)
    X = np.c_[ (lon - lon0)/float(gx_deg),
               (lat - lat0)/float(gy_deg),
               (dep - z0)/float(gz_m) ]

    model = DBSCAN(eps=1.05, min_samples=int(min_count)).fit(X)
    df["_cid"] = model.labels_

    rows = []
    for cid, sub in df.groupby("_cid"):
        if cid < 0 or len(sub) < min_count:
            continue
        # weight by SCORE if present
        if "SCORE" in sub.columns:
            vals = pd.to_numeric(sub["SCORE"], errors="coerce").to_numpy()
            w = np.nan_to_num(vals, nan=0.0) + 1e-9
        else:
            vals = np.zeros(len(sub), dtype=float)
            w = np.ones(len(sub), dtype=float)

        cx = float(np.average(as_1d_numeric(sub, "LONGITUDE"), weights=w))
        cy = float(np.average(as_1d_numeric(sub, "LATITUDE"),  weights=w))
        cz = float(np.average(as_1d_numeric(sub, "DEPTH"),     weights=w))

        rows.append(dict(
            cluster_id=int(cid),
            size=len(sub),
            score_mean=float(np.nanmean(vals)) if len(vals) else 0.0,
            score_median=float(np.nanmedian(vals)) if len(vals) else 0.0,
            score_p95=float(np.nanpercentile(vals, 95)) if len(vals) else 0.0,
            lon_center=cx, lat_center=cy, z_center=cz
        ))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["size","score_mean"], ascending=[False, False]).reset_index(drop=True)
        out["cluster_id"] = out.index + 1
    return out

def cluster_voxel(df: pd.DataFrame, score_col: str, min_count: int) -> pd.DataFrame:
    """
    Simple voxel clustering: each occupied voxel is one hotspot.
    """
    if df.empty: return pd.DataFrame()
    rows = []
    for vox, sub in df.groupby("_voxel"):
        if len(sub) < min_count:
            continue
        vals = pd.to_numeric(sub[score_col], errors="coerce").to_numpy()
        w = np.nan_to_num(vals, nan=0.0) + 1e-9
        cx = float(np.average(sub["LONGITUDE"].to_numpy(), weights=w))
        cy = float(np.average(sub["LATITUDE"].to_numpy(),  weights=w))
        cz = float(np.average(sub["DEPTH"].to_numpy(),     weights=w))
        rows.append(dict(
            cluster_id=len(rows)+1,
            size=len(sub),
            score_mean=float(np.nanmean(vals)),
            score_median=float(np.nanmedian(vals)),
            score_p95=float(np.nanpercentile(vals, 95)),
            lon_center=cx, lat_center=cy, z_center=cz
        ))
    return pd.DataFrame(rows)

# ---------- Ranking / legend helpers ----------
TOP_COLORS = ["#ef4444", "#f59e0b", "#10b981"]  # red, orange, green
OTHER_COLOR = "#9ca3af"                         # gray

def rank_and_color(df_clusters: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Rank by size desc then score_mean desc; assign Top1..TopK vs Others and colors."""
    if df_clusters is None or df_clusters.empty:
        return df_clusters
    df = df_clusters.sort_values(["size","score_mean"], ascending=[False, False]).reset_index(drop=True).copy()
    df["rank"] = np.arange(1, len(df)+1)
    df["legend"] = df["rank"].apply(lambda r: f"Top {r}" if r <= top_k else "Others")
    df["color"] = df["rank"].apply(lambda r: TOP_COLORS[r-1] if r <= top_k and r-1 < len(TOP_COLORS) else OTHER_COLOR)
    return df

def plot_star_clusters_ranked(df_clusters: pd.DataFrame, title: str,
                              text_size: int = 16, anchor_size: int = 6):
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

        if label == "Others":
            fig.add_trace(go.Scatter3d(
                x=sub["lon_center"], y=sub["lat_center"], z=sub["z_center"],
                mode="markers",
                marker=dict(symbol="diamond-open", size=anchor_size, color=color_this, opacity=0.6),
                name="Others",
                hovertemplate=(
                    "Center: (%{x:.5f}, %{y:.5f}, %{z:.1f})<br>"
                    "Size: %{customdata[0]}<br>"
                    "Mean: %{customdata[1]:.3f} · Median: %{customdata[2]:.3f} · P95: %{customdata[3]:.3f}"
                    "<extra></extra>"
                ),
                customdata=np.c_[
                    sub["size"].to_numpy(),
                    sub["score_mean"].to_numpy(),
                    sub["score_median"].to_numpy(),
                    sub["score_p95"].to_numpy(),
                ],
                showlegend=True,
            ))
        else:
            labels = [f"★{cid}" for cid in sub["cluster_id"]]
            fig.add_trace(go.Scatter3d(
                x=sub["lon_center"], y=sub["lat_center"], z=sub["z_center"],
                mode="markers+text",
                marker=dict(symbol="diamond-open", size=anchor_size, color=color_this, opacity=0.95),
                text=labels,
                textposition="top center",
                textfont=dict(size=text_size, color=color_this),
                name=label,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Center: (%{x:.5f}, %{y:.5f}, %{z:.1f})<br>"
                    "Size: %{customdata[0]}<br>"
                    "Mean: %{customdata[1]:.3f} · Median: %{customdata[2]:.3f} · P95: %{customdata[3]:.3f}"
                    "<extra></extra>"
                ),
                customdata=np.c_[
                    sub["size"].to_numpy(),
                    sub["score_mean"].to_numpy(),
                    sub["score_median"].to_numpy(),
                    sub["score_p95"].to_numpy(),
                ],
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
        title=title,
        legend=dict(title="Hotspot rank")
    )
    st.plotly_chart(fig, use_container_width=True)

def export_table(df: pd.DataFrame, filename: str):
    """Render a table and a CSV download button."""
    if df is None or df.empty:
        return
    st.dataframe(df, use_container_width=True, height=420, hide_index=True)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download CSV", data=buf.getvalue().encode("utf-8"),
                       file_name=filename, mime="text/csv", type="primary")

def percentile_threshold(arr: np.ndarray, p: float) -> float:
    return float(np.nanpercentile(arr, p)) if len(arr) else float("nan")

def summarize_clusters_topk(df: pd.DataFrame, metric_name: str, top_k: int = 3) -> str:
    """Plain-language summary focusing on Top-K hotspots."""
    if df is None or df.empty:
        return "No hotspots were found under the current filters."
    if "rank" not in df.columns:
        df = rank_and_color(df, top_k=top_k)
    n = int(df["cluster_id"].nunique())
    top = df[df["legend"].str.startswith("Top")].sort_values("rank").head(top_k)
    lines = [f"Found **{n}** hotspots in total."]
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
    lines.append(f"Overall average {metric_name}: **{mean_val:.3f}** (max P95 **{max_p95:.3f}**).")
    return "\n".join(lines)

def voxel_coverage_stats_deg(base_df: pd.DataFrame, sub_df_vox: pd.DataFrame,
                             gx_deg: float, gy_deg: float, gz_m: float) -> tuple[int,int]:
    """
    Count estimated total voxels within the current bbox (in degrees/meters),
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
    """Three small metric cards."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Total voxels (in window)", f"{total_vox:,}")
    c2.metric("Occupied voxels (selected)", f"{occ_vox:,}")
    c3.metric("Hotspots (clusters)", f"{n_hot:,}")

# ---------------------------- Tabs ----------------------------
tabA, tabB, tabC = st.tabs([
    "DL-only peaks (drillhole)",
    "DL uplift hotspots (overlap)",
    "Surface vs Drill discrepancy",
])

# ==============================================================
# A) DL-only high-value hotspots (score = CU_DL)
# ==============================================================
with tabA:
    st.caption("These areas only exist in the DL dataset, " \
    "with no records in the original data. We highlight new high-value regions discovered by DL, showing the additional insights it brings.")
    lc, rc = st.columns([0.36, 0.64])
    with lc:
        method = st.selectbox("Clustering", ["Voxel (fast)", "DBSCAN (sklearn)"], index=0,
                              help="DBSCAN does not need the number of clusters.")
        st.caption("Voxel: divide space into fixed cubes; DBSCAN: group nearby points without fixed cubes.")
        min_count = st.number_input("Min samples/cluster", 1, 1000, 5, 1)
        thr_mode = st.radio("Threshold mode", ["Percentile", "Absolute"], index=0, horizontal=True)
        pctl = st.slider("Percentile (on CU_DL)", 50.0, 99.9, 90.0, 0.1, disabled=(thr_mode=="Absolute"))
        thr_abs = st.number_input("Absolute threshold (CU_DL)", 0.0, 1e12, 0.0, 1.0, disabled=(thr_mode=="Percentile"))
        star_size = st.slider("Star size", 6, 24, 14)
        st.button("Recalculate", type="primary")
    with rc:
        base = apply_window_3d(dh_dlonly)
        if base.empty or "CU_DL" not in base.columns:
            st.info("No DL-only drillhole rows in current window.")
        else:
            base["SCORE"] = pd.to_numeric(base["CU_DL"], errors="coerce")
            thr = percentile_threshold(base["SCORE"].to_numpy(), pctl) if thr_mode=="Percentile" else float(thr_abs)
            sub = base[base["SCORE"] >= thr].copy()
            if sub.empty:
                st.info("No points meet the threshold.")
            else:
                if method.startswith("Voxel"):
                    sub_vox = voxelize_deg(sub, grid_lon, grid_lat, grid_dep)
                    clusters = cluster_voxel(sub_vox, "SCORE", min_count)
                else:
                    sub_vox = None
                    clusters = dbscan_on_deg(sub, grid_lon, grid_lat, grid_dep, min_count)

                total_vox, occ_vox = voxel_coverage_stats_deg(base, sub_vox, grid_lon, grid_lat, grid_dep)
                n_hot = 0 if clusters is None or clusters.empty else int(clusters.shape[0])
                render_hotspot_metrics(total_vox, occ_vox, n_hot)

                clusters = rank_and_color(clusters, top_k=3)
                plot_star_clusters_ranked(clusters, "DL-only peaks", text_size=star_size, anchor_size=max(5, star_size//2))

                export_table(clusters, "insights_A_dl_only_clusters.csv")
                st.markdown(summarize_clusters_topk(clusters, "CU_DL", top_k=3))


# ==============================================================
# B) Overlap uplift hotspots (score = DIFF = DL - ORIG)
# ==============================================================
with tabB:
    st.caption("These areas have both DL and ORIG data, but DL values are much higher. Such uplift indicates DL captures stronger anomalies even in regions already covered by ORIG.")
    lc, rc = st.columns([0.36, 0.64])
    with lc:
        method = st.selectbox("Clustering", ["Voxel (fast)", "DBSCAN (sklearn)"], index=0, key="b_method")
        st.caption("Voxel: divide space into fixed cubes; DBSCAN: group nearby points without fixed cubes.")
        min_count = st.number_input("Min samples/cluster", 1, 1000, 5, 1, key="b_min")
        thr_mode = st.radio("Threshold mode", ["Percentile", "Absolute"], index=0, horizontal=True, key="b_thr_mode")
        pctl = st.slider("Percentile (on DIFF)", 50.0, 99.9, 90.0, 0.1, disabled=(thr_mode=="Absolute"), key="b_pctl")
        thr_abs = st.number_input("Absolute threshold (DIFF)", 0.0, 1e12, 0.0, 1.0, disabled=(thr_mode=="Percentile"), key="b_thr_abs")
        star_size = st.slider("Star size", 6, 24, 14, key="b_star")
        st.button("Recalculate ", type="primary", key="b_go")
    with rc:
        base = apply_window_3d(dh_overlap)
        if base.empty or "DIFF" not in base.columns:
            st.info("No overlap rows with DIFF in current window.")
        else:
            base["SCORE"] = pd.to_numeric(base["DIFF"], errors="coerce")
            thr = percentile_threshold(base["SCORE"].to_numpy(), pctl) if thr_mode=="Percentile" else float(thr_abs)
            sub = base[base["SCORE"] >= thr].copy()
            if sub.empty:
                st.info("No points meet the threshold.")
            else:
                if method.startswith("Voxel"):
                    sub_vox = voxelize_deg(sub, grid_lon, grid_lat, grid_dep)
                    clusters = cluster_voxel(sub_vox, "SCORE", min_count)
                else:
                    sub_vox = None
                    clusters = dbscan_on_deg(sub, grid_lon, grid_lat, grid_dep, min_count)

                total_vox, occ_vox = voxel_coverage_stats_deg(base, sub_vox, grid_lon, grid_lat, grid_dep)
                n_hot = 0 if clusters is None or clusters.empty else int(clusters.shape[0])
                render_hotspot_metrics(total_vox, occ_vox, n_hot)

                clusters = rank_and_color(clusters, top_k=3)
                plot_star_clusters_ranked(clusters, "DL uplift hotspots", text_size=star_size, anchor_size=max(5, star_size//2))

                export_table(clusters, "insights_B_uplift_clusters.csv")
                st.markdown(summarize_clusters_topk(clusters, "uplift (DL − ORIG)", top_k=3))

# ==============================================================
# C) Surface vs Drill discrepancy at a Z slice (score = |surface - drill|)
# ==============================================================
with tabC:
    st.caption("At a chosen depth, we compare surface predictions against drillhole measurements. Significant differences reveal where surface models and drill samples do not align.")
    lc, rc = st.columns([0.36, 0.64])
    with lc:
        z_center = st.slider("Target Z (depth)", float(zr[0]), float(zr[1]), 100.0, 1.0)
        z_halfwin = st.number_input("±Z window (drill)", 0.0, 500.0, 25.0, 1.0)
        nx = st.slider("Surface NX (lon bins)", 20, 400, 120, 10)
        ny = st.slider("Surface NY (lat bins)", 20, 400, 120, 10)
        min_tile_cnt = st.number_input("Min samples/tile (surface)", 1, 500, 3, 1)
        method = st.selectbox("Clustering", ["Voxel (fast)", "DBSCAN (sklearn)"], index=0, key="c_method")
        st.caption("Voxel: divide space into fixed cubes; DBSCAN: group nearby points without fixed cubes.")
        min_count = st.number_input("Min samples/cluster", 1, 1000, 5, 1, key="c_min")
        thr_mode = st.radio("Threshold mode", ["Percentile (|delta|)", "Absolute (|delta|)"], index=0, horizontal=True, key="c_thr_mode")
        pctl = st.slider("Percentile", 50.0, 99.9, 90.0, 0.1, disabled=(thr_mode.startswith("Absolute")), key="c_p")
        thr_abs = st.number_input("Absolute threshold", 0.0, 1e12, 0.0, 1.0, disabled=(thr_mode.startswith("Percentile")), key="c_abs")
        star_size = st.slider("Star size", 6, 24, 14, key="c_star")
        st.button("Recalculate  ", type="primary", key="c_go")
    with rc:
        # 1) Build surface tiles
        sbase = sf_all.copy()
        sbase = sbase[sbase["LONGITUDE"].between(*xr) & sbase["LATITUDE"].between(*yr)]
        if sbase.empty:
            st.info("No surface rows in current window.")
        else:
            col_surf = "CU_DL" if "CU_DL" in sbase.columns else ("CU_ORIG" if "CU_ORIG" in sbase.columns else None)
            if col_surf is None:
                st.info("Surface does not contain CU columns.")
            else:
                x = sbase["LONGITUDE"].to_numpy()
                y = sbase["LATITUDE"].to_numpy()
                v = pd.to_numeric(sbase[col_surf], errors="coerce").to_numpy()

                x_edges = np.linspace(x.min(), x.max(), nx + 1)
                y_edges = np.linspace(y.min(), y.max(), ny + 1)
                ix = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1)
                iy = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1)
                flat = ix + nx * iy

                dfb = pd.DataFrame({"flat": flat, "val": v, "lon": x, "lat": y})
                g = (
                    dfb.groupby("flat", as_index=False)
                    .agg(val=("val","mean"),
                         cnt=("val","size"),
                         lon_center=("lon","mean"),
                         lat_center=("lat","mean"))
                )
                g = g[g["cnt"] >= int(min_tile_cnt)]
                if g.empty:
                    st.info("No surface tiles under these parameters.")
                else:
                    # 2) Drill slice near target Z
                    z0, z1 = float(z_center - z_halfwin), float(z_center + z_halfwin)
                    dbase = apply_window_3d(dh_all)
                    dbase = dbase[dbase["DEPTH"].between(z0, z1)]
                    col_drill = "CU_DL" if "CU_DL" in dbase.columns else ("CU_ORIG" if "CU_ORIG" in dbase.columns else None)
                    if dbase.empty or col_drill is None:
                        st.info("No drillhole rows near the Z slice.")
                    else:
                        ix_d = np.clip(np.searchsorted(x_edges, dbase["LONGITUDE"].to_numpy(), side="right") - 1, 0, nx - 1)
                        iy_d = np.clip(np.searchsorted(y_edges, dbase["LATITUDE"].to_numpy(),  side="right") - 1, 0, ny - 1)
                        flat_d = ix_d + nx * iy_d

                        dtmp = pd.DataFrame({
                            "flat": flat_d,
                            "LONGITUDE": dbase["LONGITUDE"].to_numpy(),
                            "LATITUDE":  dbase["LATITUDE"].to_numpy(),
                            "DEPTH":     dbase["DEPTH"].to_numpy(),
                            "drill_val": pd.to_numeric(dbase[col_drill], errors="coerce").to_numpy(),
                        })
                        merged = dtmp.merge(g[["flat","val","lon_center","lat_center"]], on="flat", how="inner")
                        if merged.empty:
                            st.info("No drill points fell onto populated surface tiles.")
                        else:
                            merged["delta"] = merged["val"] - merged["drill_val"]  # surf - drill

                            # 3) Threshold on |delta|
                            arr = np.abs(merged["delta"].to_numpy())
                            thr = percentile_threshold(arr, pctl) if thr_mode.startswith("Percentile") else float(thr_abs)
                            sub = merged[np.abs(merged["delta"]) >= thr].copy()
                            if sub.empty:
                                st.info("No pairs meet the discrepancy threshold.")
                            else:
                                # Treat each selected pair as a 3D point at drill depth
                                sub = sub.rename(columns={"lon_center":"LONGITUDE","lat_center":"LATITUDE"})
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
                                    clusters, f"Surface vs Drill discrepancy @Z≈{z_center:.1f}",
                                    text_size=star_size, anchor_size=max(5, star_size//2)
                                )

                                export_table(clusters, "insights_C_surface_vs_drill_clusters.csv")
                                st.markdown(summarize_clusters_topk(clusters, "|delta| (surface − drill)", top_k=3))


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