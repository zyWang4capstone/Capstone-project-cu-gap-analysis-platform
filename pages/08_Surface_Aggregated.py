# 08_Surface_Aggregated.py — Surface 2D Viewer (All / Overlap / Orig-only / DL-only)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from cap_common.config import load_cfg
from cap_task2.io import read_points

from _ui_common import inject_theme
inject_theme()

st.set_page_config(layout="wide", page_title="Surface 2D Viewer — Differences")

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
        color: #fff !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        border-color: #111827 !important;
        transform: translateY(1px);
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
    .stTabs + div [data-testid="stVerticalBlock"] > div:first-child {
        border-top: 1px solid #11182710; margin-top: -1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Paths (surface outputs) -----------------
DIFF_DIR = Path("reports/task2/difference")
PATH_ALL       = DIFF_DIR / "surface_points_all.csv"
PATH_OVERLAP   = DIFF_DIR / "surface_points_overlap.csv"
PATH_ORIGONLY  = DIFF_DIR / "surface_points_origonly.csv"
PATH_DLONLY    = DIFF_DIR / "surface_points_dlonly.csv"

CAT_ORDER = ["orig_only", "overlap", "dl_only"]
CAT_COLORS = {
    "all": "#aaaaaa",
    "orig_only": "#447dd2",
    "overlap":   "#9ecbff",
    "dl_only":   "#ff554b",
}

# ----------------- Palettes / helpers -----------------
HOT_CS = [
    [0.00, "#000000"], [0.20, "#2b0000"], [0.40, "#7a0000"],
    [0.70, "#ff3b00"], [1.00, "#ffff66"],
]

def tile_points_2d(df: pd.DataFrame, nx: int, ny: int, value_col: str,
                   agg: str = "mean", min_count: int = 1):
    if df.empty or value_col not in df.columns:
        return None
    x = df["LONGITUDE"].to_numpy()
    y = df["LATITUDE"].to_numpy()
    v = pd.to_numeric(df[value_col], errors="coerce").to_numpy()

    x_edges = np.linspace(x.min(), x.max(), nx + 1)
    y_edges = np.linspace(y.min(), y.max(), ny + 1)

    ix = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1)

    flat = ix + nx * iy
    dfb = pd.DataFrame({"flat": flat, "val": v})

    if agg == "median":
        g = dfb.groupby("flat").agg(val=("val", "median"), cnt=("val", "size")).reset_index()
    else:
        g = dfb.groupby("flat").agg(val=("val", "mean"), cnt=("val", "size")).reset_index()

    g = g[g["cnt"] >= int(min_count)]
    if g.empty:
        return None

    Z = np.full((ny, nx), np.nan, dtype=float)
    gx = (g["flat"] % nx).to_numpy()
    gy = (g["flat"] // nx).to_numpy()
    Z[gy, gx] = g["val"].to_numpy()

    Xc = (x_edges[:-1] + x_edges[1:]) / 2.0
    Yc = (y_edges[:-1] + y_edges[1:]) / 2.0
    return {"Xc": Xc, "Yc": Yc, "Z": Z, "x_edges": x_edges, "y_edges": y_edges}

# ----------------- Load data -----------------
cfg = load_cfg()

def _read_and_fix_surface(split: str) -> pd.DataFrame:
    df = read_points("surface", split, cfg)

    df = df.rename(columns=lambda c: str(c).upper())
    df = df.rename(columns={"DLAT": "LATITUDE", "DLONG": "LONGITUDE"})

    need_tag = {
        "overlap":  "overlap",
        "origonly": "orig_only",
        "dlonly":   "dl_only",
        "all":      "all",
    }.get(split, "unknown")

    if (
        "SOURCE" not in df.columns
        or df["SOURCE"].isna().all()
        or (df["SOURCE"].astype(str).str.strip() == "").all()
    ):
        df["SOURCE"] = need_tag

    for c in ["LONGITUDE", "LATITUDE", "VALUE_ORIG", "VALUE_DL", "DIFF", "DIFF_PCT"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "DIFF" not in df.columns and {"VALUE_ORIG", "VALUE_DL"}.issubset(df.columns):
        df["DIFF"] = df["VALUE_DL"] - df["VALUE_ORIG"]

    if "DIFF_PCT" not in df.columns and {"DIFF", "VALUE_ORIG"}.issubset(df.columns):
        eps = 1e-9
        denom = df["VALUE_ORIG"].where(df["VALUE_ORIG"].abs() >= eps, np.nan)
        df["DIFF_PCT"] = 100.0 * df["DIFF"] / denom

    return df.dropna(subset=["LONGITUDE", "LATITUDE"]).copy()


# ----------------- Load all splits -----------------
df_all      = _read_and_fix_surface("all")
df_overlap  = _read_and_fix_surface("overlap")
df_origonly = _read_and_fix_surface("origonly")
df_dlonly   = _read_and_fix_surface("dlonly")

df_union = pd.concat([df_all, df_overlap, df_origonly, df_dlonly], ignore_index=True)

if df_union.empty:
    st.warning("No surface data found in any of the splits.")
    st.stop()

# ---------------- Header ----------------
element = st.session_state.get("element", "Element")

h1, h2, h3 = st.columns([0.6, 0.2, 0.2])
with h1:
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.25rem; font-size:2.0rem;">
        Medium/Coarse-level • Surface ({element})
        </h1>
        <p style="color:#555; margin-top:0;">
        2D Points / 2D Grid (map optional) for {element} values
        </p>
        """,
        unsafe_allow_html=True
    )
with h2:
    st.page_link("pages/04_Diff_Home.py", label="Back to Diff • Home")
with h3:
    st.page_link("pages/09_Insights.py", label="View insights")

# ───────────────────────── Sidebar controls ──────────────────────
# 1) Data source
source = st.sidebar.radio(
    "Data source",
    ["All", "Overlap", "Orig-only", "DL-only"],
    index=0, horizontal=True, key="src_mode"
)

def pick_source_df():
    if source == "All":
        # Try to build union with SOURCE tags for clarity
        parts = []
        for tag, d in [("orig_only", df_origonly), ("overlap", df_overlap), ("dl_only", df_dlonly)]:
            if not d.empty:
                dd = d.copy()
                if "SOURCE" not in dd.columns or dd["SOURCE"].isna().all() or (dd["SOURCE"].astype(str).str.strip() == "").all():
                    dd["SOURCE"] = tag
                parts.append(dd)
        if parts:
            return pd.concat(parts, ignore_index=True)
        return df_all
    elif source == "Overlap":
        return df_overlap
    elif source == "Orig-only":
        return df_origonly
    elif source == "DL-only":
        return df_dlonly
    return df_all

# 2) Value
category_mode = (source == "All")

if not category_mode:
    element = st.session_state.get("element", "Element")

    VAL_CHOICES = {
        "Overlap": [
            ("DIFF (DL - ORIG)", "DIFF"),
            ("DIFF % ((DL-ORIG)/DL*100)", "DIFF_PCT"),
            (f"{element} ORIG", "VALUE_ORIG"),
            (f"{element} DL", "VALUE_DL"),
        ],
        "Orig-only": [(f"{element} ORIG", "VALUE_ORIG")],
        "DL-only":   [(f"{element} DL", "VALUE_DL")],
    }

    opts = VAL_CHOICES[source]
    labels = [lab for lab, _ in opts]
    lab2col = dict(opts)

    prev = st.session_state.get("value_mode")
    if prev not in labels:
        prev = labels[0]

    value_mode = st.sidebar.selectbox(
        "Value to display", labels,
        index=labels.index(prev), key="value_mode"
    )
    value_col = lab2col[value_mode]

else:
    value_mode = "Source categories (color)"
    value_col = "__SOURCE__"

# 3) Filters（lon/lat）
if df_union.empty:
    st.warning("No data found. Check CSV paths.")
    st.stop()

lon_min, lon_max = float(df_union["LONGITUDE"].min()), float(df_union["LONGITUDE"].max())
lat_min, lat_max = float(df_union["LATITUDE"].min()),  float(df_union["LATITUDE"].max())

xr = st.sidebar.slider("Longitude", lon_min, lon_max, (lon_min, lon_max), step=(lon_max-lon_min)/100)
yr = st.sidebar.slider("Latitude",  lat_min, lat_max,  (lat_min, lat_max), step=(lat_max-lat_min)/100)

# 4) Basemap
st.sidebar.subheader("Basemap")
use_basemap = st.sidebar.checkbox("Use map background (OSM)", value=False)
if not category_mode:  # only relevant when showing numeric values
    map_mode    = st.sidebar.selectbox("Map layer for numeric view", ["Heat (density)", "Squares"], index=0)
    map_zoom    = st.sidebar.slider("Map zoom (visual only)", 2, 14, 7)  
    heat_radius = st.sidebar.slider("Heat radius (px)", 2, 30, 8)
else:
    map_mode, map_zoom, heat_radius = None, 7, 8

# 5) Clip mode for numeric view
if not category_mode:
    st.sidebar.subheader("Display (numeric)")
    clip_mode = st.sidebar.selectbox("Clip mode", ["Absolute", "Percentile"], index=1)
    if clip_mode == "Absolute":
        max_abs_clip = st.sidebar.number_input("Max value to show", min_value=1.0, value=500.0, step=10.0)
        pctl = None
    else:
        pctl = st.sidebar.slider("Clip range at percentile", min_value=50.0, max_value=99.9, value=95.0, step=0.1)
        max_abs_clip = None
else:
    clip_mode, max_abs_clip, pctl = None, None, None

# 6) Min abs diff filter
if not category_mode and value_col in ("DIFF","DIFF_PCT"):
    min_abs_diff = st.sidebar.number_input("Min |value| filter", 0.0, value=0.0, step=1.0)
else:
    min_abs_diff = 0.0

# ----------------- Helpers -----------------
def fix_extent_axes(fig, xr, yr):
    fig.update_xaxes(range=[float(xr[0]), float(xr[1])])
    fig.update_yaxes(range=[float(yr[0]), float(yr[1])])

def fix_extent_mapbox(fig, xr, yr, style="open-street-map"):
    west, east = float(xr[0]), float(xr[1])
    south, north = float(yr[0]), float(yr[1])
    fig.add_trace(go.Scattermapbox(
        lat=[south, south, north, north],
        lon=[west,  east,  west,  east ],
        mode="markers",
        marker=dict(size=1, opacity=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.update_layout(mapbox_style=style, margin=dict(l=0, r=0, t=50, b=10), height=800)

# ----------------- Filtered base -----------------
base = pick_source_df().copy()
if base.empty:
    st.warning(f"No rows in selected source: {source}")
    st.stop()

# Apply lon/lat filter
base = base[base["LONGITUDE"].between(*xr) & base["LATITUDE"].between(*yr)].copy()

# Apply min abs diff filter
if (not category_mode) and value_col in ("DIFF", "DIFF_PCT") and min_abs_diff > 0:
    base = base[base[value_col].abs() >= min_abs_diff]

# Safety check
if base.empty or (value_col != "__SOURCE__" and value_col not in base.columns):
    st.warning("No data under current filters/value selection.")
    st.stop()

# ----------------- Color helpers -----------------
def diverging_color_and_range(series: np.ndarray, is_abs_clip: bool, abs_max: float, p: float):
    """Return diverging colorscale + (vmin, vmax) for symmetric data like DIFF."""
    if is_abs_clip:
        vmax = float(abs_max)
    else:
        vmax = float(np.nanpercentile(np.abs(series), p)) if series.size else 1.0
        if vmax <= 0: vmax = 1.0
    return ("Portland", (-vmax, vmax))

def sequential_color_and_range(series: np.ndarray, is_abs_clip: bool, abs_max: float, p: float):
    """Return sequential colorscale + (0, vmax) for positive-only data like CU values."""
    if is_abs_clip:
        vmax = float(abs_max)
    else:
        vmax = float(np.nanpercentile(series, p)) if series.size else 1.0
        if vmax <= 0: vmax = 1.0
    return ("Portland", (0.0, vmax))

# ----------------- Tabs -----------------
tab_points, tab_grid = st.tabs(["2D Points", "2D Grid"])

with tab_points:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        point_size = st.slider("Point size", 1, 12, 5, key="surf_pts_size")
    with c2:
        limit_points = st.toggle("Limit points", value=False, key="surf_pts_limit")
    with c3:
        max_points = st.number_input("Max points", 1_000, 1_000_000, 100_000, 1_000, key="surf_pts_cap")

    view = base.copy()
    if limit_points and len(view) > max_points:
        view = view.sample(max_points, random_state=42)

    # --- Category mode (All → SOURCE colors) ---
    if category_mode:
        view["SOURCE"] = pd.Categorical(view["SOURCE"], categories=CAT_ORDER, ordered=True)

        if use_basemap:
            fig = px.scatter_mapbox(
                view, lat="LATITUDE", lon="LONGITUDE",
                color="SOURCE", color_discrete_map=CAT_COLORS, opacity=0.9
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(legend=dict(title="SOURCE", font=dict(size=18)))
            fix_extent_mapbox(fig, xr, yr)
        else:
            fig = px.scatter(
                view, x="LONGITUDE", y="LATITUDE",
                color="SOURCE", color_discrete_map=CAT_COLORS, opacity=0.9
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(
                xaxis_title="Longitude", yaxis_title="Latitude",
                legend=dict(title="SOURCE", font=dict(size=18)),
                height=800, margin=dict(l=0, r=0, t=50, b=10),
            )
            fix_extent_axes(fig, xr, yr)

        st.plotly_chart(fig, use_container_width=True)

    # --- Numeric mode (value_col → continuous colors) ---
    else:
        vals = pd.to_numeric(view[value_col], errors="coerce")

        if value_col in ("DIFF", "DIFF_PCT"):
            cs, (vmin, vmax) = diverging_color_and_range(
                vals.to_numpy(), clip_mode == "Absolute", max_abs_clip, pctl
            )
            ticksuf = "%" if value_col == "DIFF_PCT" else ""
        else:
            cs, (vmin, vmax) = sequential_color_and_range(
                vals.to_numpy(), clip_mode == "Absolute", max_abs_clip, pctl
            )
            ticksuf = ""

        if use_basemap:
            fig = px.scatter_mapbox(
                view, lat="LATITUDE", lon="LONGITUDE",
                color=value_col, color_continuous_scale=cs,
                range_color=[vmin, vmax], opacity=0.9
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(coloraxis_colorbar=dict(title=value_mode, ticksuffix=ticksuf))
            fix_extent_mapbox(fig, xr, yr)
        else:
            fig = px.scatter(
                view, x="LONGITUDE", y="LATITUDE",
                color=value_col, color_continuous_scale=cs,
                range_color=[vmin, vmax], opacity=0.9
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(
                xaxis_title="Longitude", yaxis_title="Latitude",
                coloraxis_colorbar=dict(title=value_mode, ticksuffix=ticksuf),
                height=800, margin=dict(l=0, r=0, t=50, b=10),
            )
            fix_extent_axes(fig, xr, yr)

        st.plotly_chart(fig, use_container_width=True)

with tab_grid:
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        nx = st.slider("NX (lon bins)", 20, 400, 120, 10, key="surf_grid_nx")
    with gc2:
        ny = st.slider("NY (lat bins)", 20, 400, 120, 10, key="surf_grid_ny")
    with gc3:
        min_count_tile = st.number_input("Min samples/tile", 1, 500, 3, 1, key="surf_grid_min")

    gc4, gc5 = st.columns([1, 1])
    with gc4:
        agg_stat = st.selectbox("Aggregator", ["mean", "median"], index=0, key="surf_grid_agg")
    with gc5:
        tile_opacity = st.slider("Tile opacity", 0.1, 1.0, 0.85, 0.05, key="surf_grid_opacity")

    if category_mode:
        st.info("In 'All' mode, grid is best viewed as points by SOURCE. Switch to Overlap/Orig-only/DL-only for numeric grid.")
    else:
        num = base.dropna(subset=["LONGITUDE", "LATITUDE", value_col]).copy()
        if num.empty:
            st.info("No numeric rows for tiles under current filters/value.")
        else:
            tiles = tile_points_2d(num, nx=nx, ny=ny, value_col=value_col,
                                   agg=agg_stat, min_count=min_count_tile)
            if tiles is None:
                st.info("No tiles under current parameters. Try smaller NX/NY or lower Min samples.")
            else:
                Xc, Yc, Z = tiles["Xc"], tiles["Yc"], tiles["Z"]
                vals = Z[np.isfinite(Z)]

                if value_col in ("DIFF", "DIFF_PCT"):
                    cs, (vmin, vmax) = diverging_color_and_range(
                        vals, clip_mode == "Absolute", max_abs_clip, pctl
                    )
                    ticksuf = "%" if value_col == "DIFF_PCT" else ""
                else:
                    cs, (vmin, vmax) = sequential_color_and_range(
                        vals, clip_mode == "Absolute", max_abs_clip, pctl
                    )
                    ticksuf = ""

                if use_basemap:
                    yy, xx = np.meshgrid(Yc, Xc, indexing="ij")
                    centers = pd.DataFrame({
                        "LATITUDE": yy.ravel(),
                        "LONGITUDE": xx.ravel(),
                        "VAL": Z.ravel()
                    }).dropna()

                    if map_mode == "Heat (density)":
                        fig = px.density_mapbox(
                            centers, lat="LATITUDE", lon="LONGITUDE", z="VAL",
                            radius=heat_radius, color_continuous_scale=cs
                        )
                        fig.update_layout(coloraxis=dict(
                            cmin=vmin, cmax=vmax,
                            colorbar=dict(title=value_mode, ticksuffix=ticksuf)
                        ))
                    else:
                        fig = px.scatter_mapbox(
                            centers, lat="LATITUDE", lon="LONGITUDE",
                            color="VAL", color_continuous_scale=cs, range_color=[vmin, vmax],
                            opacity=tile_opacity
                        )
                        fig.update_traces(marker=dict(size=9, symbol="square"))
                        fig.update_layout(coloraxis_colorbar=dict(
                            title=value_mode, ticksuffix=ticksuf
                        ))

                    fix_extent_mapbox(fig, xr, yr)

                else:
                    fig = go.Figure(data=go.Heatmap(
                        x=Xc, y=Yc, z=np.flipud(Z),
                        colorscale=cs, zmin=vmin, zmax=vmax,
                        colorbar=dict(title=value_mode, ticksuffix=ticksuf),
                        showscale=True, opacity=tile_opacity
                    ))
                    fig.update_layout(
                        xaxis_title="Longitude", yaxis_title="Latitude",
                        height=800, margin=dict(l=0, r=0, t=50, b=10)
                    )
                    fig.update_yaxes(autorange="reversed")
                    fix_extent_axes(fig, xr, yr)

                st.plotly_chart(fig, use_container_width=True)