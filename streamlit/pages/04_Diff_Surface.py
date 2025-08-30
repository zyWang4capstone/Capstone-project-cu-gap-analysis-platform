# 04_Diff_Surface.py — multi-source (All / Overlap / Orig-only / DL-only), 2D surface
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Surface 2D Viewer — Differences")

# ----------------- Paths (surface outputs) -----------------
DIFF_DIR = Path("reports/task2/difference")
PATH_ALL       = DIFF_DIR / "surface_points_all.csv"
PATH_OVERLAP   = DIFF_DIR / "surface_points_overlap.csv"
PATH_ORIGONLY  = DIFF_DIR / "surface_points_origonly.csv"
PATH_DLONLY    = DIFF_DIR / "surface_points_dlonly.csv"

# ----------------- Palettes -----------------
HOT_CS = [
    [0.00, "#000000"], [0.20, "#2b0000"], [0.40, "#7a0000"],
    [0.70, "#ff3b00"], [1.00, "#ffff66"],
]
CAT_COLORS = {"overlap": "#2BB673", "orig_only": "#2F80ED", "dl_only": "#F2994A"}

# ----------------- Data loader -----------------
@st.cache_data
def safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ["LONGITUDE", "LATITUDE", "CU_ORIG", "CU_DL", "DIFF", "DIFF_PCT"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "DIFF_PCT" not in df.columns and {"DIFF", "CU_DL"}.issubset(df.columns):
        eps = 1e-9
        denom = df["CU_DL"].where(df["CU_DL"].abs() >= eps, np.nan)
        df["DIFF_PCT"] = 100.0 * df["DIFF"] / denom
    if "SOURCE" in df.columns:
        df["SOURCE"] = df["SOURCE"].astype("category")
    return df.dropna(subset=["LONGITUDE", "LATITUDE"]).copy()

# ----------------- 2D tiling helper -----------------
def tile_points_2d(df: pd.DataFrame, nx: int, ny: int, value_col: str,
                   agg: str = "mean", min_count: int = 1):
    if df.empty:
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
        g = dfb.groupby("flat").agg(val=("val", "mean"),   cnt=("val", "size")).reset_index()

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
df_all      = safe_read_csv(PATH_ALL)
df_overlap  = safe_read_csv(PATH_OVERLAP)
df_origonly = safe_read_csv(PATH_ORIGONLY)
df_dlonly   = safe_read_csv(PATH_DLONLY)
df_union    = pd.concat([df_all, df_overlap, df_origonly, df_dlonly], ignore_index=True)

st.title("Surface 2D Viewer — Differences")

# ===== 1) Data source =====
st.sidebar.header("Data source")
source = st.sidebar.radio("Source", ["All", "Overlap", "Orig-only", "DL-only"],
                          index=0, horizontal=True, key="src_mode")

def pick_source_df():
    if source == "Overlap":   return df_overlap
    if source == "Orig-only": return df_origonly
    if source == "DL-only":   return df_dlonly
    return df_all

# ===== 2) Value =====
category_mode = (source == "All")
if not category_mode:
    st.sidebar.header("Value")
    VAL_CHOICES = {
        "Overlap":   [("DIFF (DL - ORIG)", "DIFF"),
                      ("DIFF % ((DL-ORIG)/DL*100)", "DIFF_PCT"),
                      ("CU_ORIG", "CU_ORIG"), ("CU_DL", "CU_DL")],
        "Orig-only": [("CU_ORIG", "CU_ORIG")],
        "DL-only":   [("CU_DL", "CU_DL")],
    }
    opts = VAL_CHOICES[source]
    labels = [lab for lab, _ in opts]
    lab2col = dict(opts)
    prev = st.session_state.get("value_mode")
    if prev not in labels: prev = labels[0]
    value_mode = st.sidebar.selectbox("Value to display", labels,
                                      index=labels.index(prev), key="value_mode")
    value_col = lab2col[value_mode]
else:
    value_mode = "Source categories (color)"
    value_col = "__SOURCE__"

# ===== 3) View =====
if category_mode:
    view_mode = "Points"
    point_size = st.sidebar.slider("Point size", 1, 12, 5)
    max_points = st.sidebar.number_input("Max points to render", 1_000, 1_000_000, 300_000, 1_000)
else:
    st.sidebar.header("View")
    view_mode = st.sidebar.radio("View mode", ["Points", "Tiles"], index=0, horizontal=True)
    if view_mode == "Points":
        point_size = st.sidebar.slider("Point size", 1, 12, 5)
        max_points = st.sidebar.number_input("Max points to render", 1_000, 1_000_000, 300_000, 1_000)
    else:
        nx = st.sidebar.slider("NX (lon bins)", 20, 400, 120, 10)
        ny = st.sidebar.slider("NY (lat bins)", 20, 400, 120, 10)
        min_count_tile = st.sidebar.number_input("Min samples per tile", 1, 500, 3, 1)
        agg_stat = st.sidebar.selectbox("Tile aggregator", ["mean", "median"], index=0)
        tile_opacity = st.sidebar.slider("Tile opacity", 0.1, 1.0, 0.85, 0.05)

# ===== Basemap (optional) =====
st.sidebar.header("Basemap")
use_basemap = st.sidebar.checkbox("Use map background (OSM)", value=False)
map_mode = st.sidebar.selectbox("Map layer for numeric view", ["Heat (density)", "Squares"], index=0)
# (Zoom slider kept for future; not used when bbox is fixed by corners.)
map_zoom = st.sidebar.slider("Map zoom", 2, 14, 7)
heat_radius = st.sidebar.slider("Heat radius (px)", 2, 30, 8)

# ===== Filters =====
st.sidebar.header("Filters")
if df_union.empty:
    st.warning("No data found. Check CSV paths."); st.stop()

xr = st.sidebar.slider(
    "Longitude",
    float(df_union["LONGITUDE"].min()), float(df_union["LONGITUDE"].max()),
    (float(df_union["LONGITUDE"].min()), float(df_union["LONGITUDE"].max()))
)
yr = st.sidebar.slider(
    "Latitude",
    float(df_union["LATITUDE"].min()), float(df_union["LATITUDE"].max()),
    (float(df_union["LATITUDE"].min()), float(df_union["LATITUDE"].max()))
)

# --- Fixed-extent helpers (keep bbox identical with/without basemap) ---
def fix_extent_axes(fig, xr, yr):
    fig.update_xaxes(range=[float(xr[0]), float(xr[1])])
    fig.update_yaxes(range=[float(yr[0]), float(yr[1])])

def fix_extent_mapbox(fig, xr, yr, style="open-street-map"):
    west, east = float(xr[0]), float(xr[1])
    south, north = float(yr[0]), float(yr[1])
    # 4 invisible corners force Plotly to auto-frame exactly this bbox
    fig.add_trace(go.Scattermapbox(
        lat=[south, south, north, north],
        lon=[west,  east,  west,  east ],
        mode="markers",
        marker=dict(size=1, opacity=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.update_layout(mapbox_style=style, margin=dict(l=0, r=0, t=50, b=10), height=800)

# ===== Numeric-only extras =====
if not category_mode and value_col in ("DIFF", "DIFF_PCT"):
    min_abs_diff = st.sidebar.number_input("Min |value| filter", 0.0, value=0.0, step=1.0)
else:
    min_abs_diff = 0.0

# ===== Display (numeric only) =====
if not category_mode:
    st.sidebar.subheader("Display (numeric)")
    palette = st.sidebar.selectbox(
        "Palette",
        ["RdBu", "Portland", "Plasma", "Viridis", "Turbo", "Hot (black→yellow)"],
        index=1,
    )
    pctl = 95.0
    max_abs_clip = 500.0
    clip_mode = st.sidebar.selectbox("Clip mode", ["Absolute", "Percentile"], index=1)
    if clip_mode == "Absolute":
        max_abs_clip = st.sidebar.number_input("Max value to show", min_value=1.0, value=max_abs_clip, step=10.0)
    else:
        pctl = st.sidebar.slider("Clip range at percentile", 50.0, 99.9, 95.0, 0.1)
else:
    palette = None; clip_mode = None
    pctl = 95.0;   max_abs_clip = 500.0

# ===== Prepare base dataframe =====
base = pick_source_df().copy()
if base.empty:
    st.warning(f"No rows in selected source: {source}"); st.stop()

base = base[base["LONGITUDE"].between(*xr) & base["LATITUDE"].between(*yr)].copy()
if (not category_mode) and value_col in ("DIFF","DIFF_PCT") and min_abs_diff > 0:
    base = base[base[value_col].abs() >= min_abs_diff]

if base.empty or (value_col != "__SOURCE__" and value_col not in base.columns):
    st.warning("No data available under current filters / value selection."); st.stop()

# ===== Color helpers =====
def pick_seq_scale():
    return HOT_CS if palette == "Hot (black→yellow)" else (palette if palette != "RdBu" else "Turbo")

def diverging_color_and_range(series: np.ndarray, is_abs_clip: bool):
    if is_abs_clip:
        vmax = float(max_abs_clip)
    else:
        vmax = float(np.nanpercentile(np.abs(series), pctl)) if len(series) else 1.0
        if vmax <= 0: vmax = 1.0
    return ("RdBu" if palette == "RdBu" else palette), (-vmax, vmax)

def sequential_color_and_range(series: np.ndarray, is_abs_clip: bool):
    if is_abs_clip:
        vmax = float(max_abs_clip)
    else:
        vmax = float(np.nanpercentile(series, pctl)) if len(series) else 1.0
        if vmax <= 0: vmax = 1.0
    return (pick_seq_scale(), (0.0, vmax))

# ===== Method note (All mode) =====
if category_mode:
    st.markdown(
        """
**How these points were built (overview)**  
- **2D grid:** `10 m (X) × 10 m (Y)` over (LONGITUDE, LATITUDE).  
- For each grid cell, we mark presence by source: `overlap`, `orig_only`, `dl_only`.  
- The plotted point is the **center of the occupied cell**.  
Use this view to understand *coverage*.
        """
    )

# ===== Render =====
if view_mode == "Points":
    view = base.dropna(subset=["LONGITUDE","LATITUDE"]).copy()
    if value_col not in view.columns and value_col != "__SOURCE__":
        st.warning("Selected value is not available in current source."); st.stop()
    if len(view) > max_points:
        view = view.sample(max_points, random_state=42)

    if category_mode:
        if use_basemap:
            fig = px.scatter_mapbox(
                view, lat="LATITUDE", lon="LONGITUDE",
                color="SOURCE", color_discrete_map=CAT_COLORS,
                opacity=0.9,
                hover_data={"LONGITUDE":":.5f","LATITUDE":":.5f",
                            "CU_ORIG":":.2f" if "CU_ORIG" in view else False,
                            "CU_DL":":.2f"   if "CU_DL"   in view else False,
                            "DIFF":":.2f"    if "DIFF"    in view else False,
                            "DIFF_PCT":":.2f"if "DIFF_PCT" in view else False,
                            "SOURCE":True}
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(legend=dict(title="SOURCE", font=dict(size=18)))
            fix_extent_mapbox(fig, xr, yr)
        else:
            fig = px.scatter(
                view, x="LONGITUDE", y="LATITUDE",
                color="SOURCE", color_discrete_map=CAT_COLORS,
                opacity=0.9,
                hover_data={"LONGITUDE":":.5f","LATITUDE":":.5f",
                            "CU_ORIG":":.2f" if "CU_ORIG" in view else False,
                            "CU_DL":":.2f"   if "CU_DL"   in view else False,
                            "DIFF":":.2f"    if "DIFF"    in view else False,
                            "DIFF_PCT":":.2f"if "DIFF_PCT" in view else False,
                            "SOURCE":True}
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(xaxis_title="LONGITUDE", yaxis_title="LATITUDE",
                              legend=dict(title="SOURCE", font=dict(size=18)),
                              height=800, margin=dict(l=0, r=0, t=50, b=10))
            fix_extent_axes(fig, xr, yr)
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        st.subheader("Summary (filtered)")
        g = (view["SOURCE"].value_counts(dropna=False)
             .rename_axis("Source").reset_index(name="Points"))
        total = int(g["Points"].sum())
        g["% of points"] = (100.0 * g["Points"] / total).round(2).astype(str) + "%"
        order = ["overlap","orig_only","dl_only"]
        g["Source"] = pd.Categorical(g["Source"], categories=order, ordered=True)
        g = g.sort_values("Source")
        st.dataframe(g[["Source","Points","% of points"]], use_container_width=True, hide_index=True)

    else:
        view = view.dropna(subset=[value_col]).copy()
        arr = view[value_col].to_numpy()
        if value_col == "DIFF_PCT":
            cs, (vmin, vmax) = diverging_color_and_range(arr, clip_mode=="Absolute")
            ticksuf = "%"
        elif value_col == "DIFF":
            cs, (vmin, vmax) = diverging_color_and_range(arr, clip_mode=="Absolute")
            ticksuf = ""
        else:
            cs, (vmin, vmax) = sequential_color_and_range(arr, clip_mode=="Absolute")
            ticksuf = ""

        if use_basemap:
            fig = px.scatter_mapbox(
                view, lat="LATITUDE", lon="LONGITUDE",
                color=value_col, color_continuous_scale=cs, range_color=[vmin, vmax],
                opacity=0.9,
                hover_data={"LONGITUDE":":.5f","LATITUDE":":.5f",
                            "CU_ORIG":":.2f" if "CU_ORIG" in view else False,
                            "CU_DL":":.2f"   if "CU_DL"   in view else False,
                            "DIFF":":.2f"    if "DIFF"    in view else False,
                            "DIFF_PCT":":.2f"if "DIFF_PCT" in view else False,
                            "SOURCE":True if "SOURCE" in view else False}
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(coloraxis_colorbar=dict(title=value_mode, ticksuffix=ticksuf))
            fix_extent_mapbox(fig, xr, yr)
        else:
            fig = px.scatter(
                view, x="LONGITUDE", y="LATITUDE",
                color=value_col, color_continuous_scale=cs, range_color=[vmin, vmax],
                opacity=0.9,
                hover_data={"LONGITUDE":":.5f","LATITUDE":":.5f",
                            "CU_ORIG":":.2f" if "CU_ORIG" in view else False,
                            "CU_DL":":.2f"   if "CU_DL"   in view else False,
                            "DIFF":":.2f"    if "DIFF"    in view else False,
                            "DIFF_PCT":":.2f"if "DIFF_PCT" in view else False,
                            "SOURCE":True if "SOURCE" in view else False}
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(xaxis_title="LONGITUDE", yaxis_title="LATITUDE",
                              coloraxis_colorbar=dict(title=value_mode, ticksuffix=ticksuf),
                              height=800, margin=dict(l=0, r=0, t=50, b=10))
            fix_extent_axes(fig, xr, yr)
        st.plotly_chart(fig, use_container_width=True)

else:
    # Tiles (numeric-only)
    need = ["LONGITUDE","LATITUDE", value_col]
    num = base.dropna(subset=need).copy()
    num[value_col] = pd.to_numeric(num[value_col], errors="coerce")
    num = num.dropna(subset=[value_col])
    if num.empty:
        st.warning("No numeric rows for tiles under current filters / value selection."); st.stop()

    tiles = tile_points_2d(num, nx=nx, ny=ny, value_col=value_col,
                           agg=agg_stat, min_count=min_count_tile)
    if tiles is None:
        st.warning("No tiles under current parameters. Try decreasing NX/NY or Min samples per tile."); st.stop()

    Xc, Yc, Z = tiles["Xc"], tiles["Yc"], tiles["Z"]
    vals = Z[np.isfinite(Z)]

    if value_col == "DIFF_PCT":
        cs, (vmin, vmax) = diverging_color_and_range(vals, clip_mode=="Absolute")
        ticksuf = "%"
    elif value_col == "DIFF":
        cs, (vmin, vmax) = diverging_color_and_range(vals, clip_mode=="Absolute")
        ticksuf = ""
    else:
        cs, (vmin, vmax) = sequential_color_and_range(vals, clip_mode=="Absolute")
        ticksuf = ""

    yy, xx = np.meshgrid(Yc, Xc, indexing="ij")
    centers = pd.DataFrame({"LATITUDE": yy.ravel(), "LONGITUDE": xx.ravel(), "VAL": Z.ravel()}).dropna()

    if use_basemap:
        if map_mode == "Heat (density)":
            fig = px.density_mapbox(
                centers, lat="LATITUDE", lon="LONGITUDE", z="VAL",
                radius=heat_radius, color_continuous_scale=cs
            )
            fig.update_layout(coloraxis=dict(cmin=vmin, cmax=vmax,
                                             colorbar=dict(title=value_mode, ticksuffix=ticksuf)))
            fix_extent_mapbox(fig, xr, yr)
        else:
            fig = px.scatter_mapbox(
                centers, lat="LATITUDE", lon="LONGITUDE",
                color="VAL", color_continuous_scale=cs, range_color=[vmin, vmax],
                opacity=tile_opacity
            )
            fig.update_traces(marker=dict(size=9, symbol="square"))
            fig.update_layout(coloraxis_colorbar=dict(title=value_mode, ticksuffix=ticksuf))
            fix_extent_mapbox(fig, xr, yr)
    else:
        fig = go.Figure(data=go.Heatmap(
            x=Xc, y=Yc, z=np.flipud(Z),
            colorscale=cs, zmin=vmin, zmax=vmax,
            colorbar=dict(title=value_mode, ticksuffix=ticksuf),
            showscale=True, opacity=tile_opacity
        ))
        fig.update_layout(xaxis_title="LONGITUDE", yaxis_title="LATITUDE",
                          height=800, margin=dict(l=0, r=0, t=50, b=10))
        fig.update_yaxes(autorange="reversed")
        fix_extent_axes(fig, xr, yr)

    st.plotly_chart(fig, use_container_width=True)