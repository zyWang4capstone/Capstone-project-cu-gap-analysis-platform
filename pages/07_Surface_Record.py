# 07_Surface_Record.py — Record-level • Surface (09-style 2D visuals)

from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from _ui_common import inject_theme
from cap_common.config import load_cfg

st.set_page_config(layout="wide", page_title="Record-level • Surface")
inject_theme()

# ---------- Tab/heading CSS (aligned with 05/06/09) ----------
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] {
        background:#f8fafc!important;color:#111827!important;border:1px solid #e5e7eb!important;
        border-bottom:none!important;border-radius:10px 10px 0 0!important;padding:.45rem 1rem!important;
        margin-right:6px!important;font-size:1.05rem!important;font-weight:500!important;
        box-shadow:none!important;outline:none!important;
    }
    .stTabs [data-baseweb="tab"] * { color:inherit!important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background:#111827!important;color:#fff!important;font-size:1.15rem!important;font-weight:700!important;
        border-color:#111827!important;transform:translateY(1px);
    }
    .stTabs [data-baseweb="tab-highlight"] { background:transparent!important; }
    .stTabs + div [data-testid="stVerticalBlock"] > div:first-child {
        border-top:1px solid #11182710;margin-top:-1px;
    }
    .stAppDeployButton { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- Paths ----------------------------
CLEAN_DIR = Path("reports/task1/cleaned")

def _resolve_surface_clean_paths():
    """
    Resolve cleaned surface CSVs with flexible naming patterns.
    Returns: (orig_path or None, dl_path or None)
    """
    def pick_one(patterns):
        for pat in patterns:
            hits = sorted(CLEAN_DIR.glob(pat))
            if hits:
                return hits[0]
        return None

    # Try flexible patterns first (supporting surf/surface + orig/original + dl/dnn)
    p_orig = pick_one([
        "*surface*orig*clean*.csv",
        "*surf*orig*clean*.csv",
        "*surface*original*clean*.csv",
        "*surf*original*clean*.csv",
    ])
    p_dl = pick_one([
        "*surface*dnn*clean*.csv",
        "*surf*dnn*clean*.csv",
        "*surface*dl*clean*.csv",
        "*surf*dl*clean*.csv",
    ])

    # Legacy hard-coded names as the last resort (back-compat)
    if p_orig is None and (CLEAN_DIR / "surface_original_clean.csv").exists():
        p_orig = CLEAN_DIR / "surface_original_clean.csv"
    if p_dl is None and (CLEAN_DIR / "surface_dnn_clean.csv").exists():
        p_dl = CLEAN_DIR / "surface_dnn_clean.csv"

    return p_orig, p_dl

PATH_SURF_ORIG, PATH_SURF_DL = _resolve_surface_clean_paths()

# ------------------------- I/O & helpers ------------------------
@st.cache_data
def load_csv_safe(p: Path, element: str = "Cu", source_tag: str | None = None) -> pd.DataFrame:
    """Safe CSV loader for surface data with dynamic element support."""
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)

        # Standardize column names (Surface files often use DLAT/DLONG)
        rename_map = {"DLAT": "LATITUDE", "DLONG": "LONGITUDE"}
        df = df.rename(columns={c: rename_map[c] for c in rename_map if c in df.columns})

        # Ensure numeric for coords (required for plotting)
        for c in ["LONGITUDE", "LATITUDE"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Element-specific numeric columns
        element_cols = [
            f"{element}_ppm",
            f"{element.upper()}_ORIG",
            f"{element.upper()}_DL",
        ]
        for c in element_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Add source tag for identification
        if source_tag is not None:
            df["_src"] = source_tag

        return df.dropna(subset=["LONGITUDE", "LATITUDE"]).copy()
    except Exception:
        return pd.DataFrame()

# ------------------------------ Data -----------------------------
element = st.session_state.get("element", "Element")
surf_orig = load_csv_safe(PATH_SURF_ORIG, element, "ORIG")
surf_dl   = load_csv_safe(PATH_SURF_DL, element, "DL")

def tile_points_2d(df: pd.DataFrame, nx: int, ny: int, value_col: str,
                   agg: str = "mean", min_count: int = 1):
    """Aggregate scattered surface points into a 2D grid (nx × ny)."""
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

def fix_extent_axes(fig, xr, yr, height: int = 700):
    fig.update_xaxes(range=[float(xr[0]), float(xr[1])])
    fig.update_yaxes(range=[float(yr[0]), float(yr[1])])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=50, b=10))

def fix_extent_mapbox(fig, xr, yr, style="open-street-map", height: int = 800):
    west, east = float(xr[0]), float(xr[1])
    south, north = float(yr[0]), float(yr[1])
    fig.add_trace(go.Scattermapbox(
        lat=[south, south, north, north],
        lon=[west, east, west, east],
        mode="markers",
        marker=dict(size=1, opacity=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.update_layout(mapbox_style=style, margin=dict(l=0, r=0, t=50, b=10), height=height)

# Data existence check
if surf_orig.empty and surf_dl.empty:

    def _read_points_all_fallback() -> pd.DataFrame:
        """
        Try reading points_all from (a) cfg.original_points_all, then (b) default locations.
        Return empty DataFrame if not found/readable.
        """
        # 1) Candidate paths from cfg if available
        try:
            cfg = load_cfg()
        except Exception:
            cfg = None

        candidates = []
        if cfg and getattr(cfg, "original_points_all", None):
            base = Path(cfg.original_points_all)
            candidates += [base.with_suffix(".parquet"), base.with_suffix(".csv")]

        # 2) Default locations under reports/task1/original/
        base2 = Path("reports/task1/original/points_all")
        candidates += [base2.with_suffix(".parquet"), base2.with_suffix(".csv")]

        # 3) Historical aliases (optional)
        candidates += [
            Path("reports/task1/original/all_points.parquet"),
            Path("reports/task1/original/all_points.csv"),
        ]

        # Read the first existing file
        for p in candidates:
            if p.exists():
                try:
                    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                except Exception:
                    # try next candidate if read failed
                    continue
        return pd.DataFrame()

    df_all = _read_points_all_fallback()

    if not df_all.empty:
        # Standardize schema: uppercase + coord rename + numeric coercion
        df_all = df_all.rename(columns=str.upper)
        df_all = df_all.rename(columns={"DLAT": "LATITUDE", "DLONG": "LONGITUDE"})
        for c in ("LONGITUDE", "LATITUDE", "VALUE"):
            if c in df_all.columns:
                df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

        # Split by SOURCE into ORIG/DL (support SURF_* and SF_* variants)
        src = df_all.get("SOURCE", pd.Series(dtype=str)).astype(str).str.upper().str.replace("-", "_")
        so = df_all[src.isin(["SURF_ORIG", "SF_ORIG"])].copy()
        sd = df_all[src.isin(["SURF_DL", "SF_DL"])].copy()

        # Build element-specific columns expected by the page:
        # - ORIG side gets both {element}_ppm and {ELEMENT}_ORIG
        # - DL   side gets {ELEMENT}_DL
        if not so.empty:
            so[f"{element}_ppm"] = pd.to_numeric(so.get("VALUE"), errors="coerce")
            so[f"{element.upper()}_ORIG"] = so[f"{element}_ppm"]
            surf_orig = so
        if not sd.empty:
            sd[f"{element.upper()}_DL"] = pd.to_numeric(sd.get("VALUE"), errors="coerce")
            surf_dl = sd

    # Still nothin? Show a clear message and stop.
    if surf_orig.empty and surf_dl.empty:
        st.warning(
            "No surface data found.\n"
            "Tried flexible names under `reports/task1/cleaned/` and a fallback to "
            "`points_all` under `reports/task1/original/`."
        )
        st.stop()

# ------------------------------ Header ---------------------------
element = st.session_state.get("element", "Element")

left, right1, right2 = st.columns([0.6, 0.2, 0.2])
with left:
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.25rem; font-size:2.0rem;">
        Record-level • Surface ({element})
        </h1>
        <p style="color:#555; margin-top:0;">
        2D Points / 2D Grid (map optional) for {element} values
        </p>
        """,
        unsafe_allow_html=True,
    )
with right1:
    st.page_link("pages/04_Diff_Home.py", label="Back to Diff • Home")
with right2:
    st.page_link("pages/09_Insights.py", label="View insights")

# ───────────────────────── Sidebar controls ─────────────────────────
show_orig = st.sidebar.checkbox("Show ORIG", True)
show_dl   = st.sidebar.checkbox("Show DL",   True)
pt_size   = st.sidebar.slider("Point size", 1, 12, 5)

# Union of ORIG + DL for coordinate ranges
u = pd.concat(
    [df.assign(_src=tag) for df, tag in [(surf_orig, "ORIG"), (surf_dl, "DL")] if not df.empty],
    ignore_index=True
)

if u.empty:
    st.warning("No data found. Check CSV paths.")
    st.stop()

# Longitude / Latitude sliders (default to +2° window where possible)
if {"LONGITUDE", "LATITUDE"}.issubset(u.columns):
    lon_min, lon_max = float(u["LONGITUDE"].min()), float(u["LONGITUDE"].max())
    lat_min, lat_max = float(u["LATITUDE"].min()),  float(u["LATITUDE"].max())
    lon_default = (lon_min, min(lon_min + 2, lon_max))
    lat_default = (lat_min, min(lat_min + 2, lat_max))
else:
    lon_min, lon_max = -180.0, 180.0
    lat_min, lat_max = -90.0, 90.0
    lon_default = (lon_min, lon_max)
    lat_default = (lat_min, lat_max)

xr = st.sidebar.slider("Longitude", lon_min, lon_max, lon_default)
yr = st.sidebar.slider("Latitude",  lat_min, lat_max, lat_default)

# View mode choice (source vs element values)
view_kind = st.sidebar.radio(
    "Color mode",
    [f"By source (ORIG/DL)", f"By value ({element})"],
    index=0, horizontal=True, key="surf_rec_colormode"
)

# Basemap controls 
st.sidebar.subheader("Basemap")
use_basemap = st.sidebar.checkbox("Use map background (OSM)", value=False)

if view_kind == f"By value ({element})":
    map_mode    = st.sidebar.selectbox("Map layer for numeric view", ["Heat (density)", "Squares"], index=0)
    map_zoom    = st.sidebar.slider("Map zoom (visual only)", 2, 14, 7)
    heat_radius = st.sidebar.slider("Heat radius (px)", 2, 30, 8)
else:
    map_mode, map_zoom, heat_radius = None, None, None
    
# Numeric display options 
if view_kind == f"By value ({element})":
    palette = "Portland"
    clip_mode = st.sidebar.selectbox("Clip mode", ["Absolute", "Percentile"], index=1)
    # dynamic default: 99th percentile of available values
    element_cols = [f"{element}_ppm", f"{element.upper()}_ORIG", f"{element.upper()}_DL"]
    col_available = next((c for c in element_cols if c in u.columns), None)
    vmax_default = float(np.nanpercentile(u[col_available], 99)) if col_available else 500.0

    max_abs_clip = vmax_default
    pctl = 95.0
    if clip_mode == "Absolute":
        max_abs_clip = st.sidebar.number_input("Max value to show", min_value=1.0, value=max_abs_clip, step=10.0)
    else:
        pctl = st.sidebar.slider("Clip range at percentile", min_value=50.0, max_value=99.9, value=95.0, step=0.1)
    min_abs_val = st.sidebar.number_input("Min |value| filter", 0.0, value=0.0, step=1.0)
else:
    palette = None; clip_mode = None
    max_abs_clip = None; pctl = None; min_abs_val = 0.0

# ------------------------------ Filtered bases ---------------------
def filter_window(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if {"LONGITUDE","LATITUDE"}.issubset(out.columns):
        out = out[out["LONGITUDE"].between(*xr) & out["LATITUDE"].between(*yr)]
    if view_kind == f"By value ({element})":
        for col in [f"{element}_ppm", f"{element.upper()}_ORIG", f"{element.upper()}_DL"]:
            if col in out.columns:
                out = out[abs(pd.to_numeric(out[col], errors="coerce")) >= min_abs_val]
                break
    return out

full_o = filter_window(surf_orig) if show_orig else pd.DataFrame()
full_d = filter_window(surf_dl)   if show_dl   else pd.DataFrame()

# build combined base depending on view kind
base = pd.concat([
    full_o.assign(SOURCE="Original") if not full_o.empty else pd.DataFrame(),
    full_d.assign(SOURCE="DL")       if not full_d.empty else pd.DataFrame(),
], ignore_index=True)

if base.empty:
    st.info("No records in current window. Adjust the filters.")
    st.stop()

# choose available numeric columns
value_col = None 
NUM_CHOICES = [
    f"{element}_ppm",
    f"{element.upper()}_DL",
    f"{element.upper()}_ORIG",
    "DIFF",
    "DIFF_PCT",
]
NUM_CHOICES = [c for c in NUM_CHOICES if c in base.columns]

default_value_col = (
    f"{element}_ppm" if f"{element}_ppm" in NUM_CHOICES
    else (NUM_CHOICES[0] if NUM_CHOICES else None)
)

if view_kind == f"By value ({element})":
    if not NUM_CHOICES:
        st.warning(
            f"No numeric value columns found (expected one of {', '.join(NUM_CHOICES)})."
        )
        st.stop()

    value_col = st.sidebar.selectbox(
        f"Value to display ({element})", NUM_CHOICES,
        index=NUM_CHOICES.index(default_value_col)
    )

    # enforce numeric
    base[value_col] = pd.to_numeric(base[value_col], errors="coerce")

    # apply absolute value filter (for all numeric columns, not just DIFF)
    if min_abs_val > 0:
        base = base[base[value_col].abs() >= float(min_abs_val)]

# ------------------------------ Color helpers ---------------------
def diverging_color_and_range(series: np.ndarray):
    if clip_mode == "Absolute":
        vmax = float(max_abs_clip)
    else:
        vmax = float(np.nanpercentile(np.abs(series), pctl)) if len(series) else 1.0
        if vmax <= 0: vmax = 1.0
    return ("Portland", (-vmax, vmax))

def sequential_color_and_range(series: np.ndarray):
    if clip_mode == "Absolute":
        vmax = float(max_abs_clip)
    else:
        vmax = float(np.nanpercentile(series, pctl)) if len(series) else 1.0
        if vmax <= 0: vmax = 1.0
    return ("Portland", (0.0, vmax))

# ------------------------------ Tabs (09-style) -------------------
tab_points, tab_grid = st.tabs(["2D Points", "2D Grid"])

# Fixed category colors like 06: Original (blue) / DL (red)
CAT_COLORS = {"Original": "#447dd2", "DL": "#ff554b"}

# Helper: human-readable labels for value_col
def label_for_value_mode(col: str) -> str:
    if col == f"{element}_ppm": return f"Observed {element}"
    if col == f"{element.upper()}_ORIG": return f"Original {element}"
    if col == f"{element.upper()}_DL": return f"DL-predicted {element}"
    if col == "DIFF": return "Difference"
    if col == "DIFF_PCT": return "Difference (%)"
    return col

with tab_points:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        point_size = st.slider("Point size", 1, 12, int(pt_size), key="surf_rec_pts_size")
    with c2:
        limit_points = st.toggle("Limit points", value=False, key="surf_rec_pts_limit")
    with c3:
        max_points = st.number_input("Max points", 1_000, 1_000_000, 100_000, 1_000, key="surf_rec_pts_cap")

    view = base.copy()

    # --- sampling safeguard ---
    if limit_points and len(view) > max_points:
        view = view.sample(max_points, random_state=42)

    # --- Source view ---
    if view_kind == f"By source (ORIG/DL)":
        if use_basemap:
            fig = px.scatter_mapbox(
                view, lat="LATITUDE", lon="LONGITUDE",
                color="SOURCE", color_discrete_map=CAT_COLORS,
                opacity=0.9,
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(legend=dict(title="SOURCE", font=dict(size=18)))
            fix_extent_mapbox(fig, xr, yr)
        else:
            fig = px.scatter(
                view, x="LONGITUDE", y="LATITUDE",
                color="SOURCE", color_discrete_map=CAT_COLORS,
                opacity=0.9,
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(
                xaxis_title="Longitude", yaxis_title="Latitude",
                legend=dict(title="SOURCE", font=dict(size=18)),
                height=800, margin=dict(l=0, r=0, t=50, b=10),
            )
            fix_extent_axes(fig, xr, yr)
        st.plotly_chart(fig, use_container_width=True)

    # --- Value view ---
    elif view_kind == f"By value ({element})" and value_col:
        view = view.dropna(subset=[value_col]).copy()
        vals = pd.to_numeric(view[value_col], errors="coerce")

        if value_col == "DIFF_PCT":
            cs, (vmin, vmax) = diverging_color_and_range(vals.to_numpy())
            ticksuf = "%"
        elif value_col == "DIFF":
            cs, (vmin, vmax) = diverging_color_and_range(vals.to_numpy())
            ticksuf = ""
        else:
            cs, (vmin, vmax) = sequential_color_and_range(vals.to_numpy())
            ticksuf = ""

        if use_basemap:
            fig = px.scatter_mapbox(
                view, lat="LATITUDE", lon="LONGITUDE",
                color=value_col, color_continuous_scale=cs, range_color=[vmin, vmax],
                opacity=0.9,
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(coloraxis_colorbar=dict(
                title=label_for_value_mode(value_col), ticksuffix=ticksuf
            ))
            fix_extent_mapbox(fig, xr, yr)
        else:
            fig = px.scatter(
                view, x="LONGITUDE", y="LATITUDE",
                color=value_col, color_continuous_scale=cs, range_color=[vmin, vmax],
                opacity=0.9,
            )
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(
                xaxis_title="Longitude", yaxis_title="Latitude",
                coloraxis_colorbar=dict(title=label_for_value_mode(value_col), ticksuffix=ticksuf),
                height=800, margin=dict(l=0, r=0, t=50, b=10),
            )
            fix_extent_axes(fig, xr, yr)
        st.plotly_chart(fig, use_container_width=True)

with tab_grid:
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        nx = st.slider("NX (lon bins)", 20, 400, 120, 10, key="surf_rec_grid_nx")
    with gc2:
        ny = st.slider("NY (lat bins)", 20, 400, 120, 10, key="surf_rec_grid_ny")
    with gc3:
        min_count_tile = st.number_input("Min samples/tile", 1, 500, 3, 1, key="surf_rec_grid_min")

    gc4, gc5 = st.columns(2)
    with gc4:
        agg_stat = st.selectbox("Aggregator", ["mean", "median"], index=0, key="surf_rec_grid_agg")
    with gc5:
        tile_opacity = st.slider("Tile opacity", 0.1, 1.0, 0.85, 0.05, key="surf_rec_grid_opacity")

    # --- Only available in value mode ---
    if view_kind == f"By source (ORIG/DL)":
        st.info("Grid is numeric-only. Switch to 'By value' to enable 2D Grid.")
    elif view_kind == f"By value ({element})" and value_col:
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
                    cs, (vmin, vmax) = diverging_color_and_range(vals)
                    ticksuf = "%" if value_col == "DIFF_PCT" else ""
                else:
                    cs, (vmin, vmax) = sequential_color_and_range(vals)
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
                            colorbar=dict(title=label_for_value_mode(value_col), ticksuffix=ticksuf)
                        ))
                    else:
                        fig = px.scatter_mapbox(
                            centers, lat="LATITUDE", lon="LONGITUDE",
                            color="VAL", color_continuous_scale=cs, range_color=[vmin, vmax],
                            opacity=tile_opacity
                        )
                        fig.update_traces(marker=dict(size=9, symbol="square"))
                        fig.update_layout(coloraxis_colorbar=dict(
                            title=label_for_value_mode(value_col), ticksuffix=ticksuf
                        ))
                    fix_extent_mapbox(fig, xr, yr)
                else:
                    fig = go.Figure(data=go.Heatmap(
                        x=Xc, y=Yc, z=np.flipud(Z),
                        colorscale=cs, zmin=vmin, zmax=vmax,
                        colorbar=dict(title=label_for_value_mode(value_col), ticksuffix=ticksuf),
                        showscale=True, opacity=tile_opacity
                    ))
                    fig.update_layout(
                        xaxis_title="Longitude", yaxis_title="Latitude",
                        height=800, margin=dict(l=0, r=0, t=50, b=10)
                    )
                    fig.update_yaxes(autorange="reversed")
                    fix_extent_axes(fig, xr, yr)

                st.plotly_chart(fig, use_container_width=True)

# ------------------------------ Table & Export --------------------
st.subheader("Records (filtered)")

def keep_cols(df: pd.DataFrame, tag: str, element: str, mode: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_LAYER"] = tag

    if mode.startswith("By value"):  
        # keep element-related columns
        element_cols = [
            f"{element}_ppm",
            f"{element.upper()}_ORIG",
            f"{element.upper()}_DL",
            "DIFF", "DIFF_PCT"
        ]
        keep = [c for c in ["SAMPLEID", "LONGITUDE", "LATITUDE", *element_cols] if c in out.columns]
    else:  
        # By source → just basic metadata
        keep = [c for c in ["SAMPLEID", "LONGITUDE", "LATITUDE"] if c in out.columns]

    keep.append("_LAYER")
    return out[keep]

tbl = pd.concat([
    keep_cols(full_o, "ORIG", element, view_kind) if not full_o.empty else pd.DataFrame(),
    keep_cols(full_d, "DL", element, view_kind)   if not full_d.empty else pd.DataFrame()
], ignore_index=True)

if tbl.empty:
    st.caption("No rows to display.")
else:
    total_o, total_d = len(full_o), len(full_d)
    total = total_o + total_d
    st.markdown(
        f"**{total:,} records in current window** (ORIG: {total_o:,} • DL: {total_d:,})"
    )
    st.dataframe(tbl, use_container_width=True, height=360, hide_index=True)

    csv_buf = io.StringIO()
    tbl.to_csv(csv_buf, index=False)

    if view_kind.startswith("By value"):
        fname = f"surface_record_filtered_{element}.csv"
    else:
        fname = "surface_record_filtered.csv"

    st.download_button(
        "Download filtered CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=fname,
        mime="text/csv",
        type="primary"
    )