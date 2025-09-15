# 06_Diff_Drillhole.py — multi-source (All / Overlap / Orig-only / DL-only)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import io
from cap_common.config import load_cfg
from cap_task2.io import read_points

from _ui_common import inject_theme
inject_theme()

st.set_page_config(layout="wide", page_title="Drillhole Differences (3D)")

# --- Tab styling: make active tab black background + white text, larger font ---
st.markdown(
    """
    <style>
    /* Scope all rules to Streamlit tab containers only */
    .stTabs [data-baseweb="tab"] {
        /* Base (idle) tab look */
        background: #f8fafc !important;                  /* subtle light bg */
        color: #111827 !important;                       /* dark gray text */
        border: 1px solid #e5e7eb !important;            /* light border */
        border-bottom: none !important;                  /* connect to panel */
        border-radius: 10px 10px 0 0 !important;         /* rounded top */
        padding: 0.45rem 1rem !important;                /* comfortable hit area */
        margin-right: 6px !important;                    /* spacing between tabs */
        font-size: 1.05rem !important;                   /* slightly bigger base */
        font-weight: 500 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    /* Inherit text color for any nested elements (Streamlit may wrap labels) */
    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
    }

    /* Active (selected) tab */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #111827 !important;                  /* black bg */
        color: #ffffff !important;                        /* white text */
        font-size: 1.15rem !important;                    /* larger font when active */
        font-weight: 700 !important;                      /* bolder */
        border-color: #111827 !important;
        box-shadow: none !important;
        transform: translateY(1px);                       /* tiny optical alignment */
    }

    /* Hide Streamlit's default red underline highlight bar */
    .stTabs [data-baseweb="tab-highlight"] {
        background: transparent !important;
    }

    /* Optional: remove extra top border from the tab content area for a flush look */
    .stTabs + div [data-testid="stVerticalBlock"] > div:first-child {
        border-top: 1px solid #11182710;                 /* hairline divider */
        margin-top: -1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Paths (difference outputs) -----------------
DIFF_DIR = Path("reports/task2/difference")
PATH_ALL       = DIFF_DIR / "drillhole_points_all.csv"
PATH_OVERLAP   = DIFF_DIR / "drillhole_points_overlap.csv"
PATH_ORIGONLY  = DIFF_DIR / "drillhole_points_origonly.csv"
PATH_DLONLY    = DIFF_DIR / "drillhole_points_dlonly.csv"

# ----------------- Surface (overlay) paths ----------------- 
SURF_ALL      = DIFF_DIR / "surface_points_all.csv"
SURF_OVERLAP  = DIFF_DIR / "surface_points_overlap.csv"
SURF_ORIGONLY = DIFF_DIR / "surface_points_origonly.csv"
SURF_DLONLY   = DIFF_DIR / "surface_points_dlonly.csv"

@st.cache_data
def load_surface(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ["LONGITUDE","LATITUDE","CU_ORIG","CU_DL","DIFF","DIFF_PCT"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# --- Overlay helper -----------------------------------------
def get_surface_df_for_overlay(overlay_src: str) -> pd.DataFrame:
    """
    Build a surface dataframe for the chosen source.
    overlay_src ∈ {"Original", "DL"}.
    We compose it from *_origonly + overlap (for Original) or *_dlonly + overlap (for DL).
    The returned df must at least contain LONGITUDE, LATITUDE, and SOURCE.
    Numeric columns (CU_ORIG, CU_DL, DIFF, DIFF_PCT) will be used if present.
    """
    parts = []
    try:
        if overlay_src == "Original":
            df_o = load_surface(SURF_ORIGONLY)
            if not df_o.empty:
                df_o = df_o.copy()
                df_o["SOURCE"] = df_o.get("SOURCE", "orig_only")
                parts.append(df_o)
            df_ov = load_surface(SURF_OVERLAP)
            if not df_ov.empty:
                df_ov = df_ov.copy()
                df_ov["SOURCE"] = df_ov.get("SOURCE", "overlap")
                parts.append(df_ov)
        else:  # "DL"
            df_d = load_surface(SURF_DLONLY)
            if not df_d.empty:
                df_d = df_d.copy()
                df_d["SOURCE"] = df_d.get("SOURCE", "dl_only")
                parts.append(df_d)
            df_ov = load_surface(SURF_OVERLAP)
            if not df_ov.empty:
                df_ov = df_ov.copy()
                df_ov["SOURCE"] = df_ov.get("SOURCE", "overlap")
                parts.append(df_ov)
    except Exception:
        return pd.DataFrame()

    if not parts:
        return pd.DataFrame()

    surf = pd.concat(parts, ignore_index=True)
    # keep only rows with both lon/lat available
    if not {"LONGITUDE", "LATITUDE"}.issubset(surf.columns):
        return pd.DataFrame()
    surf = surf.dropna(subset=["LONGITUDE", "LATITUDE"]).copy()
    return surf

# --- Overlay helper (auto-pick by current radio 'source') ---------------------
def get_surface_df_by_source(source: str) -> pd.DataFrame:
    """Return surface df that matches current drillhole 'source' selection."""
    # --- Fix for "All": build from three split files with consistent SOURCE tags ---
    if source == "All":
        parts = []
        for tag, p in [("orig_only", SURF_ORIGONLY),
                       ("overlap",   SURF_OVERLAP),
                       ("dl_only",   SURF_DLONLY)]:
            df = load_surface(p)
            if df.empty:
                continue
            df = df.copy()
          
            df = df.rename(columns={"DLAT": "LATITUDE", "DLONG": "LONGITUDE"})
            if not {"LONGITUDE","LATITUDE"}.issubset(df.columns):
                continue
            df["SOURCE"] = tag 
            parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    
    mapping = {
        "Overlap":   SURF_OVERLAP,
        "Orig-only": SURF_ORIGONLY,
        "DL-only":   SURF_DLONLY,
    }
    p = mapping.get(source)
    if p is None:
        return pd.DataFrame()
    df = load_surface(p).rename(columns={"DLAT": "LATITUDE", "DLONG": "LONGITUDE"})
    if df.empty or not {"LONGITUDE","LATITUDE"}.issubset(df.columns):
        return pd.DataFrame()
    if "SOURCE" not in df.columns:
        df = df.copy()
        df["SOURCE"] = {"Overlap":"overlap","Orig-only":"orig_only","DL-only":"dl_only"}[source]
    return df.dropna(subset=["LONGITUDE","LATITUDE"]).copy()

# ----------------- Custom palettes -----------------
HOT_CS = [
    [0.00, "#000000"], [0.20, "#2b0000"], [0.40, "#7a0000"],
    [0.70, "#ff3b00"], [1.00, "#ffff66"],
]

# ----------------- Data loader -----------------
@st.cache_data
def safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ["LONGITUDE","LATITUDE","DEPTH","CU_ORIG","CU_DL","DIFF","DIFF_PCT","SOURCE"]:
        if c in df.columns and c != "SOURCE":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # derive DIFF_PCT if possible
    if "DIFF_PCT" not in df.columns and {"DIFF","CU_DL"}.issubset(df.columns):
        eps = 1e-9
        denom = df["CU_DL"].where(df["CU_DL"].abs() >= eps, np.nan)
        df["DIFF_PCT"] = 100.0 * df["DIFF"] / denom
    return df


# ------------------------------ Header ---------------------------

# --- Page title row with three columns ---
c1, c2, c3 = st.columns([0.6, 0.2, 0.2])  # adjust proportions as needed

with c1:
    st.markdown(
        """
        <h1 style="margin-bottom:0.25rem; font-size:2.0rem;">
        Medium/Coarse-level • Drillhole
        </h1>
        <p style="color:#555; margin-top:0;">
        Aggregated (points/voxels)
        </p>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.page_link("pages/04_Diff_Home.py", label="Back to Diff • Home")

with c3:
    st.page_link("pages/09_Insights.py", label="View insights")


# ----------------- Voxelization (grid) -----------------
def voxelize_points(df: pd.DataFrame, nx: int, ny: int, nz: int,
                    value_col: str, agg: str = "mean", min_count: int = 1):
    """Aggregate points into a regular 3D grid and return voxel centers + values."""
    if df.empty:
        return None
    x = df["LONGITUDE"].to_numpy()
    y = df["LATITUDE"].to_numpy()
    z = df["DEPTH"].to_numpy()
    v = df[value_col].to_numpy()

    x_edges = np.linspace(x.min(), x.max(), nx + 1)
    y_edges = np.linspace(y.min(), y.max(), ny + 1)
    z_edges = np.linspace(z.min(), z.max(), nz + 1)

    ix = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1)
    iz = np.clip(np.searchsorted(z_edges, z, side="right") - 1, 0, nz - 1)

    flat = ix + nx * (iy + ny * iz)
    dfb = pd.DataFrame({"flat": flat, "val": v})

    if agg == "median":
        agg_df = dfb.groupby("flat").agg(val=("val","median"), cnt=("val","size")).reset_index()
    else:
        agg_df = dfb.groupby("flat").agg(val=("val","mean"),   cnt=("val","size")).reset_index()

    agg_df = agg_df[agg_df["cnt"] >= int(min_count)]
    if len(agg_df) == 0:
        return None

    flat_idx = agg_df["flat"].to_numpy()
    ix = flat_idx % nx
    iy = (flat_idx // nx) % ny
    iz = flat_idx // (nx * ny)

    Xc = (x_edges[ix] + x_edges[ix + 1]) / 2.0
    Yc = (y_edges[iy] + y_edges[iy + 1]) / 2.0
    Zc = (z_edges[iz] + z_edges[iz + 1]) / 2.0
    V  = agg_df["val"].to_numpy()

    return {"Xc":Xc, "Yc":Yc, "Zc":Zc, "V":V,
            "x_edges":x_edges, "y_edges":y_edges, "z_edges":z_edges}

# ----------------- Build voxel cubes (Mesh3d) -----------------
def build_voxel_mesh(Xc, Yc, Zc, V, dx, dy, dz,
                     cmin, cmax, colorscale,
                     opacity=0.6, colorbar_title="value", ticksuffix="",
                     max_cubes=3000, pick_top_by_abs=True) -> go.Mesh3d:
    n = len(V)
    if n == 0:
        return go.Mesh3d()
    if n > max_cubes:
        if pick_top_by_abs:
            idx = np.argsort(np.abs(V))[::-1][:max_cubes]
        else:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_cubes, replace=False)
        Xc, Yc, Zc, V = Xc[idx], Yc[idx], Zc[idx], V[idx]
        n = len(V)

    hx, hy, hz = dx/2.0, dy/2.0, dz/2.0
    local = np.array([[-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
                      [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]], dtype=float)
    local[:,0]*=hx; local[:,1]*=hy; local[:,2]*=hz
    faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],
                      [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                      [1,2,6],[1,6,5],[0,3,7],[0,7,4]], dtype=int)

    X = np.empty(n*8); Y = np.empty(n*8); Z = np.empty(n*8)
    I = np.empty(n*12, dtype=int); J = np.empty(n*12, dtype=int); K = np.empty(n*12, dtype=int)
    INT = np.empty(n*8)

    for k in range(n):
        base_v = 8*k
        verts = local + np.array([Xc[k], Yc[k], Zc[k]])
        X[base_v:base_v+8] = verts[:,0]; Y[base_v:base_v+8] = verts[:,1]; Z[base_v:base_v+8] = verts[:,2]
        INT[base_v:base_v+8] = V[k]
        tri = faces + base_v
        base_f = 12*k
        I[base_f:base_f+12] = tri[:,0]; J[base_f:base_f+12] = tri[:,1]; K[base_f:base_f+12] = tri[:,2]

    return go.Mesh3d(
        x=X, y=Y, z=Z, i=I, j=J, k=K,
        intensity=INT, colorscale=colorscale, cmin=cmin, cmax=cmax,
        flatshading=True, opacity=opacity, showscale=True,
        colorbar=dict(title=colorbar_title, ticksuffix=ticksuffix),
        lighting=dict(ambient=0.5, diffuse=0.6, specular=0.0),
    )

# ----------------- Load data -----------------
cfg = load_cfg()

def _read_and_fix(kind: str, split: str) -> pd.DataFrame:
    df = read_points(kind, split, cfg)

    
    df = df.rename(columns=lambda c: str(c).upper())
    if "DLAT" in df.columns:  df = df.rename(columns={"DLAT": "LATITUDE"})
    if "DLONG" in df.columns: df = df.rename(columns={"DLONG": "LONGITUDE"})

   
    if "SOURCE" in df.columns:
        s = df["SOURCE"].astype(str).str.strip().str.lower()
    else:
        s = pd.Series("", index=df.index)

    
    norm = s.map({
        "orig": "orig_only", "original": "orig_only", "orig_only": "orig_only", "origonly": "orig_only",
        "dl": "dl_only", "dl_only": "dl_only", "dlonly": "dl_only",
        "overlap": "overlap", "common": "overlap",
        "drill_all": "drill_all", "all": "drill_all", "surface_all":"surface_all",
    })

    
    tag_by_split = {"overlap":"overlap", "origonly":"orig_only", "dlonly":"dl_only", "all":"drill_all"}
    tag = tag_by_split.get(split, "unknown")
    if split in ("overlap","origonly","dlonly"):
        df["SOURCE"] = tag
    else:
        df["SOURCE"] = norm.fillna(tag)

   
    for c in ["LONGITUDE","LATITUDE","DEPTH","CU_ORIG","CU_DL","DIFF","DIFF_PCT"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "DIFF_PCT" not in df.columns and {"DIFF","CU_DL"}.issubset(df.columns):
        eps = 1e-9
        denom = df["CU_DL"].where(df["CU_DL"].abs() >= eps, np.nan)
        df["DIFF_PCT"] = 100.0 * df["DIFF"] / denom

    return df.dropna(subset=["LONGITUDE","LATITUDE"]).copy()



df_all      = _read_and_fix("drillhole", "all")
df_overlap  = _read_and_fix("drillhole", "overlap")
df_origonly = _read_and_fix("drillhole", "origonly")
df_dlonly   = _read_and_fix("drillhole", "dlonly")
df_union    = pd.concat([df_all, df_overlap, df_origonly, df_dlonly], ignore_index=True)


# ---- Fixed category order & colors (same as p05) ----
CAT_ORDER = ["orig_only", "overlap", "dl_only"]  
CAT_COLORS = {
    "orig_only": "#447dd2",
    "overlap":   "#9ecbff",  
    "dl_only":   "#ff554b", 
}

# ───────────────────────── Sidebar controls (organized) ─────────────────────────

# 1) Data source
source = st.sidebar.radio(
    "Data source",
    ["All", "Overlap", "Orig-only", "DL-only"],
    index=0,
    horizontal=True,
    key="src_mode"
)

def pick_source_df():
    if source == "All":
        parts = []
        for tag, d in [("orig_only", df_origonly),
                       ("overlap",   df_overlap),
                       ("dl_only",   df_dlonly)]:
            if not d.empty:
                dd = d.copy()
                dd["SOURCE"] = tag          
                parts.append(dd)

        if parts:                            
            return pd.concat(parts, ignore_index=True)

        
        if not df_all.empty:
            dd = df_all.copy()
            o = dd["CU_ORIG"].notna() if "CU_ORIG" in dd.columns else False
            d = dd["CU_DL"].notna()   if "CU_DL"   in dd.columns else False
            dd["SOURCE"] = np.select(
                [o & d, o & ~d, ~o & d],
                ["overlap", "orig_only", "dl_only"],
                default="overlap"   
            )
            return dd
        return df_all 

    if source == "Overlap":   return df_overlap
    if source == "Orig-only": return df_origonly
    if source == "DL-only":   return df_dlonly
    return df_all


# 2) Value (only when source ≠ All). In "All" we color by SOURCE categories.
category_mode = (source == "All")
if not category_mode:
    VAL_CHOICES = {
        "Overlap":   [("DIFF (DL - ORIG)", "DIFF"),
                      ("DIFF % ((DL-ORIG)/DL*100)", "DIFF_PCT"),
                      ("CU_ORIG","CU_ORIG"), ("CU_DL","CU_DL")],
        "Orig-only": [("CU_ORIG","CU_ORIG")],
        "DL-only":   [("CU_DL","CU_DL")],
    }
    opts = VAL_CHOICES[source]
    labels  = [lab for lab, _ in opts]
    lab2col = dict(opts)
    prev = st.session_state.get("value_mode")
    if prev not in labels:
        prev = labels[0]
    value_mode = st.sidebar.selectbox("Value to display", labels, index=labels.index(prev), key="value_mode")
    value_col  = lab2col[value_mode]
else:
    value_mode = "Source categories (color)"
    value_col  = "__SOURCE__"

# 3) Filters (Depth is clamped to [0, 2500] m)
if df_union.empty:
    st.warning("No data found. Check CSV paths.")
    st.stop()

lon_min, lon_max = float(df_union["LONGITUDE"].min()), float(df_union["LONGITUDE"].max())
lat_min, lat_max = float(df_union["LATITUDE"].min()),  float(df_union["LATITUDE"].max())
DEPTH_MAX = 2500.0
depth_min = max(0.0, float(df_union["DEPTH"].min())) if "DEPTH" in df_union else 0.0

xr = st.sidebar.slider("Longitude", lon_min, lon_max, (lon_min, lon_max))
yr = st.sidebar.slider("Latitude",  lat_min, lat_max,  (lat_min, lat_max))
zr = st.sidebar.slider("Depth",     0.0,     DEPTH_MAX, (depth_min, DEPTH_MAX))


# 5) Overlays (Surface) — wired later when surface layers are ready

overlay_on = st.sidebar.toggle("Overlay surface", value=False)
if overlay_on:
    # Allow negative Z so the surface can "float" above the scene
    overlay_z = st.sidebar.slider("Overlay Z (m)", -500.0, 0.0, -200.0)
else:
    overlay_z = None
    overlay_opacity = None

# 6) Display (numeric values only): palette & clipping
if not category_mode:
    palette = "Portland"
    clip_mode    = st.sidebar.selectbox("Clip mode", ["Absolute", "Percentile"], index=1)
    max_abs_clip = 500.0
    pctl         = 95.0
    if clip_mode == "Absolute":
        max_abs_clip = st.sidebar.number_input("Max value to show", min_value=1.0, value=max_abs_clip, step=10.0)
    else:
        pctl = st.sidebar.slider("Clip range at percentile", min_value=50.0, max_value=99.9, value=95.0, step=0.1)
else:
    palette = None
    clip_mode = None
    pctl = 95.0
    max_abs_clip = 500.0



# --- Guards: ensure view-specific params exist before we filter base df ---
# (so filters that reference these won't crash if the user hasn't visited a tab yet)
point_size   = int(st.session_state.get("pts_size_main", 2))
max_points   = int(st.session_state.get("pts_cap_main", 100_000))
min_abs_diff = float(st.session_state.get("pts_min_abs_main", 0.0))

# Voxels defaults (used later when rendering voxels)
nx              = int(st.session_state.get("vox_nx", 50))
ny              = int(st.session_state.get("vox_ny", 50))
nz              = int(st.session_state.get("vox_nz", 30))
min_count_voxel = int(st.session_state.get("vox_min_cnt", 3))
agg_stat        = st.session_state.get("vox_agg", "mean")
voxel_opacity   = float(st.session_state.get("vox_opacity", 0.6))

# 2D slice defaults (used later when rendering 2D)
slice_mode     = st.session_state.get("slice_mode", "Single Z")
slice_z        = float(st.session_state.get("slice_z", 100.0))
slice_z0       = float(st.session_state.get("slice_z0", 50.0))
slice_z1       = float(st.session_state.get("slice_z1", 150.0))
slice_palette  = st.session_state.get("slice_palette", "Viridis")
slice_clip_mode= st.session_state.get("slice_clip_mode", "Percentile")
slice_max_abs  = float(st.session_state.get("slice_max_abs", 500.0))
slice_pctl     = float(st.session_state.get("slice_pctl", 95.0))

# ───────────────────────── Base dataframe under current filters ─────────────────
# Step 1: start from drillhole
base_raw = pick_source_df().copy()
base_raw["_is_surface"] = False  # mark drillhole rows

# Step 2: attach surface rows if overlay is enabled
if overlay_on and (overlay_z is not None):
    surf = get_surface_df_by_source(source)
    if not surf.empty:
        surf = surf.dropna(subset=["LONGITUDE","LATITUDE"]).copy()
        surf["DEPTH"] = float(overlay_z)  # pin floating surface Z
        surf["_is_surface"] = True
        base_raw = pd.concat([base_raw, surf], ignore_index=True, sort=False)

if base_raw.empty:
    st.warning(f"No rows in selected source: {source}")
    st.stop()

# Step 3: apply lon/lat filters (all rows must pass)
mask = (
    base_raw["LONGITUDE"].between(*xr)
    & base_raw["LATITUDE"].between(*yr)
)

# Step 4: depth filter & DIFF/DIFF% filter only affect drillhole rows
if "DEPTH" in base_raw.columns:
    
    dh = base_raw.loc[~base_raw["_is_surface"], "DEPTH"].astype(float)
    use_abs = (dh.dropna().quantile(0.9) <= 0)  
    depth_for_filter = dh.abs() if use_abs else dh

   
    depth_series = base_raw["DEPTH"].astype(float)
    depth_series.loc[~base_raw["_is_surface"]] = depth_for_filter

    mask &= base_raw["_is_surface"] | depth_series.between(*zr)

if (not category_mode) and value_col in ("DIFF","DIFF_PCT") and min_abs_diff > 0:
    mask &= base_raw["_is_surface"] | (base_raw[value_col].abs() >= min_abs_diff)

# Final filtered base
base = base_raw[mask].copy()

if base.empty or (value_col != "__SOURCE__" and value_col not in base.columns):
    st.warning("No data under current filters/value selection.")
    st.stop()
if (not category_mode) and value_col in ("DIFF","DIFF_PCT") and min_abs_diff > 0:
    base = base[base[value_col].abs() >= min_abs_diff]

if base.empty or (value_col != "__SOURCE__" and value_col not in base.columns):
    st.warning("No data under current filters/value selection.")
    st.stop()

# ───────────────────────── Color helpers ─────────────────────────
def pick_seq_scale(name: str):
    return [[0.00, "#000000"], [0.20, "#2b0000"], [0.40, "#7a0000"], [0.70, "#ff3b00"], [1.00, "#ffff66"]] \
        if name == "Hot (black→yellow)" else (name if name != "RdBu" else "Turbo")

def diverging_color_and_range(series: np.ndarray, is_abs_clip: bool, abs_max: float, p: float):
    if is_abs_clip: vmax = float(abs_max)
    else:
        vmax = float(np.nanpercentile(np.abs(series), p)) if len(series) else 1.0
        if vmax <= 0: vmax = 1.0
    return ("Portland", (-vmax, vmax))

def sequential_color_and_range(series: np.ndarray, is_abs_clip: bool, abs_max: float, p: float, name: str):
    if is_abs_clip: vmax = float(abs_max)
    else:
        vmax = float(np.nanpercentile(series, p)) if len(series) else 1.0
        if vmax <= 0: vmax = 1.0
    return (pick_seq_scale(name), (0.0, vmax))



# ───────────────────────── View tabs on the RIGHT ─────────────────────────
tab_points, tab_voxels, tab_slice = st.tabs(["3D Points", "3D Voxels", "2D Slice"])

# remember which tab user is on (so sidebar can “know” the active view if needed)
def _set_active_tab(name: str):
    st.session_state["agg_view_tab"] = name

with tab_points:
    _set_active_tab("3D Points")

    c1, c2, c3 = st.columns([1,1,1.2])
    with c1:
        point_size = st.slider("Point size", 1, 10,
                            int(st.session_state.get("pts_size_main", 2)),
                            key="pts_size_main")
    with c2:
        max_points = st.number_input("Max points", 1_000, 100_000,
                                    int(st.session_state.get("pts_cap_main", 100_000)), 1_000,
                                    key="pts_cap_main")
    with c3:
        min_abs_diff = 0.0
        if (not category_mode) and (value_col in ("DIFF","DIFF_PCT")):
            min_abs_diff = st.number_input("Min |value|", 0.0,
                                        value=float(st.session_state.get("pts_min_abs_main", 0.0)), step=1.0,
                                        key="pts_min_abs_main")

    view = base.dropna(subset=["LONGITUDE","LATITUDE","DEPTH"]).copy()
    if len(view) > max_points:
        is_surf = view["_is_surface"].fillna(False)
        surf = view[is_surf]
        dh   = view[~is_surf]

        
        keep_min_dh = min(20000, max_points // 3)

        
        n_surf, n_dh = len(surf), len(dh)
        if n_surf + n_dh == 0:
            view = view.sample(max_points, random_state=42)
        else:
            
            dh_quota   = int(max_points * (n_dh / (n_surf + n_dh)))
            surf_quota = max_points - dh_quota
            
            dh_quota   = max(1, max(keep_min_dh, min(dh_quota, len(dh))))
            surf_quota = max(1, max_points - dh_quota)
            surf_quota = min(surf_quota, len(surf))

            dh_keep   = dh.sample(dh_quota, random_state=42) if len(dh)   > dh_quota   else dh
            surf_keep = surf.sample(surf_quota, random_state=42) if len(surf) > surf_quota else surf
            view = pd.concat([dh_keep, surf_keep], ignore_index=True)



    if category_mode:

        view = view.copy()
        view["SOURCE"] = pd.Categorical(view["SOURCE"], categories=CAT_ORDER, ordered=True)

        fig = px.scatter_3d(
            view, x="LONGITUDE", y="LATITUDE", z="DEPTH",
            color="SOURCE",
            color_discrete_map=CAT_COLORS,   
            opacity=0.85
        )
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(
            scene=dict(zaxis=dict(autorange="reversed")),
            legend=dict(title="SOURCE", font=dict(size=18), itemsizing="constant"),
            height=800, margin=dict(l=0, r=0, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary (filtered)")
        g = (view["SOURCE"].value_counts(dropna=False)
            .rename_axis("Source").reset_index(name="Points"))
        total = int(g["Points"].sum())
        g["% of points"] = (100.0 * g["Points"] / total).round(2).astype(str) + "%"
        g["Source"] = pd.Categorical(g["Source"], categories=CAT_ORDER, ordered=True)
        g = g.sort_values("Source")
        st.dataframe(g[["Source","Points","% of points"]], use_container_width=True, hide_index=True)
    else:
        view = view.dropna(subset=[value_col]).copy()
        arr = view[value_col].to_numpy()
        if value_col == "DIFF_PCT":
            cs, (vmin, vmax) = diverging_color_and_range(arr, clip_mode=="Absolute", abs_max=max_abs_clip, p=pctl)
            ticksuf = "%"
        elif value_col == "DIFF":
            cs, (vmin, vmax) = diverging_color_and_range(arr, clip_mode=="Absolute", abs_max=max_abs_clip, p=pctl)
            ticksuf = ""
        else:
            cs, (vmin, vmax) = sequential_color_and_range(arr, clip_mode=="Absolute", abs_max=max_abs_clip, p=pctl, name=palette)
            ticksuf = ""
        fig = px.scatter_3d(
            view, x="LONGITUDE", y="LATITUDE", z="DEPTH",
            color=value_col, color_continuous_scale=cs, range_color=[vmin, vmax],
            opacity=0.85
        )
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(
            scene=dict(zaxis=dict(autorange="reversed")),
            coloraxis_colorbar=dict(title=value_mode, ticksuffix=ticksuf),
            height=800, margin=dict(l=0, r=0, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_voxels:
    _set_active_tab("3D Voxels")

    # --- 3D Voxels settings (compact 3-column layout) ---
    # Row 1: NX / NY / NZ
    vcol1, vcol2, vcol3 = st.columns(3)
    with vcol1:
        nx = st.slider(
            "NX (lon bins)", 8, 160,
            int(st.session_state.get("vox_nx", 50)), key="vox_nx"
        )
    with vcol2:
        ny = st.slider(
            "NY (lat bins)", 8, 160,
            int(st.session_state.get("vox_ny", 50)), key="vox_ny"
        )
    with vcol3:
        nz = st.slider(
            "NZ (depth bins)", 6, 100,
            int(st.session_state.get("vox_nz", 30)), key="vox_nz"
        )

    # Row 2: min count / aggregator / opacity
    vcol4, vcol5, vcol6 = st.columns(3)
    with vcol4:
        min_count_voxel = st.number_input(
            "Min samples/voxel", 1, 200,
            int(st.session_state.get("vox_min_cnt", 3)), 1, key="vox_min_cnt"
        )
    with vcol5:
        agg_stat = st.selectbox(
            "Aggregator", ["mean", "median"],
            index=0 if st.session_state.get("vox_agg","mean")=="mean" else 1,
            key="vox_agg"
        )
    with vcol6:
        voxel_opacity = st.slider(
            "Cube opacity", 0.1, 1.0,
            float(st.session_state.get("vox_opacity", 0.6)), 0.05, key="vox_opacity"
        )

    need = ["LONGITUDE","LATITUDE","DEPTH"]
    if not category_mode:
        need += [value_col]
    num = base.dropna(subset=need).copy()

    if not category_mode:
        num[value_col] = pd.to_numeric(num[value_col], errors="coerce")
        num = num.dropna(subset=[value_col])
    if num.empty:
        st.info("No numeric rows for voxels under current filters/value.")
    else:
        vcol = value_col if not category_mode else ("CU_DL" if "CU_DL" in num.columns else "DEPTH")
        vox = voxelize_points(num, nx=nx, ny=ny, nz=nz, value_col=vcol, agg=agg_stat, min_count=min_count_voxel)
        if vox is None:
            st.info("No voxels under current parameters. Try decreasing NX/NY/NZ or Min samples per voxel.")
        else:
            Xc, Yc, Zc, V = vox["Xc"], vox["Yc"], vox["Zc"], vox["V"]
            dx = float(np.diff(vox["x_edges"]).mean()); dy = float(np.diff(vox["y_edges"]).mean()); dz = float(np.diff(vox["z_edges"]).mean())

            if (not category_mode) and value_col in ("DIFF","DIFF_PCT"):
                if value_col == "DIFF_PCT":
                    cs, (vmin, vmax) = diverging_color_and_range(V, clip_mode=="Absolute", abs_max=max_abs_clip, p=pctl); ticksuf="%"
                else:
                    cs, (vmin, vmax) = diverging_color_and_range(V, clip_mode=="Absolute", abs_max=max_abs_clip, p=pctl); ticksuf=""
            else:
                cs, (vmin, vmax) = sequential_color_and_range(V, clip_mode=="Absolute", abs_max=max_abs_clip, p=pctl, name=(palette or "Viridis")); ticksuf=""

            trace = build_voxel_mesh(
                Xc, Yc, Zc, V,
                dx=dx, dy=dy, dz=dz,
                cmin=vmin, cmax=vmax,
                colorscale=cs,
                opacity=voxel_opacity,
                colorbar_title=(value_mode if not category_mode else vcol),
                ticksuffix=ticksuf,
                max_cubes=3000, pick_top_by_abs=True
            )
            figv = go.Figure(data=[trace])
            figv.update_layout(
                scene=dict(zaxis=dict(autorange="reversed")),
                height=800, margin=dict(l=0, r=0, t=50, b=10)
            )
            st.plotly_chart(figv, use_container_width=True)


with tab_slice:
    _set_active_tab("2D Slice")

    # Row with Slice orientation and Max points side by side
    c1, c2 = st.columns([2, 1])
    with c1:
        slice_mode = st.radio(
            "Slice orientation",
            ["XY (map)", "XZ (cross-section)", "YZ (cross-section)"],
            index=0,
            horizontal=True,
            key="slice_mode_main"
        )
    with c2:
        max_points_2d = st.number_input(
            "Max points (2D slice)",
            1_000,
            100_000,
            int(st.session_state.get("pts_cap_slice", 100_000)),
            1_000,
            key="pts_cap_slice"
        )

    # Prepare data
    view2d = base.dropna(subset=["LONGITUDE", "LATITUDE", "DEPTH"]).copy()
    if view2d.empty:
        st.info("No rows available for 2D slice.")
    else:
        if len(view2d) > max_points_2d:
            view2d = view2d.sample(max_points_2d, random_state=42)

        # --- XY Map slice ---
        if slice_mode == "XY (map)":
            st.subheader(f"XY Map slice (Depth {zr[0]}–{zr[1]} m)")
            if value_col == "__SOURCE__":
                fig2d = px.scatter(
                    view2d, x="LONGITUDE", y="LATITUDE", color="SOURCE",
                    color_discrete_map=CAT_COLORS, opacity=0.8,
                    hover_data=["DEPTH"], render_mode="webgl"
                )
            else:
                fig2d = px.scatter(
                    view2d, x="LONGITUDE", y="LATITUDE", color=value_col,
                    color_continuous_scale="Viridis", opacity=0.8,
                    hover_data=["DEPTH", value_col], render_mode="webgl"
                )
            st.plotly_chart(fig2d, use_container_width=True)

        # --- XZ Cross-section ---
        elif slice_mode == "XZ (cross-section)":
            st.subheader(f"XZ Cross-section (Latitude {yr[0]}–{yr[1]})")
            if value_col == "__SOURCE__":
                fig2d = px.scatter(
                    view2d, x="LONGITUDE", y="DEPTH", color="SOURCE",
                    color_discrete_map=CAT_COLORS, opacity=0.8,
                    hover_data=["LATITUDE"], render_mode="webgl"
                )
            else:
                fig2d = px.scatter(
                    view2d, x="LONGITUDE", y="DEPTH", color=value_col,
                    color_continuous_scale="Viridis", opacity=0.8,
                    hover_data=["LATITUDE", value_col], render_mode="webgl"
                )
            fig2d.update_yaxes(autorange="reversed", title="Depth (m)")
            st.plotly_chart(fig2d, use_container_width=True)

        # --- YZ Cross-section ---
        elif slice_mode == "YZ (cross-section)":
            st.subheader(f"YZ Cross-section (Longitude {xr[0]}–{xr[1]})")
            if value_col == "__SOURCE__":
                fig2d = px.scatter(
                    view2d, x="LATITUDE", y="DEPTH", color="SOURCE",
                    color_discrete_map=CAT_COLORS, opacity=0.8,
                    hover_data=["LONGITUDE"], render_mode="webgl"
                )
            else:
                fig2d = px.scatter(
                    view2d, x="LATITUDE", y="DEPTH", color=value_col,
                    color_continuous_scale="Viridis", opacity=0.8,
                    hover_data=["LONGITUDE", value_col], render_mode="webgl"
                )
            fig2d.update_yaxes(autorange="reversed", title="Depth (m)")
            st.plotly_chart(fig2d, use_container_width=True)



csv_buf = io.StringIO()
base.to_csv(csv_buf, index=False)
st.download_button(
    "Download filtered CSV",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="drillhole_aggregated_filtered.csv",
    mime="text/csv",
    type="primary"
)
