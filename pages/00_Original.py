

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from cap_common.config import load_cfg
from cap_original.points import build_points_views

# streamlit_four_raw_appfixed3.py
# Fixes:
# - Section surface hover now shows VALUE using `text` + hovertemplate (robust across Plotly versions).
# - Section grid default set to 1.0 km (min still 0.1 km).
# Other: numeric range inputs for spatial filters, overlay points removed, wording uses "Section".
import pandas as pd
import streamlit as st


from cap_common.config import load_cfg
from cap_original.points import build_points_views



# ---- Load data (CSV/Parquet → fallback build) ----



import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pyvista_ops as pvo

st.set_page_config(layout="wide", page_title="Four Datasets — Raw Points 3D Viewer")

# ---------- Paths ----------
cfg = load_cfg()
PATH_CANDIDATES = []
if cfg.original_points_all:
    base = cfg.original_points_all
    # prefer Parquet if present
    PATH_CANDIDATES += [base.with_suffix(".parquet"), base]

# fallbacks (old locations)
PATH_CANDIDATES += [
    Path("artifacts/raw_viewer/points_all.csv"),
    Path("reports/task2/points_all.csv"),
    Path("points_all.csv"),
    Path(r"H:\capstone project\capstone_task2_template\notebooks\artifacts\raw_viewer\points_all.csv"),
]

# ---------- Palettes ----------
HOT_BLACK_YELLOW = [
    [0.00, "#000000"], [0.05, "#190000"], [0.10, "#320000"], [0.20, "#2b0000"],
    [0.40, "#7a0000"], [0.70, "#ff3b00"], [1.00, "#ffff66"],
]
SEQ_PALETTES = ["Turbo", "Viridis", "Plasma", "Hot (black→yellow)"]
def pick_seq_scale(name: str):
    if name == "Plasma": return "Plasma"
    if name == "Viridis": return "Viridis"
    if name == "Hot (black→yellow)": return HOT_BLACK_YELLOW
    return "Turbo"

# ---------- Data loader ----------
@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def load_points_all(path: Path) -> pd.DataFrame:
    """Load points_all in a memory/IO-friendly way; returns LONGITUDE, LATITUDE, DEPTH, VALUE, SOURCE."""
    need = ["LONGITUDE", "LATITUDE", "DEPTH", "VALUE", "SOURCE"]
    if str(path).lower().endswith(".parquet"):
        df = pd.read_parquet(path, columns=need)
        df.columns = [c.upper() for c in df.columns]
    else:
        df = pd.read_csv(
            path,
            usecols=lambda c: str(c).strip().upper() in set(need + ["DLAT", "DLONG"]),  # read only needed cols
            low_memory=False
        )
        df.columns = [c.upper() for c in df.columns]
        if "DLAT" in df.columns:  df = df.rename(columns={"DLAT": "LATITUDE"})
        if "DLONG" in df.columns: df = df.rename(columns={"DLONG": "LONGITUDE"})

    # dtype tighten
    for c in ["LONGITUDE", "LATITUDE", "DEPTH", "VALUE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Required columns missing: {miss}")
    df = df.dropna(subset=need).copy()

    # normalize SOURCE variants
    df["SOURCE"] = df["SOURCE"].astype(str).str.upper()
    ren = {
        "DH-ORIG": "DH_ORIG", "DH-ORIGINAL": "DH_ORIG",
        "SURF-ORIG": "SURF_ORIG", "SURF-ORIGINAL": "SURF_ORIG",
        "DH-IMPUT": "DH_DL", "DH_IMPUT": "DH_DL", "DH PRED": "DH_DL",
        "SURF-IMPUT": "SURF_DL", "SURF_IMPUT": "SURF_DL", "SURF PRED": "SURF_DL",
    }
    df["SOURCE"] = df["SOURCE"].replace(ren)
    return df


def bounds(df: pd.DataFrame):
    return (
        float(df["LONGITUDE"].min()), float(df["LONGITUDE"].max()),
        float(df["LATITUDE"].min()),  float(df["LATITUDE"].max()),
        float(df["DEPTH"].min()),     float(df["DEPTH"].max()),
    )

def apply_spatial_filter(df, lon_rng, lat_rng, dep_rng):
    return df[
        (df["LONGITUDE"] >= lon_rng[0]) & (df["LONGITUDE"] <= lon_rng[1]) &
        (df["LATITUDE"]  >= lat_rng[0]) & (df["LATITUDE"]  <= lat_rng[1]) &
        (df["DEPTH"]     >= dep_rng[0]) & (df["DEPTH"]     <= dep_rng[1])
    ].copy()

@st.cache_data(show_spinner=False)
def voxelize_single(df: pd.DataFrame, nx: int, ny: int, nz: int, min_count: int, use_log_color: bool):
    """Voxelize a single subset by mean(VALUE)."""
    if df.empty:
        return None
    x = df["LONGITUDE"].to_numpy(); y = df["LATITUDE"].to_numpy()
    z = df["DEPTH"].to_numpy(); v = df["VALUE"].to_numpy()
    v_for_color = np.log10(np.clip(v, 1e-6, None)) if use_log_color else v
    x_edges = np.linspace(x.min(), x.max(), nx + 1)
    y_edges = np.linspace(y.min(), y.max(), ny + 1)
    z_edges = np.linspace(z.min(), z.max(), nz + 1)
    ix = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 1)
    iz = np.clip(np.searchsorted(z_edges, z, side="right") - 1, 0, nz - 1)
    flat = ix + nx * (iy + ny * iz)
    dfb = pd.DataFrame({"flat": flat, "raw": v, "disp": v_for_color})
    agg = dfb.groupby("flat").agg(raw=("raw", "mean"), disp=("disp", "mean"), cnt=("raw", "size")).reset_index()
    agg = agg[agg["cnt"] >= int(min_count)]
    if len(agg) == 0:
        return None
    flat_idx = agg["flat"].to_numpy()
    ix = flat_idx % nx; iy = (flat_idx // nx) % ny; iz = flat_idx // (nx * ny)
    Xc = (x_edges[ix] + x_edges[ix + 1]) / 2.0
    Yc = (y_edges[iy] + y_edges[iy + 1]) / 2.0
    Zc = (z_edges[iz] + z_edges[iz + 1]) / 2.0
    return Xc, Yc, Zc, agg["raw"].to_numpy(), agg["disp"].to_numpy(), agg["cnt"].to_numpy()

def estimate_delta(arr):
    if len(arr) < 2:
        return 1.0
    s = np.sort(arr); d = np.diff(s); d = d[d > 0]
    return float(np.nanmedian(d)) if len(d) else 1.0

# ---------- Sidebar: data source ----------
st.sidebar.header("Data source")
path_text = st.sidebar.text_input("points_all.csv path (optional)", "")
file = st.sidebar.file_uploader("...or upload points_all.csv", type=["csv"])

resolved_path = None
if file is not None:
    tmp = Path(st.session_state.get("_uploaded_points_all", "uploaded_points_all.csv"))
    tmp.write_bytes(file.getbuffer())
    resolved_path = tmp
elif path_text.strip():
    resolved_path = Path(path_text.strip())
else:
    for p in PATH_CANDIDATES:
        if p.exists():
            resolved_path = p; break

if not resolved_path or not Path(resolved_path).exists():
    st.warning("No points_all.csv found. Provide a path or upload the file.")
    st.stop()

df = load_points_all(Path(resolved_path))
st.sidebar.success(f"Loaded {len(df):,} rows from: {resolved_path}")

# ---------- Dataset selection ----------
st.sidebar.header("Dataset(s)")
sources_all = ["DH_ORIG", "DH_DL", "SURF_ORIG", "SURF_DL"]
present = [s for s in sources_all if s in set(df["SOURCE"].unique())]
if len(present) == 0:
    st.error("No recognized SOURCE values found."); st.stop()


choice = st.sidebar.radio(
    "View which dataset(s)?",
    ["Single", "All (overlay)", "Drillhole (2)", "Surface (2)"],
    horizontal=True, index=0
)

mode = choice

if choice == "Single":
    chosen = [st.sidebar.selectbox("Dataset", present, index=0)]
elif choice == "Drillhole (2)":

    chosen = [s for s in ["DH_ORIG", "DH_DL"] if s in present]
elif choice == "Surface (2)":
    chosen = [s for s in ["SURF_ORIG", "SURF_DL"] if s in present]
else:  # "All (overlay)"
    chosen = present

if not chosen:
    st.warning("Selected group has no datasets present in the current file/filters.")
    st.stop()

# ---------- Spatial filters: numeric range inputs ----------
lon_min, lon_max, lat_min, lat_max, dep_min, dep_max = bounds(df)
st.sidebar.header("Spatial filters")
c1, c2 = st.sidebar.columns(2)
lon_from = c1.number_input("Lon from", value=float(lon_min), format="%.6f", key="lon_from")
lon_to   = c2.number_input("Lon to",   value=float(lon_max), format="%.6f", key="lon_to")
lat_from = c1.number_input("Lat from", value=float(lat_min), format="%.6f", key="lat_from")
lat_to   = c2.number_input("Lat to",   value=float(lat_max), format="%.6f", key="lat_to")
dep_from = c1.number_input("Depth from (m)", value=float(dep_min), step=1.0, key="dep_from")
dep_to   = c2.number_input("Depth to (m)",   value=float(dep_max), step=1.0, key="dep_to")
if lon_from > lon_to: lon_from, lon_to = lon_to, lon_from
if lat_from > lat_to: lat_from, lat_to = lat_to, lat_from
if dep_from > dep_to: dep_from, dep_to = dep_to, dep_from
lon_rng = (float(lon_from), float(lon_to))
lat_rng = (float(lat_from), float(lat_to))
dep_rng = (float(dep_from), float(dep_to))

# ---------- Display & mode ----------
st.sidebar.header("Display & color")
palette = st.sidebar.selectbox("Sequential palette", SEQ_PALETTES, index=0)
clip_mode = st.sidebar.selectbox("Clip mode", ["Percentile", "Absolute"], index=0)
pctl = st.sidebar.slider("Percentile (for clip)", 80.0, 99.9, 99.0, 0.1)
abs_max = st.sidebar.number_input("Max value (Absolute clip)", min_value=1.0, value=500.0, step=10.0)
use_log_color = st.sidebar.checkbox("Use log10(value) for color only", value=True)

st.sidebar.header("View mode")
view_mode = st.sidebar.radio("Mode", ["Voxel", "Points", "Map (XY slice)", "Section"],
                             horizontal=True, index=1)   # default = Points

# ---------- Points params ----------
st.sidebar.subheader("Points params")
max_points = st.sidebar.number_input("Max points to plot (per dataset)",
                                     min_value=1_000, value=30_000, step=5_000)
point_size = st.sidebar.slider("Point size", 1, 8, 3)

# ---------- Voxel params ----------
st.sidebar.subheader("Voxel params")


mid_lat = 0.5 * (lat_rng[0] + lat_rng[1])
lon_m_per_deg = 111000.0 * max(np.cos(np.deg2rad(mid_lat)), 1e-6)
lat_m_per_deg = 111000.0
dx_deg = 10.0 / lon_m_per_deg   # 10 m in degrees (lon)
dy_deg = 10.0 / lat_m_per_deg   # 10 m in degrees (lat)
dz_m  = 1.0                     # 1 m (depth)
span_x_deg = max(lon_rng[1] - lon_rng[0], dx_deg)
span_y_deg = max(lat_rng[1] - lat_rng[0], dy_deg)
span_z_m   = max(dep_rng[1] - dep_rng[0], dz_m)
auto_nx = int(np.clip(round(span_x_deg / dx_deg), 5, 120))
auto_ny = int(np.clip(round(span_y_deg / dy_deg), 5, 120))
auto_nz = int(np.clip(round(span_z_m   / dz_m ), 5, 120))


nx = st.sidebar.slider("nx", 5, 120, auto_nx)
ny = st.sidebar.slider("ny", 5, 120, auto_ny)
nz = st.sidebar.slider("nz", 5, 120, auto_nz)
min_count = st.sidebar.number_input("Min samples per voxel", min_value=1, value=3, step=1)  
vox_style = st.sidebar.selectbox("Style", ["Cubes (mesh3d)", "Dots (centers)"], index=0)
max_cubes = st.sidebar.number_input("Max cubes (total)", min_value=500, value=3000, step=500)

# ---------- Map params ----------
st.sidebar.subheader("Map slice params")
map_depth_window = st.sidebar.slider("Depth window", float(dep_min), float(dep_max),
                                     (float(dep_min), float(dep_max)), key="map_depth_win")
map_grid_km = st.sidebar.number_input(
    "Grid size (km)", min_value=0.01, value=0.01, step=0.01, key="map_grid_km"
)
map_agg = st.sidebar.selectbox("Aggregation", ["mean", "median", "max", "absmax"], index=0, key="map_agg")
map_layer = st.sidebar.selectbox("Layer", ["scatter", "density"], index=0, key="map_layer")

# ---------- Section (horizontal depth slice only) ----------
st.sidebar.subheader("Section — horizontal depth slice")
sec_depth_m = st.sidebar.number_input("Depth (meters, same as DEPTH)", value=float(np.nanmedian(df["DEPTH"])), step=1.0, key="sec_depth_m")
sec_half_m  = st.sidebar.number_input("Half window (meters)", min_value=0.0, value=5.0, step=1.0, key="sec_half_m")
sec_grid_km = st.sidebar.number_input(
    "XY grid size (km)", min_value=0.1, value=0.5, step=0.05, key="sec_grid_km"
)
sec_agg     = st.sidebar.selectbox("Aggregation (slice)", ["mean", "median", "max", "absmax"], index=0, key="sec_agg")
sec_min_cnt = st.sidebar.number_input("Min samples per cell (slice)", min_value=1, value=3, step=1, key="sec_min_cnt")
sec_topdown  = st.sidebar.checkbox("Top-down orthographic view", value=True, key="sec_topdown")

# ---------- Filter once ----------
df = apply_spatial_filter(df[df["SOURCE"].isin(chosen)], lon_rng, lat_rng, dep_rng)
if df.empty:
    st.warning("No data after filters."); st.stop()

cs = pick_seq_scale(palette)
is_surface = lambda name: str(name).upper().startswith("SURF")

# ---------- Rendering ----------
if view_mode == "Points":
    traces = []
    if clip_mode == "Percentile":
        vmax_all = float(np.nanpercentile(df["VALUE"], pctl)) if len(df) else 1.0
        vmax_all = 1.0 if (not np.isfinite(vmax_all) or vmax_all <= 0) else vmax_all
        vmin_all, vmax_all = 0.0, vmax_all
    else:
        vmin_all, vmax_all = 0.0, float(abs_max)
    for s in chosen:
        sub = df[df["SOURCE"] == s].copy()
        if len(sub) > max_points:  # random sample only if exceeding the cap
            sub = sub.sample(max_points, random_state=42)
        disp_val = np.log10(np.clip(sub["VALUE"].to_numpy(), 1e-6, None)) if use_log_color else sub["VALUE"].to_numpy()
        z_vals = np.zeros(len(sub)) if is_surface(s) else sub["DEPTH"].to_numpy()
        traces.append(go.Scatter3d(
            x=sub["LONGITUDE"], y=sub["LATITUDE"], z=z_vals, mode="markers",
            marker=dict(size=point_size, color=disp_val, colorscale=cs,
                        cmin=0.0, cmax=np.log10(vmax_all) if use_log_color else vmax_all,
                        opacity=0.85, colorbar=dict(title="VALUE") if s == chosen[0] else None),
            name=s,
            hovertemplate=("SRC=%{meta}<br>Lon=%{x:.5f}<br>Lat=%{y:.5f}"
                           "<br>Depth=%{z:.2f}<br>VALUE=%{customdata:.4g}<extra></extra>"),
            customdata=sub["VALUE"], meta=s
        ))
    fig = go.Figure(traces)
    fig.update_layout(scene=dict(zaxis=dict(autorange="reversed"),
                                 xaxis_title="Lon", yaxis_title="Lat", zaxis_title="Depth"),
                      height=780, margin=dict(l=0, r=0, t=40, b=10))
    if mode == "All (overlay)":
        fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Voxel":
    if clip_mode == "Percentile":
        vmax_all = float(np.nanpercentile(df["VALUE"], pctl)) if len(df) else 1.0
        vmax_all = 1.0 if (not np.isfinite(vmax_all) or vmax_all <= 0) else vmax_all
        cmin, cmax = (0.0, np.log10(vmax_all)) if use_log_color else (0.0, vmax_all)
    else:
        cmin, cmax = (0.0, np.log10(abs_max)) if use_log_color else (0.0, float(abs_max))
    traces = []
    for s in chosen:
        sub = df[df["SOURCE"] == s]
        nz_eff = 1 if is_surface(s) else nz
        vox = voxelize_single(sub, nx, ny, nz_eff, int(min_count), use_log_color)
        if vox is None: continue
        Xc, Yc, Zc, Vraw, Vdisp, Cnt = vox
        if is_surface(s): Zc = np.zeros_like(Zc)
        traces.append(go.Scatter3d(
            x=Xc, y=Yc, z=Zc, mode="markers",
            marker=dict(size=point_size + 1, color=Vdisp, colorscale=cs, cmin=cmin, cmax=cmax,
                        opacity=0.9, colorbar=dict(title="VALUE") if s == chosen[0] else None),
            name=f"{s} vox({len(Xc):,})",
            hovertemplate=("SRC=%{meta}<br>Lon=%{x:.5f}<br>Lat=%{y:.5f}"
                           "<br>Depth=%{z:.2f}<br>VALUE=%{customdata[0]:.4g}"
                           "<br>count=%{customdata[1]:,}<extra></extra>"),
            customdata=np.column_stack([Vraw, Cnt]), meta=s
        ))
    if not traces:
        st.warning("No voxels created.")
    else:
        fig = go.Figure(traces)
        fig.update_layout(scene=dict(zaxis=dict(autorange="reversed")),
                          height=720, margin=dict(l=0, r=0, t=40, b=10))
        if mode == "All (overlay)":
            fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Section":
    # Points in depth window just to compute tight bounds
    df_slice_pts = df[(df["DEPTH"] >= sec_depth_m - sec_half_m) &
                      (df["DEPTH"] <= sec_depth_m + sec_half_m)].copy()
    if df_slice_pts.empty:
        st.warning("No points in the chosen depth window. Increase half-window or adjust depth.")
        st.stop()

    # Tight bounds with small padding
    pad_lon = max(0.001, 0.02 * (df_slice_pts["LONGITUDE"].max() - df_slice_pts["LONGITUDE"].min()))
    pad_lat = max(0.001, 0.02 * (df_slice_pts["LATITUDE"].max()  - df_slice_pts["LATITUDE"].min()))
    bounds_auto = (float(df_slice_pts["LONGITUDE"].min() - pad_lon),
                   float(df_slice_pts["LATITUDE"].min()  - pad_lat),
                   float(df_slice_pts["LONGITUDE"].max() + pad_lon),
                   float(df_slice_pts["LATITUDE"].max()  + pad_lat))
    mid_lat = 0.5 * (bounds_auto[1] + bounds_auto[3])
    lon_deg = (sec_grid_km * 1000.0) / (111000.0 * max(np.cos(np.deg2rad(mid_lat)), 1e-6))
    lat_deg = (sec_grid_km * 1000.0) / 111000.0
    span_lon = max(bounds_auto[2] - bounds_auto[0], lon_deg)
    span_lat = max(bounds_auto[3] - bounds_auto[1], lat_deg)
    nx_est = int(np.ceil(span_lon / lon_deg))
    ny_est = int(np.ceil(span_lat / lat_deg))

    SEC_MAX_CELLS = 80_000  # keep triangles ~ < 240k
    if nx_est * ny_est > SEC_MAX_CELLS:
        scale = float(np.sqrt((nx_est * ny_est) / SEC_MAX_CELLS))
        grid_km_eff = float(sec_grid_km * scale)   # auto-coarsen
    else:
        grid_km_eff = float(sec_grid_km)

    # Ensure slice thickness is not thinner than ~half XY cell (helps fill cells)
    sec_half_m_eff = max(float(sec_half_m), 0.5 * grid_km_eff * 1000.0)

    # (optional) when grid is very fine, allow fewer samples per cell
    sec_min_cnt_eff = int(1 if grid_km_eff <= 0.02 else sec_min_cnt)

    # small HUD
    st.caption(f"Section grid: requested={sec_grid_km:.3f} km, effective={grid_km_eff:.3f} km; "
               f"half-window={sec_half_m_eff:.1f} m; min_count={sec_min_cnt_eff}")

    # Build horizontal slice
    out = pvo.make_horizontal_slice(
        df_slice_pts,
        depth_value=float(sec_depth_m),
        half_window=sec_half_m_eff,                # 
        grid_km=grid_km_eff,                       # 
        agg=sec_agg, use_log_color=use_log_color,
        lonlat_bounds=bounds_auto,
        min_count=sec_min_cnt_eff                  # 
    )
    if out["nx"] == 0 or out["ny"] == 0:
        st.warning("No cells in horizontal slice."); st.stop()

    # Color limits from valid cells
    vr = out["val_raw_2d"]
    finite = np.isfinite(vr)
    if np.any(finite):
        vals = vr[finite]
        if clip_mode == "Percentile":
            vmax = float(np.nanpercentile(vals, pctl))
            vmax = 1.0 if (not np.isfinite(vmax) or vmax <= 0) else vmax
            cmin, cmax = (0.0, np.log10(vmax)) if use_log_color else (0.0, vmax)
        else:
            cmin, cmax = (0.0, np.log10(abs_max)) if use_log_color else (0.0, float(abs_max))
    else:
        cmin, cmax = (0.0, 1.0)

    # --- prepare 2D arrays for Surface hover (strict dtypes & same shapes) ---
    # --- build 2D arrays (strict shapes & dtypes) ---
    Z2d         = np.asarray(out["Z2d"],         dtype=float)   # (ny, nx)
    val_raw_2d  = np.asarray(out["val_raw_2d"],  dtype=float)   # (ny, nx)
    val_disp_2d = np.asarray(out["val_disp_2d"], dtype=float)
    X2d         = np.asarray(out["X2d"],         dtype=float)
    Y2d         = np.asarray(out["Y2d"],         dtype=float)

    # mask no-data cells on Z (NaN -> no surface there)
    finite_val = np.isfinite(val_raw_2d)
    Z2d_masked = Z2d.copy()
    Z2d_masked[~finite_val] = np.nan

    # --- precompose hover text WITHOUT %{...} ---
    lon_txt = np.char.mod('Lon=%.5f', X2d)
    lat_txt = np.char.mod('Lat=%.5f', Y2d)
    dep_txt = np.where(np.isfinite(Z2d_masked), np.char.mod('Depth=%.2f', Z2d_masked), 'Depth=')
    val_txt = np.where(finite_val, np.char.mod('VALUE=%.4g', val_raw_2d), 'VALUE=')

    # use numpy's char ops (np.char.add) — do NOT use '+' between numpy string arrays
    t1 = np.char.add(lon_txt, '<br>')
    t2 = np.char.add(t1,     lat_txt)
    t3 = np.char.add(t2,     '<br>')
    t4 = np.char.add(t3,     dep_txt)
    t5 = np.char.add(t4,     '<br>')
    hover_txt = np.char.add(t5, val_txt)  # shape == (ny, nx), dtype unicode

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X2d, y=Y2d, z=Z2d_masked,
        surfacecolor=val_disp_2d,
        colorscale=cs, cmin=cmin, cmax=cmax,
        text=hover_txt,
        hoverinfo="text",         
        opacity=0.85, showscale=True, name="Depth slice",
    ))



    # Camera
    d0 = float(out["meta"]["depth"])
    dz = max(20.0, 0.05 * (dep_max - dep_min))
    cam = dict(eye=dict(x=0.0, y=0.0, z=2.5),
               up=dict(x=0, y=1, z=0),
               projection=dict(type="orthographic")) if sec_topdown else None

    fig.update_layout(
        scene=dict(zaxis=dict(autorange=False, range=[d0 + dz, d0 - dz]),
                   xaxis_title="Lon", yaxis_title="Lat", zaxis_title="Depth"),
        scene_camera=cam,
        height=780, margin=dict(l=0, r=0, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # Map view unchanged (grid_km min 0.1, default 1.0)
    d0, d1 = map_depth_window
    mid_lat = 0.5 * (lat_rng[0] + lat_rng[1])
    lon_deg = (map_grid_km * 1000.0) / (111000.0 * max(np.cos(np.deg2rad(mid_lat)), 1e-6))
    lat_deg = (map_grid_km * 1000.0) / 111000.0
    all_vals = []
    for s in chosen:
        sub = df[(df["SOURCE"] == s) & (df["DEPTH"] >= d0) & (df["DEPTH"] <= d1)].copy()
        if sub.empty: continue
        gx = np.floor(sub["LONGITUDE"] / lon_deg) * lon_deg
        gy = np.floor(sub["LATITUDE"]  / lat_deg) * lat_deg
        sub["_gx"] = gx; sub["_gy"] = gy
        if map_agg == "absmax":
            grp = sub.groupby(["_gx", "_gy"])["VALUE"].agg(lambda a: a.iloc[np.argmax(np.abs(a))])
        elif map_agg == "median":
            grp = sub.groupby(["_gx", "_gy"])["VALUE"].median()
        elif map_agg == "max":
            grp = sub.groupby(["_gx", "_gy"])["VALUE"].max()
        else:
            grp = sub.groupby(["_gx", "_gy"])["VALUE"].mean()
        cnt = sub.groupby(["_gx", "_gy"])["VALUE"].size()
        agg = pd.DataFrame({
            "LONGITUDE": grp.index.get_level_values(0),
            "LATITUDE":  grp.index.get_level_values(1),
            "val": grp.values, "count": cnt.values, "SOURCE": s
        })
        all_vals.append(agg)

    if len(all_vals) == 0:
        st.warning("Empty slice. Try widening the depth window or changing dataset(s).")
    else:
        agg_df = pd.concat(all_vals, ignore_index=True)
        if clip_mode == "Percentile":
            vmax = float(np.nanpercentile(agg_df["val"], pctl))
            vmax = 1.0 if (not np.isfinite(vmax) or vmax <= 0) else vmax
            vmin, vmax = 0.0, vmax
        else:
            vmin, vmax = 0.0, float(abs_max)

        lon0, lon1 = float(agg_df["LONGITUDE"].min()), float(agg_df["LONGITUDE"].max())
        lat0, lat1 = float(agg_df["LATITUDE"].min()),  float(agg_df["LATITUDE"].max())
        center_init = {"lon": (lon0 + lon1) / 2.0, "lat": (lat0 + lat1) / 2.0}
        span = max(lon1 - lon0, lat1 - lat0, 1e-6)
        zoom_init = float(np.clip(3 + np.log2(360.0 / span), 1, 12))

        figm = px.scatter_mapbox(
            agg_df, lat="LATITUDE", lon="LONGITUDE", color="val",
            size="count", size_max=28, opacity=0.85,
            color_continuous_scale=pick_seq_scale(palette), range_color=[vmin, vmax],
            height=720, hover_name="SOURCE",
        )
        figm.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=center_init, zoom=zoom_init),
            margin=dict(l=0, r=0, t=40, b=10),
            uirevision="xy-map-viewport",
        )
        if mode == "All (overlay)":
            figm.update_layout(showlegend=False)
        st.plotly_chart(figm, use_container_width=True,
                        config={"scrollZoom": True, "doubleClick": "reset", "displaylogo": False})

# ---------- Footer ----------
with st.expander("Quick stats", expanded=False):
    st.write("Datasets present:", sorted(df["SOURCE"].unique().tolist()))
    st.write("Value percentiles (overall after filters):",
             pd.Series(df["VALUE"]).quantile([0.5, 0.9, 0.99]))
