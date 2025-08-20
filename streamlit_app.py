# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Drillhole 3D Viewer")

# ----------------- Paths -----------------
POINTS_CSV = Path("reports/task2/drillhole_points.csv")

# ----------------- Custom palettes -----------------
# Hot-like sequential scale (black -> deep red -> orange -> yellow-white)
HOT_CS = [
    [0.00, "#000000"],
    [0.20, "#2b0000"],
    [0.40, "#7a0000"],
    [0.70, "#ff3b00"],
    [1.00, "#ffff66"],
]

# ----------------- Data loader -----------------
@st.cache_data
def load_points() -> pd.DataFrame:
    df = pd.read_csv(POINTS_CSV)
    for c in ["LONGITUDE", "LATITUDE", "DEPTH", "CU_ORIG", "CU_DL", "DIFF"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["LONGITUDE", "LATITUDE", "DEPTH"]).copy()

    # add DIFF_PCT = (DL - ORIG) / DL * 100, keep sign; guard divide-by-zero
    eps = 1e-9
    denom = df["CU_DL"].where(df["CU_DL"].abs() >= eps, np.nan)
    df["DIFF_PCT"] = 100.0 * df["DIFF"] / denom
    return df

# ----------------- Voxelization helper -----------------
def voxelize_points(
    df: pd.DataFrame,
    nx: int,
    ny: int,
    nz: int,
    value_col: str = "DIFF",
    agg: str = "mean",
    min_count: int = 1,
):
    """
    Bin points into a regular 3D grid over (LONGITUDE, LATITUDE, DEPTH).
    Return voxel centers (Xc, Yc, Zc), aggregated values V and bin edges.
    """
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
        agg_df = dfb.groupby("flat").agg(val=("val", "median"), cnt=("val", "size")).reset_index()
    else:
        agg_df = dfb.groupby("flat").agg(val=("val", "mean"), cnt=("val", "size")).reset_index()

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
    V = agg_df["val"].to_numpy()

    return {
        "Xc": Xc,
        "Yc": Yc,
        "Zc": Zc,
        "V": V,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
    }

# ----------------- Build voxel cubes (Mesh3d) -----------------
def build_voxel_mesh(
    Xc: np.ndarray,
    Yc: np.ndarray,
    Zc: np.ndarray,
    V:  np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    cmin: float,
    cmax: float,
    colorscale,
    opacity: float = 0.6,
    colorbar_title: str = "value",
    ticksuffix: str = "",
    max_cubes: int = 3000,
    pick_top_by_abs: bool = True,
) -> go.Mesh3d:
    """
    Build a single Mesh3d made of many axis-aligned cubes (voxels).
    Each cube is colored by its aggregated value, with adjustable opacity.
    """
    n = len(V)
    if n == 0:
        return go.Mesh3d()

    # keep at most max_cubes (performance)
    if n > max_cubes:
        if pick_top_by_abs:
            idx = np.argsort(np.abs(V))[::-1][:max_cubes]
        else:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_cubes, replace=False)
        Xc, Yc, Zc, V = Xc[idx], Yc[idx], Zc[idx], V[idx]
        n = len(V)

    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0

    local = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], dtype=float)
    local[:, 0] *= hx
    local[:, 1] *= hy
    local[:, 2] *= hz

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [0, 3, 7], [0, 7, 4],
    ], dtype=int)

    X = np.empty(n * 8, dtype=float)
    Y = np.empty(n * 8, dtype=float)
    Z = np.empty(n * 8, dtype=float)
    I = np.empty(n * 12, dtype=int)
    J = np.empty(n * 12, dtype=int)
    K = np.empty(n * 12, dtype=int)
    INT = np.empty(n * 8, dtype=float)

    for k in range(n):
        base_v = 8 * k
        cx, cy, cz = Xc[k], Yc[k], Zc[k]
        verts = local + np.array([cx, cy, cz])
        X[base_v:base_v+8] = verts[:, 0]
        Y[base_v:base_v+8] = verts[:, 1]
        Z[base_v:base_v+8] = verts[:, 2]
        INT[base_v:base_v+8] = V[k]

        base_f = 12 * k
        tri = faces + base_v
        I[base_f:base_f+12] = tri[:, 0]
        J[base_f:base_f+12] = tri[:, 1]
        K[base_f:base_f+12] = tri[:, 2]

    mesh = go.Mesh3d(
        x=X, y=Y, z=Z,
        i=I, j=J, k=K,
        intensity=INT,
        colorscale=colorscale,
        cmin=cmin, cmax=cmax,
        flatshading=True,
        opacity=opacity,
        showscale=True,
        colorbar=dict(title=colorbar_title, ticksuffix=ticksuffix),
        lighting=dict(ambient=0.5, diffuse=0.6, specular=0.0),
    )
    return mesh

# ----------------- XY slice aggregation (for map) -----------------
def aggregate_xy_slice(
    df: pd.DataFrame,
    value_col: str,
    depth_min: float,
    depth_max: float,
    grid_km: float = 1.0,       # grid size in km
    agg: str = "mean",          # mean/median/max/absmax
) -> pd.DataFrame:
    """
    Aggregate points within a depth window onto a lon/lat grid for mapping.
    Returns: lon, lat (cell centers), val (aggregated), count.
    """
    if df.empty:
        return pd.DataFrame(columns=["lon", "lat", "val", "count"])

    sub = df[(df["DEPTH"] >= depth_min) & (df["DEPTH"] <= depth_max)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["lon", "lat", "val", "count"])

    step_deg = max(grid_km / 111.0, 1e-5)
    sub["_gx"] = np.floor(sub["LONGITUDE"] / step_deg).astype(np.int64)
    sub["_gy"] = np.floor(sub["LATITUDE"]  / step_deg).astype(np.int64)

    if agg == "median":
        agg_map = {"val": (value_col, "median"), "count": (value_col, "size")}
    elif agg == "max":
        agg_map = {"val": (value_col, "max"), "count": (value_col, "size")}
    elif agg == "absmax":
        sub["_abs"] = sub[value_col].abs()
        agg_map = {"val": ("_abs", "max"), "count": ("_abs", "size")}
    else:
        agg_map = {"val": (value_col, "mean"), "count": (value_col, "size")}

    g = sub.groupby(["_gx", "_gy"], as_index=False).agg(**agg_map)
    g["lon"] = (g["_gx"] + 0.5) * step_deg
    g["lat"] = (g["_gy"] + 0.5) * step_deg
    return g[["lon", "lat", "val", "count"]]

# ----------------- App -----------------
df = load_points()
st.title("Drillhole 3D Viewer")
st.caption(f"Loaded: {POINTS_CSV} • rows={len(df):,}")

# ===== Sidebar: filters =====
st.sidebar.header("Filters")
xr = st.sidebar.slider(
    "Longitude",
    float(df["LONGITUDE"].min()), float(df["LONGITUDE"].max()),
    (float(df["LONGITUDE"].min()), float(df["LONGITUDE"].max()))
)
yr = st.sidebar.slider(
    "Latitude",
    float(df["LATITUDE"].min()), float(df["LATITUDE"].max()),
    (float(df["LATITUDE"].min()), float(df["LATITUDE"].max()))
)
zr = st.sidebar.slider(
    "Depth",
    float(df["DEPTH"].min()), float(df["DEPTH"].max()),
    (float(df["DEPTH"].min()), float(df["DEPTH"].max()))
)

# ===== What to visualize =====
st.sidebar.header("Value")
value_mode = st.sidebar.selectbox(
    "Value to display",
    ["DIFF (DL - ORIG)", "DIFF % ((DL-ORIG)/DL*100)", "CU_ORIG", "CU_DL"],
    index=0
)
if value_mode.startswith("DIFF %"):
    value_col = "DIFF_PCT"
elif value_mode.startswith("DIFF"):
    value_col = "DIFF"
else:
    value_col = "CU_ORIG" if value_mode == "CU_ORIG" else "CU_DL"

# ===== Sidebar: display options =====
st.sidebar.header("Display")
palette = st.sidebar.selectbox(
    "Palette",
    ["RdBu", "Portland", "Plasma", "Viridis", "Turbo", "Hot (black→yellow)"],
    index=1  # default: Portland
)

clip_mode = st.sidebar.selectbox("Clip mode", ["Absolute", "Percentile"], index=0)
if clip_mode == "Absolute":
    # For DIFF / DIFF % -> [-max, +max]; for CU_* -> [0, max]
    max_abs_clip = st.sidebar.number_input("Max value to show", min_value=1.0, value=500.0, step=10.0)
else:
    pctl = st.sidebar.slider("Clip range at percentile", 50.0, 99.9, 99.0, 0.1)

# Only meaningful for DIFF/DIFF%
if value_col in ("DIFF", "DIFF_PCT"):
    min_abs_diff = st.sidebar.number_input("Min |value| filter", 0.0, value=0.0, step=1.0)
else:
    min_abs_diff = 0.0

# Points controls
point_size = st.sidebar.slider("Marker size (points)", 1, 10, 2)
max_points = st.sidebar.number_input("Max points to render", 1_000, 200_000, 50_000, 1_000)

# Voxel controls
st.sidebar.subheader("Voxelization")
view_mode = st.sidebar.selectbox("View mode", ["Points", "Voxel", "Map (XY slice)"], index=0)
voxel_style = st.sidebar.radio("Voxel style", ["Dots", "Cubes"], index=0)
nx = st.sidebar.slider("NX (lon bins)", 8, 160, 50)
ny = st.sidebar.slider("NY (lat bins)", 8, 160, 50)
nz = st.sidebar.slider("NZ (depth bins)", 6, 100, 30)
min_count_voxel = st.sidebar.number_input("Min samples per voxel", 1, 200, 3, 1)
agg_stat = st.sidebar.selectbox("Voxel aggregator", ["mean", "median"], index=0)
voxel_dot_size = st.sidebar.slider("Voxel dot size", 2, 12, 5)
voxel_opacity = st.sidebar.slider("Voxel opacity", 0.1, 1.0, 0.6, 0.05)
show_dot_border = st.sidebar.checkbox("Dot border (black)", True)

# Map (XY slice) controls
st.sidebar.subheader("XY slice map")
slice_depth = st.sidebar.slider(
    "Depth window (min, max)",
    float(df["DEPTH"].min()),
    float(df["DEPTH"].max()),
    (float(df["DEPTH"].min()), float(df["DEPTH"].min()) + 50.0),
    step=1.0,
)
map_grid_km = st.sidebar.number_input("Grid size (km)", min_value=0.2, value=1.0, step=0.2)
map_agg = st.sidebar.selectbox("Aggregate", ["mean", "median", "max", "absmax"], index=0)
map_as_heat = st.sidebar.checkbox("Show as heatmap (density)", value=False)
map_point_size = st.sidebar.slider("Map point size", 4, 18, 9)

# ===== Filter base dataframe =====
base = df[
    df["LONGITUDE"].between(*xr)
    & df["LATITUDE"].between(*yr)
    & df["DEPTH"].between(*zr)
].copy()
if value_col in ("DIFF", "DIFF_PCT") and min_abs_diff > 0:
    base = base[base[value_col].abs() >= min_abs_diff]

# Helpers
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

# ===== Points mode =====
if view_mode == "Points":
    view = base.dropna(subset=[value_col]).copy()
    if len(view) > max_points:
        view = view.sample(max_points, random_state=42)

    if value_col in ("DIFF", "DIFF_PCT"):
        cs, (vmin, vmax) = diverging_color_and_range(view[value_col].to_numpy(), clip_mode == "Absolute")
    else:
        cs, (vmin, vmax) = sequential_color_and_range(view[value_col].to_numpy(), clip_mode == "Absolute")

    # colorbar config: add % suffix when DIFF_PCT
    colorbar_cfg = dict(title=value_mode)
    if value_col == "DIFF_PCT":
        colorbar_cfg["ticksuffix"] = "%"

    fig = px.scatter_3d(
        view,
        x="LONGITUDE", y="LATITUDE", z="DEPTH",
        color=value_col,
        color_continuous_scale=cs,
        range_color=[vmin, vmax],
        opacity=0.85,
        hover_data={
            "LONGITUDE": ":.5f",
            "LATITUDE": ":.5f",
            "DEPTH": ":.2f",
            "CU_ORIG": ":.2f",
            "CU_DL": ":.2f",
            "DIFF": ":.2f",
            "DIFF_PCT": ":.2f",
        },
    )
    fig.update_traces(marker=dict(size=point_size))
    fig.update_layout(
        scene=dict(zaxis=dict(autorange="reversed")),
        coloraxis_colorbar=colorbar_cfg,
        height=640,  # smaller (about 0.8 of 800)
        margin=dict(l=0, r=0, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats
    st.subheader("Stats (filtered)")
    st.metric("Points rendered", int(len(view)))
    if len(view):
        series = view[value_col].abs() if value_col in ("DIFF", "DIFF_PCT") else view[value_col]
        q = series.quantile([0.5, 0.9, 0.99]).rename("value")
        st.table(q.to_frame())
        st.write(f"Mean {value_mode}:", float(view[value_col].mean()))
    st.download_button(
        "Download filtered CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_points.csv",
        mime="text/csv",
    )

# ===== Voxel mode (Dots / Cubes) =====
elif view_mode == "Voxel":
    vox = voxelize_points(
        base.dropna(subset=[value_col]),
        nx=nx, ny=ny, nz=nz,
        value_col=value_col, agg=agg_stat, min_count=min_count_voxel
    )
    if vox is None:
        st.warning("No voxels under current filters/parameters. Try lowering NX/NY/NZ or Min samples per voxel.")
    else:
        Xc, Yc, Zc, V = vox["Xc"], vox["Yc"], vox["Zc"], vox["V"]
        x_edges, y_edges, z_edges = vox["x_edges"], vox["y_edges"], vox["z_edges"]
        dx = float(np.diff(x_edges).mean())
        dy = float(np.diff(y_edges).mean())
        dz = float(np.diff(z_edges).mean())

        if value_col in ("DIFF", "DIFF_PCT"):
            cs, (vmin, vmax) = diverging_color_and_range(V, clip_mode == "Absolute")
        else:
            cs, (vmin, vmax) = sequential_color_and_range(V, clip_mode == "Absolute")

        if voxel_style == "Dots":
            # add % suffix on colorbar when DIFF_PCT
            colorbar_cfg = dict(title=value_mode)
            if value_col == "DIFF_PCT":
                colorbar_cfg["ticksuffix"] = "%"

            line_kwargs = dict(color="rgba(0,0,0,0.65)", width=0.6) if show_dot_border else dict(width=0)
            trace = go.Scatter3d(
                x=Xc, y=Yc, z=Zc,
                mode="markers",
                marker=dict(
                    size=voxel_dot_size,
                    color=V,
                    cmin=vmin, cmax=vmax,
                    colorscale=cs,
                    opacity=voxel_opacity,
                    line=line_kwargs,
                    symbol="circle",
                    colorbar=colorbar_cfg,
                    showscale=True
                )
            )
        else:  # Cubes
            trace = build_voxel_mesh(
                Xc, Yc, Zc, V,
                dx=dx, dy=dy, dz=dz,
                cmin=vmin, cmax=vmax,
                colorscale=cs,
                opacity=voxel_opacity,
                colorbar_title=value_mode,
                ticksuffix="%" if value_col == "DIFF_PCT" else "",
                max_cubes=3000,
                pick_top_by_abs=True
            )

        vox_fig = go.Figure(data=[trace])
        vox_fig.update_layout(
            scene=dict(zaxis=dict(autorange="reversed")),
            height=640,  # smaller
            margin=dict(l=0, r=0, t=50, b=10)
        )
        st.plotly_chart(vox_fig, use_container_width=True)

        # Stats
        st.subheader("Voxel stats")
        stat_series = np.abs(V) if value_col in ("DIFF", "DIFF_PCT") else V
        q = pd.Series(stat_series).quantile([0.5, 0.9, 0.99]).rename("value")
        st.metric("Occupied voxels", int(len(V)))
        st.table(q.to_frame())
        st.download_button(
            "Download voxel centers CSV",
            pd.DataFrame({"Xc": Xc, "Yc": Yc, "Zc": Zc, "V": V})
            .to_csv(index=False).encode("utf-8"),
            file_name="voxels_centers.csv",
            mime="text/csv",
        )

# ===== Map (XY slice) =====
else:
    d0, d1 = slice_depth
    agg_df = aggregate_xy_slice(
        base.dropna(subset=[value_col]),
        value_col=value_col,
        depth_min=d0, depth_max=d1,
        grid_km=map_grid_km, agg=map_agg
    )

    if agg_df.empty:
        st.warning("No data in the chosen depth window / grid.")
    else:
        if value_col in ("DIFF", "DIFF_PCT"):
            cs, (vmin, vmax) = diverging_color_and_range(agg_df["val"].to_numpy(), clip_mode == "Absolute")
            cs_map = "RdBu" if palette == "RdBu" else palette
        else:
            cs, (vmin, vmax) = sequential_color_and_range(agg_df["val"].to_numpy(), clip_mode == "Absolute")
            cs_map = HOT_CS if palette == "Hot (black→yellow)" else (palette if palette != "RdBu" else "Turbo")

        center_lat = float(agg_df["lat"].mean())
        center_lon = float(agg_df["lon"].mean())

        colorbar_cfg = dict(title=value_mode)
        if value_col == "DIFF_PCT":
            colorbar_cfg["ticksuffix"] = "%"

        if map_as_heat:
            figm = px.density_mapbox(
                agg_df,
                lat="lat", lon="lon", z="val",
                radius=max(int(map_point_size*1.2), 3),
                center={"lat": center_lat, "lon": center_lon},
                zoom=7,
                color_continuous_scale=cs_map
            )
            figm.update_layout(coloraxis=dict(cmin=vmin, cmax=vmax))
        else:
            figm = px.scatter_mapbox(
                agg_df,
                lat="lat", lon="lon",
                color="val",
                size="count",
                size_max=map_point_size,
                color_continuous_scale=cs_map,
                hover_data={"val":":.2f","count":True,"lat":":.5f","lon":":.5f"},
                zoom=7,
            )
            figm.update_layout(coloraxis=dict(cmin=vmin, cmax=vmax))

        figm.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=50, b=10),
            height=640,  # match main chart size
            coloraxis_colorbar=colorbar_cfg
        )

        st.plotly_chart(figm, use_container_width=True)

        st.subheader("Slice stats")
        st.write(f"Depth window: {d0:.2f} – {d1:.2f}  •  Grid: {map_grid_km:.2f} km  •  Cells: {len(agg_df):,}")
        series = np.abs(agg_df["val"]) if value_col in ("DIFF", "DIFF_PCT") else agg_df["val"]
        st.table(series.quantile([0.5, 0.9, 0.99]).rename("value").to_frame())
        st.download_button(
            "Download slice CSV",
            agg_df.to_csv(index=False).encode("utf-8"),
            file_name="xy_slice.csv",
            mime="text/csv",
        )

# ======== Analytics (below main chart) ========
st.divider()
st.subheader("Analytics (linked to current filters)")

# Use filtered base for analytics (shared across modes)
analytics_df = base.dropna(subset=["DIFF", "DIFF_PCT", "CU_DL", "CU_ORIG"]).copy()
if analytics_df.empty:
    st.info("No data under current filters for analytics.")
else:
    t1, t2, t3, t4 = st.tabs(["Histogram", "DL vs ORIG / DIFF%", "Depth trend", "Top outliers"])

    # ---------- Tab 1: Histogram ----------
    with t1:
        cA, cB = st.columns(2)
        with cA:
            h1 = px.histogram(
                analytics_df, x="DIFF", nbins=60,
                title="Distribution of DIFF (DL - ORIG)",
                labels={"DIFF":"DIFF"}
            )
            h1.update_layout(bargap=0.02, margin=dict(l=0, r=0, t=40, b=10))
            st.plotly_chart(h1, use_container_width=True)
        with cB:
            h2 = px.histogram(
                analytics_df, x="DIFF_PCT", nbins=60,
                title="Distribution of DIFF % ((DL-ORIG)/DL*100)",
                labels={"DIFF_PCT":"DIFF %"}
            )
            h2.update_layout(
                bargap=0.02, margin=dict(l=0, r=0, t=40, b=10),
                xaxis=dict(ticksuffix="%")
            )
            st.plotly_chart(h2, use_container_width=True)

    # ---------- Tab 2: DL vs ORIG / DIFF% ----------
    with t2:
        cA, cB = st.columns(2)
        with cA:
            s1 = px.scatter(
                analytics_df.sample(min(len(analytics_df), 50_000), random_state=42),
                x="CU_ORIG", y="CU_DL",
                title="CU_DL vs CU_ORIG",
                labels={"CU_ORIG":"CU_ORIG", "CU_DL":"CU_DL"},
                opacity=0.6,
            )
            # 1:1 参考线
            lim = float(max(analytics_df["CU_ORIG"].max(), analytics_df["CU_DL"].max()))
            s1.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", width=1))
            s1.update_layout(margin=dict(l=0, r=0, t=40, b=10))
            st.plotly_chart(s1, use_container_width=True)
        with cB:
            s2 = px.scatter(
                analytics_df.sample(min(len(analytics_df), 50_000), random_state=42),
                x="CU_DL", y="DIFF_PCT",
                title="DIFF % vs CU_DL",
                labels={"CU_DL":"CU_DL", "DIFF_PCT":"DIFF %"},
                opacity=0.6
            )
            s2.update_layout(
                margin=dict(l=0, r=0, t=40, b=10),
                yaxis=dict(ticksuffix="%")
            )
            st.plotly_chart(s2, use_container_width=True)

    # ---------- Tab 3: Depth trend (binned) ----------
    with t3:
        # bin depth, compute median/mean of |DIFF_PCT|
        bins = st.slider("Depth bins", min_value=10, max_value=120, value=40, step=10)
        df_tr = analytics_df.copy()
        df_tr["_abs_pct"] = df_tr["DIFF_PCT"].abs()
        df_tr["_bin"] = pd.cut(df_tr["DEPTH"], bins=bins)
        g = df_tr.groupby("_bin").agg(
            depth_mid=("DEPTH", "median"),
            abs_pct_median=("_abs_pct", "median"),
            abs_pct_mean=("_abs_pct", "mean"),
            n=("DEPTH", "size")
        ).reset_index(drop=True)

        l1 = px.line(
            g, x="depth_mid", y=["abs_pct_median", "abs_pct_mean"],
            markers=True,
            labels={"value":"|DIFF %|", "depth_mid":"DEPTH"},
            title="Depth vs |DIFF %| (binned)"
        )
        l1.update_layout(
            margin=dict(l=0, r=0, t=40, b=10),
            yaxis=dict(ticksuffix="%")
        )
        st.plotly_chart(l1, use_container_width=True)
        st.caption(f"Bins: {bins}  •  Points: {len(analytics_df):,}")

    # ---------- Tab 4: Top outliers ----------
    with t4:
        mode = st.radio("Sort by", ["|DIFF %|", "|DIFF|"], horizontal=True)
        if mode == "|DIFF %|":
            top = analytics_df.assign(_abs=lambda d: d["DIFF_PCT"].abs()).nlargest(100, "_abs")
            cols = ["LONGITUDE","LATITUDE","DEPTH","CU_ORIG","CU_DL","DIFF","DIFF_PCT"]
        else:
            top = analytics_df.assign(_abs=lambda d: d["DIFF"].abs()).nlargest(100, "_abs")
            cols = ["LONGITUDE","LATITUDE","DEPTH","CU_ORIG","CU_DL","DIFF","DIFF_PCT"]
        st.dataframe(top[cols], use_container_width=True, height=360)