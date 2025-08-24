# 03_Diff_Drillhole.py — multi-source (All / Overlap / Orig-only / DL-only)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Drillhole Differences (3D)")

# ----------------- Paths (difference outputs) -----------------
DIFF_DIR = Path("reports/task2/difference")
PATH_ALL       = DIFF_DIR / "drillhole_points_all.csv"
PATH_OVERLAP   = DIFF_DIR / "drillhole_points_overlap.csv"
PATH_ORIGONLY  = DIFF_DIR / "drillhole_points_origonly.csv"
PATH_DLONLY    = DIFF_DIR / "drillhole_points_dlonly.csv"

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
df_all      = safe_read_csv(PATH_ALL)
df_overlap  = safe_read_csv(PATH_OVERLAP)
df_origonly = safe_read_csv(PATH_ORIGONLY)
df_dlonly   = safe_read_csv(PATH_DLONLY)
df_union    = pd.concat([df_all, df_overlap, df_origonly, df_dlonly], ignore_index=True)

st.title("Drillhole 3D Viewer — Differences")

# 1) Data source 
st.sidebar.header("Data source")
source = st.sidebar.radio(
    "Source", ["All", "Overlap", "Orig-only", "DL-only"],
    index=0, horizontal=True, key="src_mode"
)
def pick_source_df():
    if source == "Overlap":   return df_overlap
    if source == "Orig-only": return df_origonly
    if source == "DL-only":   return df_dlonly
    return df_all

# 2) Value 
category_mode = (source == "All")
if not category_mode:
    st.sidebar.header("Value")
    VAL_CHOICES = {
        "Overlap":   [("DIFF (DL - ORIG)", "DIFF"),
                      ("DIFF % ((DL-ORIG)/DL*100)", "DIFF_PCT"),
                      ("CU_ORIG","CU_ORIG"), ("CU_DL","CU_DL")],
        "Orig-only": [("CU_ORIG","CU_ORIG")],
        "DL-only":   [("CU_DL","CU_DL")],
    }
    opts = VAL_CHOICES[source]
    labels = [lab for lab, _ in opts]
    lab2col = dict(opts)
    prev = st.session_state.get("value_mode")
    if prev not in labels: prev = labels[0]
    value_mode = st.sidebar.selectbox("Value to display", labels, index=labels.index(prev), key="value_mode")
    value_col = lab2col[value_mode]
else:
    # In All mode we color by SOURCE categories only
    value_mode = "Source categories (color)"
    value_col = "__SOURCE__"

# 3) View 
if category_mode:
    view_mode = "Points"               # All is always Points
    point_size = st.sidebar.slider("Point size", 1, 10, 2)
    max_points = st.sidebar.number_input("Max points to render", 1_000, 300_000, 80_000, 1_000)
else:
    st.sidebar.header("View")
    view_mode = st.sidebar.radio("View mode", ["Points", "Cubes"], index=0, horizontal=True)
    if view_mode == "Points":
        point_size = st.sidebar.slider("Point size", 1, 10, 2)
        max_points = st.sidebar.number_input("Max points to render", 1_000, 300_000, 80_000, 1_000)
    else:
        nx = st.sidebar.slider("NX (lon bins)", 8, 160, 50)
        ny = st.sidebar.slider("NY (lat bins)", 8, 160, 50)
        nz = st.sidebar.slider("NZ (depth bins)", 6, 100, 30)
        min_count_voxel = st.sidebar.number_input("Min samples per voxel", 1, 200, 3, 1)
        agg_stat = st.sidebar.selectbox("Voxel aggregator", ["mean", "median"], index=0)
        voxel_opacity = st.sidebar.slider("Cube opacity", 0.1, 1.0, 0.6, 0.05)

# Filters (Depth upper bound fixed to 2500 m)
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
zr = st.sidebar.slider(
    "Depth",
    float(df_union["DEPTH"].min()), 2500.0,
    (float(df_union["DEPTH"].min()), 2500.0)
)

# Numeric-only extras
if not category_mode and value_col in ("DIFF","DIFF_PCT"):
    min_abs_diff = st.sidebar.number_input("Min |value| filter", 0.0, value=0.0, step=1.0)
else:
    min_abs_diff = 0.0

# ----- Display (numeric) -----
if not category_mode:
    st.sidebar.subheader("Display (numeric)")

    palette = st.sidebar.selectbox(
        "Palette",
        ["RdBu", "Portland", "Plasma", "Viridis", "Turbo", "Hot (black→yellow)"],
        index=1,
    )

    # defaults so variables always exist
    pctl = 95.0
    max_abs_clip = 500.0

    # default to Percentile
    clip_mode = st.sidebar.selectbox(
        "Clip mode",
        ["Absolute", "Percentile"],
        index=1
    )

    if clip_mode == "Absolute":
        max_abs_clip = st.sidebar.number_input(
            "Max value to show",
            min_value=1.0, value=max_abs_clip, step=10.0
        )
    else:
        pctl = st.sidebar.slider(
            "Clip range at percentile",
            min_value=50.0, max_value=99.9, value=95.0, step=0.1
        )
else:
    palette = None
    clip_mode = None
    # also keep placeholders so later code never crashes
    pctl = 95.0
    max_abs_clip = 500.0    

# Prepare base dataframe

base = pick_source_df().copy()
if base.empty:
    st.warning(f"No rows in selected source: {source}"); st.stop()

base = base[
    base["LONGITUDE"].between(*xr)
    & base["LATITUDE"].between(*yr)
    & base["DEPTH"].between(*zr)
].copy()
if (not category_mode) and value_col in ("DIFF","DIFF_PCT") and min_abs_diff > 0:
    base = base[base[value_col].abs() >= min_abs_diff]

if base.empty or (value_col != "__SOURCE__" and value_col not in base.columns):
    st.warning("No data available under current filters / value selection."); st.stop()

# Color helpers
def pick_seq_scale():
    return HOT_CS if palette == "Hot (black→yellow)" else (palette if palette != "RdBu" else "Turbo")
def diverging_color_and_range(series: np.ndarray, is_abs_clip: bool):
    if is_abs_clip: vmax = float(max_abs_clip)
    else:
        vmax = float(np.nanpercentile(np.abs(series), pctl)) if len(series) else 1.0
        if vmax <= 0: vmax = 1.0
    return ("RdBu" if palette == "RdBu" else palette), (-vmax, vmax)
def sequential_color_and_range(series: np.ndarray, is_abs_clip: bool):
    if is_abs_clip: vmax = float(max_abs_clip)
    else:
        vmax = float(np.nanpercentile(series, pctl)) if len(series) else 1.0
        if vmax <= 0: vmax = 1.0
    return (pick_seq_scale(), (0.0, vmax))

# Method note for All/category mode
if category_mode:
    st.markdown(
        """
**How these points were built (overview)**  
- **3D grid:** `10 m (X) × 10 m (Y) × 1 m (Z)` over (LONGITUDE, LATITUDE, DEPTH).  
- For each grid cell, we mark presence by source: `overlap`, `orig_only`, `dl_only`.  
- The plotted point is the **center of the occupied cell**.  
Use this view to understand *coverage*: where the two datasets coincide vs. where they don’t.
        """
    )

# Render
if view_mode == "Points":
    view = base.dropna(subset=["LONGITUDE","LATITUDE","DEPTH"]).copy()
    if value_col not in view.columns and value_col != "__SOURCE__":
        st.warning("Selected value is not available in current source."); st.stop()
    if len(view) > max_points:
        view = view.sample(max_points, random_state=42)

    if category_mode:
        fig = px.scatter_3d(
            view, x="LONGITUDE", y="LATITUDE", z="DEPTH",
            color="SOURCE", opacity=0.85,
            hover_data={"LONGITUDE":":.5f","LATITUDE":":.5f","DEPTH":":.2f",
                        "CU_ORIG":":.2f" if "CU_ORIG" in view else False,
                        "CU_DL":":.2f"   if "CU_DL"   in view else False,
                        "DIFF":":.2f"    if "DIFF"    in view else False,
                        "DIFF_PCT":":.2f"if "DIFF_PCT" in view else False,
                        "SOURCE":True}
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
        order = ["overlap","orig_only","dl_only"]
        g["Source"] = pd.Categorical(g["Source"], categories=order, ordered=True)
        g = g.sort_values("Source")
        st.dataframe(g[["Source","Points","% of points"]], use_container_width=True, hide_index=True)

    else:
        view = view.dropna(subset=[value_col]).copy()
        arr = view[value_col].to_numpy()
        if value_col in ("DIFF","DIFF_PCT"):
            cs, (vmin, vmax) = diverging_color_and_range(arr, clip_mode=="Absolute")
            ticksuf = "%"
        else:
            cs, (vmin, vmax) = sequential_color_and_range(arr, clip_mode=="Absolute")
            ticksuf = ""
        fig = px.scatter_3d(
            view, x="LONGITUDE", y="LATITUDE", z="DEPTH",
            color=value_col, color_continuous_scale=cs, range_color=[vmin, vmax],
            opacity=0.85,
            hover_data={"LONGITUDE":":.5f","LATITUDE":":.5f","DEPTH":":.2f",
                        "CU_ORIG":":.2f" if "CU_ORIG" in view else False,
                        "CU_DL":":.2f"   if "CU_DL"   in view else False,
                        "DIFF":":.2f"    if "DIFF"    in view else False,
                        "DIFF_PCT":":.2f"if "DIFF_PCT" in view else False,
                        "SOURCE":True if "SOURCE" in view else False}
        )
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(
            scene=dict(zaxis=dict(autorange="reversed")),
            coloraxis_colorbar=dict(title=value_mode, ticksuffix=ticksuf),
            height=800, margin=dict(l=0, r=0, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    # Cubes (numeric-only)
    need = ["LONGITUDE","LATITUDE","DEPTH", value_col]
    num = base.dropna(subset=need).copy()
    num[value_col] = pd.to_numeric(num[value_col], errors="coerce")
    num = num.dropna(subset=[value_col])
    if num.empty:
        st.warning("No numeric rows for cubes under current filters / value selection.")
        st.stop()

    vox = voxelize_points(
        num, nx=nx, ny=ny, nz=nz,
        value_col=value_col, agg=agg_stat, min_count=min_count_voxel
    )
    if vox is None:
        st.warning("No voxels under current parameters. Try decreasing NX/NY/NZ or Min samples per voxel.")
        st.stop()

    Xc, Yc, Zc, V = vox["Xc"], vox["Yc"], vox["Zc"], vox["V"]
    x_edges, y_edges, z_edges = vox["x_edges"], vox["y_edges"], vox["z_edges"]
    dx = float(np.diff(x_edges).mean()); dy = float(np.diff(y_edges).mean()); dz = float(np.diff(z_edges).mean())

    if value_col in ("DIFF","DIFF_PCT"):
        cs, (vmin, vmax) = diverging_color_and_range(V, clip_mode=="Absolute")
        ticksuf = "%"
    else:
        cs, (vmin, vmax) = sequential_color_and_range(V, clip_mode=="Absolute")
        ticksuf = ""

    trace = build_voxel_mesh(
        Xc, Yc, Zc, V,
        dx=dx, dy=dy, dz=dz,
        cmin=vmin, cmax=vmax,
        colorscale=cs,
        opacity=voxel_opacity,
        colorbar_title=value_mode,
        ticksuffix=ticksuf,
        max_cubes=3000, pick_top_by_abs=True
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(
        scene=dict(zaxis=dict(autorange="reversed")),
        height=800, margin=dict(l=0, r=0, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------- Analytics (Overlap + DIFF/DIFF%) -----------------
show_analytics = (not category_mode) and (source == "Overlap") and (value_col in ("DIFF","DIFF_PCT"))
st.divider()
if not show_analytics:
    st.caption("Analytics are available when Source = Overlap and Value is DIFF / DIFF %.")
else:
    st.subheader("Analytics (linked to current filters)")
    analytics_df = base.dropna(
        subset=[c for c in ["DIFF","DIFF_PCT","CU_DL","CU_ORIG"] if c in base.columns]
    ).copy()
    if analytics_df.empty:
        st.info("No data under current filters for analytics.")
    else:
        t1, t2, t3, t4 = st.tabs(["Histogram", "DL vs ORIG / DIFF%", "Depth trend", "Top outliers"])
        with t1:
            cA, cB = st.columns(2)
            if "DIFF" in analytics_df:
                h1 = px.histogram(analytics_df, x="DIFF", nbins=60, title="Distribution of DIFF (DL - ORIG)")
                h1.update_layout(bargap=0.02, margin=dict(l=0, r=0, t=40, b=10))
                cA.plotly_chart(h1, use_container_width=True)
            if "DIFF_PCT" in analytics_df:
                h2 = px.histogram(analytics_df, x="DIFF_PCT", nbins=60, title="Distribution of DIFF %")
                h2.update_layout(bargap=0.02, margin=dict(l=0, r=0, t=40, b=10), xaxis=dict(ticksuffix="%"))
                cB.plotly_chart(h2, use_container_width=True)
        with t2:
            if {"CU_ORIG","CU_DL"}.issubset(analytics_df.columns):
                samp = analytics_df.sample(min(len(analytics_df), 50_000), random_state=42)
                s1 = px.scatter(samp, x="CU_ORIG", y="CU_DL", title="CU_DL vs CU_ORIG", opacity=0.6)
                lim = float(np.nanmax([analytics_df["CU_ORIG"].max(), analytics_df["CU_DL"].max()]))
                s1.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", width=1))
                s1.update_layout(margin=dict(l=0, r=0, t=40, b=10))
                st.plotly_chart(s1, use_container_width=True)
            if {"CU_DL","DIFF_PCT"}.issubset(analytics_df.columns):
                samp = analytics_df.sample(min(len(analytics_df), 50_000), random_state=42)
                s2 = px.scatter(samp, x="CU_DL", y="DIFF_PCT", title="DIFF % vs CU_DL", opacity=0.6)
                s2.update_layout(margin=dict(l=0, r=0, t=40, b=10), yaxis=dict(ticksuffix="%"))
                st.plotly_chart(s2, use_container_width=True)
        with t3:
            if "DIFF_PCT" in analytics_df:
                bins = st.slider("Depth bins", 10, 120, 40, 10)
                df_tr = analytics_df.copy()
                df_tr["_abs_pct"] = df_tr["DIFF_PCT"].abs()
                df_tr["_bin"] = pd.cut(df_tr["DEPTH"], bins=bins)
                g = df_tr.groupby("_bin").agg(
                    depth_mid=("DEPTH","median"),
                    abs_pct_median=("_abs_pct","median"),
                    abs_pct_mean=("_abs_pct","mean"),
                    n=("DEPTH","size")
                ).reset_index(drop=True)
                l1 = px.line(g, x="depth_mid", y=["abs_pct_median","abs_pct_mean"], markers=True,
                             labels={"value":"|DIFF %|", "depth_mid":"DEPTH"},
                             title="Depth vs |DIFF %| (binned)")
                l1.update_layout(margin=dict(l=0, r=0, t=40, b=10), yaxis=dict(ticksuffix="%"))
                st.plotly_chart(l1, use_container_width=True)
                st.caption(f"Bins: {bins}  •  Points: {len(analytics_df):,}")
        with t4:
            mode = st.radio("Sort by", ["|DIFF %|", "|DIFF|"], horizontal=True)
            if mode == "|DIFF %|" and "DIFF_PCT" in analytics_df:
                top = analytics_df.assign(_abs=lambda d: d["DIFF_PCT"].abs()).nlargest(100, "_abs")
            else:
                top = analytics_df.assign(_abs=lambda d: d["DIFF"].abs()).nlargest(100, "_abs") if "DIFF" in analytics_df else analytics_df.head(0)
            cols = [c for c in ["LONGITUDE","LATITUDE","DEPTH","CU_ORIG","CU_DL","DIFF","DIFF_PCT","SOURCE"] if c in top.columns]
            st.dataframe(top[cols], use_container_width=True, height=360)