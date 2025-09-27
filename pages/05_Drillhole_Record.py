# pages/05_Drillhole_Record.py

from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from _ui_common import inject_theme

st.set_page_config(layout="wide", page_title="Record Level • Drillhole")
inject_theme()

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
    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #111827 !important;
        color: #ffffff !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        border-color: #111827 !important;
        box-shadow: none !important;
        transform: translateY(1px);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: transparent !important;
    }
    .stTabs + div [data-testid="stVerticalBlock"] > div:first-child {
        border-top: 1px solid #11182710;
        margin-top: -1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- Paths ----------------------------
CLEAN_DIR = Path("reports/task1/cleaned")
PATH_DH_ORIG = CLEAN_DIR / "drillhole_original_clean.csv"
PATH_DH_DL   = CLEAN_DIR / "drillhole_dnn_clean.csv"

# ------------------------- I/O & helpers ------------------------
@st.cache_data(show_spinner=False)
def load_csv_safe(p: Path, element: str = "Element") -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        # Always ensure numeric types for core coordinate/depth columns
        base_cols = ["LONGITUDE", "LATITUDE", "DEPTH", "FROMDEPTH", "TODEPTH"]
        for c in base_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Dynamically check for element-specific columns (e.g., Element_ppm, ELEMENT_ORIG, ELEMENT_DL)
        element_cols = [
            f"{element}_ppm",
            f"{element.upper()}_ORIG",
            f"{element.upper()}_DL",
        ]
        for c in element_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df
    except Exception:
        return pd.DataFrame()

# ------------------------------ Data -----------------------------
element = st.session_state.get("element", "Element")
dh_orig = load_csv_safe(PATH_DH_ORIG, element)
dh_dl   = load_csv_safe(PATH_DH_DL, element)

if dh_orig.empty and dh_dl.empty:
    st.warning(
        "No cleaned drillhole files found under `reports/task1/cleaned/`.\n"
        "Expected: `drillhole_original_clean.csv` and/or `drillhole_dnn_clean.csv`."
    )
    st.stop()

# --- Page title row with three columns ---
c1, c2, c3 = st.columns([0.6, 0.2, 0.2])  # adjust proportions as needed

with c1:
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.25rem; font-size:2.0rem;">
        Record-level • Drillhole ({element})
        </h1>
        <p style="color:#555; margin-top:0;">
        Raw drillhole records previewed within a small spatial window (points only).
        </p>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.page_link("pages/04_Diff_Home.py", label="Back to Diff • Home")

with c3:
    st.page_link("pages/09_Insights.py", label="View insights")

# ───────────────────────── Sidebar controls (organized) ─────────────────────────
show_orig = st.sidebar.checkbox("Show ORIG", True)
show_dl   = st.sidebar.checkbox("Show DL", True)
pt_size = st.sidebar.slider("Point size", 1, 6, 2)

# union for ranges
u = pd.concat(
    [df.assign(_src=tag) for df, tag in [(dh_orig,"ORIG"), (dh_dl,"DL")] if not df.empty],
    ignore_index=True
)

# ------------------------------ Sidebar Filters ------------------------------
if u.empty:
    st.warning("No data found. Check CSV paths.")
    st.stop()

# longitude / latitude ranges
if {"LONGITUDE","LATITUDE"}.issubset(u.columns):
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

# depth range
if "DEPTH" in u.columns:
    depth_min, depth_max = float(u["DEPTH"].min()), float(u["DEPTH"].max())
else:
    depth_min, depth_max = 0.0, 2500.0
zr = st.sidebar.slider("Depth", depth_min, depth_max, (depth_min, depth_max))

# depth range, clamp to [0, 2500]
DEPTH_MAX = 2500.0
zcand = []
for df in [dh_orig, dh_dl]:
    if df.empty:
        continue
    if "DEPTH" in df.columns:
        vals = pd.to_numeric(df["DEPTH"], errors="coerce")
        zcand += vals[(vals >= 0) & (vals <= DEPTH_MAX)].tolist()
    elif {"FROMDEPTH","TODEPTH"}.issubset(df.columns):
        vfrom = pd.to_numeric(df["FROMDEPTH"], errors="coerce")
        vto   = pd.to_numeric(df["TODEPTH"],   errors="coerce")
        zcand += vfrom[(vfrom >= 0) & (vfrom <= DEPTH_MAX)].tolist()
        zcand += vto  [(vto   >= 0) & (vto   <= DEPTH_MAX)].tolist()

if zcand:
    zmin, zmax = float(np.nanmin(zcand)), float(np.nanmax(zcand))
else:
    zmin, zmax = 0.0, DEPTH_MAX

zr = st.sidebar.slider("Depth", 0.0, DEPTH_MAX, (zmin, zmax))

# ------------------------------ Filtering fn ---------------------
def filter_window(df: pd.DataFrame) -> pd.DataFrame:
    """Filter records based on longitude/latitude and depth sliders."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    # Apply longitude/latitude filters
    if {"LONGITUDE", "LATITUDE"}.issubset(out.columns):
        out = out[out["LONGITUDE"].between(*xr) & out["LATITUDE"].between(*yr)]

    # Apply depth filters
    z0, z1 = float(zr[0]), float(zr[1])
    if "DEPTH" in out.columns:
        dep = pd.to_numeric(out["DEPTH"], errors="coerce")
        m = (dep >= 0) & (dep <= 2500.0) & dep.between(z0, z1)
        out = out[m]
    elif {"FROMDEPTH", "TODEPTH"}.issubset(out.columns):
        f = pd.to_numeric(out["FROMDEPTH"], errors="coerce")
        t = pd.to_numeric(out["TODEPTH"], errors="coerce")
        # keep intervals within [0,2500] that overlap [z0,z1]
        m = (t >= 0) & (t <= 2500.0) & (t >= z0) & (f <= z1)
        out = out[m]

    return out

full_o = filter_window(dh_orig) if show_orig else pd.DataFrame()
full_d = filter_window(dh_dl)   if show_dl   else pd.DataFrame()

# ------------------------------ 3D preview (tabs) ------------------------------
def make_figure_v1() -> go.Figure:
    fig = go.Figure()

    def add_points(df: pd.DataFrame, tag: str, color: str):
        if df.empty or not {"LONGITUDE", "LATITUDE"}.issubset(df.columns):
            return

        # depth for each record
        if "DEPTH" in df.columns:
            z = pd.to_numeric(df["DEPTH"], errors="coerce")
        elif {"FROMDEPTH", "TODEPTH"}.issubset(df.columns):
            z = (pd.to_numeric(df["FROMDEPTH"], errors="coerce") +
                 pd.to_numeric(df["TODEPTH"],   errors="coerce")) / 2.0
        else:
            z = pd.Series(np.nan, index=df.index)

        # pick element column dynamically for hover (ppm, ORIG, DL)
        element_cols = [
            f"{element}_ppm",
            f"{element.upper()}_ORIG",
            f"{element.upper()}_DL"
        ]
        val_col = next((c for c in element_cols if c in df.columns), None)
        val_vals = pd.to_numeric(df[val_col], errors="coerce") if val_col else pd.Series(np.nan, index=df.index)

        # pack hover fields into customdata
        cd = pd.DataFrame({
            "tag":       tag,
            "sampleid":  df["SAMPLEID"] if "SAMPLEID" in df.columns else pd.Series([None]*len(df), index=df.index),
            "lon":       pd.to_numeric(df["LONGITUDE"], errors="coerce"),
            "lat":       pd.to_numeric(df["LATITUDE"],  errors="coerce"),
            "stype":     df["SAMPLETYPE"] if "SAMPLETYPE" in df.columns else pd.Series([None]*len(df), index=df.index),
            "from":      pd.to_numeric(df["FROMDEPTH"], errors="coerce") if "FROMDEPTH" in df.columns else pd.Series([np.nan]*len(df), index=df.index),
            "to":        pd.to_numeric(df["TODEPTH"],   errors="coerce") if "TODEPTH"   in df.columns else pd.Series([np.nan]*len(df), index=df.index),
            "val":       val_vals,
            "depth_mid": z,
        }).to_numpy()

        hover_tmpl = (
            "<b>SOURCE: %{customdata[0]}</b><br>"
            "ID=%{customdata[1]}<br>"
            "Lon=%{customdata[2]:.5f} · Lat=%{customdata[3]:.5f}<br>"
            "SampleType=%{customdata[4]}<br>"
            "From=%{customdata[5]:.2f} · To=%{customdata[6]:.2f}<br>"
            f"{element}=%{{customdata[7]:.3f}}<br>"
            "Depth_mid=%{customdata[8]:.2f}"
            "<extra></extra>"
        )

        fig.add_trace(go.Scatter3d(
            x=df["LONGITUDE"], y=df["LATITUDE"], z=z,
            mode="markers",
            marker=dict(size=pt_size, opacity=0.85, color=color),
            name=tag,
            customdata=cd,
            hovertemplate=hover_tmpl,
            showlegend=True,
        ))

    # Add Original & DL points
    if not full_o.empty: add_points(full_o, "Original", "#447dd2")
    if not full_d.empty: add_points(full_d, "DL",       "#ff554b")

    # Layout config
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Longitude"),
            yaxis=dict(title="Latitude"),
            zaxis=dict(title="Depth", autorange="reversed"),
        ),
        height=800,
        margin=dict(l=0, r=0, t=50, b=10),
        legend=dict(
            title="SOURCE",
            orientation="v",
            xanchor="left", x=1.02, yanchor="top", y=1.0,
            font=dict(size=18),
            itemsizing="constant",
        ),
    )
    return fig

def make_figure_v2() -> go.Figure:
    """
    V2: value-colored view
    - Merge ORIG + DL; color the points by element value.
    - Element column priority: {element}_ppm > {element}_DL > {element}_ORIG.
    - Percentile clipping (98th) for stable colorbar/size scaling.
    """
    both = pd.concat([full_o.assign(_src="ORIG"), full_d.assign(_src="DL")], ignore_index=True)
    fig = go.Figure()
    if both.empty:
        return fig

    # choose element value column
    element_cols = [
        f"{element}_ppm",
        f"{element.upper()}_DL",
        f"{element.upper()}_ORIG",
    ]
    val_col = next((c for c in element_cols if c in both.columns), None)
    if not val_col:
        return fig  # nothing to color by

    # depth (z)
    if "DEPTH" in both.columns:
        z = pd.to_numeric(both["DEPTH"], errors="coerce")
    elif {"FROMDEPTH", "TODEPTH"}.issubset(both.columns):
        z = (pd.to_numeric(both["FROMDEPTH"], errors="coerce") +
             pd.to_numeric(both["TODEPTH"],   errors="coerce")) / 2.0
    else:
        z = pd.Series(np.nan, index=both.index)

    # values for coloring
    v = pd.to_numeric(both[val_col], errors="coerce")
    vmax = float(np.nanpercentile(v, 98)) if np.isfinite(v).any() else 1.0
    if vmax <= 0:
        vmax = 1.0
    v_clip = v.clip(lower=0, upper=vmax)

    fig.add_trace(go.Scatter3d(
        x=both["LONGITUDE"], y=both["LATITUDE"], z=z,
        mode="markers",
        marker=dict(
            size=pt_size,
            color=v_clip,
            colorscale="Turbo",
            cmin=0, cmax=vmax,
            colorbar=dict(title=element),
            opacity=0.9,
        ),
        name=f"{element}-valued points",
        hovertemplate=(
            f"{element} (%{{marker.color:.3f}})<br>"
            "Lon=%{x:.5f} · Lat=%{y:.5f}<br>"
            "Depth=%{z:.2f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Longitude"),
            yaxis=dict(title="Latitude"),
            zaxis=dict(title="Depth", autorange="reversed"),
        ),
        margin=dict(l=0, r=120, t=50, b=10),
        height=800,
        legend=dict(orientation="h", y=-0.08),
    )
    return fig


# Render tabs or an info block when no data
if full_o.empty and full_d.empty:
    st.info("No records in current window. Adjust the filters.")
else:
    fig_v1 = make_figure_v1()
    fig_v2 = make_figure_v2()

    tab1, tab2 = st.tabs(["Source view", "Value view"])

    with tab1:
        st.caption(f"Plotted points: **{len(full_o) + len(full_d):,}** "
                   f"(ORIG: {len(full_o):,} • DL: {len(full_d):,})")
        st.plotly_chart(fig_v1, use_container_width=True)

    with tab2:
        st.caption(
            f"Colored by {element} value · Plotted: **{len(full_o) + len(full_d):,}** "
            f"(ORIG: {len(full_o):,} • DL: {len(full_d):,})"
        )
        st.plotly_chart(fig_v2, use_container_width=True)

# ------------------------------ Table & Export --------------------
st.subheader("Records (filtered)")
def keep_cols(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    out["_LAYER"] = tag
    keep = [c for c in [
        "SAMPLEID","LONGITUDE","LATITUDE","DEPTH","FROMDEPTH","TODEPTH",
        "Cu_ppm","CU_ORIG","CU_DL","SAMPLETYPE","COLLARID"
    ] if c in out.columns] + ["_LAYER"]
    return out[keep]

tbl = pd.concat([
    keep_cols(full_o, "ORIG") if not full_o.empty else pd.DataFrame(),
    keep_cols(full_d, "DL")   if not full_d.empty else pd.DataFrame()
], ignore_index=True)

if tbl.empty:
    st.caption("No rows to display.")
else:
    total_o, total_d = len(full_o), len(full_d)
    total = total_o + total_d
    st.markdown(f"**{total:,} records in current window** (ORIG: {total_o:,} • DL: {total_d:,})")
    st.dataframe(tbl, use_container_width=True, height=360, hide_index=True)

    csv_buf = io.StringIO()
    tbl.to_csv(csv_buf, index=False)
    st.download_button(
        "Download filtered CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="record_level_filtered.csv",
        mime="text/csv",
        type="primary"
    )