# 02_Data_Manager.py — Uploads, basic checks, choose active source 

from __future__ import annotations
import io
from pathlib import Path
import pandas as pd
import streamlit as st
from _ui_common import inject_theme, ensure_session_bootstrap, load_grid_cfg, save_grid_cfg, DEFAULT_GRID

REQUIRED_DH = {"LONGITUDE", "LATITUDE", "DEPTH"}   # minimal schema for drillhole points
REQUIRED_SF = {"LONGITUDE", "LATITUDE"}            # minimal schema for surface points

st.set_page_config(layout="wide", page_title="Data Manager")
inject_theme()
ensure_session_bootstrap()

st.title("Data Manager")

# ---------------------------------------------------------------------------
# Dataset manager (upload + quick checks)
# ---------------------------------------------------------------------------
st.subheader("Dataset manager")

c1, c2 = st.columns(2)

with c1:
    dh_file = st.file_uploader("Drillhole CSV/Parquet", type=["csv", "parquet"], key="dm_dh")
with c2:
    sf_file = st.file_uploader("Surface CSV/Parquet", type=["csv", "parquet"], key="dm_sf")

def _read_any(f) -> pd.DataFrame:
    if f is None:
        return None
    name = f.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(f)
    return pd.read_csv(f)

def _schema_ok(df: pd.DataFrame, required: set[str]) -> tuple[bool, list[str]]:
    missing = [c for c in required if c not in df.columns] if df is not None else list(required)
    return (len(missing) == 0, missing)

def _bbox(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    return (
        float(df["LONGITUDE"].min()), float(df["LATITUDE"].min()),
        float(df["LONGITUDE"].max()), float(df["LATITUDE"].max())
    )

# Read and cache in session
if dh_file:
    try:
        st.session_state["uploaded_dh"] = _read_any(dh_file)
        st.success("Drillhole file loaded.")
    except Exception as e:
        st.error(f"Failed to read drillhole file: {e}")

if sf_file:
    try:
        st.session_state["uploaded_sf"] = _read_any(sf_file)
        st.success("Surface file loaded.")
    except Exception as e:
        st.error(f"Failed to read surface file: {e}")

# Report status
dh = st.session_state.get("uploaded_dh")
sf = st.session_state.get("uploaded_sf")

c3, c4 = st.columns(2)
with c3:
    st.markdown("**Uploaded drillhole**")
    if isinstance(dh, pd.DataFrame):
        ok, missing = _schema_ok(dh, REQUIRED_DH)
        st.write(f"Rows: {len(dh):,}")
        st.write(f"Schema OK: {ok} ({'missing: ' + ', '.join(missing) if not ok else 'all required columns present'})")
        bb = _bbox(dh)
        if bb:
            st.caption(f"BBox: [{bb[0]:.3f}, {bb[1]:.3f}] → [{bb[2]:.3f}, {bb[3]:.3f}]")
    else:
        st.info("No drillhole file in session.")

with c4:
    st.markdown("**Uploaded surface**")
    if isinstance(sf, pd.DataFrame):
        ok, missing = _schema_ok(sf, REQUIRED_SF)
        st.write(f"Rows: {len(sf):,}")
        st.write(f"Schema OK: {ok} ({'missing: ' + ', '.join(missing) if not ok else 'all required columns present'})")
        bb = _bbox(sf)
        if bb:
            st.caption(f"BBox: [{bb[0]:.3f}, {bb[1]:.3f}] → [{bb[2]:.3f}, {bb[3]:.3f}]")
    else:
        st.info("No surface file in session.")

# Choose active source
st.markdown("---")
st.subheader("Active data source")

c5, c6 = st.columns(2)
with c5:
    if st.button("Use default dataset", use_container_width=True):
        st.session_state["data_src"] = "default"
        st.success("Default dataset is now active for all analysis pages.")

with c6:
    disabled = not (isinstance(dh, pd.DataFrame) or isinstance(sf, pd.DataFrame))
    if st.button("Use uploaded dataset", use_container_width=True, disabled=disabled):
        st.session_state["data_src"] = "uploaded"
        st.success("Uploaded dataset is now active for all analysis pages.")

# ---------------------------------------------------------------------------
# Template download
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Download templates")
cols = ["LONGITUDE", "LATITUDE", "DEPTH", "CU_ORIG", "CU_DL", "SOURCE"]
buf = io.StringIO()
pd.DataFrame(columns=cols).to_csv(buf, index=False)
st.download_button("Download CSV template", buf.getvalue(), file_name="template.csv", mime="text/csv")