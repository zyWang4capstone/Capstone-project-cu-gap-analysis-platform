# 01_Home.py
import streamlit as st

st.page_link("pages/02_Original.py",            label="Original data (baseline)")
st.page_link("pages/03_Diff_Drillhole.py",      label="Drillhole — Differences (3D)")
st.page_link("pages/04_Diff_Surface.py",        label="Surface — Differences (2D)")
st.page_link("pages/05_Slicing_Drillhole.py",   label="Slicing — Drillhole (2D)")
st.page_link("pages/06_Overlay_DH_Surface.py",  label="Overlay: DH + Surface")
st.page_link("pages/07_Insights.py",            label="Insights")