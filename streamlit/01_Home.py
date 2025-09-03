import streamlit as st
from _ui_common import inject_theme, render_top_right_logo, load_grid_cfg, ensure_session_bootstrap

# Page Setup
st.set_page_config(layout="wide", page_title="Home • Copper Modeling Dashboard")
inject_theme()
ensure_session_bootstrap()

# ----------------------------- GLOBAL STYLING ---------------------------------------
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        html, body, div, p, h1, h2, h3, h4 {
            font-family: 'Inter', sans-serif;
        }

        .cta-button {
            background-color: #005f73;
            color: white !important;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            transition: all 0.2s ease-in-out;
        }

        .cta-button:hover {
            background-color: #c87533 !important;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }

        .upload-button {
            background-color: #c87533;
        }

        .status-card {
            background-color: #f9f9f9;
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 1rem;
        }

        .divider-bar {
            height: 4px;
            background-color: #b87333;
            border-radius: 3px;
            margin: 2rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------- LOGO AT TOP RIGHT ------------------------------------
with st.container():
    cols = st.columns([8, 1.5])
    with cols[1]:
        render_top_right_logo()

# ----------------------------- HERO SECTION -----------------------------------------
st.markdown("""
    <div style="background-color: #f0fafa; padding: 2.5rem; border-radius: 10px; margin-bottom: 2rem;
                box-shadow: 0 4px 16px rgba(0,0,0,0.05);", "background: linear-gradient(135deg, #f0fafa 0%, #f9f9f9 100%);">
        <div style="flex: 1; padding-right: 2rem;", "max-width: 850px;">
            <h1 style="margin-bottom:0.5rem; font-size:2.1rem; font-weight:600; color:#005f73;">
                Data-Driven Discovery: 3D Gap Analysis of Imputed vs 
                <span style="color:#c87533; font-weight:500;">Original Cu Assays</span>
            </h1>
            <p style="font-size:1.05rem; color:#333;">
                Explore, compare, and discover copper anomalies across drillhole and surface datasets.  
                This dashboard brings together record-level data, voxel-based aggregation, and deep-learning 
                imputation results, so you can identify overlaps, differences, and potential anomalies.  
                Switch between multiple levels of granularity, overlay drillhole with surface values, 
                and generate insights into where imputed predictions diverge from original assays.
            </p>
            <div style="margin-top:2rem; display: flex; gap: 1rem;">
                <a href="/Original" class="cta-button">Start with default dataset</a>
                <a href="/Data Manager" class="cta-button upload-button">Upload your data</a>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ----------------------------- CURRENT STATUS ---------------------------------------
cfg = load_grid_cfg()
ds = st.session_state.get("data_src", "default")

st.markdown("""
    <div class="status-card">
        <h3 style="color:#005f73;">Current Status</h3>
        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
            <div>
                <strong>Data Source:</strong><br>
                <span style="color:#333;">{}</span>
            </div>
            <div>
                <strong>Global Grid:</strong><br>
                <span style="color:#333;">{} × {} × {} m, {}, min_count={}</span>
            </div>
        </div>
    </div>
""".format(
    "Default dataset" if ds == "default" else "Uploaded dataset (set in Data Manager)",
    cfg['cell_x_m'], cfg['cell_y_m'], cfg['cell_z_m'],
    cfg['agg'], cfg['min_count']
), unsafe_allow_html=True)

st.caption(
    "Use **Original** to start exploring the default data, or go to **Data Manager** to upload and validate your own files."
)

# --------------------------- HOW IT WORKS SECTION -----------------------------------
st.markdown("""
    <h3 style="margin-bottom:0.2rem;">How it works</h3>
    <hr style="border:1px solid #b87333; width:180px; margin-top:0.2rem; margin-bottom:2rem;">
""", unsafe_allow_html=True)

steps = [
    {
        "icon": "📦",
        "title": "Collect Geological Data",
        "desc": "Import drillhole and surface records from trusted geological sources or upload your own."
    },
    {
        "icon": "🧪",
        "title": "Compare Assays",
        "desc": "Examine original and imputed copper assays to detect discrepancies and patterns."
    },
    {
        "icon": "🤖",
        "title": "Run Predictions",
        "desc": "Apply machine learning models to fill gaps in assay data and predict values."
    },
    {
        "icon": "🗺️",
        "title": "Visualize in 3D",
        "desc": "Explore multi-layered 3D visualizations of copper anomalies and interpolated grids."
    },
    {
        "icon": "🏁",
        "title": "Make Informed Decisions",
        "desc": "Use the dashboard insights to drive exploration or validate hypotheses."
    }
]

cols = st.columns(len(steps))
for step, col in zip(steps, cols):
    with col:
        st.markdown(
            f"<div style='text-align:center;font-size:38px'>{step['icon']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='text-align:center;font-weight:600; margin-top:0.5rem;'>{step['title']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center; font-size:0.9rem; color:#555;'>{step['desc']}</p>",
            unsafe_allow_html=True,
        )

# --------------------------- COPPER STRIP FOOTER ------------------------------------
st.markdown("<div class='divider-bar'></div>", unsafe_allow_html=True)

# --------------------------- FLOATING HELP BUTTON ------------------------------------
st.markdown("""
    <style>
        .floating-help {
            position: fixed;
            bottom: 24px;
            right: 24px;
            background-color: #005f73;
            color: white;
            border: none;
            padding: 12px 18px;
            border-radius: 50px;
            font-weight: 600;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
            z-index: 9999;
            cursor: pointer;
        }
        .floating-help:hover {
            background-color: #c87533;
        }
    </style>

    <a href="https://example.com/help" target="_blank">
        <button class="floating-help">❓ Help</button>
    </a>
""", unsafe_allow_html=True)
