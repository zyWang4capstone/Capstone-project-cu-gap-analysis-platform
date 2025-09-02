# _ui_common.py
from __future__ import annotations
from pathlib import Path
import json
import streamlit as st

# -----------------------------------------------------------------------------
# Theme / chrome
# -----------------------------------------------------------------------------
def inject_theme() -> None:
    st.markdown(
        """
        <style>
          .hero-cta {
            display: flex;
            gap: 28px;
            flex-wrap: wrap;
            margin-top: 1.0rem;
            margin-bottom: 0.5rem;
          }

          .stButton > button {
            background: #111827 !important;   /* black */
            # background: #5C4033 !important;   /* dark brown */
            color: #ffffff !important;        /* white text */
            border: 0 !important;
            border-radius: 18px !important;
            padding: 18px 28px !important;
            font-weight: 800 !important;
            font-size: 1.12rem !important;
            /* removed box-shadow */
          }
          .stButton > button:hover {
            filter: brightness(0.95);
            transform: translateY(-1px);
          }
          .stButton > button:active {
            transform: translateY(0);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_top_right_logo() -> None:
    """Render assets/logo.png at the top-right if present."""
    logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
    st.markdown('<div class="topbar" style="display:flex;justify-content:flex-end;align-items:center;margin-top:-0.35rem;">', unsafe_allow_html=True)
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=False)
    else:
        st.caption(f"Add your logo at {logo_path.as_posix()}")
    st.markdown("</div>", unsafe_allow_html=True)

def render_cta_row() -> None:
    st.markdown('<div class="hero-cta">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start with default dataset", key="cta_default"):
            st.switch_page("pages/03_Original.py")
    with c2:
        if st.button("Upload your data", key="cta_upload"):
            st.switch_page("pages/02_Data_Manager.py")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Global grid config (persisted JSON used across pages)
# -----------------------------------------------------------------------------
CFG_PATH = Path("reports/config/grid_settings.json")
DEFAULT_GRID = dict(cell_x_m=10.0, cell_y_m=10.0, cell_z_m=1.0, agg="mean", min_count=3)

def load_grid_cfg() -> dict:
    """Load grid config; fall back to defaults if file is missing or malformed."""
    if CFG_PATH.exists():
        try:
            raw = json.loads(CFG_PATH.read_text())
            return {
                "cell_x_m": float(raw.get("cell_x_m", DEFAULT_GRID["cell_x_m"])),
                "cell_y_m": float(raw.get("cell_y_m", DEFAULT_GRID["cell_y_m"])),
                "cell_z_m": float(raw.get("cell_z_m", DEFAULT_GRID["cell_z_m"])),
                "agg":      str(raw.get("agg", DEFAULT_GRID["agg"])),
                "min_count": int(raw.get("min_count", DEFAULT_GRID["min_count"])),
            }
        except Exception:
            pass
    return DEFAULT_GRID.copy()

def save_grid_cfg(cfg: dict) -> None:
    """Persist grid config to disk and mirror it in session state."""
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(json.dumps(cfg, indent=2))
    st.session_state["grid_cfg"] = cfg

def ensure_session_bootstrap() -> None:
    """Initialize common session flags just once."""
    st.session_state.setdefault("grid_cfg", load_grid_cfg())
    st.session_state.setdefault("data_src", "default")  # "default" | "uploaded"
    st.session_state.setdefault("uploaded_dh", None)
    st.session_state.setdefault("uploaded_sf", None)