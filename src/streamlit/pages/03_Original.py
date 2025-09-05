# 03_Original.py — Original dataset visualization (minimal; no filters)

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from _ui_common import inject_theme, ensure_session_bootstrap

st.set_page_config(layout="wide", page_title="Original • Copper Dashboard")
inject_theme()
ensure_session_bootstrap()

st.title("Original dataset")
