from __future__ import annotations
import pandas as pd
import pyvista_ops as pvo 

def make_horizontal(points_df: pd.DataFrame, depth_value: float, **kw):
    """pyvista_ops"""
    return pvo.make_horizontal_slice(points_df, depth_value, **kw)

def make_section(points_df: pd.DataFrame, p1_deg, p2_deg, **kw):
    """"""
    return pvo.make_section_grid(points_df, p1_deg, p2_deg, **kw)
