# -*- coding: utf-8 -*-
"""
Task2 — difference helpers
- Compute DIFF and DIFF_PCT between DL and ORIG.
- Small utilities used by overlap.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_diff_cols(df: pd.DataFrame,
                      col_dl: str = "VALUE_DL",
                      col_orig: str = "VALUE_ORIG") -> pd.DataFrame:
    """Add DIFF and DIFF_PCT columns in-place and return df."""
    if col_dl in df and col_orig in df:
        dl = pd.to_numeric(df[col_dl], errors="coerce")
        og = pd.to_numeric(df[col_orig], errors="coerce")
        diff = dl - og
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(np.isfinite(og) & (og != 0), diff / og, np.nan)
        df["DIFF"] = diff
        df["DIFF_PCT"] = pct
    else:
        if "DIFF" not in df.columns:
            df["DIFF"] = np.nan
        if "DIFF_PCT" not in df.columns:
            df["DIFF_PCT"] = np.nan
    return df


def as_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce given columns to float; ignore missing columns."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
