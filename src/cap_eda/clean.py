# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import pandas as pd

# --- VALUE column detection ---
def detect_value_column(
    df: pd.DataFrame,
    *,
    value_aliases: list[str] | None = None,
    value_regex: str | None = None,
) -> str:
    up = {str(c).upper(): c for c in df.columns}
    # 1) 
    if value_aliases:
        for a in value_aliases:
            A = a.upper()
            if A in up:
                return up[A]
    # 2) 
    if value_regex:
        pat = re.compile(value_regex, re.I)
        for c in df.columns:
            if pat.search(str(c)):
                return c
    # 3) 
    def looks_like_value(name: str) -> bool:
        u = str(name).upper()
        return ("CU" in u) and (("PPM" in u) or ("PCT" in u) or ("PERCENT" in u))
    for c in df.columns:
        if looks_like_value(c):
            return c
    # 4) 
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
       
        df2 = df.copy()
        for c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="ignore")
        num_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
    if num_cols:
        return max(num_cols, key=lambda c: df[c].notna().sum())
    raise KeyError("Unable to detect VALUE column; please provide aliases/regex.")

# --- Minimal cleaning:  ---
def clean_minimal(
    df: pd.DataFrame,
    *,
    value_aliases: list[str] | None = None,
    value_regex: str | None = None,
    value_min: float | None = 1e-5,
    value_max: float | None = 346000.0,
) -> pd.DataFrame:
    col = detect_value_column(df, value_aliases=value_aliases, value_regex=value_regex)
    s = pd.to_numeric(df[col], errors="coerce")
    mask = s.notna()
    if value_min is not None:
        mask &= s >= float(value_min)
    if value_max is not None:
        mask &= s <= float(value_max)
    out = df.loc[mask].copy()
    # 
    out[col] = s.loc[mask].values
    return out

# 
def clean_surface(**kwargs) -> pd.DataFrame:
    return clean_minimal(**kwargs)

def clean_drillhole(**kwargs) -> pd.DataFrame:
    return clean_minimal(**kwargs)
