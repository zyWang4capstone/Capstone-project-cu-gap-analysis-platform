# cap_original/points.py  —— drop-in patch

from __future__ import annotations
import re
import pandas as pd
from cap_common.config import AppCfg
from cap_common.io import read_task1_clean
from cap_common.schema import standardize

# --- helpers: column normalization + tolerant matching -------------------
def _normalize_names(df: pd.DataFrame) -> dict[str, str]:
    """Map NORMALIZED -> original column name.
       NORMALIZED = uppercased, stripped spaces/quotes, dropped trailing units like '(m)'. """
    mapping = {}
    for c in df.columns:
        u = re.sub(r"\s+|['\"`]", "", str(c)).upper()
        u = re.sub(r"\([^)]*\)$", "", u)  # drop '(m)', '(M)', '(meters)' etc at tail
        mapping[u] = c
    return mapping

def _find_first(df: pd.DataFrame, patterns: list[str]) -> str | None:
    norm = _normalize_names(df)
    for pat in patterns:
        rgx = re.compile(pat, re.I)
        for u, orig in norm.items():
            if rgx.search(u):
                return orig
    return None

def _scale_to_meters(s: pd.Series, unit: str | None) -> pd.Series:
    """Scale numeric series to meters according to unit label."""
    s = pd.to_numeric(s, errors="coerce")
    if unit == "M":
        return s
    if unit == "CM":
        return s * 0.01
    if unit == "MM":
        return s * 0.001
    return s  # unknown -> no-op (may be fixed by heuristic later)

def _detect_pair_with_unit(df: pd.DataFrame):
    """Return (from_col, to_col, unit) where unit in {'M','CM','MM'} or None."""
    # Highest priority: explicit unit-suffixed columns
    for unit in ("M", "CM", "MM"):
        fc = _find_first(df, [fr"^FROM_?{unit}$"])
        tc = _find_first(df, [fr"^TO_?{unit}$"])
        if fc and tc:
            return fc, tc, unit
    # Fallback: generic FROM/TO family (unknown unit)
    fc = _find_first(df, [r"^FROM(_?DEPTH)?$", r"^DEPTH_?FROM$", r"^FDEPTH$", r"^F_?DEPTH$"])
    tc = _find_first(df, [r"^TO(_?DEPTH)?$", r"^DEPTH_?TO$", r"^TDEPTH$", r"^T_?DEPTH$"])
    if fc and tc:
        return fc, tc, None
    return None, None, None

def _heuristic_fix_depth_units(depth: pd.Series) -> pd.Series:
    """Heuristically convert cm/mm → m if the magnitude strongly suggests it."""
    s = pd.to_numeric(depth, errors="coerce")
    q99 = s.quantile(0.99)
    # Typical hard limits for WA drilling: few km at most.
    if q99 > 50_000 and q99 <= 5_000_000:
        # very likely centimeters (e.g., 120,000 → 1200 m)
        return s * 0.01
    if q99 > 5_000_000:
        # very likely millimeters
        return s * 0.001
    return s

def _clean_depth(depth: pd.Series) -> pd.Series:
    s = pd.to_numeric(depth, errors="coerce")
    # Remove sentinel/invalid values (negative, absurdly large)
    s = s.where(s >= 0)
    s = s.where(s <= 10_000)  # 10 km hard cap for safety
    return s

def _ensure_depth_midpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer middle depth from FROM/TO (with unit normalization); fallback to DEPTH; else 0."""
    fcol, tcol, unit = _detect_pair_with_unit(df)

    if fcol and tcol:
        f = _scale_to_meters(df[fcol], unit)
        t = _scale_to_meters(df[tcol], unit)
        depth = (f + t) / 2.0
        df["DEPTH"] = _clean_depth(depth)
        return df

    # strict DEPTH fallback (only if no FROM/TO found)
    dcol = _find_first(df, [r"^DEPTH$"])
    if dcol:
        d = pd.to_numeric(df[dcol], errors="coerce")
        d = _heuristic_fix_depth_units(d)  # if cm/mm slipped in
        df["DEPTH"] = _clean_depth(d)
        return df

    # surface-like fallback
    df["DEPTH"] = 0.0
    return df

def _view(df: pd.DataFrame, source: str, schema: dict) -> pd.DataFrame:
    d = standardize(df.copy(), schema)          # ensure lon/lat/value presence (doesn't touch FROM/TO)
    d = _ensure_depth_midpoint(d)               # robust depth with unit normalization
    d["SOURCE"] = source
    return (
        d.rename(columns={
            schema["longitude"]: "LONGITUDE",
            schema["latitude"]:  "LATITUDE",
            schema["value"]:     "VALUE",
        })[["LONGITUDE", "LATITUDE", "DEPTH", "VALUE", "SOURCE"]]
        .astype({"LONGITUDE":"float64","LATITUDE":"float64","DEPTH":"float64","VALUE":"float64"})
    )

def build_points_views(cfg: AppCfg) -> dict[str, pd.DataFrame]:
    dh = read_task1_clean("drillhole", cfg)
    sf = read_task1_clean("surface",   cfg)
    tables = {
        "points_dh_orig": _view(dh["orig"], "DH_ORIG",  cfg.schema),
        "points_dh_dl":   _view(dh["dl"],   "DH_DL",    cfg.schema),
        "points_sf_orig": _view(sf["orig"], "SURF_ORIG", cfg.schema).assign(DEPTH=0.0),
        "points_sf_dl":   _view(sf["dl"],   "SURF_DL",   cfg.schema).assign(DEPTH=0.0),
    }
    all_df = pd.concat(tables.values(), ignore_index=True)
    tables["points_all"] = all_df

    # Self-check summaries to catch future regressions
    chk = (all_df.groupby("SOURCE")["DEPTH"]
        .agg(n="size", nnz=lambda s: s.notna().sum(),
             vmin=lambda s: pd.to_numeric(s, errors="coerce").min(),
             vmax=lambda s: pd.to_numeric(s, errors="coerce").max()))
    print("[points_all] DEPTH summary by SOURCE:\n", chk)
    return tables
