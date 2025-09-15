# -*- coding: utf-8 -*-
"""
Task2 — overlap & split to points
This module turns the cleaned "drillhole/surface" pairs (ORIG vs DL) into
four point tables: overlap / origonly / dlonly / all, matching the front-end.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Iterable

import numpy as np
import pandas as pd

from cap_common.config import AppCfg, load_cfg
from cap_common.io import read_task1_clean
from cap_common.schema import standardize
from .diff import compute_diff_cols, as_float


# -----------------------
# Parameters dataclass
# -----------------------
@dataclass
class Params:
    grid_step_deg: float = 1e-4   # ≈ 11m at equator; keep consistent with notebook
    z_step_m: float = 1.0         # step when turning intervals to points
    overlap_min_len_m: float = 0  # minimal overlap length to keep
    include_nonoverlap_in_only: bool = True  # include same-XY but non-overlapping depth as 'only'
    only_sampling: Literal["mid","zstep"] = "zstep"  # how to sample 'only' intervals to points

# -----------------------
# Column helpers
# -----------------------
_DH_FROM_CANDS = ("FROM", "FROMDEPTH", "DEPTH_FROM", "FROM_M")
_DH_TO_CANDS   = ("TO", "TODEPTH", "DEPTH_TO", "TO_M")
_VAL_CANDS     = ("VALUE", "Cu_ppm", "CU_PPM", "CU")

def _first_existing(df: pd.DataFrame, cands: Iterable[str]) -> str | None:
    cols = {c.upper(): c for c in df.columns}
    for c in cands:
        if c in cols:
            return cols[c]
    return None

def _snap(v: np.ndarray, step: float) -> np.ndarray:
    """Snap coordinate to a degree grid."""
    return np.round(v / step) * step


# -----------------------
# Standardization
# -----------------------
def standardize_drillhole(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Ensure LONGITUDE, LATITUDE, FROM, TO, CU_ORIG/VALUE-style exist.
    Keeps the original numeric columns; returns a new standardized frame.
    """
    df = standardize(df.copy(), schema)  # ensures LONGITUDE/LATITUDE/VALUE when possible

    # FROM/TO resolution
    f = _first_existing(df, _DH_FROM_CANDS)
    t = _first_existing(df, _DH_TO_CANDS)
    if f is None or t is None:
        raise ValueError("Drillhole table must contain FROM/TO depth columns.")
    df = df.rename(columns={f: "FROM", t: "TO"})

    # VALUE column: if missing, try from candidates
    if "VALUE" not in df.columns:
        v = _first_existing(df, _VAL_CANDS)
        if v: df["VALUE"] = df[v]

    # types
    df = as_float(df, ["LONGITUDE", "LATITUDE", "FROM", "TO", "VALUE"])
    # enforce FROM <= TO
    swap = df["FROM"] > df["TO"]
    if swap.any():
        df.loc[swap, ["FROM", "TO"]] = df.loc[swap, ["TO", "FROM"]].values
    return df


def standardize_surface(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Ensure LONGITUDE, LATITUDE, VALUE (depth implied as 0)."""
    df = standardize(df.copy(), schema)
    if "VALUE" not in df.columns:
        v = _first_existing(df, _VAL_CANDS)
        if v: df["VALUE"] = df[v]
    df = as_float(df, ["LONGITUDE", "LATITUDE", "VALUE"])
    return df


# -----------------------
# Drillhole — interval overlap (two-pointer per XY cell)
# -----------------------
def _overlap_rows_for_cell(dfo: pd.DataFrame, dfd: pd.DataFrame,
                           min_len: float) -> list[tuple[float, float, float, float]]:
    """
    For one XY cell: return list of (from, to, value_orig, value_dl) for overlapped intervals.
    Assumes dfo/dfd have columns: FROM, TO, VALUE and are sorted by FROM.
    """
    out: list[tuple[float, float, float, float]] = []
    i = j = 0
    n, m = len(dfo), len(dfd)
    while i < n and j < m:
        fo, to = dfo.iloc[i].FROM, dfo.iloc[i].TO
        fd, td = dfd.iloc[j].FROM, dfd.iloc[j].TO
        lo = max(fo, fd); hi = min(to, td)
        if lo < hi and (hi - lo) >= min_len:
            out.append((lo, hi, dfo.iloc[i].VALUE, dfd.iloc[j].VALUE))
        # advance the one that ends first
        if to <= td:
            i += 1
        else:
            j += 1
    return out


def align_drillhole_overlap(df_orig: pd.DataFrame,
                            df_dl: pd.DataFrame,
                            pr: Params) -> pd.DataFrame:
    """
    Align ORIG and DL drillhole intervals by XY grid & depth overlap.
    Returns interval overlaps with columns:
        LONGITUDE, LATITUDE, FROM, TO, CU_ORIG, CU_DL
    """
    # snap XY
    for d in (df_orig, df_dl):
        d["LONGITUDE"] = _snap(d["LONGITUDE"].to_numpy(), pr.grid_step_deg)
        d["LATITUDE"]  = _snap(d["LATITUDE"].to_numpy(),  pr.grid_step_deg)

    # group by XY cell
    g_orig = df_orig.sort_values(["LONGITUDE","LATITUDE","FROM"]).groupby(["LONGITUDE","LATITUDE"], sort=False)
    g_dl   = df_dl.sort_values(["LONGITUDE","LATITUDE","FROM"]).groupby(["LONGITUDE","LATITUDE"], sort=False)

    # iterate intersection of XY keys
    keys = sorted(set(g_orig.groups.keys()).intersection(g_dl.groups.keys()))
    rows = []
    for lon, lat in keys:
        a = g_orig.get_group((lon, lat))[["FROM","TO","VALUE"]].reset_index(drop=True)
        b = g_dl.get_group((lon, lat))[["FROM","TO","VALUE"]].reset_index(drop=True)
        for lo, hi, v_o, v_d in _overlap_rows_for_cell(a, b, pr.overlap_min_len_m):
            rows.append((lon, lat, lo, hi, v_o, v_d))
    if not rows:
        return pd.DataFrame(columns=["LONGITUDE","LATITUDE","FROM","TO","CU_ORIG","CU_DL"])

    out = pd.DataFrame.from_records(
        rows, columns=["LONGITUDE","LATITUDE","FROM","TO","CU_ORIG","CU_DL"]
    )
    return out


def _intervals_to_points(df_int: pd.DataFrame, z_step_m: float) -> pd.DataFrame:
    """
    Turn intervals (FROM, TO, CU_*) to points at midpoints every z_step_m.
    If z_step_m <= 0, use single midpoint per interval.
    """
    if df_int.empty:
        return df_int.assign(DEPTH=pd.Series(dtype="float64"))

    use_multi = z_step_m and z_step_m > 0
    recs = []
    for r in df_int.itertuples(index=False):
        z0, z1 = float(r.FROM), float(r.TO)
        if not use_multi:
            zs = [0.5 * (z0 + z1)]
        else:
            # include both ends (open interval inside plotting is fine)
            n = max(1, int(np.ceil((z1 - z0) / z_step_m)))
            zs = (z0 + (np.arange(n) + 0.5) * (z1 - z0) / n).tolist()
        for z in zs:
            rec = dict(r._asdict())
            rec["DEPTH"] = z
            recs.append(rec)
    return pd.DataFrame.from_records(recs, columns=list(df_int.columns) + ["DEPTH"])

def _merge_intervals(spans: list[tuple[float,float]]) -> list[tuple[float,float]]:
    """Merge overlapping [lo,hi] spans; return sorted disjoint spans."""
    if not spans:
        return []
    spans = sorted((float(a), float(b)) for a,b in spans if float(a) < float(b))
    out: list[list[float]] = []
    for a,b in spans:
        if not out or a > out[-1][1]:
            out.append([a,b])
        else:
            out[-1][1] = max(out[-1][1], b)
    return [(a,b) for a,b in out]


def _subtract_spans(intervals: list[tuple[float,float,float]],
                    overlapped: list[tuple[float,float]]) -> list[tuple[float,float,float]]:
    """Subtract union(overlapped) from *intervals*.
    intervals: [(from,to,value)], overlapped: [(from,to)]
    Return remaining [(from,to,value)] with from < to.
    """
    ovl = _merge_intervals(overlapped)
    res: list[tuple[float,float,float]] = []
    for a,b,v in intervals:
        cur = float(a)
        end = float(b)
        for lo,hi in ovl:
            if hi <= cur or lo >= end:
                continue
            if lo > cur:
                res.append((cur, min(lo, end), v))
            cur = max(cur, hi)
            if cur >= end:
                break
        if cur < end:
            res.append((cur, end, v))
    return [(x,y,v) for (x,y,v) in res if y > x]


def split_drillhole_overlap_and_only(df_o: pd.DataFrame,
                                     df_d: pd.DataFrame,
                                     pr: Params) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produce three point tables for drillholes:
        overlap_pts: LONGITUDE, LATITUDE, DEPTH, CU_ORIG, CU_DL, DIFF, DIFF_PCT
        origonly_pts: LONGITUDE, LATITUDE, DEPTH, CU_ORIG
        dlonly_pts:   LONGITUDE, LATITUDE, DEPTH, CU_DL
    """
    # 1) overlap intervals → points
    ovl_int = align_drillhole_overlap(df_o, df_d, pr)
    ovl_pts = _intervals_to_points(ovl_int, pr.z_step_m)
    ovl_pts = compute_diff_cols(ovl_pts)

    # 2) origonly / dlonly
    # snap first, then find XY cells with no counterpart OR intervals not overlapped
    for d in (df_o, df_d):
        d["LONGITUDE"] = _snap(d["LONGITUDE"].to_numpy(), pr.grid_step_deg)
        d["LATITUDE"]  = _snap(d["LATITUDE"].to_numpy(),  pr.grid_step_deg)

    # mark overlapped (by FROM/TO intersection existence)
    if not ovl_int.empty:
        key_ovl = set(zip(ovl_int["LONGITUDE"], ovl_int["LATITUDE"]))
    else:
        key_ovl = set()

    def _midpoints(df, colname):
        mid = (df["FROM"] + df["TO"]) / 2.0
        return pd.DataFrame({
            "LONGITUDE": df["LONGITUDE"],
            "LATITUDE":  df["LATITUDE"],
            "DEPTH":     mid,
            colname:     df["VALUE"].astype(float),
        })

    # Cells that appear only in one side (strict by XY)
    cells_o = set(zip(df_o["LONGITUDE"], df_o["LATITUDE"]))
    cells_d = set(zip(df_d["LONGITUDE"], df_d["LATITUDE"]))

    only_o_cells = cells_o.difference(cells_d)
    only_d_cells = cells_d.difference(cells_o)

    origonly_pts = _midpoints(df_o[df_o[["LONGITUDE","LATITUDE"]].apply(tuple, axis=1).isin(only_o_cells)], "CU_ORIG")
    dlonly_pts   = _midpoints(df_d[df_d[["LONGITUDE","LATITUDE"]].apply(tuple, axis=1).isin(only_d_cells)], "CU_DL")

    # Note: intervals that are in same XY cell but non-overlapping along depth are not counted as "only".
    # If you want to also include them, add logic here to subtract overlapped [FROM,TO] ranges from each side.

    return ovl_pts, origonly_pts, dlonly_pts


# -----------------------
# Surface — overlap by XY grid only
# -----------------------
def split_surface_overlap_and_only(df_o: pd.DataFrame,
                                   df_d: pd.DataFrame,
                                   pr: Params) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produce three 2D tables for surface:
        overlap_pts: LONGITUDE, LATITUDE, CU_ORIG, CU_DL, DIFF, DIFF_PCT (DEPTH=0)
        origonly_pts: LONGITUDE, LATITUDE, CU_ORIG (DEPTH=0)
        dlonly_pts:   LONGITUDE, LATITUDE, CU_DL   (DEPTH=0)
    Aggregation: mean per XY cell.
    """
    for d in (df_o, df_d):
        d["LONGITUDE"] = _snap(d["LONGITUDE"].to_numpy(), pr.grid_step_deg)
        d["LATITUDE"]  = _snap(d["LATITUDE"].to_numpy(),  pr.grid_step_deg)

    agg_o = df_o.groupby(["LONGITUDE","LATITUDE"], as_index=False)["VALUE"].mean().rename(columns={"VALUE":"CU_ORIG"})
    agg_d = df_d.groupby(["LONGITUDE","LATITUDE"], as_index=False)["VALUE"].mean().rename(columns={"VALUE":"CU_DL"})

    ovl = pd.merge(agg_o, agg_d, on=["LONGITUDE","LATITUDE"], how="inner")
    ovl["DEPTH"] = 0.0
    ovl = compute_diff_cols(ovl)

    only_o = pd.merge(agg_o, agg_d, on=["LONGITUDE","LATITUDE"], how="left", indicator=True)
    only_o = only_o[only_o["_merge"] == "left_only"].drop(columns=["_merge"]).rename(columns={"CU_ORIG":"CU_ORIG"})
    only_o["DEPTH"] = 0.0

    only_d = pd.merge(agg_d, agg_o, on=["LONGITUDE","LATITUDE"], how="left", indicator=True)
    only_d = only_d[only_d["_merge"] == "left_only"].drop(columns=["_merge"]).rename(columns={"CU_DL":"CU_DL"})
    only_d["DEPTH"] = 0.0

    return ovl, only_o[["LONGITUDE","LATITUDE","DEPTH","CU_ORIG"]], only_d[["LONGITUDE","LATITUDE","DEPTH","CU_DL"]]


# -----------------------
# Orchestrators (read -> split -> write)
# -----------------------
def _ensure_outdir(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)

def recompute_drillhole(cfg: AppCfg, pr: Params | None = None) -> dict[str, Path]:
    pr = pr or Params()
    d = read_task1_clean("drillhole", cfg)
    df_o = standardize_drillhole(d["orig"], cfg.schema)
    df_d = standardize_drillhole(d["dl"],   cfg.schema)

    ovl, only_o, only_d = split_drillhole_overlap_and_only(df_o, df_d, pr)

    _ensure_outdir(cfg.task2_diff_dir)
    out = {}
    out["overlap"]  = (cfg.task2_diff_dir / "drillhole_points_overlap.csv")
    out["origonly"] = (cfg.task2_diff_dir / "drillhole_points_origonly.csv")
    out["dlonly"]   = (cfg.task2_diff_dir / "drillhole_points_dlonly.csv")
    out["all"]      = (cfg.task2_diff_dir / "drillhole_points_all.csv")

    ovl[["LONGITUDE","LATITUDE","DEPTH","CU_ORIG","CU_DL","DIFF","DIFF_PCT"]].to_csv(out["overlap"], index=False)
    only_o.to_csv(out["origonly"], index=False)
    only_d.to_csv(out["dlonly"], index=False)

    # all = union (keep available columns)
    all_tbl = pd.concat([
        ovl[["LONGITUDE","LATITUDE","DEPTH","CU_ORIG","CU_DL","DIFF","DIFF_PCT"]],
        only_o[["LONGITUDE","LATITUDE","DEPTH","CU_ORIG"]].assign(CU_DL=np.nan, DIFF=np.nan, DIFF_PCT=np.nan),
        only_d[["LONGITUDE","LATITUDE","DEPTH","CU_DL"]].assign(CU_ORIG=np.nan, DIFF=np.nan, DIFF_PCT=np.nan),
    ], ignore_index=True)
    all_tbl.to_csv(out["all"], index=False)

    return out


def recompute_surface(cfg: AppCfg, pr: Params | None = None) -> dict[str, Path]:
    pr = pr or Params()
    d = read_task1_clean("surface", cfg)
    df_o = standardize_surface(d["orig"], cfg.schema)
    df_d = standardize_surface(d["dl"],   cfg.schema)

    ovl, only_o, only_d = split_surface_overlap_and_only(df_o, df_d, pr)

    _ensure_outdir(cfg.task2_diff_dir)
    out = {}
    out["overlap"]  = (cfg.task2_diff_dir / "surface_points_overlap.csv")
    out["origonly"] = (cfg.task2_diff_dir / "surface_points_origonly.csv")
    out["dlonly"]   = (cfg.task2_diff_dir / "surface_points_dlonly.csv")
    out["all"]      = (cfg.task2_diff_dir / "surface_points_all.csv")

    ovl[["LONGITUDE","LATITUDE","DEPTH","CU_ORIG","CU_DL","DIFF","DIFF_PCT"]].to_csv(out["overlap"], index=False)
    only_o.to_csv(out["origonly"], index=False)
    only_d.to_csv(out["dlonly"], index=False)

    all_tbl = pd.concat([
        ovl[["LONGITUDE","LATITUDE","DEPTH","CU_ORIG","CU_DL","DIFF","DIFF_PCT"]],
        only_o[["LONGITUDE","LATITUDE","DEPTH","CU_ORIG"]].assign(CU_DL=np.nan, DIFF=np.nan, DIFF_PCT=np.nan),
        only_d[["LONGITUDE","LATITUDE","DEPTH","CU_DL"]].assign(CU_ORIG=np.nan, DIFF=np.nan, DIFF_PCT=np.nan),
    ], ignore_index=True)
    all_tbl.to_csv(out["all"], index=False)

    return out


# -----------------------
# Convenience CLI
# -----------------------
def recompute_all(cfg: AppCfg | None = None,
                  pr: Params | None = None) -> dict[str, dict[str, Path]]:
    cfg = cfg or load_cfg()
    pr = pr or Params()
    return {
        "drillhole": recompute_drillhole(cfg, pr),
        "surface":   recompute_surface(cfg, pr),
    }

if __name__ == "__main__":
    cfg = load_cfg()
    out = recompute_all(cfg)
    for k, d in out.items():
        print(f"[{k}]")
        for name, p in d.items():
            print(f"  {name}: {p}")
