from __future__ import annotations
import argparse, sys, re
from pathlib import Path
import pandas as pd

# Ensure backend src on path (repo structure: this_script/../src)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cap_common.config import load_cfg
from cap_eda.ingest import process_data_dir
from cap_original.points import build_points_views
from cap_task2.overlap import recompute_all

CAT_ORDER = ["DH_ORIG","DH_DL","SURF_ORIG","SURF_DL"]
REQ_COLS = ["LONGITUDE","LATITUDE","DEPTH","VALUE","SOURCE"]

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _write_parquet(df: pd.DataFrame, p: Path, engine: str, compression: str | None) -> None:
    _ensure_dir(p)
    df.to_parquet(p, index=False, engine=engine, compression=compression)

def _write_csv(df: pd.DataFrame, p: Path) -> None:
    _ensure_dir(p)
    df.to_csv(p, index=False)

def _resolve_default_csv_path(cfg) -> Path:
    try:
        base = Path(cfg.original_points_all) if getattr(cfg, "original_points_all", None) else None
    except Exception:
        base = None
    if not base:
        base = Path("reports/task1/original/points_all.csv")
    return base

def _coerce_dtypes(df: pd.DataFrame, dtype_num: str) -> pd.DataFrame:
    cast = {"float32": "float32", "float64": "float64"}[dtype_num]
    df = df.copy()
    for c in ["LONGITUDE","LATITUDE","DEPTH","VALUE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(cast)
    if "SOURCE" in df.columns:
        df["SOURCE"] = df["SOURCE"].astype(str)
    return df

def _normalize_source_labels(df: pd.DataFrame, surf_style: str) -> pd.DataFrame:
    df = df.copy()
    if surf_style.upper() == "SURF":
        df["SOURCE"] = (df["SOURCE"]
                        .replace({"SF_ORIG":"SURF_ORIG","SF_DL":"SURF_DL","SURF-ORIG":"SURF_ORIG","SURF-IMPUT":"SURF_DL"}))
    else:
        df["SOURCE"] = (df["SOURCE"]
                        .replace({"SURF_ORIG":"SF_ORIG","SURF_DL":"SF_DL","SURF-ORIG":"SF_ORIG","SURF-IMPUT":"SF_DL"}))
    return df

def _stable_sort(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none":
        return df
    if mode == "lex":
        order = {k:i for i,k in enumerate(CAT_ORDER)}
        key = df["SOURCE"].map(order).fillna(9999)
        return df.assign(_k=key).sort_values(["_k","LONGITUDE","LATITUDE","DEPTH"], kind="mergesort").drop(columns="_k")
    return df.reset_index(drop=True)  # 'notebook'

def _run_eda(cfg, *, value_col: str | None, value_regex: str | None, value_min: float | None, value_max: float | None):
    DEFAULT_REGEX = r"^cu_?ppm(_pred)?$"
    DEFAULT_ALIASES = ["CU_PPM","VALUE"]
    aliases = DEFAULT_ALIASES[:]
    regex = value_regex or DEFAULT_REGEX

    if value_col:
        aliases = [value_col] + [a for a in aliases if a.lower()!=value_col.lower()]
        if not value_regex:
            regex = rf"^(?i:{re.escape(value_col)})$"

    out = process_data_dir(
        cfg,
        value_aliases=aliases,
        value_regex=regex,
        value_min=value_min if value_min is not None else 1e-5,
        value_max=value_max if value_max is not None else 346000.0,
    )
    print("[EDA] Cleaned CSVs written:")
    for k, p in out.items():
        print(f"  - {k:15s} -> {p}")
    return out

def _run_points(cfg, *, fmt: str, engine: str, compression: str, dtype: str, surf_label: str, sort: str, alias: bool):
    tables = build_points_views(cfg)
    df = tables.get("points_all")
    if df is None:
        raise RuntimeError("build_points_views(cfg) did not return 'points_all'")

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns {missing}; got {list(df.columns)}")
    df = df[REQ_COLS]

    df = _normalize_source_labels(df, surf_label)
    df = _coerce_dtypes(df, dtype)
    df = _stable_sort(df, sort)

    out_csv = _resolve_default_csv_path(cfg)
    out_parq = out_csv.with_suffix(".parquet")

    if fmt in ("csv","both"):
        _write_csv(df, out_csv)
        print(f"[POINTS] CSV: {out_csv}  rows={len(df):,}")
    if fmt in ("parquet","both"):
        comp = None if compression=="none" else compression
        _write_parquet(df, out_parq, engine=engine, compression=comp)
        print(f"[POINTS] Parquet({engine},{compression}): {out_parq}  rows={len(df):,}")

    if alias:
        alias_base = out_csv.parent / "all_points"
        if fmt in ("csv","both"):
            _write_csv(df, alias_base.with_suffix(".csv"))
            print(f"[POINTS] alias CSV: {alias_base.with_suffix('.csv')}")
        if fmt in ("parquet","both"):
            comp = None if compression=="none" else compression
            _write_parquet(df, alias_base.with_suffix(".parquet"), engine=engine, compression=comp)
            print(f"[POINTS] alias Parquet: {alias_base.with_suffix('.parquet')}")

    return {"csv": out_csv if fmt in ("csv","both") else None,
            "parquet": out_parq if fmt in ("parquet","both") else None}

def _run_task2(cfg):
    out = recompute_all(cfg)
    for kind, files in out.items():
        print(f"[TASK2:{kind}]")
        for name, p in files.items():
            print(f"  {name}: {p}")
    return out

def main():
    ap = argparse.ArgumentParser(description="One-click pipeline: EDA → points_all → Task2 recompute")
    ap.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")

    # EDA controls
    ap.add_argument("--value-col", type=str, default=None, help="Exact value column name in raw CSVs (e.g., CU_PPM)")
    ap.add_argument("--value-regex", type=str, default=None, help="Custom regex for value column detection")
    ap.add_argument("--value-min", type=float, default=None, help="Minimum allowed value (default 1e-5)")
    ap.add_argument("--value-max", type=float, default=None, help="Maximum allowed value (default 346000.0)")

    # Points controls
    ap.add_argument("--fmt", choices=["csv","parquet","both"], default="both")
    ap.add_argument("--engine", choices=["pyarrow","fastparquet"], default="pyarrow")
    ap.add_argument("--compression", choices=["snappy","gzip","brotli","zstd","none"], default="snappy")
    ap.add_argument("--dtype", choices=["float32","float64"], default="float64")
    ap.add_argument("--surf-label", choices=["SURF","SF"], default="SURF")
    ap.add_argument("--sort", choices=["none","notebook","lex"], default="notebook")
    ap.add_argument("--alias", action="store_true")

    # Control flow
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--skip-points", action="store_true")
    ap.add_argument("--skip-task2", action="store_true")

    args = ap.parse_args()
    cfg = load_cfg(path=args.config) if args.config else load_cfg()

    if not args.skip_eda:
        _run_eda(cfg,
                 value_col=args.value_col,
                 value_regex=args.value_regex,
                 value_min=args.value_min,
                 value_max=args.value_max)
    else:
        print("[SKIP] EDA")

    if not args.skip_points:
        _run_points(cfg,
                    fmt=args.fmt,
                    engine=args.engine,
                    compression=args.compression,
                    dtype=args.dtype,
                    surf_label=args.surf_label,
                    sort=args.sort,
                    alias=args.alias)
    else:
        print("[SKIP] points_all")

    if not args.skip_task2:
        _run_task2(cfg)
    else:
        print("[SKIP] Task2")

if __name__ == "__main__":
    main()
