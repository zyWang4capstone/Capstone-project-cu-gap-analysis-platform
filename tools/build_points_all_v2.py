# tools/build_points_all_v2.py
from __future__ import annotations
from pathlib import Path
import sys, argparse
import pandas as pd

# import from ./src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from cap_common.config import load_cfg  # type: ignore
from cap_original.points import build_points_views  # type: ignore

CAT_ORDER = ["DH_ORIG","DH_DL","SURF_ORIG","SURF_DL"]
REQ_COLS = ["LONGITUDE","LATITUDE","DEPTH","VALUE","SOURCE"]

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _write_parquet(df: pd.DataFrame, p: Path, engine: str, compression: str) -> None:
    _ensure_dir(p)
    df.to_parquet(p, index=False, engine=engine, compression=compression)

def _write_csv(df: pd.DataFrame, p: Path) -> None:
    _ensure_dir(p)
    df.to_csv(p, index=False)

def _resolve_default_csv_path(cfg) -> Path:
    try:
        base = Path(cfg.original_points_all) if cfg.original_points_all else None
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
    # keep SOURCE as plain string to match notebook
    if "SOURCE" in df.columns:
        df["SOURCE"] = df["SOURCE"].astype(str)
    return df

def _normalize_source_labels(df: pd.DataFrame, surf_style: str) -> pd.DataFrame:
    # surf_style: 'SURF' or 'SF'
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
        # Sort by SOURCE order, then Lon/Lat/Depth
        order = {k:i for i,k in enumerate(CAT_ORDER)}
        key = df["SOURCE"].map(order).fillna(9999)
        return df.assign(_k=key).sort_values(["_k","LONGITUDE","LATITUDE","DEPTH"], kind="mergesort").drop(columns="_k")
    # 'notebook' -> keep concat order: already stable; just reset index
    return df.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Build points_all with notebook-compatible options")
    ap.add_argument("--fmt", choices=["csv","parquet","both"], default="both")
    ap.add_argument("--engine", choices=["pyarrow","fastparquet"], default="pyarrow",
                    help="Parquet engine (default: pyarrow)")
    ap.add_argument("--compression", choices=["snappy","gzip","brotli","zstd","none"], default="snappy")
    ap.add_argument("--dtype", choices=["float32","float64"], default="float64",
                    help="Numeric dtype for LONGITUDE/LATITUDE/DEPTH/VALUE")
    ap.add_argument("--surf-label", choices=["SURF","SF"], default="SURF",
                    help="SOURCE label style for surface (default: SURF)")
    ap.add_argument("--sort", choices=["none","notebook","lex"], default="notebook",
                    help="Row ordering strategy")
    ap.add_argument("--alias", action="store_true", help="Also write all_points.* aliases")
    args = ap.parse_args()

    cfg = load_cfg()
    tables = build_points_views(cfg)
    df = tables.get("points_all")
    if df is None:
        raise RuntimeError("build_points_views(cfg) did not return 'points_all'")

    # Standardize columns / order
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns {missing}; got {list(df.columns)}")
    df = df[REQ_COLS]

    # Normalize SOURCE labels and dtypes to match notebook
    df = _normalize_source_labels(df, args.surf_label)
    df = _coerce_dtypes(df, args.dtype)
    df = _stable_sort(df, args.sort)

    out_csv = _resolve_default_csv_path(cfg)
    out_parq = out_csv.with_suffix(".parquet")

    if args.fmt in ("csv","both"):
        _write_csv(df, out_csv)
        print(f"[OK] CSV: {out_csv}  rows={len(df):,}")
    if args.fmt in ("parquet","both"):
        comp = None if args.compression=="none" else args.compression
        _write_parquet(df, out_parq, engine=args.engine, compression=comp)
        print(f"[OK] Parquet({args.engine},{args.compression}): {out_parq}  rows={len(df):,}")

    if args.alias:
        alias = out_csv.parent / "all_points"
        if args.fmt in ("csv","both"):
            _write_csv(df, alias.with_suffix(".csv"))
            print(f"[OK] alias CSV: {alias.with_suffix('.csv')}")
        if args.fmt in ("parquet","both"):
            comp = None if args.compression=="none" else args.compression
            _write_parquet(df, alias.with_suffix(".parquet"), engine=args.engine, compression=comp)
            print(f"[OK] alias Parquet: {alias.with_suffix('.parquet')}")

    print("[SUMMARY] dtypes:", dict(df.dtypes))

if __name__ == "__main__":
    main()
