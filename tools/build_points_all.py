# tools/build_points_all.py
# One-click builder for points_all.{csv,parquet} (and optional all_points.* alias)
from __future__ import annotations
from pathlib import Path
import sys
import argparse
import pandas as pd

# Make ./src importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cap_common.config import load_cfg  # type: ignore
from cap_original.points import build_points_views  # type: ignore

REQ_COLS = {"LONGITUDE", "LATITUDE", "DEPTH", "VALUE", "SOURCE"}

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _write_csv(df: pd.DataFrame, p: Path) -> None:
    _ensure_dir(p)
    df.to_csv(p, index=False)

def _write_parquet(df: pd.DataFrame, p: Path) -> None:
    _ensure_dir(p)
    try:
        df.to_parquet(p, index=False, engine="pyarrow")
    except Exception:
        try:
            df.to_parquet(p, index=False, engine="fastparquet")
        except Exception as e:
            print(f"[WARN] Failed to write parquet at {p}: {e}")

def resolve_default_csv_path(cfg) -> Path:
    # Prefer app.yaml's original_points_all; fallback to reports/task1/original/points_all.csv
    try:
        base = Path(cfg.original_points_all) if cfg.original_points_all else None
    except Exception:
        base = None
    if not base:
        base = Path("reports/task1/original/points_all.csv")
    return base

def main():
    ap = argparse.ArgumentParser(description="Build points_all.* from cleaned Task1 tables")
    ap.add_argument("--fmt", choices=["csv", "parquet", "both"], default="both",
                    help="Which formats to write (default: both)")
    ap.add_argument("--alias", action="store_true",
                    help="Also write all_points.{csv,parquet} alongside points_all.*")
    args = ap.parse_args()

    cfg = load_cfg()
    tables = build_points_views(cfg)
    df = tables.get("points_all")
    if df is None:
        raise RuntimeError("build_points_views(cfg) did not return 'points_all'")

    # Validate required columns (case-insensitive)
    cols_upper = {c.upper() for c in df.columns}
    missing = REQ_COLS - cols_upper
    if missing:
        raise RuntimeError(f"points_all is missing required columns: {missing} — got {list(df.columns)}")

    out_csv = resolve_default_csv_path(cfg)
    out_parq = out_csv.with_suffix(".parquet")

    # Write outputs
    if args.fmt in ("csv", "both"):
        _write_csv(df, out_csv)
        print(f"[OK] CSV written: {out_csv}  (rows={len(df):,})")
    if args.fmt in ("parquet", "both"):
        _write_parquet(df, out_parq)
        if out_parq.exists():
            print(f"[OK] Parquet written: {out_parq}  (rows={len(df):,})")
        else:
            print(f"[WARN] Parquet not written (missing engine).")

    # Optional alias all_points.* in the same directory
    if args.alias:
        alias_base = out_csv.parent / "all_points"
        if args.fmt in ("csv", "both"):
            _write_csv(df, alias_base.with_suffix(".csv"))
            print(f"[OK] CSV alias written: {alias_base.with_suffix('.csv')}")
        if args.fmt in ("parquet", "both"):
            _write_parquet(df, alias_base.with_suffix(".parquet"))
            if alias_base.with_suffix(".parquet").exists():
                print(f"[OK] Parquet alias written: {alias_base.with_suffix('.parquet')}")
            else:
                print(f"[WARN] Parquet alias not written (missing engine).")

    # Print a short schema summary
    print("""[SUMMARY]
Columns: {}
Dtypes : {}
""".format(list(df.columns), dict(df.dtypes)))

if __name__ == "__main__":
    main()
