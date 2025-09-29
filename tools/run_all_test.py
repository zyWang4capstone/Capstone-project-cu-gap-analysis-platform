from __future__ import annotations
import argparse, sys, re
from pathlib import Path
import pandas as pd

#light imports for SHP→CSV staging (kept optional)
import zipfile, tempfile, shutil, warnings

# Ensure backend src on path (repo structure: this_script/../src)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cap_common.config import load_cfg
from cap_eda.ingest import process_data_dir
from cap_original.points import build_points_views
from cap_task2.overlap import recompute_all

# ---------------------------------------------------------------------
# Constants 
CAT_ORDER = ["DH_ORIG","DH_DL","SURF_ORIG","SURF_DL"]
REQ_COLS = ["LONGITUDE","LATITUDE","DEPTH","VALUE","SOURCE"]

# ---------------------------------------------------------------------
# Small utilities 
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _write_parquet(df: pd.DataFrame, p: Path, engine: str, compression: str | None) -> None:
    _ensure_dir(p)
    df.to_parquet(p, index=False, engine=engine, compression=compression)

def _write_csv(df: pd.DataFrame, p: Path) -> None:
    _ensure_dir(p)
    df.to_csv(p, index=False)

def _resolve_default_csv_path(cfg) -> Path:
    """
    Returns the canonical output path for points_all CSV.
    Falls back to reports/task1/... when cfg doesn't specify it.
    """
    try:
        base = Path(cfg.original_points_all) if getattr(cfg, "original_points_all", None) else None
    except Exception:
        base = None
    if not base:
        base = Path("reports/task1/original/points_all.csv")
    return base

def _coerce_dtypes(df: pd.DataFrame, dtype_num: str) -> pd.DataFrame:
    """
    Cast numeric columns to the chosen float dtype, keep SOURCE as string.
    """
    cast = {"float32": "float32", "float64": "float64"}[dtype_num]
    df = df.copy()
    for c in ["LONGITUDE","LATITUDE","DEPTH","VALUE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(cast)
    if "SOURCE" in df.columns:
        df["SOURCE"] = df["SOURCE"].astype(str)
    return df

def _normalize_source_labels(df: pd.DataFrame, surf_style: str) -> pd.DataFrame:
    """
    Normalize SOURCE labels to the expected style (SURF_* vs SF_*).
    """
    df = df.copy()
    if surf_style.upper() == "SURF":
        df["SOURCE"] = (df["SOURCE"]
                        .replace({"SF_ORIG":"SURF_ORIG","SF_DL":"SURF_DL",
                                  "SURF-ORIG":"SURF_ORIG","SURF-IMPUT":"SURF_DL"}))
    else:
        df["SOURCE"] = (df["SOURCE"]
                        .replace({"SURF_ORIG":"SF_ORIG","SURF_DL":"SF_DL",
                                  "SURF-ORIG":"SF_ORIG","SURF-IMPUT":"SF_DL"}))
    return df

def _stable_sort(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Reproduce notebook/lexicographic order without changing the schema.
    """
    if mode == "none":
        return df
    if mode == "lex":
        order = {k:i for i,k in enumerate(CAT_ORDER)}
        key = df["SOURCE"].map(order).fillna(9999)
        return df.assign(_k=key).sort_values(
            ["_k","LONGITUDE","LATITUDE","DEPTH"], kind="mergesort"
        ).drop(columns="_k")
    return df.reset_index(drop=True)  # 'notebook'

# ---------------------------------------------------------------------


def _find_latest_zip(data_dir: Path) -> Path | None:
    """
    Pick the most recent *.zip under data_dir (if any).
    """
    zips = sorted(data_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None

def _zip_contains_shapefile(zip_path: Path) -> bool:
    """
    Heuristic: if any member endswith .shp, we treat it as a Shapefile bundle.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return any(n.lower().endswith(".shp") for n in zf.namelist())
    except Exception:
        return False

def _unpack_zip(zip_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)

def _find_shp_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.shp"))

def _read_gdf(shp: Path):
    """
    Read a .shp using GeoPandas. If CRS is missing, we assume EPSG:4326 (WGS84).
    Always project to EPSG:4326 so we can safely derive LONGITUDE/LATITUDE.
    """
    try:
        import geopandas as gpd
    except Exception as e:
        raise RuntimeError(
            "GeoPandas is required for Shapefile conversion. "
            "Install with: pip install geopandas pyogrio (or geopandas fiona)"
        ) from e

    gdf = gpd.read_file(shp)

    if gdf.crs is None:
        warnings.warn(f"[CRS] {shp.name} has no CRS; assuming EPSG:4326 (WGS84).")
        gdf = gdf.set_crs(epsg=4326)

    # Reproject to EPSG:4326 if needed
    try:
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
    except Exception:
        # Some proj strings may not map cleanly; force to 4326
        gdf = gdf.to_crs(4326)
    return gdf

def _geometry_to_lonlat(gdf) -> tuple[pd.Series, pd.Series]:
    """
    If geometries are points, use x/y. Otherwise (lines/polygons) use centroids.
    """
    geom = gdf.geometry
    if geom is None:
        raise ValueError("No geometry column found in the shapefile.")
    try:
        if all(t == "Point" for t in geom.geom_type):
            return pd.Series(geom.x, index=gdf.index), pd.Series(geom.y, index=gdf.index)
    except Exception:
        pass
    cen = geom.centroid
    return pd.Series(cen.x, index=gdf.index), pd.Series(cen.y, index=gdf.index)

def _to_upper_drop_geometry(gdf) -> pd.DataFrame:
    """
    Return a plain pandas.DataFrame:
    - drop the active geometry column by its *name* (not the GeoSeries object)
    - uppercase all attribute column names
    - as a final guard, remove any residual 'GEOMETRY' column
    """
    # GeoPandas' active geometry is accessible as a GeoSeries via gdf.geometry,
    # but DataFrame.drop expects the *column name*, which is gdf.geometry.name.
    geom_name = None
    try:
        # If a geometry is set, get its column name (usually 'geometry')
        geom_name = getattr(gdf, "geometry", None)
        geom_name = getattr(geom_name, "name", None)
    except Exception:
        geom_name = None

    if geom_name and geom_name in gdf.columns:
        df = gdf.drop(columns=[geom_name])
    else:
        # Fall back to a plain DataFrame copy
        df = pd.DataFrame(gdf)

    # Uppercase attribute names
    df.columns = [str(c).upper() for c in df.columns]

    # Safety guard: if a 'GEOMETRY' column still slipped through, drop it
    if "GEOMETRY" in df.columns:
        df = df.drop(columns=["GEOMETRY"])

    return df


def _convert_shp_zip_to_csv_zip(shp_zip: Path, data_dir: Path) -> Path:
    """
    Convert a Shapefile ZIP (with 4 .shp sets) to 4 CSVs (same stems), then pack
    them into <stem>_csv.zip under ./data for the EDA stage.

    IMPORTANT: To avoid "Multiple .zip files under ./data" errors during EDA,
    we temporarily MOVE the original SHP-zip into ./data/_quarantine BEFORE packing.
    - If packing/conversion SUCCEEDS: we DELETE the quarantined original zip.
    - If anything FAILS: we RESTORE the original zip back to its original place.

    This guarantees that by the time EDA scans ./data, there is only one .zip left.
    """
    with tempfile.TemporaryDirectory(prefix="shp2csv_") as tmpd:
        tmp = Path(tmpd)

        # ---------------- Safety move: quarantine the original zip if it's under ./data
        quarantine = None
        src_zip_for_read = shp_zip  # path we will actually read from
        try:
            if shp_zip.resolve().parent.resolve() == data_dir.resolve():
                qdir = data_dir / "_quarantine"
                qdir.mkdir(parents=True, exist_ok=True)
                quarantine = qdir / shp_zip.name
                # Move the original SHP zip OUT of ./data so EDA won't see two zips later
                shutil.move(str(shp_zip), str(quarantine))
                print(f"[CLEANUP] Temporarily moved source ZIP to: {quarantine}")
                src_zip_for_read = quarantine
        except Exception as e:
            warnings.warn(f"[CLEANUP] Failed to quarantine source zip: {e}. Continuing without move.")
            quarantine = None
            src_zip_for_read = shp_zip

        try:
            # -------------- Unpack (read from the quarantine if we moved it)
            _unpack_zip(src_zip_for_read, tmp)

            # -------------- Find and convert all .shp

            shp_files_all = _find_shp_files(tmp)

            # Drop entries inside "__MACOSX" and AppleDouble companions starting with "._"
            # This works across Windows/macOS/Linux paths.
            def _is_macos_junk(p: Path) -> bool:
                return p.name.startswith("._") or ("__MACOSX" in p.parts)

            filtered = [p for p in shp_files_all if not _is_macos_junk(p)]

            # (Optional but recommended) keep only complete shapefile trios: .shp + .dbf + .shx
            def _is_complete_shapefile(p: Path) -> bool:
                return p.with_suffix(".dbf").exists() and p.with_suffix(".shx").exists()

            shp_files = [p for p in filtered if _is_complete_shapefile(p)]

            skipped = len(shp_files_all) - len(shp_files)
            if not shp_files:
                raise RuntimeError("No valid .shp files found in the ZIP after filtering macOS metadata and incomplete sets.")
            if skipped > 0:
                warnings.warn(f"[INFO] Skipped {skipped} macOS metadata/partial entries (.__/.__MACOSX/._*).")
            if len(shp_files) != 4:
                warnings.warn(f"[WARN] Found {len(shp_files)} real .shp files (expected 4). Converting all found.")

            shp_files = _find_shp_files(tmp)
            if not shp_files:
                raise RuntimeError("No .shp files found in the provided ZIP.")
            if len(shp_files) != 4:
                warnings.warn(f"[WARN] Found {len(shp_files)} .shp files (expected 4). Converting all found.")


            staged = data_dir / f"staged_{shp_zip.stem}"
            staged.mkdir(parents=True, exist_ok=True)
            out_csvs: list[Path] = []

            for shp in shp_files:
                gdf = _read_gdf(shp)
                lon, lat = _geometry_to_lonlat(gdf)
                df = _to_upper_drop_geometry(gdf)

                # Always use geometry-derived LONGITUDE/LATITUDE
                for col in ("LONGITUDE", "LATITUDE"):
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True)

                df.insert(0, "LATITUDE",  pd.to_numeric(lat, errors="coerce"))
                df.insert(0, "LONGITUDE", pd.to_numeric(lon, errors="coerce"))

                out_csv = staged / (shp.stem + ".csv")
                _write_csv(df, out_csv)
                print(f"[SHP→CSV] {shp.name} -> {out_csv.name}  rows={len(df):,}")
                out_csvs.append(out_csv)

            # -------------- Pack CSVs into a single zip under ./data
            out_zip = data_dir / f"{shp_zip.stem}_csv.zip"
            with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in out_csvs:
                    zf.write(p, arcname=p.name)
            print(f"[PACK] {len(out_csvs)} CSVs packed -> {out_zip}")

            # -------------- Success: remove the quarantined original if present
            if quarantine and quarantine.exists():
                try:
                    quarantine.unlink()
                    print(f"[CLEANUP] Deleted source SHP ZIP: {quarantine.name}")
                except Exception as e:
                    warnings.warn(f"[CLEANUP] Failed to delete quarantined zip: {e}")

            return out_zip

        except Exception as e:
            # If anything failed and we had moved the original, restore it back
            if quarantine and quarantine.exists() and not shp_zip.exists():
                try:
                    shutil.move(str(quarantine), str(shp_zip))
                    print(f"[CLEANUP] Restored original SHP ZIP to: {shp_zip}")
                except Exception as ee:
                    warnings.warn(f"[CLEANUP] Failed to restore original zip: {ee}")
            raise


def _auto_stage_raw_zip_for_eda(raw_zip: Path | None) -> None:
    """
    Optional pre-EDA adapter:
    - If raw_zip is provided and is a SHP-zip, convert and produce a CSV-zip under ./data.
    - If raw_zip is None, look for the latest *.zip in ./data and, if it's a SHP-zip, convert it.
    - If it's already a CSV-zip, do nothing — EDA will proceed as before.
    This preserves the original downstream behavior (no schema/flow changes).
    """
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Choose candidate zip
    candidate = raw_zip if raw_zip else _find_latest_zip(data_dir)
    if not candidate:
        # No zip at all — nothing to stage; EDA may rely on existing unpacked CSVs.
        print("[RAW] No ZIP found to stage; proceeding with existing data.")
        return

    candidate = candidate.resolve()
    if _zip_contains_shapefile(candidate):
        print(f"[RAW] Detected Shapefile ZIP: {candidate.name} -> converting to CSV ZIP…")
        _convert_shp_zip_to_csv_zip(candidate, data_dir)
    else:
        # If it's already under ./data, keep as-is; otherwise copy it there
        target = data_dir / candidate.name
        if target.resolve() != candidate:
            shutil.copy2(candidate, target)
        print(f"[RAW] CSV ZIP staged: {target}")

# ---------------------------------------------------------------------
# EDA stage 
def _run_eda(cfg, *, value_col: str | None, value_regex: str | None,
             value_min: float | None, value_max: float | None):
    """
    Run the ingestion/cleaning stage with a gentle default:
    - If value_col is given, prefer exact-match regex and keep common aliases.
    - Otherwise fall back to the legacy default regex.
    """
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

# ---------------------------------------------------------------------
# Points stage 
def _run_points(cfg, *, fmt: str, engine: str, compression: str,
                dtype: str, surf_label: str, sort: str, alias: bool):
    """
    Build unified points view and write CSV/Parquet according to args.
    """
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

# ---------------------------------------------------------------------
# Task2 stage 
def _run_task2(cfg):
    out = recompute_all(cfg)
    for kind, files in out.items():
        print(f"[TASK2:{kind}]")
        for name, p in files.items():
            print(f"  {name}: {p}")
    return out

# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="One-click pipeline: (optional SHP→CSV staging) → EDA → points_all → Task2")
    ap.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")


    ap.add_argument("--raw-zip", type=str, default=None,
                    help="Optional path to a raw ZIP. If it's a Shapefile ZIP, "
                         "we convert to a CSV ZIP under ./data before EDA.")

    # EDA controls
    ap.add_argument("--value-col", type=str, default=None,
                    help="Exact value column name in raw CSVs (e.g., CU_PPM)")
    ap.add_argument("--value-regex", type=str, default=None,
                    help="Custom regex for value column detection")
    ap.add_argument("--value-min", type=float, default=None,
                    help="Minimum allowed value (default 1e-5)")
    ap.add_argument("--value-max", type=float, default=None,
                    help="Maximum allowed value (default 346000.0)")

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

    #stage raw ZIP if needed (SHP→CSV zip); kept optional and non-breaking.
    _auto_stage_raw_zip_for_eda(Path(args.raw_zip) if args.raw_zip else None)

    # Downstream stages remain exactly the same:
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
