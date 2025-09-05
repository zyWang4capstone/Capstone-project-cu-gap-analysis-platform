# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile
import io, re
import pandas as pd
from typing import Literal

from cap_common.config import AppCfg, load_cfg
from .clean import clean_surface, clean_drillhole

Kind = Literal["drillhole", "surface"]
Side = Literal["orig", "dl"]

def find_single_zip(data_dir: Path = Path("data")) -> Path:
    zips = sorted(p for p in data_dir.glob("*.zip") if p.is_file())
    if len(zips) == 1:
        return zips[0]
    if len(zips) == 0:
        raise FileNotFoundError("No .zip found under ./data")
    raise RuntimeError(f"Multiple .zip files under ./data: {zips}")

def _read_csv_from_zip(zf: ZipFile, member: str) -> pd.DataFrame:
    buf = io.BytesIO(zf.read(member))
    try:
        return pd.read_csv(buf)
    except UnicodeDecodeError:
        buf.seek(0); return pd.read_csv(buf, encoding="latin-1")

def _tokenize(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", Path(name).name.lower()) if t]

def _detect_kind(df: pd.DataFrame, name: str) -> Kind:
    cols = {c.upper() for c in df.columns}
    if {"FROMDEPTH","TODEPTH","FROM","TO","DEPTH_FROM","DEPTH_TO"}.intersection(cols):
        return "drillhole"
    toks = _tokenize(name)
    return "drillhole" if any(k in toks for k in ("drill","hole","dh")) else "surface"

def _detect_side(name: str) -> Side:
    toks = _tokenize(name)
    if any(k in toks for k in ("dnn","dl","pred","imput","model")): return "dl"
    if any(k in toks for k in ("orig","original","baseline")):      return "orig"
    return "orig"

def _is_clean_name(name: str) -> bool:
    s = name.lower(); return ("clean" in s) or ("cln" in s)

def process_zip(
    zip_path: Path,
    cfg: AppCfg,
    *,
    value_aliases: list[str] | None = None,
    value_regex: str | None = None,
    value_min: float | None = 1e-5,
    value_max: float | None = 346000.0,
    mapping: dict[str, dict] | None = None,  
) -> dict[str, Path]:
    outdir = cfg.task1_clean_dir
    outdir.mkdir(parents=True, exist_ok=True)

    targets = {
        ("drillhole","orig"): outdir / "drillhole_original_clean.csv",
        ("drillhole","dl"):   outdir / "drillhole_dnn_clean.csv",
        ("surface","orig"):   outdir / "surface_original_clean.csv",
        ("surface","dl"):     outdir / "surface_dnn_clean.csv",
    }

    chosen: dict[tuple[str,str], tuple[str,pd.DataFrame]] = {}
    with ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"): continue
            df_raw = _read_csv_from_zip(zf, name)

            # mapping 
            k: Kind | None = None; s: Side | None = None
            if mapping:
                for pat, spec in mapping.items():
                    if re.search(pat, name, re.I):
                        k = spec.get("kind"); s = spec.get("side"); break
            k = k or _detect_kind(df_raw, name)
            s = s or _detect_side(name)
            key = (k, s)

            if key not in chosen:
                chosen[key] = (name, df_raw)
            else:
                prev_name, _ = chosen[key]
               
                if _is_clean_name(name) and not _is_clean_name(prev_name):
                    chosen[key] = (name, df_raw)

 
    missing = [ks for ks in targets if ks not in chosen]
    if missing:
        miss = ", ".join([f"{k}_{s}" for k,s in missing])
        raise RuntimeError(f"ZIP does not contain all four CSVs (missing: {miss}).")

    written: dict[str, Path] = {}
    for (k,s), outp in targets.items():
        name, df_in = chosen[(k,s)]
        if k == "surface":
            cleaned = clean_surface(
                df=df_in,
                value_aliases=value_aliases,
                value_regex=value_regex,
                value_min=value_min,
                value_max=value_max,
            )
        else:
            cleaned = clean_drillhole(
                df=df_in,
                value_aliases=value_aliases,
                value_regex=value_regex,
                value_min=value_min,
                value_max=value_max,
            )
        
        cleaned.to_csv(outp, index=False)
        written[f"{k}_{s}"] = outp

    return written

def process_data_dir(cfg: AppCfg, data_dir: Path = Path("data"), **kwargs) -> dict[str, Path]:
    zp = find_single_zip(data_dir)
    return process_zip(zp, cfg, **kwargs)

if __name__ == "__main__":
    cfg = load_cfg()
    out = process_data_dir(
        cfg,
        value_aliases=["CU_PPM","VALUE"],
        value_regex=r"^cu_?ppm(_pred)?$",
        value_min=1e-5,       
        value_max=346000.0,   
        mapping=None,
    )
    for k,p in out.items():
        print(f"{k}: {p}")
