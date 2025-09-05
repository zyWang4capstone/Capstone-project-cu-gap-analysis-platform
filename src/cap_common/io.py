from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import AppCfg

def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

def read_task1_clean(kind: str, cfg: AppCfg) -> dict[str, pd.DataFrame]:
    mp = {
        "drillhole": ("drillhole_original_clean.csv", "drillhole_dnn_clean.csv"),
        "surface":   ("surface_original_clean.csv",   "surface_dnn_clean.csv"),
    }
    o, d = mp[kind]
    return {
        "orig": _read_csv(cfg.task1_clean_dir / o),
        "dl":   _read_csv(cfg.task1_clean_dir / d),
    }

def read_task2_points(kind: str, split: str, cfg: AppCfg) -> pd.DataFrame:
    p = cfg.task2_diff_dir / f"{kind}_points_{split}.csv"
    return _read_csv(p)
