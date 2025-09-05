# src/cap_task2/io.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from cap_common.config import AppCfg

def _first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of the candidate files exist:\n" + "\n".join(map(str, paths)))

def read_points(kind: str, split: str, cfg: AppCfg) -> pd.DataFrame:
    """
    kind: 'drillhole' | 'surface'
    split: 'all' | 'overlap' | 'origonly' | 'dlonly'

    Tries (in order):
      1) reports/task2/difference/{kind}_points_{split}.csv        #
      2) reports/task2/{kind}_points_{split}.csv                    # 
      3) reports/task2/{kind}_points.csv  (only if split == 'all')  # 
    """
    base_new = cfg.task2_diff_dir                 # reports/task2/difference
    base_old = base_new.parent                    # reports/task2
    candidates = [
        base_new / f"{kind}_points_{split}.csv",
        base_old / f"{kind}_points_{split}.csv",
    ]
    if split == "all":
        candidates.append(base_old / f"{kind}_points.csv")

    path = _first_existing(candidates)
    return pd.read_csv(path)
