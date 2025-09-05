from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class AppCfg:
    root: Path
    task1_clean_dir: Path
    task2_diff_dir: Path
    original_points_all: Path | None
    cache_dir: Path
    schema: dict
    grid: dict
    diff: dict
    viz_defaults: dict

def load_cfg(path: str = "configs/app.yaml") -> AppCfg:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = Path(".").resolve()
    d = data["data"]
    return AppCfg(
        root=root,
        task1_clean_dir=root / d["task1_clean_dir"],
        task2_diff_dir=root / d["task2_diff_dir"],
        original_points_all=(root / d["original_points_all"]) if d.get("original_points_all") else None,
        cache_dir=root / d.get("cache_dir", ".cache"),
        schema=data["schema"],
        grid=data["grid"],
        diff=data["diff"],
        viz_defaults=data["viz_defaults"],
    )
