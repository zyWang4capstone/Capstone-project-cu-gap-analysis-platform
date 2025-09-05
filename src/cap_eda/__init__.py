# -*- coding: utf-8 -*-
"""cap_eda public API (no side effects on import)."""

from .ingest import (
    process_zip,
    process_data_dir,    # new: auto-pick the only zip under ./data
    find_single_zip,
)
from .clean import (
    clean_surface,
    clean_drillhole,
    detect_value_column, # export detector for advanced users
)

__all__ = [
    "process_zip",
    "process_data_dir",
    "find_single_zip",
    "clean_surface",
    "clean_drillhole",
    "detect_value_column",
]
