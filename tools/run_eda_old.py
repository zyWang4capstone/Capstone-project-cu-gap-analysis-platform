from __future__ import annotations
from pathlib import Path
import sys

#./src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cap_common.config import load_cfg
from cap_eda.ingest import process_data_dir

def main() -> None:
    cfg = load_cfg()
    
    out = process_data_dir(
        cfg,
        value_aliases=["CU_PPM","VALUE"],
        value_regex=r"^cu_?ppm(_pred)?$",
        value_min=1e-5,
        value_max=346000.0,
    )
    print("[EDA] Cleaned CSVs written:")
    for k, p in out.items():
        print(f"  - {k:15s} -> {p}")

if __name__ == "__main__":
    main()
