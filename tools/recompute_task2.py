# tools/recompute_task2.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cap_common.config import load_cfg
from cap_task2.overlap import recompute_all

if __name__ == "__main__":
    out = recompute_all(load_cfg())
    for kind, files in out.items():
        print(f"[{kind}]")
        for name, p in files.items():
            print(f"  {name}: {p}")
