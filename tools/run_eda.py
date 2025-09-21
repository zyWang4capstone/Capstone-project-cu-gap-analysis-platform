from __future__ import annotations
import argparse
import sys, re
from pathlib import Path

# Ensure ./src is importable (script assumed under repo/tools/)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cap_common.config import load_cfg
from cap_eda.ingest import process_data_dir

DEFAULT_REGEX = r"^cu_?ppm(_pred)?$"
DEFAULT_ALIASES = ["CU_PPM", "VALUE"]
DEFAULT_MIN = 1e-5
DEFAULT_MAX = 346000.0

def main() -> None:
    ap = argparse.ArgumentParser(description="Task1 EDA: process ZIP in ./data → 4 cleaned CSVs (robust value-column control)")
    ap.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    ap.add_argument("--value-col", type=str, default=None, help="Exact name of the value column (e.g., CU_PPM)")
    ap.add_argument("--value-regex", type=str, default=None, help=f"Regex to detect value column (default: {DEFAULT_REGEX})")
    ap.add_argument("--value-min", type=float, default=None, help=f"Minimum allowed value (default: {DEFAULT_MIN})")
    ap.add_argument("--value-max", type=float, default=None, help=f"Maximum allowed value (default: {DEFAULT_MAX})")
    ap.add_argument("--interactive", action="store_true", help="Prompt for column & range in terminal if not provided")
    args = ap.parse_args()

    # Build aliases & regex from inputs
    aliases = DEFAULT_ALIASES[:]
    value_regex = args.value_regex or DEFAULT_REGEX
    if args.value_col:
        aliases = [args.value_col] + [a for a in aliases if a.lower() != args.value_col.lower()]
        if not args.value_regex:
            value_regex = rf"^(?i:{re.escape(args.value_col)})$"
    vmin = DEFAULT_MIN if args.value_min is None else float(args.value_min)
    vmax = DEFAULT_MAX if args.value_max is None else float(args.value_max)

    # Optional prompts (only when interactive flag is set and no explicit CLI values were given)
    def _isatty() -> bool:
        try:
            return sys.stdin.isatty()
        except Exception:
            return False

    if args.interactive and _isatty():
        if not args.value_col and not args.value_regex:
            user_col = input(f"[run_eda] Enter value column name (default aliases {aliases} / regex {value_regex}): ").strip()
            if user_col:
                aliases = [user_col] + [a for a in aliases if a.lower() != user_col.lower()]
                value_regex = rf"^(?i:{re.escape(user_col)})$"
        try:
            s = input(f"[run_eda] Enter min,max range (default {vmin},{vmax}): ").strip()
            if s:
                smin, smax = s.split(",")
                vmin, vmax = float(smin), float(smax)
        except Exception:
            print("[run_eda] Range input ignored; using defaults.")

    cfg = load_cfg(path=args.config) if args.config else load_cfg()
    out = process_data_dir(
        cfg,
        value_aliases=aliases,
        value_regex=value_regex,
        value_min=vmin,
        value_max=vmax,
    )
    print("[EDA] Cleaned CSVs written:")
    for k, p in out.items():
        print(f"  - {k:15s} -> {p}")

if __name__ == "__main__":
    main()
