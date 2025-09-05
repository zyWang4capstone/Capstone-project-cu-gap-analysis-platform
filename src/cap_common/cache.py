from __future__ import annotations
from pathlib import Path
import hashlib, json, pandas as pd

def _mk_key(name: str, params: dict, src_files: list[Path]) -> str:
    h = hashlib.md5(name.encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    for f in src_files:
        st = f.stat()
        h.update(f"{f}:{st.st_mtime_ns}:{st.st_size}".encode())
    return h.hexdigest()[:12]

def maybe_cache_csv(df: pd.DataFrame, cache_dir: Path, key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{key}.csv"
    df.to_csv(out, index=False)
    return out
