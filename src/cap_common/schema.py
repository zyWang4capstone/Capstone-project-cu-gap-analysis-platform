from __future__ import annotations
import re
import pandas as pd

def standardize(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Standardize column names using aliases and optional regex.
    
    Behavior:
    1) Uppercase + trim incoming headers.
    2) Map known aliases to canonical keys: longitude, latitude, value.
    3) If the value column is still missing and schema['value_regex'] is set,
       pick the first column whose name matches the regex (case-insensitive),
       preferring names listed in aliases['value'] when multiple match.
    """
    # 1) Uppercase/trim first
    renamed = {c: str(c).upper().strip() for c in df.columns}
    df = df.rename(columns=renamed)

    # 2) Exact alias mapping
    alias = schema.get("aliases", {}) or {}

    def _ensure(col: str, candidates: list[str]):
        if col in df.columns:
            return
        for c in candidates or []:
            u = str(c).upper()
            if u in df.columns:
                df[col] = df[u]
                return

    lon_key = schema.get("longitude", "LONGITUDE")
    lat_key = schema.get("latitude",  "LATITUDE")
    val_key = schema.get("value",     "VALUE")

    _ensure(lon_key, (alias.get("lon") or []))
    _ensure(lat_key, (alias.get("lat") or []))
    _ensure(val_key, (alias.get("value") or []))

    # 3) Fuzzy regex fallback for value column
    if val_key not in df.columns:
        val_regex = (schema.get("value_regex") or "").strip()
        if val_regex:
            # find all matching columns
            cand = [c for c in df.columns if re.search(val_regex, c, flags=re.IGNORECASE)]
            if cand:
                # prefer aliases order if provided
                pref = [str(c).upper() for c in (alias.get("value") or [])]
                chosen = None
                for p in pref:
                    if p in cand:
                        chosen = p
                        break
                if chosen is None:
                    chosen = cand[0]
                df[val_key] = df[chosen]
                print(f"[schema.standardize] value_regex matched '{chosen}' → '{val_key}'")
    return df
