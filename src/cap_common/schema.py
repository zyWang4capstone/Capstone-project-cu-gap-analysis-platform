from __future__ import annotations
import pandas as pd

def standardize(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    # Uppercase/trim first
    renamed = {c: c.upper().strip() for c in df.columns}
    df = df.rename(columns=renamed)
    alias = schema.get("aliases", {})

    def _ensure(col: str, candidates: list[str]):
        if col in df.columns: 
            return
        for c in candidates:
            if c in df.columns:
                df[col] = df[c]; break

    _ensure(schema["longitude"], alias.get("lon", []))
    _ensure(schema["latitude"],  alias.get("lat", []))
    _ensure(schema["value"],     alias.get("value", []))
    return df
