from __future__ import annotations
import pandas as pd

def get_row(df: pd.DataFrame, idx: int) -> dict:
    """Return one row as a dict (used for simulated streaming)."""
    return df.iloc[idx].to_dict()

