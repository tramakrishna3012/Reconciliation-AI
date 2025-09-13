from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import pandas as pd
from loguru import logger


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        # Try datetime
        try:
            df[c] = pd.to_datetime(df[c])
            continue
        except Exception:
            pass
        # Try numeric
        try:
            df[c] = pd.to_numeric(df[c])
            continue
        except Exception:
            pass
        # Fallback to string
        df[c] = df[c].astype(str).fillna("")
    return df


def load_path(path: str | Path) -> pd.DataFrame:
    """Load a CSV or JSON file into a normalized DataFrame.

    - CSV: header inferred
    - JSON: either list[dict] or line-delimited JSON objects
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in {".json", ".jsonl"}:
        try:
            # Try JSON array
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        except json.JSONDecodeError:
            # JSON Lines
            records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    df = _normalize_columns(df)
    df = _coerce_types(df)
    return df


def load_many(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [load_path(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    # Align columns
    all_cols = sorted({c for f in frames for c in f.columns})
    frames = [f.reindex(columns=all_cols) for f in frames]
    return pd.concat(frames, ignore_index=True)


def make_record_text(row: pd.Series, fields: Optional[List[str]] = None) -> str:
    """Create a canonical string representation of a record for embedding.

    If fields is None, use all columns sorted by name.
    """
    if fields is None:
        fields = sorted(list(row.index))
    parts = []
    for f in fields:
        val = row.get(f, "")
        if pd.isna(val):
            val = ""
        parts.append(f"{f}: {val}")
    return " | ".join(map(str, parts))


def dataframe_to_texts(df: pd.DataFrame, fields: Optional[List[str]] = None) -> List[str]:
    return [make_record_text(r, fields) for _, r in df.iterrows()]


def ensure_id(df: pd.DataFrame, id_col: str = "_id") -> pd.DataFrame:
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = range(1, len(df) + 1)
    return df
