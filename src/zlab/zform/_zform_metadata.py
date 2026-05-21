"""
_zform_metadata.py — attaches and extracts zform metadata safely

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .zform_functions import get_zform_functions


def make_metadata(custom_funcs=None):
    """Return a metadata dict containing version, time, and function info."""
    if custom_funcs is None:
        custom_funcs = []

    metadata = {
        # "zlab_version": __version__,
        "zlab_version": "0.0.1",
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "custom_functions": [
            {
                "name": f.__name__,
                "source": getattr(f, "__source__", None)
                or getattr(f, "__doc__", "")
                or "lambda",
            }
            for f in custom_funcs
        ],
    }
    metadata["sha256"] = None

    return metadata


def attach_metadata(df: pd.DataFrame, metadata: dict):
    """Embed metadata into a DataFrame._zmeta attribute and a column note (for exports)."""
    df = df.copy()
    df.attrs["zlab_metadata"] = metadata
    # safe for JSON export — will not interfere with columns
    df["__zform_metadata__"] = json.dumps(metadata)
    return df


def extract_metadata(df: pd.DataFrame):
    """Retrieve embedded metadata if present."""
    if "zlab_metadata" in df.attrs:
        return df.attrs["zlab_metadata"]
    if "__zform_metadata__" in df.columns:
        try:
            return json.loads(df["__zform_metadata__"].iloc[0])
        except Exception:
            return None
    return None


def compute_sha256(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA256 hash for Zforms data."""
    df_no_meta = df.drop(columns=["__zform_metadata__"], errors="ignore").copy()
    df_no_meta = df_no_meta.reindex(sorted(df_no_meta.columns), axis=1)

    for col in df_no_meta.select_dtypes(include=[np.number]).columns:
        df_no_meta[col] = df_no_meta[col].astype(float).round(10)

    df_normalized = df_no_meta.fillna("NaN").astype(str)

    canonical = json.dumps(
        df_normalized.to_dict(orient="records"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
