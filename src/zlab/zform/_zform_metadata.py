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
import pandas as pd
# from zlab import __version__
from zlab.zform_functions import get_zform_functions


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
                or getattr(f, "__doc__", "") or "lambda",
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
    """Compute a stable SHA256 hash for a DataFrame, ignoring dtype and index noise."""
    # Sort columns and rows for consistent ordering
    df_sorted = df.sort_index(axis=1).sort_values(list(df.columns), axis=0, ignore_index=True)
    
    # Convert all floats to consistent string precision
    df_normalized = df_sorted.round(8).astype(str)
    
    # Serialize deterministically
    data_bytes = df_normalized.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()
