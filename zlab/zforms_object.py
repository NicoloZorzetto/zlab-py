"""
zforms_object.py — Object wrapper for zform results.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from zlab._zform_metadata import extract_metadata
from zlab.zforms_behaviors import (
    apply_zforms,
    export_zforms,
    import_zforms,
    validate_zforms,
)
from zlab.warnings import ZformWarning


class Zforms:
    """Wrapper around zform results (DataFrame + metadata + helper methods)."""

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Zforms must be initialized with a pandas DataFrame.")
        self.df = df
        self.metadata = extract_metadata(df)

        if not self.metadata:
            raise ValueError(
                "Missing zform metadata — cannot safely create Zforms object."
            )

    #  --- DataFrame Delegation ---
    def __getattr__(self, name):
        """Delegate attribute access to underlying DataFrame unless overridden."""
        # Only delegate if the attribute is not part of this class
        if name in self.__dict__ or hasattr(Zforms, name):
            return self.__dict__[name]
        return getattr(self.df, name)

    def __getitem__(self, key):
        return self.df[key]

    def __len__(self):
        return len(self.df)

    #  --- High-level Behaviors ---
    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply stored transformations to a DataFrame."""
        return apply_zforms(df, self.df, **kwargs)

    def export_to(self, path: str | Path) -> Path:
        """Export results to a supported file format (CSV, Excel, JSON, etc.)."""
        return export_zforms(self.df, self.metadata, path)

    def summary(self):
        """Print basic summary stats about the fitted transformations."""
        print(f"\nZforms Summary:")
        print(f"  • Records: {len(self.df)}")
        print(f"  • Unique models: {self.df['Best Model'].nunique()}")

        best_cols = [c for c in self.df.columns if c.lower().startswith("best")]
        for c in best_cols:
            vals = pd.to_numeric(self.df[c], errors="coerce")
            if len(vals.dropna()) > 0:
                print(f"  • Avg {c}: {np.nanmean(vals):.4f}")

        print(f"  • Created: {self.metadata.get('created_at', '?')}")
        print(f"  • zlab version: {self.metadata.get('zlab_version', '?')}\n")

    def validate(self) -> bool:
        """Validate internal hash and metadata integrity."""
        return validate_zforms(self.df, self.metadata)

    def to_dataframe(self) -> pd.DataFrame:
        """Return raw pandas DataFrame."""
        return self.df.copy()

    @classmethod
    def from_file(cls, path: str | Path):
        """Import zforms results and automatically validate integrity."""
        df = import_zforms(path)
        obj = cls(df)
        ok = obj.validate()
        if not ok:
            ZformWarning("⚠️ Integrity validation failed after import.")
        return obj

    #  --- Representation ---
    def __repr__(self):
        meta = self.metadata or {}
        return (
            f"<Zforms | {len(self.df)} rows, "
            f"zlab {meta.get('zlab_version', '?')}, "
            f"created {meta.get('created_at', '?')}>"
        )
