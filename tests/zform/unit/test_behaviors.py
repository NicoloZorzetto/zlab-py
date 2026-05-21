"""
Tests or Zforms metadata helper behavior.

Run with:
  pytest tests/zform/unit/test_behaviors.py
or:
  python tests/zform/unit/test_behaviors.py
"""

import pytest
import pandas as pd

from zlab.zform._zform_metadata import attach_metadata, compute_sha256
from zlab.zform.zforms_behaviors import validate_zforms


def test_compute_sha256_ignores_metadata_column():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    with_meta = df.copy()
    with_meta["__zform_metadata__"] = ["{}", "{}"]

    assert compute_sha256(df) == compute_sha256(with_meta)


def test_compute_sha256_is_stable_under_column_order():
    left = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    right = pd.DataFrame({"y": ["a", "b"], "x": [1.0, 2.0]})

    assert compute_sha256(left) == compute_sha256(right)


def test_compute_sha256_normalizes_numeric_dtype_precision():
    left = pd.DataFrame({"x": pd.Series([1, 2], dtype="int64")})
    right = pd.DataFrame({"x": pd.Series([1.0, 2.0], dtype="float64")})

    assert compute_sha256(left) == compute_sha256(right)


def test_validate_zforms_detects_data_tampering():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    metadata = {"sha256": compute_sha256(df), "custom_functions": []}
    zforms_df = attach_metadata(df, metadata)

    assert validate_zforms(zforms_df, metadata)

    tampered = zforms_df.copy()
    tampered.loc[0, "x"] = 99.0

    with pytest.raises(ValueError, match="integrity validation failed"):
        validate_zforms(tampered, metadata)
