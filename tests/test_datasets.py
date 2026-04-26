"""
Tests for zlab dataset imports.

Run with:
    pytest tests/test_datasets.py
or:
    python tests/test_datasets.py
"""

import hashlib
import pandas as pd

from zlab.datasets import load_iris, load_penguins


# --- Basic tests ---
def test_datasets_iris_first_last():
    iris = load_iris()
    assert iris.iloc[0, 0] == 5.1
    assert iris.iloc[-1, -1] == 2


def test_penguins_first_last():
    penguins = load_penguins()
    assert penguins.iloc[0, 0] == "Adelie"
    assert penguins.iloc[-1, -1] == 2009


# --- Test hash ---
def hash_df(df: pd.DataFrame) -> str:
    cols = sorted(df.columns.tolist())
    h = (
        df.loc[:, cols]
        .sort_values(by=cols)
        .reset_index(drop=True)
        .round(10)
        .fillna("Na")
        .astype(str)
        .to_json(orient="records", date_format="iso", date_unit="s")
    )
    return hashlib.sha256(h.encode("utf-8")).hexdigest()


def test_datasets_iris_hash():
    iris = load_iris()
    irishash = "861d77203855a18d5c2c5528f42364d9d3701ebb441fabe98d7068368f6428ce"
    assert hash_df(iris) == irishash


def test_datasets_penguins_hash():
    penguins = load_penguins()
    penguinshash = "5a04ed9ed7029207246ff05b52d941d448695ec1a24db8e23fd6f28b06f1ea6e"
    assert hash_df(penguins) == penguinshash
