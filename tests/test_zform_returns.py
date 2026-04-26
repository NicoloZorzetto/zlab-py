"""
Returns tests for zform.

Run with:
    pytest tests/test_zform_returns.py
or:
    python tests/test_zform_returns.py
"""

import pandas as pd

from zlab import zform, Zforms
from zlab.datasets import load_iris


def test_zform_returns_iris_simple():
    iris = load_iris()
    z = zform(iris)

    assert isinstance(z, Zforms)
    assert z.validate()


def test_zform_returns_iris_apply():
    iris = load_iris()
    df, z = zform(iris, apply=True)
    assert isinstance(z, Zforms)
    assert isinstance(df, pd.DataFrame)
    assert z.validate()
