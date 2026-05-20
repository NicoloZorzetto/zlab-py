"""
Tests for zform engines.

Run with:
  pytest tests/zform/unit/test_engines.py
or:
  python tests/zform/unit/test_engines.py
"""

import importlib
import builtins

import numpy as np
import pandas as pd
import pytest

from zlab import zform
from zlab.zform import _zform_eval_engines as eval_engines


def test_eval_engine_ols_returns_predictions():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0

    y_pred, info = eval_engines.evaluate_engine(x, y, "ols")

    np.testing.assert_allclose(y_pred, y, rtol=1e-10, atol=1e-10)
    assert info["engine"] == "ols"
    assert info["coef"][0] == pytest.approx(2.0)
    assert info["intercept"] == pytest.approx(1.0)


def zero_engine(x, y, **kwargs):
    return np.array([0.0]), 0.0, np.zeros_like(y)


def test_eval_engine_custom_callable():
    y_pred, info = eval_engines.evaluate_engine(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        zero_engine,
    )

    assert np.all(y_pred == 0.0)
    assert info["engine"] == "zero_engine"


def test_eval_engines_handles_missing_sklearn(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sklearn"):
            raise ImportError("boom")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    module = importlib.reload(eval_engines)
    assert module.SKLEARN_AVAILABLE is False

    monkeypatch.undo()
    importlib.reload(eval_engines)


def test_ridge_engine_requires_sklearn(monkeypatch):
    monkeypatch.setattr(eval_engines, "SKLEARN_AVAILABLE", False)

    with pytest.raises(ImportError, match="Ridge"):
        eval_engines._ridge_engine(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_lasso_engine_requires_sklearn(monkeypatch):
    monkeypatch.setattr(eval_engines, "SKLEARN_AVAILABLE", False)

    with pytest.raises(ImportError, match="Lasso"):
        eval_engines._lasso_engine(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_get_eval_engine_unknown_name():
    with pytest.raises(KeyError, match="Unknown engine"):
        eval_engines.get_eval_engine("does_not_exist")


def test_zform_uses_named_engine():
    df = pd.DataFrame(
        {
            "y": np.linspace(1.0, 5.0, 20),
            "x": np.linspace(1.0, 5.0, 20),
        }
    )

    z = zform(
        df,
        y="y",
        x="x",
        transformations=["linear"],
        engine="ridge",
        strategy="best",
        verbose=False,
        min_obs=3,
    )

    assert len(z) == 1
    assert "Best R2" in z.columns
