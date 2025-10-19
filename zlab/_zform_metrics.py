"""
zform_metrics.py — Metrics and evaluation utilities for zform.

This module defines model evaluation metrics used to assess transformation quality.
All metrics must accept (y_true, y_pred, k) and return a scalar score.

Higher-is-better metrics:  r2, adjr2  
Lower-is-better metrics:   rmse, mae, aic, bic
"""

import numpy as np
from zlab.warnings import ZformWarning


# --- Metric functions ---

def _r2(y, yhat, k):
    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum((y - yhat) ** 2)
    return 1 - rss / tss if tss != 0 else np.nan


def _adjr2(y, yhat, k):
    n = len(y)
    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum((y - yhat) ** 2)
    r2 = 1 - rss / tss if tss != 0 else np.nan
    return 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan


def _rmse(y, yhat, k):
    return np.sqrt(np.mean((y - yhat) ** 2))


def _mae(y, yhat, k):
    return np.mean(np.abs(y - yhat))


def _aic(y, yhat, k):
    n = len(y)
    rss = np.sum((y - yhat) ** 2)
    return n * np.log(rss / n) + 2 * k if rss > 0 else np.inf


def _bic(y, yhat, k):
    n = len(y)
    rss = np.sum((y - yhat) ** 2)
    return n * np.log(rss / n) + k * np.log(n) if rss > 0 else np.inf


# --- Registry and helpers ---

METRICS = {
    "r2": _r2,
    "adjr2": _adjr2,
    "rmse": _rmse,
    "mae": _mae,
    "aic": _aic,
    "bic": _bic,
}

_METRIC_DIRECTION = {
    "r2": True,
    "adjr2": True,
    "rmse": False,
    "mae": False,
    "aic": False,
    "bic": False,
}

def is_higher_better(metric_name: str) -> bool:
    """Return True if the metric should be maximized."""
    return _METRIC_DIRECTION.get(metric_name.lower(), True)


# --- Metric computation ---

def compute_metric(name: str, y_true, y_pred, k: int = 0):
    """Compute a given metric by name.

    Parameters
    ----------
    name : str
        Metric name (e.g. 'r2', 'aic', 'rmse', ...).
    y_true, y_pred : array-like
        True and predicted values.
    k : int
        Number of fitted parameters (used in adjusted R², AIC, BIC).

    Returns
    -------
    float
        Metric score.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    if y_true.size == 0 or np.any(~np.isfinite(y_pred)):
        return np.nan

    func = METRICS.get(name.lower())
    if func is None:
        ZformWarning(f"Unknown eval_metric '{name}', falling back to R².")
        func = METRICS["r2"]

    return func(y_true, y_pred, k)
