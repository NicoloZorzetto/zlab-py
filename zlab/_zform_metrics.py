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


# --- Reference scales for normalization ---

_METRIC_RANGES = {
    "r2": (0.0, 1.0),
    "adjr2": (0.0, 1.0),
    "rmse": (0.0, 10.0),
    "mae": (0.0, 10.0),
    "aic": (-1000, 1000),
    "bic": (-1000, 1000),
}

_METRIC_STATS = {
    "r2": (0.5, 0.25),
    "adjr2": (0.5, 0.25),
    "rmse": (2.0, 1.0),
    "mae": (2.0, 1.0),
    "aic": (0.0, 500.0),
    "bic": (0.0, 500.0),
}


def _normalize_score(metric, value, method="minmax"):
    """Normalize a metric value globally to [0,1] or Z-score, handling direction."""
    if not np.isfinite(value):
        return np.nan

    # Flip so that "higher is better" for all metrics
    flipped = -value if not is_higher_better(metric) else value

    if method == "zscore":
        mean, std = _METRIC_STATS.get(metric, (0.0, 1.0))
        return (flipped - mean) / (std + 1e-12)
    else:  # minmax
        low, high = _METRIC_RANGES.get(metric, (flipped - 1, flipped + 1))
        return (flipped - low) / (high - low + 1e-12)


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


def compute_multi_metrics(eval_metric, y_true, y_pred, k=0):
    """
    Compute one or more evaluation metrics.

    Supports:
      - str: single metric ("r2")
      - list[str]: multiple metrics, equal weight
      - dict[str, float]: weighted metrics
      - callable: custom scoring function (y_true, y_pred, k) -> float

    Returns
    -------
    dict[str, float]
        A dictionary of metric_name -> value.
        If eval_metric is callable, returns {'custom': value}.
    """
    if callable(eval_metric):
        try:
            return {"custom": float(eval_metric(y_true, y_pred, k))}
        except Exception:
            return {"custom": np.nan}

    if isinstance(eval_metric, str):
        return {eval_metric: compute_metric(eval_metric, y_true, y_pred, k)}

    if isinstance(eval_metric, (list, tuple)):
        return {m: compute_metric(m, y_true, y_pred, k) for m in eval_metric}

    if isinstance(eval_metric, dict):
        return {m: compute_metric(m, y_true, y_pred, k) for m in eval_metric.keys()}

    raise TypeError(
        f"Unsupported eval_metric type: {type(eval_metric)}. Must be str, list, dict, or callable."
    )


def compute_composite_score(metrics_dict, eval_metric, normalize=False, normalize_method="minmax"):
    """
    Combine multiple metric scores into a composite score.

    metrics_dict: dict of {metric_name: score}
    eval_metric: str | list[str] | dict[str, float]
    normalize: bool
    normalize_method: {'minmax', 'zscore'}
    """
    from zlab._zform_metrics import is_higher_better  # local import for safety
    import numpy as np

    # --- prepare metrics & weights ---
    if isinstance(eval_metric, str):
        return metrics_dict.get(eval_metric, np.nan)
    elif isinstance(eval_metric, list):
        weights = {m: 1.0 for m in eval_metric}
    elif isinstance(eval_metric, dict):
        weights = eval_metric
    else:
        raise TypeError("eval_metric must be str, list, or dict")

    # --- normalization & direction handling ---
    scores = {}
    for m, w in weights.items():
        if m not in metrics_dict or not np.isfinite(metrics_dict[m]):
            continue
        val = metrics_dict[m]

        # Direction-aware normalization
        if normalize:
            val = _normalize_score(m, val, method=normalize_method)
        else:
            if not is_higher_better(m):
                val = -val

        scores[m] = val * w

    if not scores:
        return np.nan

    return sum(scores.values()) / sum(abs(w) for w in weights.values())
