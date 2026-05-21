"""
Tests for zform eval metrics.

Run with:
  pytest tests/zform/unit/test_eval_metrics.py
or:
  python tests/zform/unit/test_eval_metrics.py
"""

import numpy as np
import pandas as pd
import pytest

from zlab import zform
from zlab.zform._zform_metrics import (
    compute_composite_score,
    compute_multi_metrics,
)
from zlab.warnings import ZformWarning


def test_compute_composite_score_uses_custom_combiner():
    def combine(metrics):
        return metrics["r2"] - 0.1 * metrics["aic"]

    metrics = {"r2": 0.8, "aic": 2.0}

    score = compute_composite_score(
        metrics,
        ["r2", "aic"],
        composite_metric_func=combine,
    )

    assert score == pytest.approx(0.6)


def test_compute_composite_score_custom_combiner_exception_warns():
    def bad_combiner(metrics):
        raise RuntimeError("boom")

    with pytest.warns(ZformWarning):
        score = compute_composite_score(
            {"r2": 0.8},
            ["r2"],
            composite_metric_func=bad_combiner,
        )

    assert np.isnan(score)


def test_compute_composite_score_accepts_callable_eval_metric_result():
    def custom_metric(y_true, y_pred, k):
        return 42.0

    metrics = compute_multi_metrics(
        custom_metric,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        k=0,
    )

    score = compute_composite_score(metrics, custom_metric)

    assert score == pytest.approx(42.0)


def test_compute_composite_score_allows_negative_infinity():
    def combine(metrics):
        return metrics["r2"] - 0.1 * metrics["aic"]

    score = compute_composite_score(
        {"r2": 1.0, "aic": np.inf},
        ["r2", "aic"],
        composite_metric_func=combine,
    )

    assert score == -np.inf
