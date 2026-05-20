"""
Tests for the ZformConfig validations and helpers.

Run with:
    pytest tests/zform/unit/test_config.py
or:
    python tests/zform/unit/test_config.py
"""

import pytest

from zlab.warnings import ZformRuntimeWarning
from zlab.zform._zform_config import ZformConfig


def test_zform_config_invalid_strategy():
    with pytest.raises(ValueError, match="Invalid strategy"):
        ZformConfig(strategy="random")


def test_zform_config_min_obs_guard():
    with pytest.raises(ValueError, match="min_obs must be ≥ 3"):
        ZformConfig(min_obs=2)


def test_zform_config_requires_positive_maxfev():
    with pytest.raises(ValueError, match="maxfev must be a positive integer"):
        ZformConfig(maxfev=0)


def test_zform_config_requires_integer_n_jobs():
    with pytest.raises(ValueError, match="n_jobs must be an integer"):
        ZformConfig(n_jobs=1.5)


def test_zform_config_rejects_negative_core_counts():
    with pytest.raises(ValueError, match="n_jobs must be -1"):
        ZformConfig(n_jobs=-3)


def test_zform_config_coerces_zero_jobs_with_warning():
    with pytest.warns(ZformRuntimeWarning, match="Coercing n_jobs=1"):
        cfg = ZformConfig(n_jobs=0)
        assert cfg.n_jobs == 1


def test_zform_config_eval_metric_cannot_be_empty():
    with pytest.raises(ValueError, match="eval_metric cannot be an empty string"):
        ZformConfig(eval_metric="")


def test_zform_config_requires_callable_composite_metric():
    with pytest.raises(ValueError, match="composite_metric_func must be a callable"):
        ZformConfig(composite_metric_func="not_callable")


def test_zform_config_engine_type_checks():
    with pytest.raises(ValueError, match="engine must be a string or callable"):
        ZformConfig(engine=123)


def test_zform_config_engine_string_must_be_known():
    with pytest.raises(ValueError, match="engine must be one of"):
        ZformConfig(engine="svm")


def test_zform_config_engine_kwargs_requires_mapping():
    with pytest.raises(ValueError, match="engine_kwargs must be a mapping"):
        ZformConfig(engine_kwargs=[("alpha", 1.0)])


def test_zform_config_override_and_repr_helpers():
    cfg = ZformConfig(strategy="best", min_obs=12, n_jobs=1)
    updated = cfg.override(strategy="fixed", min_obs=30)

    assert updated.strategy == "fixed"
    assert updated.min_obs == 30
    assert updated.n_jobs == 1  # untouched fields propagate
    assert updated.as_dict()["strategy"] == "fixed"
    assert "strategy='fixed'" in repr(updated)
