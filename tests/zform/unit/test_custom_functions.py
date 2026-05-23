"""
Tests for custom zform function registration.

Run with:
    pytest tests/zform/unit/test_custom_functions.py
or:
    python tests/zform/unit/test_custom_functions.py
"""

import numpy as np
import pandas as pd
import pytest

from zlab.datasets import load_penguins
from zlab.zform import zform
from zlab.zform.zform_functions import (
    zform_function,
    register_zform_function,
    ZFORM_FUNCTIONS,
    SQRT_MAX_FLOAT,
)
from zlab.zform import zform_functions


@zform_function(
    "custom_ratio_for_test1",
    n_params=1,
    description="Simple ratio transformation",
    bounds=([1e-8], [1e8]),
    fixed_params=[1.0],
    init_func=lambda x, y: [1.0],
    register=False,
)
def custom_ratio_for_test1(x, k):
    return x / k


def test_function_custom_ratio_fixed_penguins():
    penguins = load_penguins()
    z = zform(
        penguins,
        strategy="fixed",
        transformations="custom_ratio_for_test1",
    )

    assert z.validate()
    assert all(
        z[(z["y"] == "bill_depth_mm") & (z["x"] == "bill_length_mm")]["Best R2"]
        == 0.055
    )
    assert all(
        z[(z["y"] == "bill_depth_mm") & (z["x"] == "body_mass_g")]["Parameters"]
        == (1.0,)
    )


def test_function_custom_ratio_best_penguins():
    penguins = load_penguins()
    z = zform(penguins, transformations="custom_ratio_for_test1")

    assert z.validate()

    assert all(
        z[(z["y"] == "bill_depth_mm") & (z["x"] == "bill_length_mm")]["Best R2"]
        == 0.055
    )
    assert all(
        z[(z["y"] == "bill_depth_mm") & (z["x"] == "body_mass_g")]["Parameters"]
        == (1.0,)
    )


def test_function_custom_ratio_recovers_linear_relationship():
    df = pd.DataFrame(
        {
            "linear": np.arange(1, 200, dtype=float),
            "half": np.arange(1, 200, dtype=float) / 2,
            "log": np.log(np.arange(1, 200, dtype=float)),
        }
    )

    z = zform(df, transformations="custom_ratio_for_test1")

    assert z.validate()
    assert all(z[(z["y"] == "linear") & (z["x"] == "half")]["Best R2"] == 1.0)
    assert all(z[(z["y"] == "half") & (z["x"] == "linear")]["Parameters"] == (1.0,))


@zform_function(
    "custom_tanh",
    n_params=2,
    description="Centered tanh curve",
    init_func=lambda x, y: [
        1.0 / (np.std(x) or 1.0),
        float(np.median(x)),
    ],
    bounds=([1e-6, -SQRT_MAX_FLOAT], [10.0, SQRT_MAX_FLOAT]),
    fixed_params=[1.0, 0.0],
    register=False,
)
def custom_tanh(x, gain, shift):
    return np.tanh(gain * (x - shift))


def test_function_custom_tanh_best_fits_parameters():
    x = np.linspace(-3, 3, 200)
    rng = np.random.default_rng(42)
    y = np.tanh(2.0 * (x - 0.5)) + 0.05 * rng.standard_normal(len(x))

    df = pd.DataFrame({"y": y, "x": x})
    z = zform(df, y="y", x="x", transformations="custom_tanh")

    assert z.validate()

    best = z[(z["y"] == "y") & (z["x"] == "x")]
    assert all(best["Best R2"] >= 0.9)

    params = best["Parameters"].iloc[0]
    assert params[0] == pytest.approx(1.0, rel=0.1)
    assert params[1] == pytest.approx(0.0, rel=0.1)


def test_function_custom_tanh_fixed_uses_fixed_params():
    x = np.linspace(-3, 3, 200)
    y = np.tanh(2.0 * (x - 0.5))

    df = pd.DataFrame({"y": y, "x": x})
    z = zform(df, strategy="fixed", transformations="custom_tanh")

    assert z.validate()
    assert all(z[(z["y"] == "y") & (z["x"] == "x")]["Parameters"] == (1.0, 0.0))


@pytest.fixture
def global_sigmoid():
    def _sigmoid(x, a, b):
        return 1.0 / (1.0 + np.exp(-a * (x - b)))

    register_zform_function(
        name="global_sigmoid",
        func=_sigmoid,
        n_params=2,
        init_func=lambda x, y: [1.0 / (np.std(x) or 1.0), float(np.median(x))],
        bounds=([1e-6, -SQRT_MAX_FLOAT], [10.0, SQRT_MAX_FLOAT]),
        fixed_params=[1.0, 0.0],
    )

    try:
        yield "global_sigmoid"
    finally:
        ZFORM_FUNCTIONS.pop("global_sigmoid", None)


def test_function_global_registered_function_reachable(global_sigmoid):
    x = np.linspace(-3, 3, 200)
    rng = np.random.default_rng(123)
    y = np.tanh(1.5 * (x + 0.2)) + 0.05 * rng.standard_normal(len(x))

    df = pd.DataFrame({"y": y, "x": x})
    z = zform(df, y="y", x="x", transformations=[global_sigmoid])

    assert z.validate()

    params = z[(z["y"] == "y") & (z["x"] == "x")]["Parameters"].iloc[0]
    assert params[0] == pytest.approx(1.0, rel=0.2)


def test_get_zform_functions_handles_callables(monkeypatch):
    def custom(x):
        return x

    monkeypatch.setattr(
        zform_functions,
        "ZFORM_FUNCTIONS",
        {"linear": {"func": lambda x: x}},
        raising=False,
    )

    funcs = zform_functions.get_zform_functions([custom])

    assert "custom" in funcs and funcs["custom"] is custom


@pytest.fixture
def global_shifted_abs():
    @zform_function(
        "shifted_abs",
        n_params=0,
        description="Abs with callable guard",
        init_func=lambda x, y: [],
        bounds=([], []),
        fixed_params=[],
        requires_positive_x=lambda x: np.any(x < -2),
        register=True,
    )
    def shifted_abs(x):
        return np.abs(x)

    try:
        yield "shifted_abs"
    finally:
        ZFORM_FUNCTIONS.pop("shifted_abs", None)


def test_function_requires_positive_x_callable_respected(global_shifted_abs):
    df = pd.DataFrame({"y": [2, 1, 0, 1, 2], "x": [-2, -1, 0, 1, 2]})
    zf = zform(
        df,
        y="y",
        x="x",
        transformations=[global_shifted_abs, "linear"],
        strategy="fixed",
        min_obs=3,
    )
    assert "shifted_abs" in zf["Best Transformation"].values

    @zform_function(
        "shifted_abs_guarded",
        n_params=0,
        description="Abs with always-skip guard",
        init_func=lambda x, y: [],
        bounds=([], []),
        fixed_params=[],
        requires_positive_x=lambda x: True,
        register=False,
    )
    def shifted_abs_guarded(x):
        return np.abs(x)

    df2 = pd.DataFrame({"y": [5, 4, 3], "x": [-5, -4, -3]})
    zf2 = zform(
        df2,
        y="y",
        x="x",
        transformations=[shifted_abs_guarded, "linear"],
        strategy="fixed",
        min_obs=3,
    )

    assert "shifted_abs_guarded" not in zf2["Best Transformation"].values


# --- Focused coverage for registry helpers ---


def test_get_zform_functions_raises_key_error_for_unknown(monkeypatch):
    monkeypatch.setattr(
        zform_functions,
        "ZFORM_FUNCTIONS",
        {"linear": {"func": lambda x: x}},
        raising=False,
    )

    with pytest.raises(KeyError, match="Unknown transformation"):
        zform_functions.get_zform_functions(["missing"])


def test_get_zform_functions_raises_type_error_for_non_supported(monkeypatch):
    monkeypatch.setattr(
        zform_functions,
        "ZFORM_FUNCTIONS",
        {"linear": {"func": lambda x: x}},
        raising=False,
    )

    with pytest.raises(TypeError, match="Unsupported transformation"):
        zform_functions.get_zform_functions([object()])


def test_guess_initial_params_warns_when_no_init_func(monkeypatch):
    registry = {"foo": {"func": lambda x, y: x + y, "init_func": None}}
    monkeypatch.setattr(zform_functions, "ZFORM_FUNCTIONS", registry, raising=False)

    warnings = []
    monkeypatch.setattr(zform_functions, "ZformRuntimeWarning", warnings.append)

    params = zform_functions.guess_initial_params([1.0], [2.0], "foo")

    assert params == [1.0]
    assert warnings and "no init function" in warnings[0]
