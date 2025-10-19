"""
Model-specific parameter bounds for zform.

This module defines lower and upper bounds for curve fitting parameters
based on model type and data scale. It provides a consistent interface
for retrieving safe, scale-aware limits that improve optimization
stability and prevent numerical overflows.

Part of the zlab library by Nicolò Zorzetto.
"""

import numpy as np
from zlab.warnings import ZformRuntimeWarning

# --- Model-specific bounds ---

def _bounds_linear(x, y):
    y_range = np.ptp(y) or 1.0
    x_range = np.ptp(x) or 1.0
    slope_bound = 10 * (y_range / x_range)
    intercept_bound = 10 * np.abs(np.mean(y))
    return ([-slope_bound, -intercept_bound], [slope_bound, intercept_bound])


def _bounds_power(x, y):
    y_range = np.ptp(y) or 1.0
    x_range = np.ptp(x) or 1.0
    a_bound = 10 * (y_range / x_range)
    return ([-a_bound, 0], [a_bound, 5])


def _bounds_log_dynamic(x, y):
    y_range = np.ptp(y) or 1.0
    b_bound = 10 * np.abs(np.mean(y))
    return ([-y_range, -b_bound, 1.001], [y_range, b_bound, 1e4])


def _bounds_logistic(x, y):
    L_max = 2 * np.max(np.abs(y))
    return ([0, -10, np.min(x)], [L_max, 10, np.max(x)])


def _bounds_default(x, y):
    return (-1e8, 1e8)


# --- Registry ---

BOUNDS_FUNCS = {
    "linear": _bounds_linear,
    "power": _bounds_power,
    "log_dynamic": _bounds_log_dynamic,
    "logistic": _bounds_logistic,
}


# --- Bounds computation ---

def get_model_bounds(model_name, x, y):
    """
    Retrieve safe, model-specific parameter bounds for optimization.

    Automatically detects non-finite, reversed, or excessively wide bounds
    and falls back to defaults with a runtime warning.
    """
    func = BOUNDS_FUNCS.get(model_name, _bounds_default)
    try:
        bounds = func(x, y)
        lower, upper = np.array(bounds[0]), np.array(bounds[1])

        # Sanity checks
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            ZformRuntimeWarning(f"Non-finite bounds for model '{model_name}'. Using defaults (-1e8, 1e8).")
            return _bounds_default(x, y)

        if np.any(upper <= lower):
            ZformRuntimeWarning(f"Reversed bounds for model '{model_name}' — check data scaling.")
            return _bounds_default(x, y)

        if np.max(upper - lower) > 1e6:
            ZformRuntimeWarning(f"Extremely wide bounds for '{model_name}' — data may be poorly scaled.")

        return bounds

    except Exception as e:
        ZformRuntimeWarning(f"Failed to compute bounds for '{model_name}': {e}. Using defaults.")
        return _bounds_default(x, y)
