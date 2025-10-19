"""
Transformation functions and parameter initialization
for zform models.
"""

import numpy as np

def linear_func(x, a, b):
    return a * x + b

def log_func(x, a, b, base=np.e):
    return a * np.emath.logn(base, x + 1) + b

def log_func_dynamic(x, a, b, base):
    base = np.abs(base) + 1e-5
    return a * (np.log(x + 1) / np.log(base)) + b

def power_func(x, a, b):
    return a * (x ** b)

def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


def guess_initial_params(x, y, model_name):
    """Generate rough initial guesses for curve fitting."""
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)
    x_std = x_std or 1.0
    y_std = y_std or 1.0

    if model_name == "linear":
        a0 = y_std / x_std
        b0 = y_mean - a0 * x_mean
        return [a0, b0]
    elif model_name == "log_dynamic":
        a0 = y_std / (np.std(np.log(np.abs(x) + 1)) or 1.0)
        b0 = y_mean
        return [a0, b0, np.e]
    elif model_name == "power":
        return [y_mean / (x_mean if x_mean != 0 else 1.0), 1.0]
    elif model_name == "logistic":
        return [float(np.max(y)), 1.0 / (x_std or 1.0), float(np.median(x))]
    else:
        raise ValueError(f"Unknown model '{model_name}'")
