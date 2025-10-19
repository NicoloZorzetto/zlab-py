"""
Transformation functions and parameter initialization
for zform models.
"""

try:
    import numpy as np
except ImportError as e:
    msg = (
        f"Missing dependency: {e.name}. Please install all requirements via "
        "'pip install -r requirements.txt'"
    )
    raise ImportError(msg)

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


# --- Model-specific initial guess functions ---

def _init_linear(x, y):
    x_mean, x_std = np.mean(x), np.std(x) or 1.0
    y_mean, y_std = np.mean(y), np.std(y) or 1.0
    a0 = y_std / x_std
    b0 = y_mean - a0 * x_mean
    return [a0, b0]

def _init_log_dynamic(x, y):
    y_std = np.std(y) or 1.0
    a0 = y_std / (np.std(np.log(np.abs(x) + 1)) or 1.0)
    b0 = np.mean(y)
    return [a0, b0, np.e]

def _init_power(x, y):
    x_mean = np.mean(x) or 1.0
    y_mean = np.mean(y)
    return [y_mean / x_mean, 1.0]

def _init_logistic(x, y):
    x_std = np.std(x) or 1.0
    return [float(np.max(y)), 1.0 / x_std, float(np.median(x))]

# --- Registry of initial guess functions ---
INIT_GUESS_FUNCS = {
    "linear": _init_linear,
    "log_dynamic": _init_log_dynamic,
    "power": _init_power,
    "logistic": _init_logistic,
}

def guess_initial_params(x, y, model_name):
    """Wrapper that retrieves the model-specific initialization function."""
    func = INIT_GUESS_FUNCS.get(model_name)
    if func is None:
        raise ValueError(f"Unknown model '{model_name}' — no init function registered.")
    return func(x, y)
