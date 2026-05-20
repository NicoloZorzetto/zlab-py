"""
_zform_eval_engines.py — Modular regression engines for zform.

This module defines interchangeable regression "engines" that zforms
can use to compute the evaluation metrics for the transformations.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

import numpy as np

# Optional scikit-learn dependency for non-OLS engines
try:
    from sklearn.linear_model import Ridge, Lasso

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False


# --- Engine Failure Error class ---
class EngineFailure(RuntimeError):
    """Wrap engine execution errors so callers can report engine + reason for error."""

    def __init__(self, engine, reason):
        super().__init__(reason)
        self.engine = engine
        self.reason = reason


# --- Engine registry ---
EVAL_ENGINES = {}


# --- Decorator for registration ---
def zform_engine(name, description=None):
    """
    Decorator to register a regression engine.

    Parameters
    ----------
    name : str
        Engine name (e.g. "ols", "ridge").
    description : str, optional
        Short human-readable description.

    Usage
    -----
    >>> @zform_engine("ols", description="Ordinary least squares regression.")
    ... def _ols_engine(x, y):
    ...     ...
    """

    def decorator(func):
        EVAL_ENGINES[name] = {"func": func, "description": description or ""}
        return func

    return decorator


# --- Built-in engines ---


@zform_engine("ols", description="Ordinary least squares regression.")
def _ols_engine(x: np.ndarray, y: np.ndarray):
    """OLS regression (y ~ x) using normal equations."""
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    x_design_matrix = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(x_design_matrix, y, rcond=None)
    intercept, slope = coef[0], coef[1:]
    y_pred = intercept + x @ slope
    return slope, intercept, y_pred


@zform_engine("ridge", description="Ridge regression (L2 regularized).")
def _ridge_engine(x: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """Ridge regression via scikit-learn."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for Ridge regression.")
    model = Ridge(alpha=alpha)
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    return model.coef_, model.intercept_, y_pred


@zform_engine("lasso", description="Lasso regression (L1 regularized).")
def _lasso_engine(x: np.ndarray, y: np.ndarray, alpha: float = 0.1):
    """Lasso regression via scikit-learn."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for Lasso regression.")
    model = Lasso(alpha=alpha, max_iter=10_000)
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    return model.coef_, model.intercept_, y_pred


# --- Accessors & Execution Helpers ---


def get_eval_engine(name):
    """Retrieve a registered engine function by name."""
    if callable(name):
        return name
    if name not in EVAL_ENGINES:
        raise KeyError(
            f"Unknown engine '{name}'. Available engines: {list(EVAL_ENGINES.keys())}"
        )
    return EVAL_ENGINES[name]["func"]


def fit_engine(engine_name, x, y, **kwargs):
    """
    Execute a regression engine with safe fallback.

    Parameters
    ----------
    engine_name : str | callable
        Engine name or custom callable returning (coef, intercept, y_pred).
    x, y : array-like
        Input data.
    kwargs : dict
        Extra parameters for the engine (e.g., alpha for Ridge/Lasso).

    Returns
    -------
    coef : array-like
        Model coefficients (slope).
    intercept : float
        Model intercept.
    y_pred : array-like
        Predicted target values.
    """
    try:
        engine_func = get_eval_engine(engine_name)
        return engine_func(x, y, **kwargs)
    except (KeyError, ImportError, ValueError) as exc:
        raise EngineFailure(engine_name, str(exc)) from exc
        # return _ols_engine(x, y)


def evaluate_engine(z, y, engine, **kwargs):
    """Run the requested regression engine and return predictions + summary."""
    coef, intercept, y_pred = fit_engine(engine, np.asarray(z), np.asarray(y), **kwargs)

    engine_label = (
        engine
        if isinstance(engine, str)
        else getattr(engine, "__name__", "custom_engine")
    )
    model_info = {
        "engine": engine_label,
        "coef": np.atleast_1d(coef).tolist(),
        "intercept": float(np.asarray(intercept).squeeze()),
    }
    return np.asarray(y_pred, float), model_info
