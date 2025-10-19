"""
Model fitting logic for zform.
Handles the evaluation and optimization of parametric transformations.
"""

import warnings

try:
    import numpy as np
    from scipy.optimize import curve_fit, OptimizeWarning
except ImportError as e:
    msg = (
        f"Missing dependency: {e.name}. Please install all requirements via "
        "'pip install -r requirements.txt'"
    )
    raise ImportError(msg)

from zlab._zform_metrics import compute_metric, is_higher_better
from zlab.zform_functions import (
    linear_func, power_func, log_func_dynamic, logistic_func, guess_initial_params
)
from zlab._zform_model_bounds import get_model_bounds

def compute_best_model(x, y, eval_metric="r2", transformations=None, strategy="best"):
    """
    Fit or evaluate all candidate transformations between x and y.

    Parameters
    ----------
    x, y : array-like
        Input numeric data for fitting.
    eval_metric : str, default="r2"
        Evaluation metric used to compare fits.
    transformations : list[str] | None
        Subset of transformations to test; if None, all are tested.
    strategy : {'best', 'fixed'}, default="best"
        'best' fits full parametric models, 'fixed' uses canonical parameters.

    Returns
    -------
    (best_model, best_score, best_params, gain_vs_fixed, total_iterations)
    """
    FIXED_DEFAULTS = {
        "linear": [1.0, 0.0],
        "power": [1.0, 2.0],
        "log_dynamic": [1.0, 0.0, np.e],
        "logistic": [1.0, 1.0, 0.0],
    }

    TRANSFORMATIONS = {
        "linear": linear_func,
        "power": power_func,
        "log_dynamic": log_func_dynamic,
        "logistic": logistic_func,
    }

    if transformations is not None:
        TRANSFORMATIONS = {k: v for k, v in TRANSFORMATIONS.items() if k in transformations}

    # Clean input
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return "N/A", np.nan, None, None, 0

    zforms = {}
    total_iters = 0

    # --- Strategy: fixed (no fitting, no gain computation) ---
    if strategy == "fixed":
        for name, func in TRANSFORMATIONS.items():
            if "log" in name and np.any(x <= -1):
                continue
            p = FIXED_DEFAULTS.get(name, [])
            try:
                y_pred = func(x, *p)
                score = compute_metric(eval_metric, y, y_pred, k=len(p))
                zforms[name] = {"score": score, "params": p}
            except Exception:
                continue

        valid = {k: v for k, v in zforms.items() if np.isfinite(v["score"])}
        if not valid:
            return "N/A", np.nan, None, None, 0

        best = (
            max(valid, key=lambda k: valid[k]["score"])
            if is_higher_better(eval_metric)
            else min(valid, key=lambda k: valid[k]["score"])
        )
        return best, round(valid[best]["score"], 3), valid[best]["params"], None, 0

    # --- Strategy: best (parametric fitting + gain computation) ---
    for name, func in TRANSFORMATIONS.items():
        if "log" in name and np.any(x <= -1):
            continue
        if name == "power" and np.any(x < 0):
            continue

        # model-specific or scale-aware bounds
        bounds = get_model_bounds(name, x, y)

        # initial guess for parameters
        p0 = guess_initial_params(x, y, name)

        # two-attempt optimization (perturbed initial guess)
        for attempt in range(2):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    warnings.simplefilter("ignore", category=OptimizeWarning)
                    popt, _, infodict, _, _ = curve_fit(
                        func, x, y,
                        p0=np.array(p0) * (1 + np.random.uniform(-0.1, 0.1, len(p0)))
                        if attempt == 1 else p0,
                        bounds=bounds,
                        maxfev=80000,
                        method="trf",
                        full_output=True,
                    )
                total_iters += infodict.get("nfev", 0)
                y_pred = func(x, *popt)
                score = compute_metric(eval_metric, y, y_pred, k=len(popt))
                zforms[name] = {"score": score, "params": popt}
                break
            except Exception:
                continue

    valid = {k: v for k, v in zforms.items() if np.isfinite(v["score"])}
    if not valid:
        return "N/A", np.nan, None, None, total_iters

    # pick best model depending on metric direction
    best = (
        max(valid, key=lambda k: valid[k]["score"])
        if is_higher_better(eval_metric)
        else min(valid, key=lambda k: valid[k]["score"])
    )

    best_score = round(valid[best]["score"], 3)
    best_params = valid[best]["params"]

    # --- Compute gain vs fixed ---
    fixed_scores = {}
    for name, func in TRANSFORMATIONS.items():
        p = FIXED_DEFAULTS.get(name, [])
        try:
            if "log" in name and np.any(x <= -1):
                continue
            y_pred = func(x, *p)
            score = compute_metric(eval_metric, y, y_pred, k=len(p))
            fixed_scores[name] = score
        except Exception:
            continue

    gain = None
    if fixed_scores:
        if is_higher_better(eval_metric):
            best_fixed = max(fixed_scores, key=lambda k: fixed_scores[k])
            fixed_best_score = fixed_scores[best_fixed]
            gain = best_score - fixed_best_score
        else:
            best_fixed = min(fixed_scores, key=lambda k: fixed_scores[k])
            fixed_best_score = fixed_scores[best_fixed]
            gain = fixed_best_score - best_score  # reverse direction for lower-is-better metrics

    return best, best_score, best_params, gain, total_iters


# --- Parallel fitting wrapper ---

def _fit_pair(args):
    group_name, gdf, y_var, x_var, eval_metric, transformations, strategy, min_obs = args
    x, y = gdf[x_var], gdf[y_var]
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[valid].to_numpy(), y[valid].to_numpy()
    if len(x_clean) < min_obs:
        return (group_name, y_var, x_var, "N/A", np.nan, None, None, 0)
    model, score, params, gain, n_iter = compute_best_model(
        x_clean, y_clean, eval_metric, transformations, strategy
    )
    return (group_name, y_var, x_var, model, score, params, gain, n_iter)
