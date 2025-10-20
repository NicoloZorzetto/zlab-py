"""
Model fitting logic for zform.
Handles the evaluation and optimization of parametric transformations.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
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

from zlab._zform_metrics import (
    compute_metric, compute_multi_metrics, compute_composite_score, is_higher_better)
from zlab.zform_functions import get_zform_functions, guess_initial_params
from zlab._zform_model_bounds import get_model_bounds



def compute_best_model(
        x,
        y,
        eval_metric="r2",
        normalize_metrics=False,
        transformations=None,
        strategy="best",
        maxfev=100000):
    """
    Fit or evaluate all candidate transformations between x and y.

    Parameters
    ----------
    x, y : array-like
        Input numeric data for fitting.
    eval_metric : str | list[str] | dict[str, float] | callable, default='r2'
        Evaluation metric(s) used to assess transformation fit quality.
        Available built-in metrics: {'r2', 'adjr2', 'rmse', 'mae', 'aic', 'bic'}.
        - If a **string** is passed, a single metric is used (e.g., "r2").
        - If a **list** is passed, all metrics are averaged equally.
        - If a **dict** is passed, values are treated as weights for averaging
          (e.g. `{"r2": 1.0, "aic": -0.2}`).
        - If a **callable** is passed, it must accept `(y, y_pred, k)` and
          return a numeric score.
    normalize_metrics : bool, default=False
        When True, evaluation metrics are rescaled to [0, 1] before combining.
        This helps ensure balanced weighting when combining metrics of
        different scales or directions.
    transformations : list[str] | None
        Subset of transformations to test; if None, all are tested.
    strategy : {'best', 'fixed'}, default="best"
        'best' fits full parametric models, 'fixed' uses canonical parameters.

    Returns
    -------
    (best_model, best_score, best_params, gain_vs_fixed, total_iterations)
    """
    
    # Default parameter guesses for fixed models
    FIXED_DEFAULTS = {
        "linear": [1.0, 0.0],
        "power": [1.0, 2.0],
        "log_dynamic": [1.0, 0.0, np.e],
        "logistic": [1.0, 1.0, 0.0],
    }

    # ✅ Use registry instead of hardcoded dict
    TRANSFORMATIONS = get_zform_functions(transformations)

    # Clean input
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return "N/A", np.nan, None, None, 0

    zforms = {}
    total_iters = 0

    # --- Strategy: fixed ---
    if strategy == "fixed":
        for name, func in TRANSFORMATIONS.items():
            if "log" in name and np.any(x <= -1):
                continue
            p = FIXED_DEFAULTS.get(name, [])
            try:
                y_pred = func(x, *p)
                metrics = compute_multi_metrics(eval_metric, y, y_pred, k=len(p))
                score = compute_composite_score(metrics, eval_metric, normalize=normalize_metrics)
                zforms[name] = {"score": score, "metrics": metrics, "params": p}
            except Exception:
                continue

        valid = {k: v for k, v in zforms.items() if np.isfinite(v["score"])}
        if not valid:
            return "N/A", np.nan, None, None, 0

        best = max(valid, key=lambda k: valid[k]["score"])
        return best, round(valid[best]["score"], 3), valid[best]["params"], None, 0

    # --- Strategy: best (fit + gain) ---
    for name, func in TRANSFORMATIONS.items():
        if "log" in name and np.any(x <= -1):
            continue
        if name == "power" and np.any(x < 0):
            continue

        bounds = get_model_bounds(name, x, y)
        p0 = guess_initial_params(x, y, name)

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
                        maxfev=maxfev,
                        method="trf",
                        full_output=True,
                    )
                total_iters += infodict.get("nfev", 0)
                y_pred = func(x, *popt)
                metrics = compute_multi_metrics(eval_metric, y, y_pred, k=len(popt))
                score = compute_composite_score(metrics, eval_metric, normalize=normalize_metrics)
                zforms[name] = {"score": score, "metrics": metrics, "params": popt}
                break
            except Exception:
                continue

    valid = {k: v for k, v in zforms.items() if np.isfinite(v["score"])}
    if not valid:
        return "N/A", np.nan, None, None, total_iters

    best = max(valid, key=lambda k: valid[k]["score"])
    best_score = round(valid[best]["score"], 3)
    best_params = valid[best]["params"]

    # --- Gain vs fixed ---
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
        best_fixed = max(fixed_scores, key=lambda k: fixed_scores[k])
        gain = best_score - fixed_scores[best_fixed]

    return best, best_score, best_params, gain, total_iters

# --- Parallel fitting wrapper ---

def _fit_pair(args, normalize_metrics=False):
    group_name, gdf, y_var, x_var, eval_metric, transformations, strategy, min_obs, maxfev = args
    x, y = gdf[x_var], gdf[y_var]
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[valid].to_numpy(), y[valid].to_numpy()
    if len(x_clean) < min_obs:
        return (group_name, y_var, x_var, "N/A", np.nan, None, None, 0)

    model, score, params, gain, n_iter = compute_best_model(
        x_clean,
        y_clean,
        eval_metric=eval_metric,
        normalize_metrics=normalize_metrics,
        transformations=transformations,
        strategy=strategy,
        maxfev=maxfev,
    )
    return (group_name, y_var, x_var, model, score, params, gain, n_iter)
