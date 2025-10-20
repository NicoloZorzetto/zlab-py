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

from zlab._zform_config import ZformConfig
from zlab._zform_metrics import (
    compute_metric, compute_multi_metrics, compute_composite_score, is_higher_better)
from zlab._zform_model_defaults import FIXED_DEFAULTS
from zlab.zform_functions import get_zform_functions, guess_initial_params
from zlab._zform_model_bounds import get_model_bounds


def compute_best_model(
    x,
    y,
    transformations=None,
    config: ZformConfig | None = None,
):
    """
    Fit or evaluate all candidate transformations between x and y.

    Parameters
    ----------
    x, y : array-like
        Input numeric data for fitting.
    transformations : list[str] | None
        Subset of transformations to test; if None, all are tested.
    config : ZformConfig, optional
        Configuration object controlling metrics, penalization, and optimization settings.
    """
    if config is None:
        config = ZformConfig()

    TRANSFORMATIONS = get_zform_functions(transformations)

    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < config.min_obs or np.std(x) == 0 or np.std(y) == 0:
        return "N/A", np.nan, None, None, 0

    zforms = {}
    total_iters = 0

    # --- Strategy: fixed ---
    if config.strategy == "fixed":
        from zlab.zform_functions import ZFORM_FUNCTIONS  # local import to avoid circulars
        for name, func in TRANSFORMATIONS.items():
            # Skip invalid domains
            if "log" in name and np.any(x <= -1):
                continue

            # Retrieve canonical parameters
            # Priority: user fixed_params > built-in FIXED_DEFAULTS > empty list
            fixed_params = ZFORM_FUNCTIONS.get(name, {}).get("fixed_params")
            if fixed_params is not None:
                p = fixed_params
            else:
                p = FIXED_DEFAULTS.get(name, [])

            if not p:
                msg = (
                    f"Model '{name}' has no fixed_params or default. "
                    "Skipping for strategy='fixed'."
                )
                ZformRuntimeWarning(msg)
                continue

            try:
                # Apply directly (no optimization)
                y_pred = func(x, *p)
                k = 2 + (len(p) if config.penalize_theta_in_ic else 0)
                metrics = compute_multi_metrics(config.eval_metric, y, y_pred, k=k)
                score = compute_composite_score(metrics, config.eval_metric,
                                                normalize=config.normalize_metrics)
                zforms[name] = {"score": score, "metrics": metrics, "params": p}
            except Exception:
                continue

        valid = {k: v for k, v in zforms.items() if np.isfinite(v["score"])}
        if not valid:
            return "N/A", np.nan, None, None, 0

        best = max(valid, key=lambda k: valid[k]["score"])
        return best, round(valid[best]["score"], 3), valid[best]["params"], None, 0


    # --- Strategy: best ---
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
                        maxfev=config.maxfev,
                        method="trf",
                        full_output=True,
                    )
                total_iters += infodict.get("nfev", 0)
                y_pred = func(x, *popt)
                k = 2 + (len(popt) if config.penalize_theta_in_ic else 0)
                metrics = compute_multi_metrics(config.eval_metric, y, y_pred, k=k)
                score = compute_composite_score(metrics, config.eval_metric,
                                                normalize=config.normalize_metrics)
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
            k = 2 + (len(p) if config.penalize_theta_in_ic else 0)
            score = compute_metric(config.eval_metric, y, y_pred, k=k)
            fixed_scores[name] = score
        except Exception:
            continue

    gain = None
    if fixed_scores:
        best_fixed = max(fixed_scores, key=fixed_scores.get)
        gain = best_score - fixed_scores[best_fixed]

    return best, best_score, best_params, gain, total_iters


def _fit_pair(args, config: ZformConfig):
    group_name, gdf, y_var, x_var, transformations = args
    x, y = gdf[x_var], gdf[y_var]
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[valid].to_numpy(), y[valid].to_numpy()

    if len(x_clean) < config.min_obs:
        return (group_name, y_var, x_var, "N/A", np.nan, None, None, 0)

    model, score, params, gain, n_iter = compute_best_model(
        x_clean,
        y_clean,
        transformations=transformations,
        config=config,
    )
    return (group_name, y_var, x_var, model, score, params, gain, n_iter)
