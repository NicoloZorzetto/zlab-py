"""
Model fitting logic for zform.
Handles the evaluation and optimization of parametric transformations.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

from zlab.zform._zform_config import ZformConfig
from zlab.zform._zform_metrics import compute_multi_metrics, compute_composite_score
from zlab.zform.zform_functions import (
    get_zform_functions,
    ZFORM_FUNCTIONS,
    guess_initial_params,
    ensure_zform_functions,
)
from zlab.zform._zform_eval_engines import evaluate_engine, EngineFailure
from zlab.warnings import ZformRuntimeWarning


class EngineExhausted(RuntimeError):
    """Sentinel class for engine failures."""

    def __init__(self, engine, failures):
        self.engine = engine
        self.failures = failures
        super().__init__(engine)


def compute_best_model(
    x,
    y,
    transformations=None,
    config: ZformConfig | None = None,
):
    """
    Evaluate each candidate transformation and return the best-scoring model.

    Parameters
    ----------
    x, y : array-like
        Numeric Series/DataFrame columns converted to 1-D numpy arrays.
    transformations : list[str | callable] | None, default=None
        Subset of registry names or callables to evaluate. `None` uses all
        registered transformations.
    config : ZformConfig | None, default=None
        Full zform configuration controlling strategy, engine, metrics, etc.
        A new `ZformConfig()` is created when omitted.

    Returns
    -------
    tuple
        (best_name, score, params, gain, n_iterations) where:
        - `best_name` : str, winning transformation name or "N/A".
        - `score` : float, composite score (higher is better) or NaN.
        - `params` : list | None, fitted parameter vector.
        - `gain` : float | np.nan, relative improvement over the best fixed form (0 when the fixed form wins).
        - `n_iterations` : int, total iterations performed by the optimizer.

    Raises
    ------
    EngineExhausted
        When every transformation fails due to regression-engine errors under.
    """

    if config is None:
        config = ZformConfig()

    TRANSFORMATIONS_MAP = get_zform_functions(transformations)

    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < config.min_obs or np.std(x) == 0 or np.std(y) == 0:
        return "N/A", np.nan, np.nan, np.nan, 0

    zforms = {}
    total_iters = 0

    engine_failures = []

    # --- Strategy: fixed ---
    if config.strategy == "fixed":
        engine_failures = []
        for name, func in TRANSFORMATIONS_MAP.items():
            info = ZFORM_FUNCTIONS[name]

            flag = info.get("requires_positive_x", False)
            if flag is True:
                if np.any(x <= 0):
                    continue
            elif callable(flag):
                if flag(x):
                    continue
            elif flag:
                continue

            p = info.get("fixed_params", [])
            if (info.get("n_params") or 0) > 0 and not p:
                ZformRuntimeWarning(f"No fixed_params for '{name}', skipping.")
                continue

            try:
                z = func(x, *p)
                if not np.all(np.isfinite(z)):
                    continue
                try:
                    y_pred, model_info = evaluate_engine(
                        z, y, config.engine, **config.engine_kwargs
                    )
                except EngineFailure as exc:
                    engine_failures.append((name, exc.reason))
                    continue

                k = 2 + (len(p) if config.penalize_theta_in_ic else 0)
                metrics = compute_multi_metrics(config.eval_metric, y, y_pred, k=k)
                score = compute_composite_score(
                    metrics,
                    config.eval_metric,
                    normalize=config.normalize_metrics,
                    composite_metric_func=config.composite_metric_func,
                )
                zforms[name] = {
                    "score": score,
                    "metrics": metrics,
                    "params": p,
                    "engine": config.engine,
                    "coef": model_info["coef"],
                    "intercept": model_info["intercept"],
                }
            except (
                TypeError,
                ValueError,
                FloatingPointError,
                OverflowError,
                RuntimeError,
                EngineFailure,
            ):
                continue

        valid = {k: v for k, v in zforms.items() if np.isfinite(v["score"])}
        if not valid:
            if engine_failures:
                raise EngineExhausted(config.engine, engine_failures)
            return "N/A", np.nan, np.nan, np.nan, 0

        best = max(valid, key=lambda k: valid[k]["score"])
        return best, round(valid[best]["score"], 3), valid[best]["params"], None, 0

    # --- Strategy: best ---
    for name, func in TRANSFORMATIONS_MAP.items():
        info = ZFORM_FUNCTIONS[name]

        flag = info.get("requires_positive_x", False)
        if flag is True:
            if np.any(x <= 0):
                continue
        elif callable(flag):
            if flag(x):
                continue
        elif flag:
            continue

        p0 = guess_initial_params(x, y, name)
        bounds = info.get("bounds")

        if bounds is None:
            lower = [-np.inf] * len(p0)
            upper = [np.inf] * len(p0)
        else:
            try:
                lower = np.asarray(bounds[0], dtype=float)
                upper = np.asarray(bounds[1], dtype=float)
            except (TypeError, ValueError):
                # malformed bounds skip this transformation
                continue
        bounds = (lower, upper)

        for attempt in range(2):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.simplefilter("ignore", OptimizeWarning)

                    popt, _, infodict, _, _ = curve_fit(
                        func,
                        x,
                        y,
                        p0=(
                            np.array(p0)
                            * (1 + np.random.default_rng(0).uniform(-0.1, 0.1, len(p0)))
                            if attempt == 1
                            else p0
                        ),
                        bounds=bounds,
                        maxfev=config.maxfev,
                        method="trf",
                        full_output=True,
                    )

                total_iters += infodict.get("nfev", 0)

                z = func(x, *popt)
                if not np.all(np.isfinite(z)):
                    continue
                try:
                    y_pred, model_info = evaluate_engine(
                        z, y, config.engine, **config.engine_kwargs
                    )
                except EngineFailure as exc:
                    engine_failures.append((name, exc.reason))
                    continue

                k = 2 + (len(popt) if config.penalize_theta_in_ic else 0)
                metrics = compute_multi_metrics(config.eval_metric, y, y_pred, k=k)
                score = compute_composite_score(
                    metrics,
                    config.eval_metric,
                    normalize=config.normalize_metrics,
                    composite_metric_func=config.composite_metric_func,
                )

                zforms[name] = {
                    "score": score,
                    "metrics": metrics,
                    "params": popt,
                    "engine": config.engine,
                    "coef": model_info["coef"],
                    "intercept": model_info["intercept"],
                }
                break
            except (
                TypeError,
                ValueError,
                FloatingPointError,
                OverflowError,
                RuntimeError,
                EngineFailure,
            ):
                continue

    valid = {k: v for k, v in zforms.items() if np.isfinite(v["score"])}
    if not valid:
        if engine_failures:
            raise EngineExhausted(config.engine, engine_failures)
        return "N/A", np.nan, np.nan, np.nan, total_iters

    best = max(valid, key=lambda k: valid[k]["score"])
    best_score_raw = valid[best]["score"]
    best_score = round(valid[best]["score"], 3)
    best_params = valid[best]["params"]

    # --- Gain vs fixed ---
    fixed_scores = {}
    for name, func in TRANSFORMATIONS_MAP.items():
        info = ZFORM_FUNCTIONS[name]
        p = info["fixed_params"]
        try:
            flag = info.get("requires_positive_x", False)
            if flag is True:
                if np.any(x <= 0):
                    continue
            elif callable(flag):
                if flag(x):
                    continue
            elif flag:
                continue

            z = func(x, *p)
            if not np.all(np.isfinite(z)):
                continue
            try:
                y_pred, _ = evaluate_engine(z, y, config.engine)
            except EngineFailure as exc:
                engine_failures.append((name, exc.reason))
                continue
            k = 2 + (len(p) if config.penalize_theta_in_ic else 0)
            metrics = compute_multi_metrics(config.eval_metric, y, y_pred, k=k)
            score = compute_composite_score(
                metrics,
                config.eval_metric,
                normalize=config.normalize_metrics,
                composite_metric_func=config.composite_metric_func,
            )
            fixed_scores[name] = {"score": score, "params": p}
        except (
            TypeError,
            ValueError,
            FloatingPointError,
            OverflowError,
            RuntimeError,
            EngineFailure,
        ):
            continue

    gain = np.nan
    if fixed_scores:
        best_fixed = max(fixed_scores, key=lambda k: fixed_scores[k]["score"])
        best_fixed_score = fixed_scores[best_fixed]["score"]
        gain = round(best_score_raw - best_fixed_score, 6)

        if best_score_raw <= best_fixed_score:
            best = best_fixed
            best_score_raw = best_fixed_score
            best_score = round(best_fixed_score, 3)
            best_params = fixed_scores[best_fixed]["params"]
            gain = 0.0

    return best, best_score, best_params, gain, total_iters


def _fit_pair(args, config: ZformConfig):
    group_name, gdf, y_var, x_var, transformations, func_snapshot = args
    x, y = gdf[x_var], gdf[y_var]
    ensure_zform_functions(func_snapshot)
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[valid].to_numpy(), y[valid].to_numpy()

    if len(x_clean) < config.min_obs:
        return (group_name, y_var, x_var, "N/A", np.nan, np.nan, np.nan, 0)

    try:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            btransformation, score, params, gain, n_iter = compute_best_model(
                x_clean,
                y_clean,
                transformations=transformations,
                config=config,
            )
    except EngineExhausted as exc:
        details = "; ".join(f"{t}: {msg}" for t, msg in exc.failures)
        return (
            group_name,
            y_var,
            x_var,
            "N/A",
            np.nan,
            f"ENGINE_FAIL: {details}",
            np.nan,
            0,
        )
    return (group_name, y_var, x_var, btransformation, score, params, gain, n_iter)
