#!/usr/bin/env python

"""
zform automatically identifies and optionally applies
the best parametric transformation
that linearizes the relationship between variables.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

try:
    import warnings
    from itertools import permutations
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit, OptimizeWarning  # type: ignore[import-untyped]
    from typing import Tuple
    from concurrent.futures import ProcessPoolExecutor, as_completed
except ImportError as e:
    raise ImportError(
        f"Missing dependency: {e.name}. Please install all requirements via "
        "'pip install -r requirements.txt'"
    )

# === Transformation functions ===
def linear_func(x, a, b):
    """Linear transformation: a*x + b"""
    return a * x + b

def log_func(x, a, b, base=np.e):
    """Fixed-base logarithmic transformation: a * log_base(x + 1) + b"""
    return a * np.emath.logn(base, x + 1) + b

def log_func_dynamic(x, a, b, base):
    """Dynamic logarithmic transformation: a * log_base(x + 1) + b, base is fitted."""
    base = np.abs(base) + 1e-5
    return a * (np.log(x + 1) / np.log(base)) + b

def power_func(x, a, b):
    """Power transformation: a * x^b"""
    return a * (x ** b)

def logistic_func(x, L, k, x0):
    """Logistic transformation"""
    return L / (1 + np.exp(-k * (x - x0)))

# === Metrics ===
def compute_metric(name: str, y_true, y_pred, k: int = 0):
    """Compute R2, RMSE, MAE, AIC, or BIC."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = y_true.size
    if n == 0 or np.any(~np.isfinite(y_pred)):
        return np.nan

    resid = y_true - y_pred
    rss = np.sum(resid ** 2)
    name = name.lower()

    if name == "r2":
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (rss / tss) if tss != 0 else np.nan
    elif name == "rmse":
        return np.sqrt(rss / n)
    elif name == "mae":
        return np.mean(np.abs(resid))
    elif name == "aic":
        return n * np.log(rss / n) + 2 * k if rss > 0 else -np.inf
    elif name == "bic":
        return n * np.log(rss / n) + k * np.log(n) if rss > 0 else -np.inf
    else:
        warnings.warn(f"Unknown eval_metric '{name}', falling back to R².", UserWarning)
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (rss / tss) if tss != 0 else np.nan

# === Core fitting routine ===
def compute_best_model(x, y, eval_metric="r2", transformations=None, mode="discovery"):
    """Fit multiple functional forms using curve_fit and return the best one."""

    def guess_initial_params(x, y, model_name):
        x_mean, x_std = np.mean(x), np.std(x)
        y_mean, y_std = np.mean(y), np.std(y)
        x_std = x_std or 1.0
        y_std = y_std or 1.0

        if model_name == "linear":
            a0 = y_std / x_std
            b0 = y_mean - a0 * x_mean
            p = [a0, b0]
        elif model_name in ("log", "log_dynamic"):
            logx_std = np.std(np.log(np.abs(x) + 1)) or 1.0
            a0 = y_std / logx_std
            b0 = y_mean
            p = [a0, b0, np.e] if model_name == "log_dynamic" else [a0, b0]
        elif model_name == "power":
            b0 = 1.0
            a0 = y_mean / (x_mean if x_mean != 0 else 1.0)
            p = [a0, b0]
        elif model_name == "logistic":
            L0 = float(np.max(y))
            k0 = 1.0 / (x_std or 1.0)
            x0 = float(np.median(x))
            p = [L0, k0, x0]
        else:
            raise ValueError(f"Unknown model '{model_name}'")

        return np.clip(p, -1e6, 1e6).tolist()

    TRANSFORMATIONS = {
        "linear": linear_func,
        "power": power_func,
        "logistic": logistic_func,
    }
    if mode == "discovery":
        TRANSFORMATIONS["log_dynamic"] = log_func_dynamic
    else:
        TRANSFORMATIONS["log"] = log_func

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return "N/A", np.nan, None, None

    results = {}
    bounds = (-1e8, 1e8)

    for name, func in TRANSFORMATIONS.items():
        if "log" in name and np.any(x <= -1):
            continue
        if name == "power" and np.any(x < 0):
            continue

        p0 = guess_initial_params(x, y, name)
        success = False
        for attempt in range(2):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    warnings.simplefilter("ignore", category=OptimizeWarning)
                    popt, _ = curve_fit(
                        func, x, y,
                        p0=np.array(p0) * (1 + np.random.uniform(-0.1, 0.1, len(p0)))
                        if attempt == 1 else p0,
                        bounds=bounds,
                        maxfev=80000,
                        method="trf",
                    )
                y_pred = func(x, *popt)
                score = compute_metric(eval_metric, y, y_pred, k=len(popt))
                results[name] = {"score": score, "params": popt}
                success = True
                break
            except Exception:
                continue
        if not success:
            results[name] = {"score": np.nan, "params": None}

    valid = {k: v for k, v in results.items() if np.isfinite(v["score"])}
    if not valid:
        return "N/A", np.nan, None, None

    best = max(valid, key=lambda k: valid[k]["score"])
    return best, round(valid[best]["score"], 3), valid[best]["params"], None

def _fit_pair(args):
    group_name, gdf, y_var, x_var, eval_metric, transformations, mode, min_obs = args
    x = gdf[x_var]
    y = gdf[y_var]
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_clean = x[valid].to_numpy()
    y_clean = y[valid].to_numpy()
    if len(x_clean) < min_obs:
        return (group_name, y_var, x_var, "N/A", np.nan, None)
    model, score, params, _ = compute_best_model(x_clean, y_clean, eval_metric, transformations, mode)
    return (group_name, y_var, x_var, model, score, params)

# === High-level interface ===
def zform(
    df,
    y=None,
    x=None,
    group_col=None,
    eval_metric="r2",
    transformations=None,
    min_obs=10,
    apply=False,
    naming="standard",
    export_csv=None,
    export_csv_index=False,
    return_results=False,
    mode="discovery",
    n_jobs=1,
    verbose=True,
):
    """
    Compute and optionally apply the best-fitting transformation between variable pairs.

    """

   # --- Normalize y/x input ---
    if isinstance(y, str):
        y = [y]
    if isinstance(x, str):
        x = [x]

    # --- Select numeric variables ---
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_vars:
        raise ValueError("No numeric variables found in DataFrame.")

    # --- Filter user-specified y/x to numeric only ---
    def _filter_numeric(vars_list, name):
        if vars_list is None:
            return None
        non_numeric = [v for v in vars_list if v not in numeric_vars]
        if non_numeric:
            warnings.warn(
                f"⚠️ Skipping non-numeric {name} columns: {', '.join(non_numeric)}",
                UserWarning,
                stacklevel=2,
            )
        kept = [v for v in vars_list if v in numeric_vars]
        if not kept:
            raise ValueError(f"No numeric {name} columns remain after filtering.")
        return kept

    y = _filter_numeric(y, "y")
    x = _filter_numeric(x, "x")

    # --- Determine which variables to use ---
    if y is None and x is None:
        y_vars = x_vars = numeric_vars
        if verbose:
            warnings.warn("Neither y nor x specified — applying ALL pairwise combinations.", UserWarning)
    elif y is not None and x is None:
        y_vars = y
        x_vars = [c for c in numeric_vars if c not in y]
    elif y is None and x is not None:
        x_vars = x
        y_vars = [c for c in numeric_vars if c not in x]
    else:
        y_vars, x_vars = y, x


    groups = [("All Data", df)] if group_col is None else df.groupby(group_col)
    results = defaultdict(dict)

    if verbose:
        print(f"\n🔍 Computing optimal forms for {len(y_vars)} Y × {len(x_vars)} X combinations...\n")

    jobs = [
        (group_name, gdf, y_var, x_var, eval_metric, transformations, mode, min_obs)
        for group_name, gdf in groups
        for y_var in y_vars
        for x_var in x_vars
        if y_var != x_var
    ]

        # === Parallel or serial processing ===
    if n_jobs != 1:
        with ProcessPoolExecutor(max_workers=None if n_jobs == -1 else n_jobs) as ex:
            futures = [ex.submit(_fit_pair, j) for j in jobs]
            results_list = [f.result() for f in as_completed(futures)]
    else:
        results_list = [_fit_pair(j) for j in jobs]

    # === Store results into results dict ===
    for group_name, y_var, x_var, model, score, params in results_list:
        results[(y_var, x_var)][f"{group_name} - best model"] = model
        results[(y_var, x_var)][f"{group_name} - best {eval_metric.upper()}"] = score
        results[(y_var, x_var)][f"{group_name} - params"] = (
            ", ".join(f"{p:.5g}" for p in params) if params is not None else None
        )

    # === Build summary DataFrame ===
    records = []
    for (y_var, x_var), result_dict in results.items():
        for key, model_name in result_dict.items():
            if "best model" not in key:
                continue
            group_name = key.split(" - ")[0]
            metric_key = f"{group_name} - best {eval_metric.upper()}"
            params_key = f"{group_name} - params"
            records.append({
                "y": y_var,
                "x": x_var,
                "Group": None if group_name == "All Data" else group_name,
                "Best Model": model_name,
                f"Best {eval_metric.upper()}": result_dict.get(metric_key, np.nan),
                "Parameters": result_dict.get(params_key, None),
            })

    results_df = pd.DataFrame.from_records(records).reset_index(drop=True)
    if len(results_df.get("Group", pd.Series()).dropna().unique()) <= 1:
        results_df = results_df.drop(columns=["Group"], errors="ignore")

    # === Optionally apply transformations ===
    if apply:
        try:
            from .zform_apply import zform_apply
        except ImportError:
            from zform_apply import zform_apply

        df = zform_apply(df, results_df, naming=naming)

    # === Optional CSV export ===
    if export_csv:
        results_df.to_csv(export_csv, index=export_csv_index)

    # === Return ===
    return (df, results_df) if return_results else df
