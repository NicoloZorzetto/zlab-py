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

import warnings
import time
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

try:
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit, OptimizeWarning  # type: ignore[import-untyped]
except ImportError as e:
    msg = (
        f"Missing dependency: {e.name}. Please install all requirements via "
        "'pip install -r requirements.txt'"
    )
    raise ImportError(msg)

from zlab.warnings import ZformWarning, ZformExportWarning, ZformRuntimeWarning


# --- Transformations ---


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


# --- Metric computation ---

def compute_metric(name: str, y_true, y_pred, k: int = 0):
    """Compute evaluation metrics for model fit quality."""
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
    elif name == "adjr2":
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (rss / tss) if tss != 0 else np.nan
        return 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan
    elif name == "rmse":
        return np.sqrt(rss / n)
    elif name == "mae":
        return np.mean(np.abs(resid))
    elif name == "aic":
        return n * np.log(rss / n) + 2 * k if rss > 0 else -np.inf
    elif name == "bic":
        return n * np.log(rss / n) + k * np.log(n) if rss > 0 else -np.inf
    else:
        msg = f"Unknown eval_metric '{name}', falling back to R²."
        ZformWarning(msg)
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (rss / tss) if tss != 0 else np.nan


# --- Model computation ---

def compute_best_model(x, y, eval_metric="r2", transformations=None, mode=""):
    """Fit available transformations and select the best one."""

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

    # Define transformation pool
    TRANSFORMATIONS = {
        "linear": linear_func,
        "power": power_func,
        "logistic": logistic_func,
    }

    if mode == "discovery":
        TRANSFORMATIONS["log_dynamic"] = log_func_dynamic
    else:
        TRANSFORMATIONS["log"] = log_func

    # Validate transformations
    if transformations is not None:
        if not transformations:
            raise ValueError("Transformations list is empty — nothing to compute.")
        TRANSFORMATIONS = {t: f for t, f in TRANSFORMATIONS.items() if t in transformations}

    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return "N/A", np.nan, None, None, 0

    zforms = {}
    bounds = (-1e8, 1e8)
    total_iters = 0

    for name, func in TRANSFORMATIONS.items():
        if "log" in name and np.any(x <= -1):
            continue
        if name == "power" and np.any(x < 0):
            continue

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

    best = max(valid, key=lambda k: valid[k]["score"])
    return best, round(valid[best]["score"], 3), valid[best]["params"], None, total_iters


# --- Helper functions ---

def _fit_pair(args):
    group_name, gdf, y_var, x_var, eval_metric, transformations, mode, min_obs = args
    x, y = gdf[x_var], gdf[y_var]
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[valid].to_numpy(), y[valid].to_numpy()
    if len(x_clean) < min_obs:
        return (group_name, y_var, x_var, "N/A", np.nan, None, 0)
    model, score, params, _, n_iter = compute_best_model(x_clean, y_clean, eval_metric, transformations, mode)
    return (group_name, y_var, x_var, model, score, params, n_iter)


def export_zforms(df, path):
    """Export zforms results to a supported file format."""
    path = Path(path)
    ext = path.suffix.lower()

    try:
        if ext == ".csv":
            df.to_csv(path, index=False)
        elif ext in (".xls", ".xlsx"):
            df.to_excel(path, index=False)
        elif ext == ".json":
            df.to_json(path, orient="records", indent=2)
        elif ext == ".parquet":
            df.to_parquet(path)
        elif ext == ".html":
            df.to_html(path, index=False)
        elif ext == ".md":
            df.to_markdown(path)
        else:
            ZformExportWarning(f"Unsupported export format '{ext}', skipping export.")
    except Exception as e:
        ZformExportWarning(f"Export failed for '{path}': {e}")


# --- Main zform function ---

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
    export_zforms_to=None,
    export_zforms_index=False,
    return_zforms=False,
    mode="discovery",
    n_jobs=1,
    verbose=True,
    silence_warnings=False,
):
    """Identify and optionally apply the best-fitting transformation between variables."""

    if silence_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return _zform_core(
                df, y, x, group_col, eval_metric, transformations, min_obs, apply,
                naming, export_zforms_to, export_zforms_index, return_zforms,
                mode, n_jobs, verbose
            )
    else:
        return _zform_core(
            df, y, x, group_col, eval_metric, transformations, min_obs, apply,
            naming, export_zforms_to, export_zforms_index, return_zforms,
            mode, n_jobs, verbose
        )


def _zform_core(
    df, y, x, group_col, eval_metric, transformations, min_obs, apply,
    naming, export_zforms_to, export_zforms_index, return_zforms,
    mode, n_jobs, verbose
):
    """Core logic for zform (separated for clarity)."""

    if isinstance(y, str):
        y = [y]
    if isinstance(x, str):
        x = [x]

    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_vars:
        raise ValueError("No numeric variables found in DataFrame.")

    def _filter_numeric(vars_list, name):
        if vars_list is None:
            return None
        non_numeric = [v for v in vars_list if v not in numeric_vars]
        if non_numeric and verbose:
            ZformWarning(f"Skipping non-numeric {name} columns: {', '.join(non_numeric)}")
        kept = [v for v in vars_list if v in numeric_vars]
        if not kept:
            raise ValueError(f"No numeric {name} columns remain after filtering.")
        return kept

    y = _filter_numeric(y, "y")
    x = _filter_numeric(x, "x")

    if y is None and x is None:
        y_vars = x_vars = numeric_vars
        ZformWarning("Neither y nor x specified — applying ALL pairwise combinations.")
    elif y is not None and x is None:
        y_vars = y
        x_vars = [c for c in numeric_vars if c not in y]
    elif y is None and x is not None:
        x_vars = x
        y_vars = [c for c in numeric_vars if c not in x]
    else:
        y_vars, x_vars = y, x

    groups = [("All Data", df)] if group_col is None else df.groupby(group_col)
    zforms = defaultdict(dict)

    if verbose:
        print(f"\nComputing optimal forms for {len(y_vars)} Y × {len(x_vars)} X combinations...\n")

    jobs = [
        (group_name, gdf, y_var, x_var, eval_metric, transformations, mode, min_obs)
        for group_name, gdf in groups
        for y_var in y_vars
        for x_var in x_vars
        if y_var != x_var
    ]

    start_time = time.time()
    if n_jobs and n_jobs != 1:
        with ProcessPoolExecutor(max_workers=None if n_jobs == -1 else n_jobs) as ex:
            futures = [ex.submit(_fit_pair, j) for j in jobs]
            zforms_list = [f.result() for f in as_completed(futures)]
    else:
        zforms_list = [_fit_pair(j) for j in jobs]

    elapsed = time.time() - start_time
    total_iterations = sum(r[-1] for r in zforms_list)
    total_specs = len(jobs)
    n_transforms = len(transformations or ["linear", "power", "logistic", "log_dynamic"])
    cores_used = multiprocessing.cpu_count() if n_jobs == -1 else (n_jobs if n_jobs != 1 else 1)

    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            m, s = divmod(seconds, 60)
            return f"{int(m)} minutes {s:.1f} seconds"
        else:
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{int(h)} hours {int(m)} minutes {int(s)} seconds"

    # Build summary table
    for group_name, y_var, x_var, model, score, params, _ in zforms_list:
        zforms[(y_var, x_var)][f"{group_name} - best model"] = model
        zforms[(y_var, x_var)][f"{group_name} - best {eval_metric.upper()}"] = score
        zforms[(y_var, x_var)][f"{group_name} - params"] = (
            ", ".join(f"{p:.5g}" for p in params) if params is not None else None
        )

    records = []
    for (y_var, x_var), result_dict in zforms.items():
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

    zforms_df = pd.DataFrame.from_records(records).reset_index(drop=True)
    if len(zforms_df.get("Group", pd.Series()).dropna().unique()) <= 1:
        zforms_df = zforms_df.drop(columns=["Group"], errors="ignore")

    if verbose:
        print(
            f"Computation completed over {total_specs:,} specifications × {n_transforms} transformations.\n"
            f"Total function evaluations: {total_iterations:,}\n"
            f"Elapsed time: {format_time(elapsed)}\n"
            f"Used {cores_used} core{'s' if cores_used > 1 else ''}.\n"
        )

    # Optional application
    if apply:
        try:
            from .zform_apply import zform_apply
        except ImportError:
            from zform_apply import zform_apply
        df = zform_apply(df, zforms_df, naming=naming)

    # Optional export
    if export_zforms_to:
        export_zforms(zforms_df, export_zforms_to)

    return (df, zforms_df) if return_zforms else df
