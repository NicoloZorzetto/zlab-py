"""
zform automatically identifies and optionally applies
the best parametric transformation
that linearizes the relationship between variables.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

import __main__
import warnings
import time
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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

try:
    from rich.console import Console
    from rich.progress import Progress
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


from zlab.warnings import ZformWarning, ZformExportWarning, ZformRuntimeWarning


# Use a safe process start method to avoid fork() warnings in Python 3.12+
try:
    multiprocessing.set_start_method("forkserver", force=True)
except RuntimeError:
    # The start method can only be set once per session
    pass


# ------------------ Transformation families ------------------

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


# ------------------ Evaluation metric ------------------

def compute_metric(name: str, y_true, y_pred, k: int = 0):
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
        ZformWarning(f"Unknown eval_metric '{name}', falling back to R².")
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (rss / tss) if tss != 0 else np.nan


# ------------------ Model fitting ------------------

def compute_best_model(x, y, eval_metric="r2", transformations=None, strategy="best"):
    """
    Fit or evaluate all candidate transformations between x and y.
    strategy='best' -> full parametric fit
    strategy='fixed' -> evaluate canonical parameter sets only
    """

    # canonical fixed parameters for each model (no fitting)
    FIXED_DEFAULTS = {
        "linear": [1.0, 0.0],
        "power": [1.0, 2.0],
        "log_dynamic": [1.0, 0.0, np.e],
        "logistic": [1.0, 1.0, 0.0],
    }

    def guess_initial_params(x, y, model_name):
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

    TRANSFORMATIONS = {
        "linear": linear_func,
        "power": power_func,
        "log_dynamic": log_func_dynamic,
        "logistic": logistic_func,
    }

    if transformations is not None:
        TRANSFORMATIONS = {k: v for k, v in TRANSFORMATIONS.items() if k in transformations}

    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return "N/A", np.nan, None, None, 0

    zforms = {}
    bounds = (-1e8, 1e8)
    total_iters = 0

    # ---- FIXED STRATEGY ----
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
        best = max(valid, key=lambda k: valid[k]["score"])
        return best, round(valid[best]["score"], 3), valid[best]["params"], None, 0

    # ---- BEST STRATEGY ----
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
    best_score = round(valid[best]["score"], 3)
    best_params = valid[best]["params"]

    # ---- Compute gain vs fixed ----
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
        fixed_best_score = fixed_scores[best_fixed]
        gain = best_score - fixed_best_score

    return best, best_score, best_params, gain, total_iters


# ------------------ Parallel fitting wrapper ------------------

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


# ------------------ Export ------------------

def export_zforms(df, path):
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
            return None
    except Exception as e:
        ZformExportWarning(f"Export failed for '{path}': {e}")
        return None

    return path


# ------------------ Main zform() API ------------------

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
    strategy="best",
    n_jobs=-1,
    verbose=True,
    silence_warnings=False,
):
    """
    Automatically identifies and optionally applies the best parametric transformations
    that linearize relationships between variables in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing numeric columns.
    y : str | list[str] | None, default=None
        Dependent variable(s). If None, all numeric columns are considered.
        One y will not be used as x for another unless it is explicitly included in x as well.
    x : str | list[str] | None, default=None
        Independent variable(s). If None, all numeric columns are considered.
        If both y and x are None, zform tests all pairwise combinations.
    group_col : str | list[str] | None, default=None
        Optional column(s) to group data by before fitting (e.g., "species").
    eval_metric : {'r2', 'adjr2', 'rmse', 'mae', 'aic', 'bic'}, default='r2'
        Metric used to evaluate transformation fit quality.
    transformations : list[str] | None, default=None
        Subset of transformations to test.
        Available: {'linear', 'power', 'log_dynamic', 'logistic'}.
        If None, all are tested.
    min_obs : int, default=10
        Minimum number of valid observations required per (y, x) pair.
    apply : bool, default=False
        If True, applies the best transformations to the DataFrame and returns
        transformed columns (requires 'Parameters' in fitted forms).
    naming : {'standard', 'compact', 'minimal'}, default='standard'
        Naming convention for transformed columns when apply=True.
    export_zforms_to : str | Path | None, default=None
        Optional export path (.csv, .xlsx, .json, .parquet, etc.) for fitted forms.
    export_zforms_index : bool, default=False
        Whether to include the index when exporting fitted forms.
    return_zforms : bool, default=False
        If True, returns both the transformed DataFrame and a DataFrame of
        fitted models and parameters.
    strategy : {'best', 'fixed'}, default='best'
        Transformation strategy:
          - "best": fits each model’s parameters via nonlinear optimization.
          - "fixed": evaluates canonical parameter sets (e.g., ln(x+1), x², logistic).
        In "best" mode, zform also reports the gain in R² versus fixed transformations.
    n_jobs : int, default=-1
        Number of parallel processes to use (-1 = all available cores).
    verbose : bool, default=True
        If True, prints progress and timing information.
    silence_warnings : bool, default=False
        If True, suppresses warnings.

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, pandas.DataFrame)
        - If return_zforms=False: the input DataFrame.
        - If return_zforms=True: a tuple (the input DataFrame, zforms_df), where:
            zforms_df includes columns:
                ['y', 'x', 'Group', 'Best Model', 'Best R2', 'Gain_vs_Fixed_R2', 'Parameters']
        The input DataFrame may be modified if apply=True.

    Notes
    -----
    - When `strategy='best'`, parameters are estimated via `scipy.optimize.curve_fit`
      using dynamically generated starting points.
    - When `strategy='fixed'`, pre-defined transformations are evaluated directly
      without optimization.
    - Grouped fitting and parallel execution are supported.
    - The “Gain_vs_Fixed” column quantifies improvement over standard functional forms
      that would be obtained when `strategy='fixed'`.

    Examples
    --------
    >>> from zlab import zform
    >>> import seaborn as sbn
    >>> df = sbn.load_dataset("penguins").dropna()
    >>> df_out, zforms = zform(
    ...     df, group_col="species", return_zforms=True,
    ...     strategy="best", apply=True, naming="standard"
    ... )
    >>> zforms.head()
    """

    if silence_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return _zform_core(
                df, y, x, group_col, eval_metric, transformations, min_obs, apply,
                naming, export_zforms_to, export_zforms_index, return_zforms,
                strategy, n_jobs, verbose
            )
    else:
        return _zform_core(
            df, y, x, group_col, eval_metric, transformations, min_obs, apply,
            naming, export_zforms_to, export_zforms_index, return_zforms,
            strategy, n_jobs, verbose
        )


# ------------------ Zform core ------------------

def _zform_core(
    df, y, x, group_col, eval_metric, transformations, min_obs, apply,
    naming, export_zforms_to, export_zforms_index, return_zforms,
    strategy, n_jobs, verbose
):
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

    # --- setup ---
    groups = [("All Data", df)] if group_col is None else df.groupby(group_col)
    zforms = defaultdict(dict)

    # Define all computation jobs first
    jobs = [
        (group_name, gdf, y_var, x_var, eval_metric, transformations, strategy, min_obs)
        for group_name, gdf in groups
        for y_var in y_vars
        for x_var in x_vars
        if y_var != x_var
    ]

    if verbose:
        if RICH_AVAILABLE:
            console.print(f"\n[cyan]Computing optimal forms for "
                          f"{len(y_vars)} Y × {len(x_vars)} X combinations...[/cyan]\n")
        else:
            print(f"\nComputing optimal forms for "
                  f"{len(y_vars)} Y × {len(x_vars)} X combinations...\n")

    # --- SAFETY: disable parallel mode when run from a top-level script ---
    if getattr(__main__, "__file__", None) and __main__.__file__.endswith(".py"):
        if n_jobs != 1:
            msg = ("Running from a top-level script — parallel mode disabled")
            ZformWarning(msg)
            n_jobs = 1
                
    # --- execution ---
    start_time = time.time()

    if n_jobs and n_jobs != 1:
        max_workers = None if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_fit_pair, j) for j in jobs]

            if verbose and RICH_AVAILABLE:
                with Progress(console=console) as progress:
                    task = progress.add_task(f"[cyan]Fitting models...", total=len(futures))
                    zforms_list = []
                    for f in as_completed(futures):
                        zforms_list.append(f.result())
                        progress.advance(task)
            else:
                if verbose:
                    print(f"Computing {len(futures)} pairwise transformations...")
                zforms_list = [f.result() for f in as_completed(futures)]
    else:
        if verbose and RICH_AVAILABLE:
            with Progress(console=console) as progress:
                task = progress.add_task(f"[cyan]Fitting models (sequential)...", total=len(jobs))
                zforms_list = []
                for j in jobs:
                    zforms_list.append(_fit_pair(j))
                    progress.advance(task)
        else:
            if verbose:
                print(f"Computing {len(jobs)} transformations sequentially...")
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

    for group_name, y_var, x_var, model, score, params, gain, _ in zforms_list:
        zforms[(y_var, x_var)][f"{group_name} - best model"] = model
        zforms[(y_var, x_var)][f"{group_name} - best {eval_metric.upper()}"] = score
        zforms[(y_var, x_var)][f"{group_name} - gain_vs_fixed"] = gain
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
            gain_key = f"{group_name} - gain_vs_fixed"
            params_key = f"{group_name} - params"
            records.append({
                "y": y_var,
                "x": x_var,
                "Group": None if group_name == "All Data" else group_name,
                "Best Model": model_name,
                f"Best {eval_metric.upper()}": result_dict.get(metric_key, np.nan),
                f"Gain_vs_Fixed_{eval_metric.upper()}": result_dict.get(gain_key, np.nan),
                "Parameters": result_dict.get(params_key, None),
            })

    zforms_df = pd.DataFrame.from_records(records).reset_index(drop=True)
    if len(zforms_df.get("Group", pd.Series()).dropna().unique()) <= 1:
        zforms_df = zforms_df.drop(columns=["Group"], errors="ignore")

    if verbose:
        console.print(
            f"Computation completed over {total_specs:,} specifications × "
            f"{n_transforms} transformations.\n"
            f"Total function evaluations: {total_iterations:,}\n"
            f"Elapsed time: {format_time(elapsed)}\n"
            f"Used {cores_used} core{'s' if cores_used > 1 else ''}.\n"
        )

    if apply:
        try:
            from .zform_apply import zform_apply
        except ImportError:
            from zform_apply import zform_apply
        df = zform_apply(df, zforms_df, naming=naming)

    if export_zforms_to:
        export_zforms(zforms_df, export_zforms_to)

    return (df, zforms_df) if return_zforms else df
