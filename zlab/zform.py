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
from zlab._zform_config import ZformConfig
from zlab.zform_functions import get_zform_functions, zform_function
from zlab.zform_compute_models import _fit_pair
from zlab._zform_metadata import make_metadata, attach_metadata, compute_sha256
from zlab.zforms_object import Zforms


# Use a safe process start method to avoid fork() warnings in Python 3.12+
try:
    multiprocessing.set_start_method("forkserver", force=True)
except RuntimeError:
    # The start method can only be set once per session
    pass


# --- Transformation argument parser ---
def _parse_transformations(transformations):
    custom_funcs = []
    if transformations is not None:
        # Normalize to list
        if callable(transformations):
            transformations = [transformations]

        # Expand 'default' keyword to all built-ins
        if any(t == "default" for t in transformations):
            base = list(get_zform_functions().keys())
            transformations = [t for t in transformations if t != "default"]
            transformations = base + transformations

        # Register any callables dynamically
        for t in transformations:
            if callable(t):
                zform_function(t.__name__)(t)
                custom_funcs.append(t)

    else:
        # No user-specified transformations → use all defaults
        transformations = list(get_zform_functions().keys())

    return transformations, custom_funcs


# --- Export ---

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


# --- Main zform() API ---

def zform(
    df,
    y=None,
    x=None,
    group_col=None,
    transformations=None,
    apply=False,
    naming="standard",
    export_zforms_to=None,
    export_zforms_index=False,
    return_zforms=True,
    verbose=True,
    silence_warnings=False,
    config: ZformConfig | None = None,
    **kwargs,
):
    """
    Automatically identifies, fits, and optionally applies the best parametric
    transformations that linearize relationships between variables in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing numeric columns.
    y : str | list[str] | None, default=None
        Dependent variable(s). If None, all numeric columns are considered.
        One y will not be used as x for another unless explicitly included in x.
    x : str | list[str] | None, default=None
        Independent variable(s). If None, all numeric columns are considered.
        If both y and x are None, zform tests all pairwise combinations.
    group_col : str | list[str] | None, default=None
        Optional column(s) to group data by before fitting (e.g., "species").
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
        Helps balance metrics with different scales or directions.
    penalize_theta_in_ic : bool, default=False
        When False AIC/BIC penalize only intercept and slope (k=2).
        When True AIC/BIC penalize intercept, slope and number of transformation
        parameters (k = 2 + len(theta))
    transformations : list[str | callable] | None, default=None
        Subset of transformations to test.
        Default: all built-in forms (`['linear', 'power', 'log_dynamic', 'logistic']`).
        - If a **callable** is passed, it is registered and tested as a custom transformation.
        - You can include custom functions alongside defaults, e.g.:
          `[default, my_func]`.
        - If None, all default transformations are tested.
    min_obs : int, default=10
        Minimum number of valid observations required per (y, x) pair.
    apply : bool, default=False
        If True, applies the best transformations to the DataFrame and returns
        transformed columns (requires fitted parameters).
    naming : {'standard', 'compact', 'minimal'}, default='standard'
        Naming convention for transformed columns when `apply=True`.
    export_zforms_to : str | Path | None, default=None
        Optional export path (.csv, .xlsx, .json, .parquet, etc.) for fitted transformations.
    export_zforms_index : bool, default=False
        Whether to include the index when exporting fitted forms.
    return_zforms : bool, default=True
        If True, returns both a `Zforms` object and the (optionally) transformed input DataFrame.
        The `Zforms` contains fitted models, parameters, and embedded metadata.
        If False, returns just the (optionally) transformed input DataFrame.
    strategy : {'best', 'fixed'}, default='best'
        Transformation strategy:
          - "best": fits each model’s parameters via nonlinear optimization.
          - "fixed": evaluates canonical parameter sets (e.g., ln(x+1), x², logistic).
        In "best" mode, zform also reports the gain in R² versus fixed transformations.
    n_jobs : int, default=-1
        Number of parallel processes to use (-1 = all available cores).
        Automatically reduced to sequential mode when run from a top-level script.
    maxfev : int, default=100000
        Maximum number of function evaluations during optimization.
        Increasing this value can improve convergence at the cost of computation time.
    verbose : bool, default=True
        If True, prints progress, timing, and summary information.
    silence_warnings : bool, default=False
        If True, suppresses warnings.

    Returns
    -------
    (Zforms, pandas.DataFrame) or pandas.DataFrame 
        - If `return_zforms=True`: returns a tuple `(zforms_obj, df_out)`, where:
            * `zforms_obj` is a `Zforms` instance containing:
                - the full fitted forms table
                - metadata (version, timestamp, hash, and any custom functions)
                - methods for validation, export, reapplication, and summary display.
            * `df_out` is the transformed DataFrame.
        - If `return_zforms=False`: returns the transformed DataFrame (or original if apply=False).

    Notes
    -----
    - Fitted results are validated via a SHA256 integrity hash and include
      metadata about creation time, zlab version, and any registered custom functions.
    - When exported and re-imported via `Zforms.from_file()`, the hash is checked
      to ensure the results are unmodified.
    - Grouped fitting and parallel execution are supported.

    Examples
    --------
    # --- Fit only ---
    >>> from zlab import zform
    >>> import pandas as pd
    >>> df = pd.read_csv("data.csv")
    >>> zf = zform(
    ...     df, group_col="species", return_zforms=True,
    ...     strategy="best"
    ... )
    >>> zf.summary()
    >>> zf.export_to("results.json")
    >>> zf.validate()
    True

    # --- Fit and apply ---
    >>> from zlab import zform
    >>> import pandas as pd
    >>> df = pd.read_csv("data.csv")
    >>> zf, df_out = zform(
    ...     df, group_col="species", return_zforms=True,
    ...     strategy="best", apply=True, naming="standard"
    ... )
    >>> zf.summary()
    >>> zf.export_to("results.json")
    >>> zf.validate()
    True
    """

    # If no user config pull defaults
    config = config.override(**kwargs) if config else ZformConfig(**kwargs)

    eval_metric = config.eval_metric
    normalize_metrics = config.normalize_metrics
    penalize_theta_in_ic = config.penalize_theta_in_ic
    strategy = config.strategy
    min_obs = config.min_obs
    maxfev = config.maxfev

    
    if silence_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return _zform_core(
                df, y, x, group_col, transformations,
                apply, naming, export_zforms_to, export_zforms_index,
                return_zforms, verbose, config
            )

    else:
        return _zform_core(
            df, y, x, group_col, transformations,
            apply, naming, export_zforms_to, export_zforms_index,
            return_zforms, verbose, config
        )


# --- Zform core ---

def _zform_core(
    df, y, x, group_col, transformations,
    apply, naming, export_zforms_to, export_zforms_index,
    return_zforms, verbose, config: ZformConfig
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

    # --- Setup transformations ---

    transformations, custom_funcs = _parse_transformations(transformations)
    
    # --- Comp setup ---
    
    groups = [("All Data", df)] if group_col is None else df.groupby(group_col)
    zforms = defaultdict(dict)

    # Define all computation jobs first
    jobs = [
        (group_name, gdf, y_var, x_var, transformations)
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
        if config.n_jobs != 1:
            msg = ("Running from a top-level script — parallel mode disabled")
            ZformWarning(msg)
            config.n_jobs = 1
                
    # --- execution ---
    start_time = time.time()

    if config.n_jobs and config.n_jobs != 1:
        max_workers = None if config.n_jobs == -1 else config.n_jobs
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_fit_pair, j, config) for j in jobs]

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
                    zforms_list.append(_fit_pair(j, config=config))
                    progress.advance(task)
        else:
            if verbose:
                print(f"Computing {len(jobs)} transformations sequentially...")
            zforms_list = [_fit_pair(j, config=config) for j in jobs]

    elapsed = time.time() - start_time


    total_iterations = sum(r[-1] for r in zforms_list)
    total_specs = len(jobs)
    n_transforms = len(transformations or ["linear", "power", "logistic", "log_dynamic"])
    cores_used = multiprocessing.cpu_count() if config.n_jobs == -1 else (config.n_jobs if config.n_jobs != 1 else 1)

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
        metric_label = (config.eval_metric.upper()
                if isinstance(config.eval_metric, str) else "SCORE")
        zforms[(y_var, x_var)][f"{group_name} - best {metric_label}"] = score
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
            metric_label = (config.eval_metric.upper()
                if isinstance(config.eval_metric, str) else "SCORE")
            metric_key = f"{group_name} - best {metric_label}"
            gain_key = f"{group_name} - gain_vs_fixed"
            params_key = f"{group_name} - params"
            metric_label = (config.eval_metric.upper()
                if isinstance(config.eval_metric, str) else "SCORE")
            records.append({
                "y": y_var,
                "x": x_var,
                "Group": None if group_name == "All Data" else group_name,
                "Best Model": model_name,
                f"Best {metric_label}": result_dict.get(metric_key, np.nan),
                f"Gain_vs_Fixed_{metric_key}": result_dict.get(gain_key, np.nan),
                "Parameters": result_dict.get(params_key, None),
            })

    zforms_df = pd.DataFrame.from_records(records).reset_index(drop=True)
    if len(zforms_df.get("Group", pd.Series()).dropna().unique()) <= 1:
        zforms_df = zforms_df.drop(columns=["Group"], errors="ignore")

    # Add metadata for future use, reference and validation
    metadata = make_metadata(custom_funcs)
    metadata["config"] = config.as_dict()
    metadata["sha256"] = compute_sha256(zforms_df)
    zforms_df = attach_metadata(zforms_df, metadata)
    zforms = Zforms(zforms_df)
        
    if verbose:
        console.print(
            f"Computation completed over {total_specs:,} specifications × "
            f"{n_transforms} transformations.\n"
            f"Total function evaluations: {total_iterations:,}\n"
            f"Elapsed time: {format_time(elapsed)}\n"
            f"Used {cores_used} core{'s' if cores_used > 1 else ''}.\n"
        )

    if apply:
        df = zforms.apply(df, naming=naming, y=y, x=x, group_col=group_col, verbose=verbose)

    if export_zforms_to:
        zforms.export_to(export_zforms_to)

    return (zforms, df) if return_zforms else df
