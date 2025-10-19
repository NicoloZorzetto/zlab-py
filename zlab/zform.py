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
from zlab._zform_metrics import compute_metric, is_higher_better
from zlab.zform_compute_models import compute_best_model, _fit_pair


# Use a safe process start method to avoid fork() warnings in Python 3.12+
try:
    multiprocessing.set_start_method("forkserver", force=True)
except RuntimeError:
    # The start method can only be set once per session
    pass


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
    eval_metric="r2",
    normalize_metrics=False,
    transformations=None,
    min_obs=10,
    apply=False,
    naming="standard",
    export_zforms_to=None,
    export_zforms_index=False,
    return_zforms=False,
    strategy="best",
    n_jobs=-1,
    maxfev=100000,
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
    maxfev : int, default=100000
        Maximum number of function evaluations during optimization.
        Increasing this value can improve convergence at the cost of computation time.
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
    >>> import pandas as pd
    >>> df = pd.read_csv('path/to/file.csv')
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
                strategy, n_jobs, maxfev, verbose
            )
    else:
        return _zform_core(
            df, y, x, group_col, eval_metric, transformations, min_obs, apply,
            naming, export_zforms_to, export_zforms_index, return_zforms,
            strategy, n_jobs, maxfev, verbose
        )


# --- Zform core ---

def _zform_core(
    df, y, x, group_col, eval_metric, transformations, min_obs, apply,
    naming, export_zforms_to, export_zforms_index, return_zforms,
        strategy, n_jobs, maxfev, verbose
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
        (group_name, gdf, y_var, x_var, eval_metric, transformations, strategy, min_obs, maxfev)
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
                    zforms_list.append(_fit_pair(j, normalize_metrics))
                    progress.advance(task)
        else:
            if verbose:
                print(f"Computing {len(jobs)} transformations sequentially...")
            zforms_list = [_fit_pair(j, normalize_metrics) for j in jobs]

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
