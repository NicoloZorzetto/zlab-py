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
import sys
import warnings
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
from ._zform_config import ZformConfig
from .zform_functions import (
    get_zform_functions,
    register_zform_function,
    snapshot_zform_functions,
    ZFORM_FUNCTIONS,
    LOCAL_ZFORM_FUNCTIONS,
)
from .zform_compute_models import _fit_pair
from ._zform_metadata import make_metadata, attach_metadata, compute_sha256
from .zforms_object import Zforms


# Use a safe process start method to avoid fork() warnings in Python 3.12+
if sys.platform != "win32":
    try:
        multiprocessing.set_start_method("forkserver", force=True)
    except RuntimeError:
        # The start method can only be set once per session
        pass


# --- Transformation argument parser ---
def _parse_transformations(transformations):
    if isinstance(transformations, str):
        transformations = [transformations]

    custom_funcs = []
    temp_registrations = {}

    if transformations is None:
        return list(get_zform_functions().keys()), custom_funcs, temp_registrations

    if callable(transformations):
        transformations = [transformations]

    if any(t == "default" for t in transformations):
        base = list(get_zform_functions().keys())
        transformations = [t for t in transformations if t != "default"]
        transformations = base + transformations

    normalized = []
    for item in list(transformations):
        if callable(item):
            meta = getattr(item, "__zform_registration__", None)
            name = (
                meta.get("name") if meta else getattr(item, "__name__", "custom_func")
            )
            prev = ZFORM_FUNCTIONS.get(name)

            register_zform_function(
                func=meta["func"] if meta else item,
                name=name,
                n_params=meta.get("n_params") if meta else None,
                description=meta.get("description") if meta else None,
                init_func=meta.get("init_func") if meta else None,
                bounds=meta.get("bounds") if meta else None,
                fixed_params=meta.get("fixed_params") if meta else None,
                requires_positive_x=meta.get("requires_positive_x") if meta else False,
            )
            temp_registrations.setdefault(name, prev)
            normalized.append(name)
            custom_funcs.append(item)

        elif isinstance(item, str):
            if item not in ZFORM_FUNCTIONS and item in LOCAL_ZFORM_FUNCTIONS:
                payload = LOCAL_ZFORM_FUNCTIONS[item]
                prev = ZFORM_FUNCTIONS.get(item)
                register_zform_function(
                    func=payload["func"],
                    name=item,
                    n_params=payload.get("n_params"),
                    description=payload.get("description"),
                    init_func=payload.get("init_func"),
                    bounds=payload.get("bounds"),
                    fixed_params=payload.get("fixed_params"),
                    requires_positive_x=payload.get("requires_positive_x"),
                )
                temp_registrations.setdefault(item, prev)
            normalized.append(item)

        else:
            normalized.append(item)

    unknown = [t for t in normalized if isinstance(t, str) and t not in ZFORM_FUNCTIONS]
    if unknown:
        available = ", ".join(sorted(ZFORM_FUNCTIONS))
        raise ValueError(
            f"Unknown transformation(s): {unknown}. Available: {available}"
        )

    return normalized, custom_funcs, temp_registrations


# --- Multisystem multithreading support ---


def _in_interactive_session() -> bool:
    # No __main__.__file__ in notebooks/REPL, so treat as interactive
    return not getattr(__main__, "__file__", None)


def _has_dynamic_funcs(custom_funcs) -> bool:
    # Anything defined in __main__ or with a <locals> qualname is not safely pickleable
    for f in custom_funcs or []:
        if getattr(f, "__module__", "") == "__main__":
            return True
        if "<locals>" in getattr(f, "__qualname__", ""):
            return True
        if getattr(f, "__name__", "") == "<lambda>":
            return True
    return False


def _select_executor(custom_funcs, n_jobs, *, extra_callables=None):
    """
    Return (ExecutorClass, reason_string) based on portability/pickling constraints.
    Threads are chosen when user-defined functions are present or in interactive contexts.
    """
    if n_jobs in (None, 1):
        return None, "sequential"

    extra_callables = extra_callables or []
    dynamic = _has_dynamic_funcs(custom_funcs) or _has_dynamic_funcs(extra_callables)

    if dynamic or _in_interactive_session() or sys.platform.startswith("win"):
        return ThreadPoolExecutor, "threads (dynamic funcs / interactive / Windows)"

    return ProcessPoolExecutor, "processes"


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
    Zforms or (pandas.DataFrame, Zforms)
        Returns the fitted `Zforms` object and optionally the transformed DataFrame:
          - When `apply=False` -> `zforms_obj`
          - When `apply=True` -> `(df_out, zforms_obj)`
            * `zforms_obj` is a `Zforms` instance containing:
                - the full fitted forms table
                - metadata (version, timestamp, hash, and any custom functions)
                - methods for validation, export, reapplication, and summary display.
            * `df_out` is the transformed DataFrame.

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
    ...     df, group_col="species",
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
    >>> df_out, zf = zform(
    ...     df, group_col="species", strategy="best",
    ...     apply=True, naming="standard"
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
                df,
                y,
                x,
                group_col,
                transformations,
                apply,
                naming,
                export_zforms_to,
                export_zforms_index,
                verbose,
                config,
            )

    else:
        return _zform_core(
            df,
            y,
            x,
            group_col,
            transformations,
            apply,
            naming,
            export_zforms_to,
            export_zforms_index,
            verbose,
            config,
        )


# --- Zform core ---


def _zform_core(
    df,
    y,
    x,
    group_col,
    transformations,
    apply,
    naming,
    export_zforms_to,
    export_zforms_index,
    verbose,
    config: ZformConfig,
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
            ZformWarning(
                f"Skipping non-numeric {name} columns: {', '.join(non_numeric)}"
            )
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

    transformations, custom_funcs, temp_registrations = _parse_transformations(
        transformations
    )

    try:
        transform_names = [
            t if isinstance(t, str) else getattr(t, "__name__", "custom_func")
            for t in transformations
        ]
        func_snapshot = snapshot_zform_functions(transform_names)

        for entry in func_snapshot.values():
            custom_funcs.append(entry["func"])
            init_fn = entry.get("init_func")
            if callable(init_fn):
                custom_funcs.append(init_fn)

        transformations = transform_names

        groups = [("All Data", df)] if group_col is None else df.groupby(group_col)

        jobs = [
            (group_name, gdf, y_var, x_var, transformations, func_snapshot)
            for group_name, gdf in groups
            for y_var in y_vars
            for x_var in x_vars
            if y_var != x_var
        ]

        if verbose:
            if RICH_AVAILABLE:
                console.print(
                    f"\n[cyan]Computing optimal forms for "
                    f"{len(y_vars)} Y × {len(x_vars)} X combinations...[/cyan]\n"
                )
            else:
                print(
                    f"\nComputing optimal forms for "
                    f"{len(y_vars)} Y × {len(x_vars)} X combinations...\n"
                )

        # --- execution ---
        start_time = time.time()

        ExecutorClass, backend_reason = _select_executor(
            custom_funcs,
            config.n_jobs,
            extra_callables=[
                config.eval_metric,
                config.composite_metric_func,
                config.engine,
            ],
        )
        max_workers = None if config.n_jobs == -1 else config.n_jobs

        if ExecutorClass is not None:
            with ExecutorClass(max_workers=max_workers) as ex:
                futures = [ex.submit(_fit_pair, j, config) for j in jobs]

                if verbose and RICH_AVAILABLE:
                    with Progress(console=console) as progress:
                        task = progress.add_task(
                            "[cyan]Fitting transformations...", total=len(futures)
                        )
                        zforms_list = []
                        for f in as_completed(futures):
                            zforms_list.append(f.result())
                            progress.advance(task)
                else:
                    if verbose:
                        print(
                            f"Computing {len(futures)} pairwise transformations in {backend_reason}..."
                        )
                    zforms_list = [f.result() for f in as_completed(futures)]
        else:
            if verbose and RICH_AVAILABLE:
                with Progress(console=console) as progress:
                    task = progress.add_task(
                        "[cyan]Fitting transformations (sequential)...", total=len(jobs)
                    )
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
        n_transforms = len(
            transformations or ["linear", "power", "logistic", "log_dynamic"]
        )
        cores_used = (
            multiprocessing.cpu_count()
            if config.n_jobs == -1
            else (config.n_jobs if config.n_jobs != 1 else 1)
        )

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


        records = []
        metric_label = (
            config.eval_metric.upper()
            if isinstance(config.eval_metric, str)
            else "SCORE"
        )

        def _canonical_params(params):
            if params is None:
                return np.nan
            arr = np.asarray(params, float).round(10)
            return tuple(arr.tolist()) if arr.ndim == 1 else float(arr)

        for (
            group_name,
            y_var,
            x_var,
            transformation,
            score,
            params,
            gain,
            _,
        ) in zforms_list:
            records.append(
                {
                    "y": y_var,
                    "x": x_var,
                    "Group": np.nan if group_name == "All Data" else group_name,
                    "Best Transformation": transformation,
                    f"Best {metric_label}": score,
                    "Gain vs Fixed": gain,
                    "Parameters": _canonical_params(params),
                }
            )

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
            msg = (
                f"Computation completed over {total_specs:,} specifications × "
                f"{n_transforms} transformations.\n"
                f"Total function evaluations: {total_iterations:,}\n"
                f"Elapsed time: {format_time(elapsed)}\n"
                f"Used {cores_used} core{'s' if cores_used > 1 else ''}.\n"
            )
            if RICH_AVAILABLE:
                console.print(msg)
            else:
                print(msg)

        df_out = None
        if apply:
            df_out = zforms.apply(
                df,
                naming=naming,
                y=y,
                x=x,
                group_col=group_col,
                verbose=verbose,
            )

        if export_zforms_to:
            zforms.export_to(export_zforms_to)

    finally:
        for name, previous in temp_registrations.items():
            if previous is None:
                ZFORM_FUNCTIONS.pop(name, None)
            else:
                ZFORM_FUNCTIONS[name] = previous

    if apply:
        return df_out, zforms
    return zforms
