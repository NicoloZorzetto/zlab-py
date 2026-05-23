"""
Transformation functions and parameter initialization
for zform functions.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

import numpy as np

from zlab.warnings import ZformRuntimeWarning
from zlab.zform._zform_functions_bounds import (
    SQRT_MAX_FLOAT,
    MIN_LOG_BASE,
    MIN_POS_FLOAT,
    MIN_EXP_ARGUMENT,
    MAX_EXP_ARGUMENT,
    MAX_EXP_K,
    MIN_EXP_K,
)


# --- Registration helpers ---
def snapshot_zform_functions(names: list[str]) -> dict[str, dict]:
    """Return a shallow copy of registry entries for the requested names."""
    snapshot = {}
    for name in names:
        entry = ZFORM_FUNCTIONS.get(name)
        if entry:
            snapshot[name] = {
                "func": entry["func"],
                "n_params": entry.get("n_params"),
                "description": entry.get("description"),
                "fixed_params": entry.get("fixed_params"),
                "bounds": entry.get("bounds"),
                "init_func": entry.get("init_func"),
                "requires_positive_x": entry.get("requires_positive_x"),  # add this
            }
    return snapshot


def ensure_zform_functions(snapshot: dict[str, dict]):
    """Restore registry entries from previously captured snapshot."""
    for name, entry in snapshot.items():
        ZFORM_FUNCTIONS[name] = entry


# --- Default class for initialization of customs ---
class _DefaultInit:
    """Pickle-safe default initializer for custom transforms."""

    def __init__(self, n_params=None):
        self.n_params = n_params or 1

    def __call__(self, x, y):
        return [1.0] * self.n_params


# --- Registry of all known zform functions ---
ZFORM_FUNCTIONS = {}

# --- Registry for local zform functions ---
LOCAL_ZFORM_FUNCTIONS = {}


# --- Decorator for registration ---
def zform_function(
    name,
    n_params=None,
    description=None,
    init_func=None,
    bounds=None,
    fixed_params=None,
    register=False,
    requires_positive_x=False,
):
    """
    Decorator to register a new zform transformation.

    Parameters
    ----------
    name : str
        Transformation name used in registry.
    n_params : int | None
        Number of free parameters (0 for parameter-free forms).
    description : str | None
        Human-readable description.
    init_func : callable | None
        Initializer (x, y) -> list/tuple of starting params; defaults to `fixed_params` if `None`.
        Lambdas/closures work but may not serialize safely; prefer named, module-level functions
        for export/import and pull-from-export.
    bounds : tuple[list|tuple, list|tuple] | None
        Parameter bounds (lower, upper) for curve fitting.
    fixed_params : list | tuple | None
        Canonical parameter set for `strategy="fixed"`.
    register : bool
        If True, immediately register in the global registry; otherwise only define locally.
    requires_positive_x : bool | callable
        If True, skip when x has nonpositive values;
        if callable, skip when callable(x) returns True.

    Notes
    ----------
    Built-in names: ``linear``, ``power``, ``inverse``, ``root``, ``log_dynamic``,
    ``exponential``, ``logistic``. Override only if intentional.
    """

    def decorator(func):
        if init_func and init_func.__code__.co_freevars:
            msg = (
                f"init_func for '{name}' uses closure variables: "
                f"{init_func.__code__.co_freevars}. "
                "Restoration may fail. Prefer explicit parameters."
            )
            ZformRuntimeWarning(msg)
        # embed init_func
        if init_func is None:
            if fixed_params:

                def actual_init(x, y):
                    return list(fixed_params)

            else:
                actual_init = _DefaultInit(n_params)
        else:
            actual_init = init_func

        entry = {
            "func": func,
            "n_params": n_params,
            "description": description,
            "fixed_params": fixed_params,
            "bounds": bounds,
            "init_func": actual_init,
            "requires_positive_x": requires_positive_x,
        }

        if register:
            ZFORM_FUNCTIONS[name] = entry
        else:
            entry["name"] = name
            LOCAL_ZFORM_FUNCTIONS[name] = entry
            setattr(func, "__zform_registration__", entry)

        return func

    return decorator


# --- Built-in transformations ---


@zform_function(
    "linear",
    n_params=0,
    description="Identity transformation: x",
    init_func=lambda x, y: [],
    bounds=([], []),
    fixed_params=[],
    requires_positive_x=False,
    register=True,
)
def linear_func(x):
    """
    Identity transformation.
    Returns: x.
    """
    return x


@zform_function(
    "power",
    n_params=1,
    description="Power-law transformation: x**b",
    init_func=lambda x, y: [1.0],
    bounds=([-SQRT_MAX_FLOAT], [SQRT_MAX_FLOAT]),
    fixed_params=[1.0],
    requires_positive_x=False,
    register=True,
)
def power_func(x, b):
    """
    Power-law transformation.
    Returns: np.power(x, b).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.power(x, b)


@zform_function(
    "inverse",
    n_params=1,
    description="Reciprocal transform: 1 / (x**k)",
    init_func=lambda x, y: [1.0],
    bounds=([MIN_POS_FLOAT], [SQRT_MAX_FLOAT]),
    fixed_params=[1.0],
    requires_positive_x=False,
    register=True,
)
def inverse_func(x, k):
    """
    Reciprocal power transformation.
    Returns: np.power(x, -abs(k)).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.power(x, -abs(k))


@zform_function(
    "root",
    n_params=1,
    description="Root transform: x ** (1 / k)",
    init_func=lambda x, y: [2.0],  # start from square root
    bounds=([MIN_POS_FLOAT], [SQRT_MAX_FLOAT]),
    fixed_params=[2.0],
    requires_positive_x=False,
    register=True,
)
def root_func(x, k):
    """
    K-th root transformation.
    Returns: np.power(x, 1.0 / (k or 1.0)).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.power(x, 1.0 / (k or 1.0))


@zform_function(
    "log_dynamic",
    n_params=1,
    description="Logarithmic transformation with variable base: log_base(x)",
    init_func=lambda x, y: [np.e],  # start from natual log
    bounds=([MIN_LOG_BASE], [SQRT_MAX_FLOAT]),
    fixed_params=[np.e],
    requires_positive_x=True,
    register=True,
)
def log_func_dynamic(x, base):
    """
    Logarithmic transformation with variable base.
    Returns: np.log(x)/np.log(base).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x) / np.log(base)


@zform_function(
    "exponential",
    n_params=1,
    description="Exponential transformation: e^(k * x)",
    init_func=lambda x, y: [1.0],  # normal start
    bounds=([MIN_EXP_K], [MAX_EXP_K]),
    fixed_params=[1.0],
    requires_positive_x=False,
    register=True,
)
def exponential_func(x, k):
    """
    Exponential transformation.
    Returns: np.exp(np.clip(k * x, MIN_EXP_ARGUMENT, MAX_EXP_ARGUMENT)).
    """
    argument = np.clip(k * x, MIN_EXP_ARGUMENT, MAX_EXP_ARGUMENT)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp(argument)


@zform_function(
    "logistic",
    n_params=2,
    description="Logistic transformation: 1 / (1 + exp(-k*(x - x0)))",
    init_func=lambda x, y: [
        1.0 / (np.std(x) or 1.0),  # k
        float(np.median(x)),  # x0
    ],
    bounds=([1e-12, -SQRT_MAX_FLOAT], [1e12, SQRT_MAX_FLOAT]),
    fixed_params=[1.0, 0.0],
    requires_positive_x=False,
    register=True,
)
def logistic_func(x, k, x0):
    """
    Logistic (sigmoid) transformation.
    Returns: 1.0 / (1.0 + np.exp(argument)).
    """
    argument = np.clip(-k * (x - x0), MIN_EXP_ARGUMENT, MAX_EXP_ARGUMENT)
    with np.errstate(over="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.exp(argument))


# --- Register builtin function explicitly ---

BUILTIN_TRANSFORM_NAMES = set(ZFORM_FUNCTIONS.keys())

# --- API helpers ---


def get_zform_functions(transformations=None):
    """Return dict of transformation name -> callable."""
    if transformations is None:
        return {n: v["func"] for n, v in ZFORM_FUNCTIONS.items()}
    funcs = {}
    for t in transformations:
        if isinstance(t, str):
            if t not in ZFORM_FUNCTIONS:
                msg = (
                    f"Unknown transformation '{t}' "
                    f"— available: {list(ZFORM_FUNCTIONS.keys())}"
                )
                raise KeyError(msg)
            funcs[t] = ZFORM_FUNCTIONS[t]["func"]
        elif callable(t):
            funcs[getattr(t, "__name__", "custom_func")] = t
        else:
            raise TypeError(f"Unsupported transformation: {t}")
    return funcs


def guess_initial_params(x, y, transformation_name):
    """Retrieve the model-specific initialization function."""
    entry = ZFORM_FUNCTIONS.get(transformation_name)
    if entry is None or entry.get("init_func") is None:
        msg = (
            f"Unknown model '{transformation_name}' — "
            "no init function registered.\n"
            "Defaulting to initial guess = 1.0 for all parameters.\n"
            "This may cause convergence issues.\n"
            "Please specify an 'init_func' in your custom "
            "transformation definition."
        )
        ZformRuntimeWarning(msg)
        return [1.0]
    return entry["init_func"](x, y)


# -- Programmatic helper --
def register_zform_function(
    *,
    func,
    name,
    n_params=None,
    description=None,
    init_func=None,
    bounds=None,
    fixed_params=None,
    register=True,
    requires_positive_x=False,
):
    """
    Programmatic registration of a zform transformation.

    Equivalent to @zform_function(...) but for dynamic injection.

    Parameters
    ----------
    func : callable
        Transformation function ``f(x, *theta)``.
    name : str
        Registry name.
    n_params : int | None
        Number of free parameters (0 for parameter-free forms).
    description : str | None
        Human-readable description.
    init_func : callable | None
        Initializer (x, y) -> list/tuple of starting params; defaults to `fixed_params` if `None`.
        Lambdas/closures work but may not serialize safely; prefer named, module-level functions
        for export/import and pull-from-export.
    bounds : tuple[list|tuple, list|tuple] | None
        Parameter bounds (lower, upper) for curve fitting.
    fixed_params : list | tuple | None
        Canonical parameter set for `strategy="fixed"`.
    register : bool
        If True, immediately register in the global registry; otherwise only define locally.
    requires_positive_x : bool | callable
        If True, skip when x has nonpositive values;
        if callable, skip when callable(x) returns True.

    Notes
    ----------
    Built-in names: ``linear``, ``power``, ``inverse``, ``root``, ``log_dynamic``,
    ``exponential``, ``logistic``. Override only if intentional.
    """

    # reuse the decorator
    decorator = zform_function(
        name=name,
        n_params=n_params,
        description=description,
        init_func=init_func,
        bounds=bounds,
        fixed_params=fixed_params,
        requires_positive_x=requires_positive_x,
        register=True,
    )
    decorator(func)
    return func
