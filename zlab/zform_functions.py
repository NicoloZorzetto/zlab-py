"""
Transformation functions and parameter initialization
for zform models.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

try:
    import numpy as np
except ImportError as e:
    msg = (
        f"Missing dependency: {e.name}. Please install all requirements via "
        "'pip install -r requirements.txt'"
    )
    raise ImportError(msg)


# --- Registry of all known zform models ---
ZFORM_FUNCTIONS = {}
INIT_GUESS_FUNCS = {}


# --- Decorator for registration ---
def zform_function(name, n_params=None, description=None, init_func=None, bounds=None):
    """
    Decorator to register a new zform transformation.

    Parameters
    ----------
    name : str
        Unique name of the transformation (e.g. "linear", "power").
    n_params : int, optional
        Number of parameters (for reference/documentation).
    description : str, optional
        Short human-readable description.
    init_func : callable, optional
        Optional initialization function: init_func(x, y) -> list of params.
    bounds : tuple[list, list] | None
        Optional (lower_bounds, upper_bounds) for parameters.
        If not provided, wide defaults are used.

    Example
    -------
    >>> @zform_function("quadratic", n_params=2, description="y = a*x^2 + b")
    ... def quadratic_func(x, a, b):
    ...     return a * x**2 + b
    ...
    >>> @zform_function("weird", init_func=lambda x, y: [1, 0])
    ... def weird_func(x, a, b):
    ...     return a*np.sin(x) + b
    """
    def decorator(func):
        # --- Register transformation ---
        ZFORM_FUNCTIONS[name] = {
            "func": func,
            "n_params": n_params,
            "description": description,
        }

        # --- Register init function if provided ---
        if init_func:
            INIT_GUESS_FUNCS[name] = init_func
        else:
            # Safe fallback initializer: scale + offset or single scale
            def _default_init(x, y):
                if len(np.shape(x)) == 0:
                    return [1.0]
                try:
                    return [np.std(y) / (np.std(x) + 1e-8), np.mean(y)]
                except Exception:
                    return [1.0]
            INIT_GUESS_FUNCS[name] = _default_init

        # --- Register bounds if provided ---
        if bounds:
            BOUNDS_REGISTRY[name] = bounds

        return func

    return decorator


# --- Built-in transformations ---

@zform_function("linear", n_params=2, description="Linear model: a*x + b",
                init_func=lambda x, y: [
                    (np.std(y) or 1.0) / (np.std(x) or 1.0),
                    np.mean(y) - (np.std(y) or 1.0)/(np.std(x) or 1.0) * np.mean(x)
                ])
def linear_func(x, a, b):
    return a * x + b


@zform_function("power", n_params=2, description="Power model: a*x^b",
                init_func=lambda x, y: [np.mean(y) / (np.mean(x) or 1.0), 1.0])
def power_func(x, a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a * np.power(x, b)


@zform_function("log_dynamic", n_params=3, description="Dynamic log model: a*log(b*x + c)",
                init_func=lambda x, y: [
                    (np.std(y) or 1.0) / (np.std(np.log(np.abs(x) + 1)) or 1.0),
                    np.mean(y),
                    np.e
                ])
def log_func_dynamic(x, a, b, c):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a * np.log(b * x + c)


@zform_function("logistic", n_params=3, description="Logistic model: L / (1 + exp(-k*(x - x0)))",
                init_func=lambda x, y: [
                    float(np.max(y)),
                    1.0 / (np.std(x) or 1.0),
                    float(np.median(x))
                ])
def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


# --- API helpers ---

def get_zform_functions(transformations=None):
    """Return dict of transformation name -> callable."""
    if transformations is None:
        return {n: v["func"] for n, v in ZFORM_FUNCTIONS.items()}
    funcs = {}
    for t in transformations:
        if isinstance(t, str):
            if t not in ZFORM_FUNCTIONS:
                raise KeyError(f"Unknown transformation '{t}' — available: {list(ZFORM_FUNCTIONS.keys())}")
            funcs[t] = ZFORM_FUNCTIONS[t]["func"]
        elif callable(t):
            funcs[getattr(t, "__name__", "custom_func")] = t
        else:
            raise TypeError(f"Unsupported transformation: {t}")
    return funcs


def guess_initial_params(x, y, model_name):
    """Retrieve the model-specific initialization function."""
    func = INIT_GUESS_FUNCS.get(model_name)
    if func is None:
        msg = (f"Unknown model '{model_name}' — no init function registered.\n"
               "Defaulting to initial guess = 1.0 for all parameters.\n"
               "This may cause convergence issues.\n"
               "Please specify an 'init_func' in your custom transformation definition.")
        ZformRuntimeWarning(msg)
        return [1.0]
    return func(x, y)
