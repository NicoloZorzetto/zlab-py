"""
_zform_config.py — Class for configuration of zform.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

from dataclasses import dataclass, asdict
from typing import Callable
from collections.abc import Mapping

from zlab.warnings import ZformRuntimeWarning


@dataclass
class ZformConfig:
    """
    Configuration container for zform() and related subroutines.

    Centralizes parameters that control model fitting, metric normalization,
    penalty behavior, computational strategy, and regression engine.
    """

    eval_metric: str | list[str] | dict[str, float] | Callable = "r2"
    normalize_metrics: bool = False
    composite_metric_func: Callable | None = None
    penalize_theta_in_ic: bool = False
    strategy: str = "best"
    engine: str | Callable = "ols"
    engine_kwargs: Mapping[str, any] | None = None
    maxfev: int = 100_000
    min_obs: int = 10
    n_jobs: int = -1

    def __post_init__(self):
        """Validate configuration fields."""
        if self.strategy not in {"best", "fixed"}:
            raise ValueError(
                f"Invalid strategy '{self.strategy}', must be 'best' or 'fixed'."
            )
        if self.min_obs < 3:
            raise ValueError("min_obs must be ≥ 3.")
        if not isinstance(self.maxfev, int) or self.maxfev <= 0:
            raise ValueError("maxfev must be a positive integer.")
        if not isinstance(self.n_jobs, int):
            raise ValueError("n_jobs must be an integer.")
        if self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 (all cores), 1, or a positive integer.")
        if self.n_jobs == 0:
            self.n_jobs = 1
            ZformRuntimeWarning("n_jobs cannot be set to 0. Coercing n_jobs=1.")
        if isinstance(self.eval_metric, str) and not self.eval_metric:
            raise ValueError("eval_metric cannot be an empty string.")
        if self.composite_metric_func is not None and not callable(
            self.composite_metric_func
        ):
            raise ValueError("composite_metric_func must be a callable or None.")
        if not (isinstance(self.engine, str) or callable(self.engine)):
            raise ValueError("engine must be a string or callable.")
        if isinstance(self.engine, str) and self.engine.lower() not in {
            "ols",
            "ridge",
            "lasso",
        }:
            raise ValueError(
                "engine must be one of {'ols', 'ridge', 'lasso'} or a callable."
            )
        if self.engine_kwargs is None:
            self.engine_kwargs = {}
        elif not isinstance(self.engine_kwargs, Mapping):
            raise ValueError("engine_kwargs must be a mapping of keyword arguments.")

    def override(self, **kwargs) -> "ZformConfig":
        """Return a new ZformConfig with selected attributes overridden."""
        data = self.__dict__.copy()
        data.update({k: v for k, v in kwargs.items() if k in data})
        return ZformConfig(**data)

    def as_dict(self) -> dict:
        """Return a dict representation for easy unpacking."""
        return asdict(self)

    def __repr__(self):
        """Readable representation for logs and metadata."""
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ZformConfig({fields})"
