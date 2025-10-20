"""
_zform_config.py — Class for configuration of zform.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

from dataclasses import dataclass, asdict
from typing import Callable

@dataclass
class ZformConfig:
    """
    Configuration container for zform() and related subroutines.

    Centralizes parameters that control model fitting, metric normalization,
    penalty behavior, and computational strategy.
    """
    eval_metric: str | list[str] | dict[str, float] | Callable = "r2"
    normalize_metrics: bool = False
    penalize_theta_in_ic: bool = False
    strategy: str = "best"
    maxfev: int = 100_000
    min_obs: int = 10
    n_jobs: int = -1

    def __post_init__(self):
        """Validate configuration fields."""
        if self.strategy not in {"best", "fixed"}:
            raise ValueError(f"Invalid strategy '{self.strategy}', must be 'best' or 'fixed'.")
        if self.min_obs < 3:
            raise ValueError("min_obs must be ≥ 3.")
        if not isinstance(self.maxfev, int) or self.maxfev <= 0:
            raise ValueError("maxfev must be a positive integer.")
        if not isinstance(self.n_jobs, int):
            raise ValueError("n_jobs must be an integer.")
        if isinstance(self.eval_metric, str) and not self.eval_metric:
            raise ValueError("eval_metric cannot be an empty string.")

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
