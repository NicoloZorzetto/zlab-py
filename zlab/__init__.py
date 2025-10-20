"""
zlab: Parametric transformation and linearization toolkit.

This package provides:
- zform(): automatic transformation fitting
- Zforms: container object for fitted transformations with metadata, validation, and export utilities
"""

from .zform import zform
from .zforms_object import Zforms

__all__ = [
    "zform",
    "Zforms",
]
