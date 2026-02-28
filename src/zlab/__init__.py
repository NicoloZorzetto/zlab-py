"""
zlab: Parametric transformation and linearization toolkit.

This package provides:
- zform(): automatic transformation fitting
- Zforms: container object for fitted transformations with metadata, validation, and export utilities
"""

__version__ = "0.0.1.dev0"


from .zform.zform import zform
from .zform.zforms_object import Zforms

__all__ = [
    "zform",
    "Zforms",
]
