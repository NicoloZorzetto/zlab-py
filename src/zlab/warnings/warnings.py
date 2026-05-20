"""
Custom warning classes for zlab.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

import warnings
import inspect
from pathlib import Path

_PKG_DIR = str(Path(__file__).resolve().parents[1])


def find_stacklevel():
    """Return the first stacklevel outside zlab internals."""
    frame = inspect.currentframe()
    level = 1
    while frame:
        filename = frame.f_code.co_filename
        if not filename.startswith(_PKG_DIR):
            break
        frame = frame.f_back
        level += 1
    return level


class ZlabWarning(Warning):
    """Base warning class for all zlab warnings."""

    default_stacklevel = 2
    _emitting = False  # recursion guard

    def __init__(self, message=None, *, stacklevel=None):
        # Prevent recursion: only emit once per construction
        if message and not self.__class__._emitting:
            self.__class__._emitting = True
            try:
                warnings.warn(
                    message, self.__class__, stacklevel=stacklevel or find_stacklevel()
                )
            finally:
                self.__class__._emitting = False
        super().__init__(message)


class ZformWarning(ZlabWarning):
    """Generic warning for zform-related issues."""


class ZformExportWarning(ZformWarning):
    """Warning for export or file handling issues in zform."""


class ZformRuntimeWarning(ZformWarning):
    """Warning for runtime issues (e.g., missing data, convergence) in zform."""


class ZformApplyWarning(ZformWarning):
    """Warning raised when a transformation cannot be applied (e.g. missing forms)."""


class ZformFunctionWarning(ZformWarning):
    """Warning for custom function registration collisions."""
