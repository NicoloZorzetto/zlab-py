"""
Custom warning classes for zlab.
"""

import warnings


class ZlabWarning(Warning):
    """Base warning class for all zlab warnings."""
    default_stacklevel = 3
    _emitting = False  # recursion guard

    def __init__(self, message=None, *, stacklevel=None):
        # Prevent recursion: only emit once per construction
        if message and not self.__class__._emitting:
            self.__class__._emitting = True
            try:
                warnings.warn(message, self.__class__, stacklevel=stacklevel or self.default_stacklevel)
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
