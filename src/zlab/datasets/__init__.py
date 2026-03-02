"""
Example datasets distributed with zlab.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

from .load_penguins import load_penguins
from .load_iris import load_iris

__all__ = ["load_penguins", "load_iris"]
