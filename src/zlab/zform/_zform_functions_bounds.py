"""
_zform_function_bounds.py

Safe numerical parameter bounds for zform transformations.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

import math
import numpy as np

# --- Numerical constants ---

MAX_FLOAT = np.finfo(float).max
MIN_POS_FLOAT = np.finfo(float).tiny
SQRT_MAX_FLOAT = math.sqrt(MAX_FLOAT)  # ~1e154 — avoids single-step overflow
MIN_LOG_BASE = 1.0 + math.ulp(1.0)  # Just above 1.0 (log undefined at 1)
MAX_EXP_ARGUMENT = np.log(np.finfo(float).max)
MIN_EXP_ARGUMENT = np.log(np.finfo(float).tiny)
MAX_EXP_K = 10.0
MIN_EXP_K = 1e-8
