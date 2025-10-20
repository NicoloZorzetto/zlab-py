"""
_zform_model_defaults.py — Default parameters for zform fixed strategy.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""


import numpy as np

FIXED_DEFAULTS = {
    "linear": [1.0, 0.0],
    "power": [1.0, 2.0],
    "log_dynamic": [1.0, 0.0, np.e],
    "logistic": [1.0, 1.0, 0.0],
}
