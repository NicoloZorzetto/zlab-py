"""
load_penguins.py — Offers the penguins dataset.

Source:
    Horst, A. M., Hill, A. P., & Gorman, K. B. (2020).
    *palmerpenguins: Palmer Archipelago (Antarctica) penguin data.*
    License: CC0 1.0 Universal.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

from importlib.resources import files
import pandas as pd


def load_penguins():
    """Return the Palmer Penguins dataset as a pandas DataFrame.

    Source:
        Horst, A. M., Hill, A. P., & Gorman, K. B. (2020).
        *palmerpenguins: Palmer Archipelago (Antarctica) penguin data.*
        License: CC0 1.0 Universal.

    Returns
    -------
    pandas.DataFrame
        The Palmer Penguins dataset with species, island, bill dimensions,
        flipper length, body mass, and sex columns.
    """
    path = files("zlab.datasets.data").joinpath("penguins.csv")
    return pd.read_csv(path)
