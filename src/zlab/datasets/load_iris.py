"""
load_iris.py — Offers the iris dataset.

Source:
    Fisher, R. A. (1936).
    *The use of multiple measurements in taxonomic problems.*
    Public domain.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

from importlib.resources import files
import pandas as pd


def load_iris():
    """Return the Iris dataset as a pandas DataFrame.

    Source:
        Fisher, R. A. (1936).
        *The use of multiple measurements in taxonomic problems.*
        Public domain.

    Returns
    -------
    pandas.DataFrame
        The classic Iris flower dataset with 4 features and target species.
    """
    path = files("zlab.datasets.data").joinpath("iris.csv")
    return pd.read_csv(path)
