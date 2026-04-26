"""
Tests for zform verbosity.

Run with:
    pytest tests/test_zform_verbose.py
or:
    python tests/test_zform_verbose.py
"""

from zlab import zform
from zlab.datasets import load_penguins


def test_zform_verbose_synthetic_outputs(capsys):
    penguins = load_penguins()
    zform(
        penguins,
        y="bill_depth_mm",
        x="bill_length_mm",
        transformations=["linear"],
        strategy="fixed",
        verbose=True,
    )
    out = capsys.readouterr().out
    assert out  # some output when verbose


def test_zform_verbose_synthetic_suppresses_output(capsys):
    penguins = load_penguins()
    zform(
        penguins,
        y="bill_depth_mm",
        x="bill_length_mm",
        transformations=["linear"],
        strategy="fixed",
        verbose=False,
    )
    out = capsys.readouterr().out
    assert out == ""
