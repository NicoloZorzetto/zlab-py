#!/usr/bin/env python
"""
zform_apply applies previously computed transformations from zform
to a DataFrame, using stored parameters.

This module is part of the zlab library by Nicolò Zorzetto.

License
-------
GPL v3
"""

try:
    import warnings
    import numpy as np
    import pandas as pd
except ImportError as e:
    raise ImportError(
        f"Missing dependency: {e.name}. Please install all requirements via "
        "'pip install -r requirements.txt'"
    )

try:
    from .zform import (
        linear_func,
        log_func,
        log_func_dynamic,
        power_func,
        logistic_func,
    )
except ImportError as e:
    raise ImportError(
        f"Could not import {e.name}."
        "Please ensure the zlab package is intact."
        "If you are working on a prototype version, "
        "ensure you are running your script from the project root."
    )

def _func_from_name(name):
    """Map model name to transformation function."""
    return {
        "linear": linear_func,
        "power": power_func,
        "logistic": logistic_func,
        "log": log_func,
        "log_dynamic": log_func_dynamic,
    }.get(name, linear_func)


def zform_apply(
    df,
    forms,
    y=None,
    x=None,
    group_col=None,
    naming="standard",
):
    """
    Apply one or multiple previously computed zform transformations
    to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Original data to transform.
    forms : pandas.DataFrame
        Output of zform() or a CSV with the same structure.
        Must include ['y', 'x', 'Best Model', 'Parameters'].
    y : str | list[str] | None
        Dependent variable(s) (targets). If None, applies to all y in forms.
    x : str | list[str] | None
        Independent variable(s) (predictors). If None, applies to all x in forms.
    group_col : None | str | list[str], optional
        Column(s) defining groups used during zform fitting.
        If provided, zform_apply will apply group-specific transformations
        based on the 'Group' column in `forms`.
    naming : {'standard', 'compact', 'minimal'}, default='standard'
        - 'standard': adds columns like x_z_logistic_for_y
        - 'compact':  adds columns like x_z_logistic_y
        - 'minimal':  adds columns like x_z_logistic (⚠ overwrites if reused)

    Returns
    -------
    pandas.DataFrame
        A copy of df with transformed predictor columns added.

    Notes
    -----
    - The transformation is applied to the *predictor variable (x)*,
      not to the dependent variable (y).
    - If groups were used in zform, they must match those in `df[group_col]`.
    """
    df = df.copy()

    # === Validate input ===
    required_cols = {"y", "x", "Best Model", "Parameters"}
    if not required_cols.issubset(forms.columns):
        raise ValueError(f"Forms DataFrame must contain columns: {required_cols}")

        # --- Normalize y/x input ---
    if isinstance(y, str):
        y = [y]
    if isinstance(x, str):
        x = [x]

    # --- Determine which y/x pairs exist in forms ---
    available_y = forms["y"].unique().tolist()
    available_x = forms["x"].unique().tolist()

    # --- Filter user-specified y/x to only those present in forms ---
    def _filter_valid(vars_list, available, name):
        if vars_list is None:
            return None
        missing = [v for v in vars_list if v not in available]
        if missing:
            warnings.warn(
                f"⚠️ Skipping {name} not found in forms: {', '.join(missing)}",
                UserWarning,
                stacklevel=2,
            )
        kept = [v for v in vars_list if v in available]
        if not kept:
            raise ValueError(f"No valid {name} remain after filtering (none found in forms).")
        return kept

    y = _filter_valid(y, available_y, "y")
    x = _filter_valid(x, available_x, "x")

    # --- Determine subset of forms to apply ---
    if y is None and x is None:
        warnings.warn(
            "Neither y nor x specified — applying ALL pairwise transformations.\n"
            "This may create a column for every y~x combination found in 'forms', "
            "potentially squaring your dataset's width.",
            UserWarning,
        )
        subset = forms.copy()
    elif y is not None and x is None:
        subset = forms.query("y in @y")
    elif y is None and x is not None:
        subset = forms.query("x in @x")
    else:
        subset = forms.query("y in @y and x in @x")

    if subset.empty:
        raise ValueError(
            "No matching transformations found for the given y/x pairs in forms. "
            "Ensure they were fitted by zform()."
        )

    # === Handle grouping ===
    if group_col is not None and "Group" in forms.columns:
        if isinstance(group_col, str):
            df["_zgroup"] = df[group_col].astype(str)
        elif isinstance(group_col, list):
            df["_zgroup"] = df[group_col].astype(str).agg("_".join, axis=1)
        else:
            raise ValueError("group_col must be None, str, or list of str.")

        groups = df["_zgroup"].unique()
        print(f"Applying group-specific transformations for {len(groups)} groups...")

        for group_name, gdf in df.groupby("_zgroup"):
            subforms = subset.query("Group == @group_name or Group.isnull()")
            if subforms.empty:
                continue

            for _, row in subforms.iterrows():
                yv, xv, model, params_str = row["y"], row["x"], row["Best Model"], row["Parameters"]
                if model == "N/A" or not isinstance(params_str, str):
                    continue
                if xv not in df.columns or yv not in df.columns:
                    warnings.warn(f"Skipping ({yv}, {xv}) — missing column in df.", UserWarning)
                    continue

                try:
                    params = [float(p.strip()) for p in params_str.split(",")]
                    func = _func_from_name(model)

                    # Naming logic
                    if naming == "standard":
                        col_name = f"{xv}_z_{model}_for_{yv}_{group_name}"
                    elif naming == "compact":
                        col_name = f"{xv}_z_{model}_{yv}_{group_name}"
                    elif naming == "minimal":
                        col_name = f"{xv}_z_{model}_{group_name}"
                    else:
                        raise ValueError("Invalid naming mode. Use 'standard', 'compact', or 'minimal'.")

                    mask = df["_zgroup"] == group_name
                    df.loc[mask, col_name] = func(df.loc[mask, xv].to_numpy(), *params)

                except Exception as e:
                    warnings.warn(f"Failed applying {model} to {xv} ~ {yv} (group={group_name}): {e}", UserWarning)

        df.drop(columns="_zgroup", inplace=True)

    else:
        # === Apply globally (no grouping) ===
        print(f"Applying {len(subset)} global transformations...")

        for _, row in subset.iterrows():
            yv, xv, model, params_str = row["y"], row["x"], row["Best Model"], row["Parameters"]
            if model == "N/A" or not isinstance(params_str, str):
                continue
            if xv not in df.columns or yv not in df.columns:
                warnings.warn(f"Skipping ({yv}, {xv}) — missing column in df.", UserWarning)
                continue

            try:
                params = [float(p.strip()) for p in params_str.split(",")]
                func = _func_from_name(model)

                # Naming logic
                if naming == "standard":
                    col_name = f"{xv}_z_{model}_for_{yv}"
                elif naming == "compact":
                    col_name = f"{xv}_z_{model}_{yv}"
                elif naming == "minimal":
                    col_name = f"{xv}_z_{model}"
                else:
                    raise ValueError("Invalid naming mode. Use 'standard', 'compact', or 'minimal'.")

                if naming == "minimal" and col_name in df.columns:
                    warnings.warn(f"Overwriting {col_name} (multiple y for same x).", UserWarning)

                df[col_name] = func(df[xv].to_numpy(), *params)

            except Exception as e:
                warnings.warn(f"Failed applying {model} to {xv} ~ {yv}: {e}", UserWarning)

    return df
