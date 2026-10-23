"""Forecast accuracy metrics and their translation into business impact (MXN).

Accuracy uses WAPE (weighted absolute percentage error = sum |error| / sum actual):
it is scale-free across series of very different size and, unlike MAPE, it does not
explode on small actuals. Bias (sum error / sum actual) shows whether a forecast
systematically orders too much (positive) or too little (negative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_analytics.features import SERIES_KEYS


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute WAPE = sum |y_pred - y_true| / sum y_true, ignoring null actuals.

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    float
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    m = ~np.isnan(y_true)
    return float(np.abs(y_pred[m] - y_true[m]).sum() / y_true[m].sum())


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute bias = sum (y_pred - y_true) / sum y_true, ignoring null actuals.

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    float
        Positive: over-forecast (excess stock). Negative: under-forecast (lost sales).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    m = ~np.isnan(y_true)
    return float((y_pred[m] - y_true[m]).sum() / y_true[m].sum())


def score(preds: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Summarize WAPE and bias, overall or by groups.

    Parameters
    ----------
    preds : pandas.DataFrame
        Backtest output with ``y_true`` and ``y_pred`` (plus any grouping columns).
    by : list of str, optional
        Grouping columns (e.g. ``["model"]`` or ``["model", "store_format"]``).

    Returns
    -------
    pandas.DataFrame
        Columns ``wape``, ``bias`` and ``units`` (sum of actuals).
    """
    d = preds.dropna(subset=["y_true"])

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "wape": wape(g["y_true"], g["y_pred"]),
                "bias": bias(g["y_true"], g["y_pred"]),
                "units": g["y_true"].sum(),
            }
        )

    if not by:
        return agg(d).to_frame().T
    return d.groupby(by, observed=True)[["y_true", "y_pred"]].apply(agg)


def inventory_impact(preds: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Simulate weekly replenishment with each forecast and value the misses in MXN.

    Simplified policy, identical for every model so the comparison is fair: at each
    origin the store orders, for each category, the sum of the forecast for the
    week. Units of demand above the order are lost sales; units ordered above
    demand are excess stock. Days without an actual are excluded on both sides.

    Parameters
    ----------
    preds : pandas.DataFrame
        Backtest output with ``model``, keys, ``origin``, ``y_true`` and ``y_pred``.
    prices : pandas.DataFrame
        ``store_id``, ``category``, ``date`` and ``unit_price`` (MXN per unit).

    Returns
    -------
    pandas.DataFrame
        Per model and origin: lost units, excess units and their value in MXN.
    """
    d = preds.dropna(subset=["y_true"]).merge(prices, on=[*SERIES_KEYS, "date"], how="left")
    d["sales_mxn"] = d["y_true"] * d["unit_price"]
    week = d.groupby(["model", "origin", *SERIES_KEYS], observed=True).agg(
        demand=("y_true", "sum"), order=("y_pred", "sum"), sales_mxn=("sales_mxn", "sum")
    )
    price = week["sales_mxn"] / week["demand"]
    week["lost_units"] = (week["demand"] - week["order"]).clip(lower=0)
    week["excess_units"] = (week["order"] - week["demand"]).clip(lower=0)
    week["lost_mxn"] = week["lost_units"] * price
    week["excess_mxn"] = week["excess_units"] * price
    cols = ["demand", "sales_mxn", "lost_units", "excess_units", "lost_mxn", "excess_mxn"]
    return week.groupby(["model", "origin"], observed=True)[cols].sum().reset_index()
