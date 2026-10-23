"""Data-quality diagnostics used in the EDA.

Each function returns a small, readable table instead of printing, so the notebook
only displays results and the tests can assert on the same numbers. Column labels
of the returned tables are in Spanish because they are shown to business readers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PANEL_KEYS = ["date", "store_id", "category"]


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count nulls per column (absolute and %), only for columns with any null.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Columns ``nulos`` and ``pct``, sorted descending.
    """
    n = df.isna().sum()
    out = pd.DataFrame({"nulos": n, "pct": (n / len(df) * 100).round(2)})
    return out[out["nulos"] > 0].sort_values("nulos", ascending=False)


def null_patterns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Find null combinations that occur together (columns that fail at the same time).

    Parameters
    ----------
    df : pandas.DataFrame
    cols : list of str
        Columns with nulls to cross.

    Returns
    -------
    pandas.DataFrame
        One row per pattern, ``x`` where the column is null, plus the row count.
    """
    pat = df[cols].isna()
    counts = pat.value_counts().rename("filas").reset_index()
    counts[cols] = counts[cols].replace({True: "x", False: ""})
    return counts


def missingness_tests(df: pd.DataFrame, col: str, by: list[str]) -> pd.DataFrame:
    """Chi-square independence test between "is null" and each variable in ``by``.

    If nulls come from random POS failures (MCAR), the null rate should not depend
    on the day, the store or the promotion. A high p-value is consistent with that
    hypothesis; it does not prove it, but the absence of dependence is what we need
    so that dropping or reconstructing values does not bias results.

    Parameters
    ----------
    df : pandas.DataFrame
    col : str
        Column with nulls.
    by : list of str
        Categorical variables to test against.

    Returns
    -------
    pandas.DataFrame
        Statistic, degrees of freedom, p-value and range of the null rate by group.
    """
    rows = []
    is_null = df[col].isna()
    for var in by:
        table = pd.crosstab(df[var], is_null)
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rate = is_null.groupby(df[var], observed=True).mean() * 100
        rows.append(
            {
                "columna": col,
                "contra": var,
                "chi2": round(chi2, 1),
                "gl": dof,
                "p_valor": round(p, 4),
                "tasa_min_%": round(rate.min(), 2),
                "tasa_max_%": round(rate.max(), 2),
            }
        )
    return pd.DataFrame(rows)


def grid_gaps(tx: pd.DataFrame, stores: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """List the rows missing from the full date x store x category grid.

    Parameters
    ----------
    tx, stores, calendar : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Per store: first and last missing date, missing days and missing rows.
    """
    grid = pd.MultiIndex.from_product(
        [calendar["date"], stores["store_id"].astype("string"), tx["category"].unique()],
        names=PANEL_KEYS,
    ).to_frame(index=False)
    merged = grid.merge(tx[PANEL_KEYS], how="left", indicator=True)
    gaps = merged[merged["_merge"] == "left_only"]
    return (
        gaps.groupby("store_id")
        .agg(
            desde=("date", "min"),
            hasta=("date", "max"),
            dias=("date", "nunique"),
            filas=("date", "size"),
        )
        .reset_index()
    )


def accounting_identities(tx: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    """Check the accounting identities of the file, split by promotion / no promotion.

    Identities checked:

    - ``total_transactions = cash_transactions + card_transactions``
    - ``amount_total = amount_cash + amount_card``
    - ``avg_ticket = amount_total / total_transactions``
    - ``avg_ticket = amount_total / (cash_transactions + card_transactions)``

    Parameters
    ----------
    tx : pandas.DataFrame
        Raw transactions.
    tol : float
        Absolute tolerance (MXN) when comparing amounts and tickets.

    Returns
    -------
    pandas.DataFrame
        % of evaluable rows where each identity holds, with and without promotion.
    """

    def close(diff: pd.Series, tolerance: float) -> pd.Series:
        # Null when there is nothing to compare: a row without cash does not "fail", it is not evaluable.
        return (diff.abs() < tolerance).astype("boolean").mask(diff.isna())

    d = tx.copy()
    paid = d["cash_transactions"] + d["card_transactions"]
    checks = {
        "total_tx = efectivo + tarjeta": d["total_transactions"] == paid,
        "monto = efectivo + tarjeta": close(
            d["amount_total"] - d["amount_cash"] - d["amount_card"], tol
        ),
        "ticket = monto / total_tx": close(
            d["avg_ticket"] - d["amount_total"] / d["total_transactions"].astype(float), tol
        ),
        "ticket = monto / (efectivo + tarjeta)": close(
            d["avg_ticket"] - d["amount_total"] / paid.astype(float), tol + 0.005
        ),
    }
    rows = []
    for name, ok in checks.items():
        ok = ok.astype("boolean")
        for promo in (0, 1):
            m = (d["has_promotion"] == promo) & ok.notna()
            pct = round(float(ok[m].astype(float).mean()) * 100, 2) if m.any() else np.nan
            rows.append(
                {"identidad": name, "promocion": promo, "filas": int(m.sum()), "se_cumple_%": pct}
            )
    return (
        pd.DataFrame(rows)
        .pivot(index="identidad", columns="promocion", values="se_cumple_%")
        .rename(columns={0: "sin promo (%)", 1: "con promo (%)"})
    )


def phantom_transactions(tx: pd.DataFrame) -> pd.Series:
    """Compute reported transactions without a payment method: ``total - (cash + card)``.

    Parameters
    ----------
    tx : pandas.DataFrame

    Returns
    -------
    pandas.Series
        Difference per row (null when cash is missing).
    """
    return tx["total_transactions"] - tx["cash_transactions"] - tx["card_transactions"]


def replenishment_window_check(
    tx: pd.DataFrame, window: int = 7, tol: float = 0.01
) -> pd.DataFrame:
    """Find which window of ``total_transactions`` reproduces ``replenishment_signal``.

    Moving averages of ``window`` days ending on different days relative to ``t`` are
    compared against the signal. If the matching window ends after ``t``, the signal
    uses information that did not exist when it was computed.

    Parameters
    ----------
    tx : pandas.DataFrame
        Raw transactions.
    window : int
        Length of the moving average, in days.
    tol : float
        Absolute tolerance to count a row as an exact match.

    Returns
    -------
    pandas.DataFrame
        Per candidate window: % of exact matches and median absolute error.
    """
    d = tx.sort_values(["store_id", "category", "date"])
    g = d.groupby(["store_id", "category"])["total_transactions"]
    rows = []
    for end in (-1, 0, 1, 2):
        # end = days after t where the window ends (end=1 -> includes t+1)
        est = g.transform(
            lambda s, e=end: s.astype(float).shift(-e).rolling(window, min_periods=1).mean()
        )
        err = (d["replenishment_signal"] - est).abs()
        rows.append(
            {
                "ventana": f"t{end - window + 1:+d} … t{end:+d}".replace("+0", ""),
                "termina_en": end,
                "coincidencia_exacta_%": round(
                    (err < tol).mean() / d["replenishment_signal"].notna().mean() * 100, 2
                ),
                "error_abs_mediano": round(err.median(), 3),
            }
        )
    return pd.DataFrame(rows)


def ratio_to_baseline(values: pd.Series, groups: list[pd.Series]) -> pd.Series:
    """Divide each value by the median of its group (to spot peaks and drops).

    Parameters
    ----------
    values : pandas.Series
    groups : list of pandas.Series
        Grouping keys aligned with ``values``.

    Returns
    -------
    pandas.Series
    """
    return values / values.groupby(groups).transform("median")


def within_cell_lift(df: pd.DataFrame, value: str, flag: str, cell: list[str]) -> pd.Series:
    """Estimate the relative effect of a binary flag, comparing within the same cell.

    For each cell (e.g. store x category x month) computes
    ``mean(value | flag=1) / mean(value | flag=0) - 1``. Comparing within the cell
    controls for store level, category and seasonality without a model.

    Parameters
    ----------
    df : pandas.DataFrame
    value : str
        Outcome column.
    flag : str
        Binary column (0/1).
    cell : list of str
        Columns that define the comparison cell.

    Returns
    -------
    pandas.Series
        One effect per cell (only cells with both cases).
    """
    m = df.groupby([*cell, flag], observed=True)[value].mean().unstack(flag)
    return (m[1] / m[0] - 1).dropna()


def bootstrap_ci(
    x: pd.Series, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    """Compute the mean and a percentile bootstrap confidence interval.

    Parameters
    ----------
    x : pandas.Series
    n_boot : int
    alpha : float
    seed : int

    Returns
    -------
    tuple of float
        (mean, lower bound, upper bound).
    """
    rng = np.random.default_rng(seed)
    arr = x.dropna().to_numpy()
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)
