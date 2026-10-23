"""Cleaning of the transactions panel.

Principle: first reconstruct what can be *deduced* (accounting identities that hold
in 100% of checkable rows), and only then, flagged, what can be *estimated*.
``units_sold`` is not imputed here: any imputation that learns from the data
(medians, models) must be fitted inside the training pipeline on training data only,
otherwise it would leak information from the test period.

Decisions are documented in ``notebooks/01_eda.ipynb`` (section 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_analytics.quality import PANEL_KEYS


def complete_grid(tx: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Add the rows missing from the date x store x category grid.

    New rows keep null metrics and ``is_missing_row=True``. They are flagged instead
    of filled because we do not know whether the store opened that day; filling with
    zeros would invent a demand drop that never happened.

    Parameters
    ----------
    tx : pandas.DataFrame
        Raw transactions.
    calendar : pandas.DataFrame
        Calendar with every date of the period.

    Returns
    -------
    pandas.DataFrame
        Complete panel, sorted by store, category and date.
    """
    grid = pd.MultiIndex.from_product(
        [calendar["date"], tx["store_id"].unique(), tx["category"].unique()], names=PANEL_KEYS
    ).to_frame(index=False)
    grid["store_id"] = grid["store_id"].astype("string")
    grid["category"] = grid["category"].astype("string")
    out = grid.merge(tx, on=PANEL_KEYS, how="left", validate="one_to_one")
    out["is_missing_row"] = out["total_transactions"].isna()
    return out.sort_values(["store_id", "category", "date"]).reset_index(drop=True)


def reconstruct_payments(tx: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct cash transactions, the real transaction count and the ticket.

    Order used to fill ``cash_transactions`` (recorded in ``cash_source``):

    1. ``reported``: present in the file.
    2. ``identity``: day without promotion -> ``total - card`` (exact identity in
       100% of non-promotion rows).
    3. ``ticket``: promotion day with ticket available -> ``amount / ticket - card``
       (exact in 100% of checkable rows; ``total`` is inflated on promotion days).
    4. ``estimated``: promotion day without ticket -> ``cash_amount / card_ticket``,
       assuming the same ticket for cash and card (observed difference +/-5%).

    ``amount_cash`` is always deduced as ``amount_total - amount_card``.

    Parameters
    ----------
    tx : pandas.DataFrame
        Panel with the raw columns.

    Returns
    -------
    pandas.DataFrame
        Copy with ``cash_transactions``, ``amount_cash`` and ``avg_ticket`` filled
        where possible, plus ``transactions_clean``, ``phantom_transactions`` and
        ``cash_source``.
    """
    d = tx.copy()
    d["amount_cash"] = d["amount_cash"].fillna(d["amount_total"] - d["amount_card"])

    source = pd.Series(
        np.where(d["cash_transactions"].notna(), "reported", "missing"), index=d.index
    )
    cash = d["cash_transactions"].astype("Float64")
    no_promo = d["has_promotion"] == 0
    promo = d["has_promotion"] == 1

    m = cash.isna() & no_promo
    cash[m] = d.loc[m, "total_transactions"] - d.loc[m, "card_transactions"]
    source[m] = "identity"

    m = cash.isna() & promo & d["avg_ticket"].notna()
    cash[m] = (d.loc[m, "amount_total"] / d.loc[m, "avg_ticket"]).round() - d.loc[
        m, "card_transactions"
    ]
    source[m] = "ticket"

    m = cash.isna() & promo
    card_ticket = d.loc[m, "amount_card"] / d.loc[m, "card_transactions"]
    cash[m] = (d.loc[m, "amount_cash"] / card_ticket).round()
    source[m] = "estimated"

    # Rows added by complete_grid have nothing to reconstruct.
    source[d["total_transactions"].isna()] = "missing"

    d["cash_transactions"] = cash.round().astype("Int64")
    d["cash_source"] = source.astype(str)
    d["transactions_clean"] = (d["cash_transactions"] + d["card_transactions"]).astype("Int64")
    d["phantom_transactions"] = (d["total_transactions"] - d["transactions_clean"]).astype("Int64")
    d["avg_ticket"] = d["avg_ticket"].fillna(
        d["amount_total"] / d["transactions_clean"].astype(float)
    )
    return d


def build_clean_panel(
    tx: pd.DataFrame, stores: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """Build the clean panel: full grid + reconstructed payments + attributes.

    ``total_transactions`` is kept as ``total_transactions_reported`` for traceability,
    but the count to use is ``transactions_clean``. ``replenishment_signal`` is kept
    only as a reference of the current system; ``configs/config.yaml`` bans it as a
    model feature.

    Parameters
    ----------
    tx, stores, calendar : pandas.DataFrame
        Output of ``io.load_raw``.

    Returns
    -------
    pandas.DataFrame
        One row per date x store x category, with store and calendar attributes.
    """
    panel = reconstruct_payments(complete_grid(tx, calendar))
    panel = panel.rename(columns={"total_transactions": "total_transactions_reported"})
    panel = panel.merge(stores, on="store_id", how="left", validate="many_to_one")
    panel = panel.merge(calendar, on="date", how="left", validate="many_to_one")
    panel["units_missing"] = panel["units_sold"].isna() & ~panel["is_missing_row"]
    return panel
