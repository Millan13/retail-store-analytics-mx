"""Store potential: how much each store sells compared with comparable stores.

A store's *expected* sales come from a regression of log daily sales on the
attributes it cannot change in the short term (format, region, socioeconomic level
of the area and size). The ratio actual / expected is its *performance index*.
Raising under-performing stores to the median (P50) or the 75th percentile (P75)
of the index within their format gives two opportunity scenarios, the same
conservative / ambitious pair used for market-potential work.

Stores have no coordinates, so geography is shown at region level with an explicit,
documented state-to-region mapping (``REGION_STATES``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Assumed mapping of Mexican states to the five regions in stores.csv (the data does
# not include it). Used only to draw maps; every number is computed per store.
REGION_STATES: dict[str, list[str]] = {
    "Norte": [
        "Baja California",
        "Baja California Sur",
        "Sonora",
        "Chihuahua",
        "Coahuila",
        "Nuevo León",
        "Tamaulipas",
        "Sinaloa",
        "Durango",
    ],
    "Occidente": [
        "Jalisco",
        "Nayarit",
        "Colima",
        "Michoacán",
        "Aguascalientes",
        "Zacatecas",
        "Guanajuato",
    ],
    "Centro": ["Ciudad de México", "México", "Morelos", "Hidalgo", "Querétaro", "San Luis Potosí"],
    "Oriente": ["Veracruz", "Puebla", "Tlaxcala"],
    "Sur": ["Guerrero", "Oaxaca", "Chiapas", "Tabasco", "Campeche", "Yucatán", "Quintana Roo"],
}

FORMULA = (
    "np.log(daily_sales) ~ C(store_format, Treatment('Bodega')) + C(region, Treatment('Centro'))"
    " + C(socioeconomic_level, Treatment('C')) + np.log(size_sqm)"
)


def store_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the clean panel to one row per store.

    Parameters
    ----------
    panel : pandas.DataFrame
        Output of ``cleaning.build_clean_panel``.

    Returns
    -------
    pandas.DataFrame
        Store attributes plus average daily sales (all categories), average daily
        units, annualized sales and average daily transactions per checkout.
    """
    real = panel[~panel["is_missing_row"]]
    daily = real.groupby(["store_id", "date"], observed=True).agg(
        sales=("amount_total", "sum"), units=("units_sold", "sum"), tx=("transactions_clean", "sum")
    )
    per_store = daily.groupby("store_id").agg(
        daily_sales=("sales", "mean"), daily_units=("units", "mean"), daily_tx=("tx", "mean")
    )
    attrs = panel.drop_duplicates("store_id").set_index("store_id")[
        [
            "store_format",
            "region",
            "socioeconomic_level",
            "size_sqm",
            "num_checkouts",
            "opening_year",
        ]
    ]
    out = per_store.join(attrs).reset_index()
    out["annual_sales"] = out["daily_sales"] * 365
    out["tx_per_checkout"] = out["daily_tx"] / out["num_checkouts"]
    out["sales_per_sqm"] = out["annual_sales"] / out["size_sqm"]
    return out


def performance_index(stores: pd.DataFrame, formula: str = FORMULA) -> pd.DataFrame:
    """Fit the expected-sales model and add each store's performance index.

    Parameters
    ----------
    stores : pandas.DataFrame
        Output of ``store_table``.
    formula : str
        Patsy formula for log daily sales.

    Returns
    -------
    pandas.DataFrame
        Copy with ``expected_sales`` and ``index`` (actual / expected).
    """
    fit = smf.ols(formula, data=stores).fit()
    out = stores.copy()
    out["expected_sales"] = np.exp(fit.fittedvalues)
    out["index"] = out["daily_sales"] / out["expected_sales"]
    out.attrs["r2"] = fit.rsquared
    return out


def opportunity(stores: pd.DataFrame, peer_col: str = "store_format") -> pd.DataFrame:
    """Annual sales gap if each store reached the P50 / P75 index of its peers.

    Stores already above the target have no gap (the target never lowers sales).

    Parameters
    ----------
    stores : pandas.DataFrame
        Output of ``performance_index``.
    peer_col : str
        Column that defines the peer group.

    Returns
    -------
    pandas.DataFrame
        Copy with ``target_p50``, ``target_p75``, ``gap_p50`` and ``gap_p75`` (MXN/year).
    """
    out = stores.copy()
    by_peer = out.groupby(peer_col, observed=True)["index"]
    for q, name in [(0.5, "p50"), (0.75, "p75")]:
        out[f"target_{name}"] = by_peer.transform(lambda s, q=q: s.quantile(q))
        out[f"gap_{name}"] = (out[f"target_{name}"] / out["index"] - 1).clip(lower=0) * out[
            "annual_sales"
        ]
    return out


def quadrant(stores: pd.DataFrame) -> pd.Series:
    """Classify stores by commercial performance and checkout load versus their format.

    - **Crecer**: sells below peers with spare checkout capacity -> commercial action.
    - **Revisar operación**: sells below peers with saturated checkouts.
    - **Ampliar capacidad**: sells above peers with saturated checkouts.
    - **Mantener y replicar**: sells above peers with spare capacity.

    Both axes are split at the median of the store's format.

    Parameters
    ----------
    stores : pandas.DataFrame
        Output of ``performance_index`` (needs ``index`` and ``tx_per_checkout``).

    Returns
    -------
    pandas.Series
        Quadrant label per store.
    """
    g = stores.groupby("store_format", observed=True)
    high_perf = stores["index"] >= g["index"].transform("median")
    high_load = stores["tx_per_checkout"] >= g["tx_per_checkout"].transform("median")
    return pd.Series(
        np.select(
            [~high_perf & ~high_load, ~high_perf & high_load, high_perf & high_load],
            ["Crecer", "Revisar operación", "Ampliar capacidad"],
            default="Mantener y replicar",
        ),
        index=stores.index,
    )


def state_regions() -> pd.DataFrame:
    """Return the assumed state -> region mapping as a table."""
    return pd.DataFrame(
        [(s, r) for r, states in REGION_STATES.items() for s in states],
        columns=["estado", "region"],
    )
