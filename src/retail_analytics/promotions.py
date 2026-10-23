"""Promotion effectiveness, analysed as a randomized experiment.

Promotions cover ~20% of store-category-days with no visible link to weekday, format
or season (EDA, section 11). If assignment is as-good-as-random, comparing promotion
and non-promotion days *within the same store x category x month* identifies the
causal effect. The functions below (1) check that assumption, (2) estimate the effect
with fixed effects and store-clustered standard errors, (3) report the minimum
detectable effect and (4) look for heterogeneity with a multiple-testing correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

CELL = ["store_id", "category", "ym"]


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep observed rows and add the variables used in the analysis.

    Parameters
    ----------
    panel : pandas.DataFrame
        Output of ``cleaning.build_clean_panel``.

    Returns
    -------
    pandas.DataFrame
        Rows with sales and units, plus ``ym`` (month), ``unit_price``, logs of the
        outcomes and ``prev_week_units`` (same weekday one week earlier, for balance).
    """
    d = panel[~panel["is_missing_row"]].copy()
    d["ym"] = d["date"].dt.to_period("M").astype(str)
    d["unit_price"] = d["amount_total"] / d["units_sold"]
    for col in ["amount_total", "units_sold", "transactions_clean", "unit_price", "avg_ticket"]:
        d[f"log_{col}"] = np.log(d[col].astype(float))
    d = d.sort_values(["store_id", "category", "date"])
    d["prev_week_units"] = d.groupby(["store_id", "category"])["units_sold"].shift(7)
    d["log_prev_week_units"] = np.log(d["prev_week_units"])
    return d


def assignment_balance(d: pd.DataFrame) -> pd.DataFrame:
    """Check that promotions were not targeted (as-good-as-random assignment).

    Compares the promotion rate across groups (chi-square) and tests whether the
    previous week's sales differ between promotion and non-promotion days within the
    same cell (an A/A test: a promotion cannot change last week's sales, so any
    difference would mean promotions were assigned on purpose to weak or strong days).

    Parameters
    ----------
    d : pandas.DataFrame
        Output of ``prepare``.

    Returns
    -------
    pandas.DataFrame
        One row per check with its statistic and p-value.
    """
    rows = []
    for var in [
        "day_of_week",
        "store_format",
        "category",
        "region",
        "month",
        "is_payday",
        "is_holiday",
    ]:
        table = pd.crosstab(d[var], d["has_promotion"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rate = d.groupby(var, observed=True)["has_promotion"].mean()
        rows.append(
            {
                "prueba": f"Tasa de promoción por {var}",
                "estadístico": round(chi2, 2),
                "p_valor": round(p, 4),
                "detalle": f"tasa entre {rate.min():.1%} y {rate.max():.1%}",
            }
        )
    aa = fixed_effects_effect(d.dropna(subset=["log_prev_week_units"]), "log_prev_week_units")
    rows.append(
        {
            "prueba": "Venta de la semana previa (prueba A/A)",
            "estadístico": round(aa["t"], 2),
            "p_valor": round(aa["p_valor"], 4),
            "detalle": f"diferencia {aa['efecto']:+.2%}",
        }
    )
    return pd.DataFrame(rows)


def _demean(
    d: pd.DataFrame, cols: list[str], groups: list[list[str]], n_iter: int = 10
) -> pd.DataFrame:
    """Remove several sets of fixed effects by alternating projections."""
    out = d[cols].astype(float).copy()
    for _ in range(n_iter):
        for g in groups:
            out = out - out.groupby([d[k] for k in g], observed=True).transform("mean")
    return out


def fixed_effects_effect(d: pd.DataFrame, outcome: str) -> dict:
    """Estimate the promotion effect on a log outcome with fixed effects.

    Model: ``log(y) = beta * promotion + cell FE + weekday FE + holiday FE + error``,
    with cells = store x category x month. Fixed effects are absorbed by alternating
    demeaning (equivalent to including the dummies). Standard errors are clustered by
    store, because days of the same store share shocks.

    Parameters
    ----------
    d : pandas.DataFrame
        Output of ``prepare``.
    outcome : str
        Log outcome column (e.g. ``log_amount_total``).

    Returns
    -------
    dict
        ``efecto`` (exp(beta) - 1), its 95% CI, ``se`` of beta, ``t``, ``p_valor``
        and ``n``.
    """
    d = d.dropna(subset=[outcome]).copy()
    d["holiday_fe"] = d["holiday_name"].fillna("none")
    dm = _demean(d, [outcome, "has_promotion"], [CELL, ["day_of_week"], ["holiday_fe"]])
    fit = sm.OLS(dm[outcome], dm[["has_promotion"]]).fit(
        cov_type="cluster", cov_kwds={"groups": d["store_id"].astype("category").cat.codes}
    )
    beta, se = fit.params.iloc[0], fit.bse.iloc[0]
    return {
        "efecto": float(np.expm1(beta)),
        "ic_inf": float(np.expm1(beta - 1.96 * se)),
        "ic_sup": float(np.expm1(beta + 1.96 * se)),
        "beta": float(beta),
        "se": float(se),
        "t": float(beta / se),
        "p_valor": float(fit.pvalues.iloc[0]),
        "n": int(len(d)),
    }


def minimum_detectable_effect(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Return the smallest effect detectable with the given standard error.

    ``MDE = (z_{1-alpha/2} + z_{power}) * se`` on the log scale, returned as a
    percentage change.

    Parameters
    ----------
    se : float
        Standard error of the log-scale coefficient.
    alpha : float
        Two-sided significance level.
    power : float
        Desired statistical power.

    Returns
    -------
    float
    """
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(np.expm1(z * se))


def heterogeneity(d: pd.DataFrame, outcome: str, segments: list[str]) -> pd.DataFrame:
    """Estimate the effect separately by segment, with Benjamini-Hochberg correction.

    Testing many segments makes some look significant by chance; the adjusted p-value
    controls the false discovery rate across all segments tested.

    Parameters
    ----------
    d : pandas.DataFrame
        Output of ``prepare``.
    outcome : str
        Log outcome column.
    segments : list of str
        Columns to split by.

    Returns
    -------
    pandas.DataFrame
        Effect, 95% CI, raw and adjusted p-values per segment value.
    """
    rows = []
    for seg in segments:
        for value, g in d.groupby(seg, observed=True):
            r = fixed_effects_effect(g, outcome)
            rows.append({"segmento": seg, "valor": str(value), **r})
    out = pd.DataFrame(rows)
    out["p_ajustado"] = multipletests(out["p_valor"], method="fdr_bh")[1]
    return out


def business_impact(d: pd.DataFrame, effect_amount: float, effect_price: float) -> pd.Series:
    """Translate the estimated effects into MXN over the observed period.

    Parameters
    ----------
    d : pandas.DataFrame
        Output of ``prepare``.
    effect_amount : float
        Estimated relative effect of the promotion on sales (MXN).
    effect_price : float
        Estimated relative effect on unit price (negative = discount).

    Returns
    -------
    pandas.Series
        Sales on promotion days, incremental sales attributable to the promotion and
        the discount given (price reduction times units sold on promotion days).
    """
    promo = d[d["has_promotion"] == 1]
    sales = promo["amount_total"].sum()
    incremental = sales - sales / (1 + effect_amount)
    # Discount given: what the same units would have cost at the non-promotion price.
    discount = promo["amount_total"].sum() / (1 + effect_price) - promo["amount_total"].sum()
    return pd.Series(
        {
            "venta_en_dias_promo": sales,
            "venta_incremental": incremental,
            "descuento_otorgado": discount,
            "dias_tienda_categoria_con_promo": len(promo),
        }
    )
