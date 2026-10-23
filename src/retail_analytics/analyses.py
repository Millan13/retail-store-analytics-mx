"""Command-line runs for the promotion and store-potential analyses.

The forecast has its own pipeline (``retail_analytics.pipeline``). This module runs
the other two analyses end to end from the raw CSVs and writes their result tables
to ``data/processed/``, so neither depends on executing a notebook::

    python -m retail_analytics.analyses promotions
    python -m retail_analytics.analyses potential
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from retail_analytics import potential as pt
from retail_analytics import promotions as pm
from retail_analytics.cleaning import build_clean_panel
from retail_analytics.config import project_path
from retail_analytics.io import load_raw

log = logging.getLogger(__name__)

PROMO_OUTCOMES = {
    "log_amount_total": "monto_vendido",
    "log_units_sold": "unidades",
    "log_unit_price": "precio_por_unidad",
    "log_transactions_clean": "transacciones_reales",
    "log_avg_ticket": "ticket_promedio",
}
PROMO_SEGMENTS = ["store_format", "category", "region", "socioeconomic_level"]


def promotion_results(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate the promotion effect on each outcome and on sales amount by segment.

    Parameters
    ----------
    panel : pandas.DataFrame
        Output of ``cleaning.build_clean_panel``.

    Returns
    -------
    effects : pandas.DataFrame
        One row per outcome: effect, 95% CI, p-value, minimum detectable effect, n.
    segments : pandas.DataFrame
        Effect on sales amount per segment value, with Benjamini-Hochberg p-values.
    """
    d = pm.prepare(panel)
    effects = pd.DataFrame(
        {name: pm.fixed_effects_effect(d, col) for col, name in PROMO_OUTCOMES.items()}
    ).T
    effects["mde"] = effects["se"].map(pm.minimum_detectable_effect)
    effects = effects[["efecto", "ic_inf", "ic_sup", "p_valor", "mde", "n"]]
    segments = pm.heterogeneity(d, "log_amount_total", PROMO_SEGMENTS)
    return effects.rename_axis("metrica").reset_index(), segments


def potential_results(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute performance index, P50/P75 opportunity and matrix quadrant per store.

    Parameters
    ----------
    panel : pandas.DataFrame
        Output of ``cleaning.build_clean_panel``.

    Returns
    -------
    pandas.DataFrame
        One row per store, sorted by P50 opportunity (largest first).
    """
    stores = pt.opportunity(pt.performance_index(pt.store_table(panel)))
    stores["cuadrante"] = pt.quadrant(stores)
    cols = [
        "store_id",
        "store_format",
        "region",
        "socioeconomic_level",
        "annual_sales",
        "expected_sales",
        "daily_sales",
        "index",
        "gap_p50",
        "gap_p75",
        "cuadrante",
    ]
    return stores[cols].sort_values("gap_p50", ascending=False).reset_index(drop=True)


def main(argv: list[str]) -> None:
    """Run the analysis named in ``argv`` (``promotions`` or ``potential``)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if len(argv) != 1 or argv[0] not in {"promotions", "potential"}:
        sys.exit("Uso: python -m retail_analytics.analyses [promotions | potential]")

    raw = load_raw()
    panel = build_clean_panel(raw.transactions, raw.stores, raw.calendar)
    out = project_path("processed")

    if argv[0] == "promotions":
        effects, segments = promotion_results(panel)
        effects.to_csv(out / "promociones_efectos.csv", index=False)
        segments.to_csv(out / "promociones_segmentos.csv", index=False)
        amount = effects.set_index("metrica").loc["monto_vendido"]
        log.info(
            "Efecto de la promoción en monto vendido: %+.2f%% (IC 95%%: %+.2f%% a %+.2f%%)",
            100 * amount["efecto"],
            100 * amount["ic_inf"],
            100 * amount["ic_sup"],
        )
        log.info("Escrito: promociones_efectos.csv, promociones_segmentos.csv en %s", out)
    else:
        stores = potential_results(panel)
        stores.to_csv(out / "potencial_oportunidad.csv", index=False)
        log.info(
            "Oportunidad anual: P50 %s M MXN · P75 %s M MXN · %d tiendas >15%% abajo de su perfil",
            f"{stores['gap_p50'].sum() / 1e6:,.0f}",
            f"{stores['gap_p75'].sum() / 1e6:,.0f}",
            (stores["index"] < 0.85).sum(),
        )
        log.info("Escrito: potencial_oportunidad.csv en %s", out)


if __name__ == "__main__":
    main(sys.argv[1:])
