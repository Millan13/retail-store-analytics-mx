"""Limpieza del panel de transacciones.

Principio: primero se reconstruye lo que es *deducible* (identidades contables
que se cumplen en el 100% de las filas comprobables) y solo después, y marcado,
lo que es *estimable*. ``units_sold`` no se imputa aquí: cualquier imputación que
aprenda de los datos (medianas, modelos) debe ajustarse dentro del pipeline de
entrenamiento con datos de train, o filtraría información del periodo de prueba.

Decisiones documentadas en ``notebooks/01_eda.ipynb`` (sección 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_analytics.quality import PANEL_KEYS


def complete_grid(tx: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Agrega las filas faltantes de la malla fecha × tienda × categoría.

    Las filas nuevas quedan con métricas nulas y ``is_missing_row=True``. Se
    marcan en vez de rellenarse porque no sabemos si la tienda abrió ese día;
    rellenar con cero inventaría una caída de demanda que no ocurrió.

    Parameters
    ----------
    tx : pandas.DataFrame
        Transacciones crudas.
    calendar : pandas.DataFrame
        Calendario con todas las fechas del periodo.

    Returns
    -------
    pandas.DataFrame
        Panel completo, ordenado por tienda, categoría y fecha.
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
    """Reconstruye efectivo, conteo real de transacciones y ticket.

    Orden de reconstrucción de ``cash_transactions`` (columna ``cash_source``):

    1. ``reportado``: venía en el archivo.
    2. ``identidad``: día sin promoción → ``total - tarjeta`` (identidad exacta en
       el 100% de filas sin promoción).
    3. ``ticket``: día con promoción y ticket disponible → ``monto/ticket - tarjeta``
       (exacta en el 100% de filas comprobables; ``total`` está inflado en promo).
    4. ``estimado``: promoción sin ticket → ``monto_efectivo / ticket_tarjeta``,
       supone ticket igual en efectivo y tarjeta (diferencia observada ±5%).

    ``amount_cash`` se deduce siempre como ``amount_total - amount_card``.

    Parameters
    ----------
    tx : pandas.DataFrame
        Panel con las columnas crudas.

    Returns
    -------
    pandas.DataFrame
        Copia con ``cash_transactions``, ``amount_cash`` y ``avg_ticket`` completos
        donde fue posible, más ``transactions_clean``, ``phantom_transactions`` y
        ``cash_source``.
    """
    d = tx.copy()
    d["amount_cash"] = d["amount_cash"].fillna(d["amount_total"] - d["amount_card"])

    source = pd.Series(
        np.where(d["cash_transactions"].notna(), "reportado", "sin_dato"), index=d.index
    )
    cash = d["cash_transactions"].astype("Float64")
    no_promo = d["has_promotion"] == 0
    promo = d["has_promotion"] == 1

    m = cash.isna() & no_promo
    cash[m] = d.loc[m, "total_transactions"] - d.loc[m, "card_transactions"]
    source[m] = "identidad"

    m = cash.isna() & promo & d["avg_ticket"].notna()
    cash[m] = (d.loc[m, "amount_total"] / d.loc[m, "avg_ticket"]).round() - d.loc[
        m, "card_transactions"
    ]
    source[m] = "ticket"

    m = cash.isna() & promo
    card_ticket = d.loc[m, "amount_card"] / d.loc[m, "card_transactions"]
    cash[m] = (d.loc[m, "amount_cash"] / card_ticket).round()
    source[m] = "estimado"

    # Las filas agregadas por complete_grid no tienen nada que reconstruir.
    source[d["total_transactions"].isna()] = "sin_dato"

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
    """Panel limpio listo para análisis: malla completa + pagos reconstruidos + atributos.

    ``total_transactions`` se conserva renombrada como ``total_transactions_reported``
    para trazabilidad, pero el conteo a usar es ``transactions_clean``.
    ``replenishment_signal`` se conserva solo como referencia del sistema actual;
    ``configs/config.yaml`` la declara prohibida como variable de modelo.

    Parameters
    ----------
    tx, stores, calendar : pandas.DataFrame
        Salida de ``io.load_raw``.

    Returns
    -------
    pandas.DataFrame
        Una fila por fecha × tienda × categoría con atributos de tienda y calendario.
    """
    panel = reconstruct_payments(complete_grid(tx, calendar))
    panel = panel.rename(columns={"total_transactions": "total_transactions_reported"})
    panel = panel.merge(stores, on="store_id", how="left", validate="many_to_one")
    panel = panel.merge(calendar, on="date", how="left", validate="many_to_one")
    panel["units_missing"] = panel["units_sold"].isna() & ~panel["is_missing_row"]
    return panel
