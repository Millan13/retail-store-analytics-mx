"""Diagnósticos de calidad de datos usados en el EDA.

Cada función devuelve una tabla pequeña y legible en vez de imprimir, para que el
notebook solo muestre resultados y los tests puedan verificar los mismos números.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PANEL_KEYS = ["date", "store_id", "category"]


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo y porcentaje de nulos por columna, solo columnas con algún nulo.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Columnas ``nulos`` y ``pct``, ordenado de mayor a menor.
    """
    n = df.isna().sum()
    out = pd.DataFrame({"nulos": n, "pct": (n / len(df) * 100).round(2)})
    return out[out["nulos"] > 0].sort_values("nulos", ascending=False)


def null_patterns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Combinaciones de nulos que aparecen juntas (qué columnas fallan a la vez).

    Parameters
    ----------
    df : pandas.DataFrame
    cols : list of str
        Columnas con nulos a cruzar.

    Returns
    -------
    pandas.DataFrame
        Una fila por patrón, con ``x`` donde la columna es nula, y el número de filas.
    """
    pat = df[cols].isna()
    counts = pat.value_counts().rename("filas").reset_index()
    counts[cols] = counts[cols].replace({True: "x", False: ""})
    return counts


def missingness_tests(df: pd.DataFrame, col: str, by: list[str]) -> pd.DataFrame:
    """Prueba chi-cuadrada de independencia entre "es nulo" y cada variable de ``by``.

    Si los nulos fueran por fallas aleatorias de los POS (MCAR), la tasa de nulos
    no debería depender del día, la tienda o la promoción. Un p-valor alto es
    consistente con esa hipótesis (no la prueba, pero la falta de dependencia es
    lo que necesitamos para no sesgar al imputar o descartar).

    Parameters
    ----------
    df : pandas.DataFrame
    col : str
        Columna con nulos.
    by : list of str
        Variables categóricas contra las cuales probar.

    Returns
    -------
    pandas.DataFrame
        Estadístico, grados de libertad, p-valor y rango de la tasa de nulos por grupo.
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
    """Filas que faltan respecto a la malla completa fecha × tienda × categoría.

    Parameters
    ----------
    tx, stores, calendar : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Por tienda: primera y última fecha faltante, días y filas faltantes.
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
    """Verifica las identidades contables del archivo, separando días con y sin promoción.

    Identidades revisadas:

    - ``total_transactions = cash_transactions + card_transactions``
    - ``amount_total = amount_cash + amount_card``
    - ``avg_ticket = amount_total / total_transactions``
    - ``avg_ticket = amount_total / (cash_transactions + card_transactions)``

    Parameters
    ----------
    tx : pandas.DataFrame
        Transacciones crudas.
    tol : float
        Tolerancia absoluta (MXN) para comparar montos y tickets.

    Returns
    -------
    pandas.DataFrame
        % de filas evaluables donde se cumple cada identidad, con y sin promoción.
    """
    d = tx.copy()
    paid = d["cash_transactions"] + d["card_transactions"]
    checks = {
        "total_tx = efectivo + tarjeta": d["total_transactions"] == paid,
        "monto = efectivo + tarjeta": (
            d["amount_total"] - d["amount_cash"] - d["amount_card"]
        ).abs()
        < tol,
        "ticket = monto / total_tx": (
            d["avg_ticket"] - d["amount_total"] / d["total_transactions"]
        ).abs()
        < tol,
        "ticket = monto / (efectivo + tarjeta)": (d["avg_ticket"] - d["amount_total"] / paid).abs()
        < tol + 0.005,
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
    """Transacciones reportadas que no tienen medio de pago: ``total - (efectivo + tarjeta)``.

    Parameters
    ----------
    tx : pandas.DataFrame

    Returns
    -------
    pandas.Series
        Diferencia por fila (nula si falta el efectivo).
    """
    return tx["total_transactions"] - tx["cash_transactions"] - tx["card_transactions"]


def replenishment_window_check(
    tx: pd.DataFrame, window: int = 7, tol: float = 0.01
) -> pd.DataFrame:
    """Prueba qué ventana de ``total_transactions`` reproduce ``replenishment_signal``.

    Se comparan promedios móviles de ``window`` días que terminan en distintos días
    relativos a ``t``. Si la ventana que reproduce la señal termina después de ``t``,
    la señal usa información que no existía al momento de calcularla.

    Parameters
    ----------
    tx : pandas.DataFrame
        Transacciones crudas.
    window : int
        Días del promedio móvil.
    tol : float
        Tolerancia absoluta para considerar coincidencia exacta.

    Returns
    -------
    pandas.DataFrame
        Por ventana candidata: % de coincidencia exacta y error absoluto mediano.
    """
    d = tx.sort_values(["store_id", "category", "date"])
    g = d.groupby(["store_id", "category"])["total_transactions"]
    rows = []
    for end in (-1, 0, 1, 2):
        # end = días después de t en que termina la ventana (end=1 -> incluye t+1)
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
    """Valor dividido entre la mediana de su grupo (para ubicar picos y caídas).

    Parameters
    ----------
    values : pandas.Series
    groups : list of pandas.Series
        Llaves de agrupación alineadas con ``values``.

    Returns
    -------
    pandas.Series
    """
    return values / values.groupby(groups).transform("median")


def within_cell_lift(df: pd.DataFrame, value: str, flag: str, cell: list[str]) -> pd.Series:
    """Efecto relativo de una bandera comparando dentro de la misma celda.

    Para cada celda (por ejemplo tienda × categoría × mes) calcula
    ``media(valor | bandera=1) / media(valor | bandera=0) - 1``. Comparar dentro de
    la celda controla nivel de tienda, categoría y estacionalidad sin modelo.

    Parameters
    ----------
    df : pandas.DataFrame
    value : str
        Columna de resultado.
    flag : str
        Columna binaria (0/1).
    cell : list of str
        Columnas que definen la celda de comparación.

    Returns
    -------
    pandas.Series
        Un efecto por celda (solo celdas con ambos casos).
    """
    m = df.groupby([*cell, flag], observed=True)[value].mean().unstack(flag)
    return (m[1] / m[0] - 1).dropna()


def bootstrap_ci(
    x: pd.Series, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    """Media e intervalo de confianza bootstrap percentil.

    Parameters
    ----------
    x : pandas.Series
    n_boot : int
    alpha : float
    seed : int

    Returns
    -------
    tuple of float
        (media, límite inferior, límite superior).
    """
    rng = np.random.default_rng(seed)
    arr = x.dropna().to_numpy()
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)
