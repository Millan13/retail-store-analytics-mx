"""Feature engineering for the 7-day demand forecast.

Design rule: every feature built from the target or from other observed sales uses
data from at least ``horizon`` days before the row's date. With that rule a single
model can forecast any of the next ``horizon`` days from a forecast origin, and no
feature can contain information that is unknown at the origin (no leakage).

Calendar and store attributes are known in advance, so they need no lag. The
promotion flag is treated as known in advance too (promotions are planned).
"""

from __future__ import annotations

import pandas as pd

from retail_analytics.config import load_config

SERIES_KEYS = ["store_id", "category"]
LAGS = (7, 14, 21, 28)
ROLLING_WINDOWS = (7, 28)

# Feature groups, used for group-wise selection (ablation) in the forecast notebook.
FEATURE_GROUPS: dict[str, list[str]] = {
    "calendar": [
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "is_payday",
        "is_holiday",
        "holiday",
        "is_buen_fin",
        "is_navidad_season",
        "is_semana_santa",
    ],
    "lags": [f"lag_{k}" for k in LAGS] + ["lag_7_was_holiday", "lag_7_was_payday"],
    "rolling": [
        "roll_mean_7",
        "roll_mean_28",
        "roll_median_28",
        "roll_std_28",
        "same_weekday_mean_4w",
    ],
    "store_traffic": ["store_tx_roll_7"],
    "store": [
        "store_format",
        "region",
        "socioeconomic_level",
        "size_sqm",
        "num_checkouts",
        "has_pharmacy",
        "has_fuel_station",
    ],
    "category": ["category_code"],
    "promotion": ["has_promotion"],
}
# Holiday families defined by what people do that day (business criterion), so that a
# holiday never seen in training can borrow the effect of others in its family.
# Experimental: not part of FEATURE_GROUPS, so the official model is unchanged.
HOLIDAY_FAMILIES = {
    "Año Nuevo": "descanso",
    "Día de la Constitución": "descanso",
    "Natalicio Benito Juárez": "descanso",
    "Jueves Santo": "descanso",
    "Viernes Santo": "descanso",
    "Día del Trabajo": "descanso",
    "Día de la Independencia": "celebracion",
    "Día de Muertos": "celebracion",
    "Virgen de Guadalupe": "celebracion",
    "Nochebuena": "celebracion",
    "Navidad": "celebracion",
    "Fin de Año": "celebracion",
    "Buen Fin": "evento_comercial",
    "Revolución Mexicana / Buen Fin": "evento_comercial",
}
EXPERIMENTAL_FEATURES = ["holiday_family"]

CATEGORICAL_FEATURES = [
    "holiday",
    "holiday_family",
    "store_format",
    "region",
    "socioeconomic_level",
    "category_code",
]


def _check_horizon(horizon: int) -> None:
    if min(LAGS) < horizon:
        raise ValueError(f"All lags must be >= horizon ({horizon}); got {LAGS}.")


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features known in advance for any future date.

    Parameters
    ----------
    panel : pandas.DataFrame
        Clean panel with the calendar columns merged in.

    Returns
    -------
    pandas.DataFrame
        Copy with ``day_of_month``, ``holiday`` (holiday name or ``"none"``) and
        ``holiday_family`` (``descanso``, ``celebracion``, ``evento_comercial`` or ``"none"``).
    """
    out = panel.copy()
    out["day_of_month"] = out["date"].dt.day
    out["holiday"] = out["holiday_name"].fillna("none").astype("category")
    family = out["holiday_name"].map(HOLIDAY_FAMILIES)
    unmapped = out.loc[out["holiday_name"].notna() & family.isna(), "holiday_name"].unique()
    if len(unmapped):
        raise ValueError(f"Holidays without a family: {list(unmapped)}")
    out["holiday_family"] = family.fillna("none").astype("category")
    return out


def add_lag_features(panel: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Add lags and rolling statistics of the target, all shifted by at least ``horizon``.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel with one row per date x store x category and no gaps in dates
        (``cleaning.complete_grid`` guarantees it).
    target : str
        Column to lag (``units_sold``).
    horizon : int
        Forecast horizon in days; the minimum shift applied to any feature.

    Returns
    -------
    pandas.DataFrame
        Copy sorted by series and date with the lag and rolling features.
    """
    _check_horizon(horizon)
    out = panel.sort_values([*SERIES_KEYS, "date"]).copy()
    by_series = out.groupby(SERIES_KEYS, observed=True)

    for k in LAGS:
        out[f"lag_{k}"] = by_series[target].shift(k)
    # Same weekday over the last 4 weeks: robust to a single atypical week.
    out["same_weekday_mean_4w"] = out[[f"lag_{k}" for k in LAGS]].mean(axis=1)

    # Rolling windows end at t - horizon, so they only see data known at the origin.
    shifted = by_series[target].shift(horizon)
    by_shifted = shifted.groupby([out[k] for k in SERIES_KEYS], observed=True)
    for w in ROLLING_WINDOWS:
        out[f"roll_mean_{w}"] = by_shifted.transform(
            lambda s, w=w: s.rolling(w, min_periods=w // 2).mean()
        )
    out["roll_median_28"] = by_shifted.transform(lambda s: s.rolling(28, min_periods=14).median())
    out["roll_std_28"] = by_shifted.transform(lambda s: s.rolling(28, min_periods=14).std())

    # Lag 7 falls on the same weekday, but it may have been a holiday or payday;
    # these flags let the model discount an atypical reference day.
    by_series = out.groupby(SERIES_KEYS, observed=True)
    out["lag_7_was_holiday"] = by_series["is_holiday"].shift(7).astype("float")
    out["lag_7_was_payday"] = by_series["is_payday"].shift(7).astype("float")
    return out


def add_store_traffic(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add the store's recent traffic (all categories), lagged by ``horizon``.

    Categories move together with store traffic (EDA, section 10), so the store
    total is a less noisy signal of the level than any single category.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel with ``transactions_clean``.
    horizon : int
        Forecast horizon in days.

    Returns
    -------
    pandas.DataFrame
        Copy with ``store_tx_roll_7``: mean daily store transactions over the
        7 days ending at t - horizon.
    """
    daily = (
        panel.groupby(["store_id", "date"], observed=True)["transactions_clean"]
        .sum(min_count=1)
        .sort_index()
    )
    traffic = (
        daily.groupby(level="store_id")
        .transform(lambda s: s.shift(horizon).rolling(7, min_periods=4).mean())
        .rename("store_tx_roll_7")
    )
    return panel.merge(traffic.reset_index(), on=["store_id", "date"], how="left")


def build_features(panel: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    """Build the modeling table from the clean panel.

    Parameters
    ----------
    panel : pandas.DataFrame
        Output of ``cleaning.build_clean_panel``.
    horizon : int, optional
        Forecast horizon. Defaults to ``forecast.horizon`` in the configuration.

    Returns
    -------
    pandas.DataFrame
        One row per date x store x category with the target, the keys and every
        column listed in ``FEATURE_GROUPS``.
    """
    cfg = load_config()["forecast"]
    horizon = horizon or cfg["horizon"]
    target = cfg["target"]

    df = add_calendar_features(panel)
    df = add_lag_features(df, target=target, horizon=horizon)
    df = add_store_traffic(df, horizon=horizon)
    df["category_code"] = df["category"].astype("category")
    for col in [
        "is_weekend",
        "is_payday",
        "is_holiday",
        "is_buen_fin",
        "is_navidad_season",
        "is_semana_santa",
        "has_pharmacy",
        "has_fuel_station",
    ]:
        df[col] = df[col].astype(int)
    df["has_promotion"] = df["has_promotion"].astype("float")
    return df.sort_values([*SERIES_KEYS, "date"]).reset_index(drop=True)


def feature_columns(groups: list[str] | None = None) -> list[str]:
    """Return the feature names for the given groups, rejecting banned columns.

    Parameters
    ----------
    groups : list of str, optional
        Keys of ``FEATURE_GROUPS``. Defaults to all groups.

    Returns
    -------
    list of str

    Raises
    ------
    ValueError
        If any feature is listed under ``leaky_columns`` in the configuration.
    """
    groups = list(FEATURE_GROUPS) if groups is None else groups
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    banned = set(load_config()["leaky_columns"]) & set(cols)
    if banned:
        raise ValueError(f"Banned (leaky) columns in the feature list: {sorted(banned)}")
    return cols
