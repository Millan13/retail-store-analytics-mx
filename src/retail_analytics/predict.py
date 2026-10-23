"""Forecast the next 7 days of units per store x category with the chosen model.

This is the production use of the forecast, as opposed to ``pipeline.py``, which
evaluates it on past weeks. It reuses the same feature code and the model and
hyperparameters chosen by the pipeline (``data/processed/forecast_setup.json``),
trains on all the available history and writes the forecast for the days after
the last date in the data::

    python -m retail_analytics.predict

Every feature uses data from at least 7 days before the forecast date, so the 7
future days only depend on data that is already known.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from retail_analytics import forecast as fc
from retail_analytics.cleaning import build_clean_panel
from retail_analytics.config import load_config, project_path
from retail_analytics.features import build_features
from retail_analytics.io import load_raw

log = logging.getLogger(__name__)

EVENT_FLAGS = ["is_holiday", "is_navidad_season", "is_buen_fin", "is_semana_santa"]

# Holidays on a fixed date (month, day).
FIXED_HOLIDAYS = {
    (1, 1): "Año Nuevo",
    (5, 1): "Día del Trabajo",
    (9, 16): "Día de la Independencia",
    (11, 2): "Día de Muertos",
    (12, 12): "Virgen de Guadalupe",
    (12, 24): "Nochebuena",
    (12, 25): "Navidad",
    (12, 31): "Fin de Año",
}
# Holidays moved to the n-th Monday of the month (Ley Federal del Trabajo, art. 74).
MONDAY_HOLIDAYS = {
    "Día de la Constitución": (2, 1),
    "Natalicio Benito Juárez": (3, 3),
    "Revolución Mexicana / Buen Fin": (11, 3),
}


def _nth_monday(year: int, month: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.Timedelta(days=(7 - first.dayofweek) % 7 + 7 * (n - 1))


def _easter(year: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    q = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * q + 114, 31)
    return pd.Timestamp(year=year, month=month, day=day + 1)


def holidays(year: int) -> dict[pd.Timestamp, str]:
    """Holidays and commercial events of a year, with the names used in the calendar.

    Parameters
    ----------
    year : int
        Calendar year.

    Returns
    -------
    dict
        Date -> holiday name.
    """
    out = {pd.Timestamp(year=year, month=m, day=d): name for (m, d), name in FIXED_HOLIDAYS.items()}
    for name, (month, n) in MONDAY_HOLIDAYS.items():
        out[_nth_monday(year, month, n)] = name
    # Buen Fin: the Friday to Sunday before the Revolution Monday (the Monday is shared).
    revolution = _nth_monday(year, 11, 3)
    for back in (3, 2, 1):
        out[revolution - pd.Timedelta(days=back)] = "Buen Fin"
    easter = _easter(year)
    out[easter - pd.Timedelta(days=3)] = "Jueves Santo"
    out[easter - pd.Timedelta(days=2)] = "Viernes Santo"
    return out


def calendar_for_dates(dates: pd.DatetimeIndex, season_by_month: pd.Series) -> pd.DataFrame:
    """Build calendar rows for ``dates`` with the conventions of ``calendar.csv``.

    Payday is the 15th and the last day of the month; Navidad season runs from
    December 15 to January 6; holidays follow ``holidays``.

    Parameters
    ----------
    dates : pandas.DatetimeIndex
        Dates to build.
    season_by_month : pandas.Series
        Season name indexed by month number.

    Returns
    -------
    pandas.DataFrame
        One row per date with the columns of ``calendar.csv``.
    """
    names = {}
    for year in sorted(set(dates.year)):
        names.update(holidays(year))
    out = pd.DataFrame({"date": dates})
    out["day_of_week"] = dates.dayofweek
    out["day_name"] = dates.day_name()
    out["week_of_year"] = dates.isocalendar().week.to_numpy().astype(int)
    out["month"] = dates.month
    out["year"] = dates.year
    out["quarter"] = dates.quarter
    out["season"] = out["month"].map(season_by_month)
    out["holiday_name"] = out["date"].map(names)
    out["is_holiday"] = out["holiday_name"].notna()
    out["is_payday"] = (dates.day == 15) | dates.is_month_end
    out["is_weekend"] = dates.dayofweek >= 5
    out["is_navidad_season"] = ((dates.month == 12) & (dates.day >= 15)) | (
        (dates.month == 1) & (dates.day <= 6)
    )
    out["is_buen_fin"] = out["holiday_name"].isin(["Buen Fin", "Revolución Mexicana / Buen Fin"])
    out["is_semana_santa"] = out["holiday_name"].isin(["Jueves Santo", "Viernes Santo"])
    return out


def future_calendar(calendar: pd.DataFrame, days: int) -> pd.DataFrame:
    """Extend the calendar ``days`` days after its last date.

    Parameters
    ----------
    calendar : pandas.DataFrame
        Calendar as returned by ``io.load_raw``.
    days : int
        Number of days to add.

    Returns
    -------
    pandas.DataFrame
        The new rows only, with the same columns as ``calendar``.
    """
    dates = pd.date_range(calendar["date"].max() + pd.Timedelta(days=1), periods=days, freq="D")
    season_by_month = calendar.groupby("month")["season"].agg(lambda s: s.mode()[0])
    return calendar_for_dates(dates, season_by_month)[calendar.columns]


def forecast_next_days(raw, setup: dict, cfg: dict) -> pd.DataFrame:
    """Train the chosen model on all the history and forecast the next ``horizon`` days.

    Parameters
    ----------
    raw : RawData
        Output of ``io.load_raw``.
    setup : dict
        Content of ``forecast_setup.json`` (``features``, ``best_params``, ``ganador``).
    cfg : dict
        ``forecast`` section of the configuration.

    Returns
    -------
    pandas.DataFrame
        ``store_id``, ``category``, ``date`` and ``units_forecast``, one row per
        series and future day.
    """
    horizon = cfg["horizon"]
    origin = raw.calendar["date"].max()
    calendar = pd.concat([raw.calendar, future_calendar(raw.calendar, horizon)], ignore_index=True)
    panel = build_clean_panel(raw.transactions, raw.stores, calendar)
    df = build_features(panel, horizon)
    # No promotion is assumed for the future days (the plan is not in the data; the
    # feature does not contribute to the forecast, see the feature selection).
    df.loc[df["date"] > origin, "has_promotion"] = 0.0

    train, future = fc.split_at_origin(df, origin, horizon, cfg["target"], cfg["train_start"])
    winner = setup["ganador"]
    model = getattr(fc, winner)
    pred = model(train, future, setup["features"], **setup["best_params"].get(winner, {}))
    out = future[["store_id", "category", "date"]].copy()
    out["units_forecast"] = np.clip(pred, 0, None)
    return out.sort_values(["store_id", "category", "date"]).reset_index(drop=True)


def main() -> None:
    """Write ``data/processed/pronostico_proxima_semana.csv``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()["forecast"]
    out_dir = project_path("processed")
    setup = json.loads((out_dir / "forecast_setup.json").read_text(encoding="utf-8"))

    raw = load_raw()
    result = forecast_next_days(raw, setup, cfg)
    result.to_csv(out_dir / "pronostico_proxima_semana.csv", index=False)

    last_week = raw.transactions[
        raw.transactions["date"] > raw.transactions["date"].max() - pd.Timedelta(days=7)
    ]["units_sold"].sum()
    log.info(
        "Pronóstico %s a %s con %s: %s filas, %s unidades (última semana real: %s)",
        result["date"].min().date(),
        result["date"].max().date(),
        setup["ganador"],
        f"{len(result):,}",
        f"{result['units_forecast'].sum():,.0f}",
        f"{last_week:,.0f}",
    )
    log.info("Escrito: pronostico_proxima_semana.csv en %s", out_dir)


if __name__ == "__main__":
    main()
