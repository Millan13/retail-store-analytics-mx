"""El comando de predicción reproduce el backtest y genera bien el calendario futuro."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from retail_analytics.config import load_config, project_path
from retail_analytics.io import load_raw
from retail_analytics.predict import (
    calendar_for_dates,
    forecast_next_days,
    future_calendar,
    holidays,
)

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_calendar_rules_reproduce_the_real_calendar(raw):
    # Las reglas (fechas fijas, lunes de la LFT, Pascua, Buen Fin, quincenas) regeneran
    # las 425 fechas del archivo real, incluidos todos los festivos y eventos.
    cal = raw.calendar.reset_index(drop=True)
    seasons = cal.groupby("month")["season"].agg(lambda s: s.mode()[0])
    generated = calendar_for_dates(pd.DatetimeIndex(cal["date"]), seasons)[cal.columns]
    pd.testing.assert_frame_equal(generated, cal, check_dtype=False)


def test_future_calendar_starts_after_the_data(raw):
    future = future_calendar(raw.calendar, 7)
    assert future["date"].min() == raw.calendar["date"].max() + pd.Timedelta(days=1)
    assert len(future) == 7 and list(future.columns) == list(raw.calendar.columns)


def test_moved_holidays_fall_on_the_legal_monday():
    h = {name: date for date, name in holidays(2026).items()}
    assert h["Día de la Constitución"] == pd.Timestamp("2026-02-02")
    assert h["Natalicio Benito Juárez"] == pd.Timestamp("2026-03-16")
    assert h["Revolución Mexicana / Buen Fin"] == pd.Timestamp("2026-11-16")
    assert h["Viernes Santo"] == pd.Timestamp("2026-04-03")


def test_prediction_reproduces_the_backtest_week(raw):
    # Con los datos cortados al 18-feb-2024, la predicción debe coincidir con el backtest de
    # esa semana. Solo difieren los días con promoción: aquí se asume que no hay promoción.
    cfg = load_config()["forecast"]
    setup = json.loads((project_path("processed") / "forecast_setup.json").read_text("utf-8"))
    cut = pd.Timestamp("2024-02-18")
    truncated = replace(
        raw,
        transactions=raw.transactions[raw.transactions["date"] <= cut],
        calendar=raw.calendar[raw.calendar["date"] <= cut],
    )
    pred = forecast_next_days(truncated, setup, cfg)
    assert len(pred) == 80 * 6 * cfg["horizon"]
    assert pred["date"].min() == cut + pd.Timedelta(days=1)
    assert pred["units_forecast"].ge(0).all() and pred["units_forecast"].notna().all()

    keys = ["store_id", "category", "date"]
    backtest = pd.read_parquet(project_path("processed") / "test_predictions.parquet")
    backtest = backtest[(backtest["model"] == setup["ganador"]) & (backtest["origin"] == cut)]
    promo = raw.transactions[[*keys, "has_promotion"]]
    m = pred.merge(backtest[[*keys, "y_pred"]], on=keys).merge(promo, on=keys, how="left")
    no_promo = m[m["has_promotion"] == 0]
    assert len(no_promo) > 2000
    assert (no_promo["units_forecast"] - no_promo["y_pred"]).abs().max() < 1e-6
    assert m["units_forecast"].sum() == pytest.approx(m["y_pred"].sum(), rel=1e-3)
