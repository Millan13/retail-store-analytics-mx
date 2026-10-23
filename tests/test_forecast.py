import numpy as np
import pandas as pd
import pytest

from retail_analytics import evaluate as ev
from retail_analytics import forecast as fc
from retail_analytics.features import (
    FEATURE_GROUPS,
    add_lag_features,
    build_features,
    feature_columns,
)

HORIZON = 7


@pytest.fixture
def panel() -> pd.DataFrame:
    """Two stores x two categories x 70 days, shuffled, with a few null days.

    Carries every column ``build_features`` needs, so tests exercise the real entry point.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=70)
    rows = []
    for store, fmt in [("S1", "Bodega"), ("S2", "Express")]:
        for cat in ["C1", "C2"]:
            for d in dates:
                rows.append(
                    {
                        "date": d,
                        "store_id": store,
                        "category": cat,
                        "units_sold": float(rng.integers(50, 150)),
                        "transactions_clean": int(rng.integers(20, 60)),
                        "holiday_name": "Feriado" if d.day == 5 else None,
                        "is_holiday": d.day == 5,
                        "day_of_week": d.dayofweek,
                        "month": d.month,
                        "is_payday": d.day == 15,
                        "is_weekend": d.dayofweek >= 5,
                        "is_buen_fin": False,
                        "is_navidad_season": False,
                        "is_semana_santa": False,
                        "has_promotion": int(rng.random() < 0.2),
                        "store_format": fmt,
                        "region": "Norte",
                        "socioeconomic_level": "C",
                        "size_sqm": 1000,
                        "num_checkouts": 5,
                        "has_pharmacy": False,
                        "has_fuel_station": False,
                    }
                )
    df = pd.DataFrame(rows)
    # POS failures: null sales on a few days, as in the real data.
    df.loc[rng.choice(len(df), 15, replace=False), "units_sold"] = np.nan
    return df.sample(frac=1, random_state=1).reset_index(drop=True)


def build(panel: pd.DataFrame) -> pd.DataFrame:
    return build_features(panel, HORIZON)


# ----------------------------------------------------------------------------- leakage
def test_features_do_not_change_when_the_future_changes(panel):
    """Core leakage test on the real entry point: perturbing every observed sales
    column after a cutoff must not alter any feature of rows up to cutoff + horizon
    (those rows are forecast at or before the cutoff)."""
    cutoff = pd.Timestamp("2023-02-10")
    base = build(panel)

    future = panel.copy()
    after = future["date"] > cutoff
    future.loc[after, "units_sold"] *= 10
    future.loc[after, "transactions_clean"] *= 10
    perturbed = build(future)

    feats = feature_columns()
    keys = ["store_id", "category", "date"]
    visible = base["date"] <= cutoff + pd.Timedelta(days=HORIZON)
    pd.testing.assert_frame_equal(
        base.loc[visible, keys + feats].sort_values(keys).reset_index(drop=True),
        perturbed.loc[perturbed["date"] <= cutoff + pd.Timedelta(days=HORIZON), keys + feats]
        .sort_values(keys)
        .reset_index(drop=True),
    )
    # And the perturbation does reach later rows (the test is not vacuous).
    later = base["date"] > cutoff + pd.Timedelta(days=HORIZON)
    assert not base.loc[later, "lag_7"].equals(perturbed.loc[later, "lag_7"])


def test_lags_point_to_the_right_day(panel):
    out = build(panel)
    s = out[(out["store_id"] == "S1") & (out["category"] == "C1")].set_index("date")
    t = pd.Timestamp("2023-02-20")
    assert s.loc[t, "lag_7"] == s.loc[t - pd.Timedelta(days=7), "units_sold"] or (
        np.isnan(s.loc[t, "lag_7"]) and np.isnan(s.loc[t - pd.Timedelta(days=7), "units_sold"])
    )
    window = s.loc[t - pd.Timedelta(days=13) : t - pd.Timedelta(days=7), "units_sold"]
    assert s.loc[t, "roll_mean_7"] == pytest.approx(window.mean())


def test_lag_shorter_than_horizon_is_rejected(panel):
    with pytest.raises(ValueError):
        add_lag_features(panel, "units_sold", horizon=8)


@pytest.mark.parametrize(
    "col",
    [
        "replenishment_signal",
        "total_transactions_reported",
        "phantom_transactions",
        "units_sold",
        "amount_total",
    ],
)
def test_banned_columns_cannot_be_features(monkeypatch, col):
    monkeypatch.setitem(FEATURE_GROUPS, "leaky", [col])
    with pytest.raises(ValueError, match=col):
        feature_columns(["lags", "leaky"])


def test_empty_group_list_means_no_features():
    assert feature_columns([]) == []


# ----------------------------------------------------------------------------- splits
def test_split_is_strictly_temporal(panel):
    df = build(panel)
    origin = pd.Timestamp("2023-02-19")
    train, test = fc.split_at_origin(df, origin, HORIZON, "units_sold", "2023-01-29")
    assert train["date"].max() <= origin < test["date"].min()
    assert test["date"].max() == origin + pd.Timedelta(days=HORIZON)
    assert test["date"].nunique() == HORIZON


def test_weekly_origins_requires_sundays():
    assert len(fc.weekly_origins("2023-08-27", "2023-10-22")) == 9
    with pytest.raises(ValueError):
        fc.weekly_origins("2023-08-28", "2023-10-22")


def test_moving_average_uses_only_history(panel):
    df = build(panel)
    origin = pd.Timestamp("2023-02-19")
    train, test = fc.split_at_origin(df, origin, HORIZON, "units_sold", "2023-01-29")
    pred = fc.moving_average(train, test, [])
    s1 = df[(df["store_id"] == "S1") & (df["category"] == "C1")]
    expected = s1[s1["date"].between(origin - pd.Timedelta(days=6), origin)]["units_sold"].mean()
    mask = (test["store_id"] == "S1").to_numpy() & (test["category"] == "C1").to_numpy()
    assert pred[mask] == pytest.approx(expected)


def test_seasonal_naive_uses_lag_7_with_fallback(panel):
    df = build(panel)
    origin = pd.Timestamp("2023-02-19")
    train, test = fc.split_at_origin(df, origin, HORIZON, "units_sold", "2023-01-29")
    pred = fc.seasonal_naive(train, test, [])
    has_lag = test["lag_7"].notna().to_numpy()
    np.testing.assert_array_equal(pred[has_lag], test["lag_7"].to_numpy()[has_lag])
    np.testing.assert_array_equal(pred[~has_lag], test["same_weekday_mean_4w"].to_numpy()[~has_lag])


def test_holt_winters_forecast_starts_after_origin_despite_gaps():
    """A series with no data on the last days before the origin must still be
    forecast on origin+1..origin+h with the right weekday pattern."""
    dates = pd.date_range("2023-01-02", periods=70)  # starts on a Monday
    weekly = np.array([100, 100, 100, 100, 100, 300, 300], dtype=float)
    train = pd.DataFrame(
        {
            "date": dates,
            "store_id": "S1",
            "category": "C1",
            "units_sold": np.tile(weekly, 10) * np.random.default_rng(0).normal(1, 0.05, 70),
        }
    )
    train.loc[train["date"] > dates[-3], "units_sold"] = np.nan  # last two days missing
    origin = dates[-1]
    test = pd.DataFrame(
        {
            "date": pd.date_range(origin + pd.Timedelta(days=1), periods=7),
            "store_id": "S1",
            "category": "C1",
        }
    )
    pred = fc.holt_winters(train, test, [], n_jobs=1)
    weekend = test["date"].dt.dayofweek.to_numpy() >= 5
    assert pred[weekend].min() > 2 * pred[~weekend].max()


# ----------------------------------------------------------------------------- metrics
def test_wape_and_bias_known_values():
    y = np.array([100.0, 200.0, np.nan])
    yhat = np.array([110.0, 170.0, 999.0])
    assert ev.wape(y, yhat) == pytest.approx(40 / 300)
    assert ev.bias(y, yhat) == pytest.approx(-20 / 300)


def test_inventory_impact_lost_and_excess():
    preds = pd.DataFrame(
        {
            "model": "m",
            "origin": pd.Timestamp("2023-01-01"),
            "store_id": ["S1", "S1", "S2", "S2"],
            "category": "C1",
            "date": pd.to_datetime(["2023-01-02", "2023-01-03"] * 2),
            "y_true": [10.0, 10.0, 10.0, 10.0],
            "y_pred": [5.0, 5.0, 15.0, 15.0],
        }
    )
    prices = preds[["store_id", "category", "date"]].assign(unit_price=2.0)
    out = ev.inventory_impact(preds, prices).iloc[0]
    # S1 ordered 10 for a demand of 20 -> 10 lost; S2 ordered 30 for 20 -> 10 excess.
    assert out["lost_units"] == 10 and out["excess_units"] == 10
    assert out["lost_mxn"] == 20 and out["excess_mxn"] == 20


def _preds(errors: dict[str, float], n_stores: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stores):
        y = rng.uniform(50, 150, 30)
        for m, e in errors.items():
            noise = rng.normal(0, e, 30) * y
            rows.append(
                pd.DataFrame({"model": m, "store_id": f"S{s}", "y_true": y, "y_pred": y + noise})
            )
    return pd.concat(rows, ignore_index=True)


def test_choose_model_prefers_simpler_model_on_a_tie():
    # Same error level -> statistical tie -> the simpler model (poisson_glm) wins.
    preds = _preds({"poisson_glm": 0.20, "lightgbm": 0.20})
    chosen, table = ev.choose_model(preds)
    assert chosen == "poisson_glm"
    assert table.loc["poisson_glm", "empate_con_mejor"]


def test_choose_model_keeps_clearly_better_complex_model():
    preds = _preds({"poisson_glm": 0.30, "lightgbm": 0.15})
    chosen, _ = ev.choose_model(preds)
    assert chosen == "lightgbm"
