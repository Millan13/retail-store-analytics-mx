import numpy as np
import pandas as pd
import pytest

from retail_analytics import promotions as pm


@pytest.fixture
def experiment() -> pd.DataFrame:
    """Randomized promotions with a known +10% effect on sales.

    Store levels differ by 10x and weekdays have their own pattern, so a naive
    comparison would be noisy; the fixed-effects estimator must recover +10%.
    """
    rng = np.random.default_rng(42)
    rows = []
    for s in range(20):
        level = 100 * (1 + 9 * rng.random())
        for day in pd.date_range("2023-01-01", periods=120):
            promo = int(rng.random() < 0.2)
            weekday = 1.4 if day.dayofweek >= 5 else 1.0
            sales = level * weekday * (1.10 if promo else 1.0) * np.exp(rng.normal(0, 0.1))
            rows.append(
                {
                    "store_id": f"S{s}",
                    "category": "C1",
                    "date": day,
                    "ym": day.strftime("%Y-%m"),
                    "day_of_week": day.dayofweek,
                    "holiday_name": None,
                    "has_promotion": promo,
                    "log_amount_total": np.log(sales),
                }
            )
    return pd.DataFrame(rows)


def test_fixed_effects_recover_known_effect(experiment):
    r = pm.fixed_effects_effect(experiment, "log_amount_total")
    assert r["efecto"] == pytest.approx(0.10, abs=0.01)
    assert r["ic_inf"] < 0.10 < r["ic_sup"]


def test_no_effect_is_not_significant(experiment):
    d = experiment.copy()
    # Remove the effect: the estimate must be ~0 and not significant.
    d.loc[d["has_promotion"] == 1, "log_amount_total"] -= np.log(1.10)
    r = pm.fixed_effects_effect(d, "log_amount_total")
    assert abs(r["efecto"]) < 0.01
    assert r["p_valor"] > 0.05


def test_minimum_detectable_effect_formula():
    # z(0.975) + z(0.8) = 1.96 + 0.8416 = 2.8016
    assert pm.minimum_detectable_effect(0.01) == pytest.approx(np.expm1(0.028016), rel=1e-3)
