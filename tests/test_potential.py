import pandas as pd
import pytest

from retail_analytics import potential as pt


@pytest.fixture
def stores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": ["A", "B", "C", "D"],
            "store_format": ["Bodega"] * 4,
            "index": [0.8, 1.0, 1.1, 1.3],
            "annual_sales": [100.0, 100.0, 100.0, 100.0],
            "tx_per_checkout": [50.0, 150.0, 160.0, 40.0],
        }
    )


def test_gap_is_zero_above_target_and_never_negative(stores):
    out = pt.opportunity(stores)
    # P50 of [0.8, 1.0, 1.1, 1.3] = 1.05 -> only A and B have a P50 gap.
    assert out.loc[out["store_id"].isin(["C", "D"]), "gap_p50"].eq(0).all()
    assert out["gap_p50"].ge(0).all() and out["gap_p75"].ge(0).all()
    assert out.loc[0, "gap_p50"] == pytest.approx((1.05 / 0.8 - 1) * 100)


def test_p75_gap_is_at_least_p50_gap(stores):
    out = pt.opportunity(stores)
    assert (out["gap_p75"] >= out["gap_p50"]).all()


def test_quadrants(stores):
    q = pt.quadrant(stores).tolist()
    assert q == ["Crecer", "Revisar operación", "Ampliar capacidad", "Mantener y replicar"]


def test_every_region_has_states_and_no_state_is_repeated():
    m = pt.state_regions()
    assert set(m["region"]) == {"Norte", "Occidente", "Centro", "Oriente", "Sur"}
    assert m["estado"].is_unique and len(m) == 32


def test_store_table_sql_aggregates_store_days_first():
    """Daily store sales must total the categories per day before averaging days."""
    panel = pd.DataFrame(
        {
            "store_id": ["S1"] * 4,
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]),
            "is_missing_row": [False, False, False, True],
            "amount_total": [100.0, 50.0, 90.0, None],
            "units_sold": [10.0, 5.0, 9.0, None],
            "transactions_clean": pd.array([4, 2, 3, None], dtype="Int64"),
            "store_format": "Bodega",
            "region": "Norte",
            "socioeconomic_level": "C",
            "size_sqm": 1000,
            "num_checkouts": 3,
            "opening_year": 2015,
        }
    )
    out = pt.store_table(panel).iloc[0]
    # Day 1 = 150, day 2 = 90 (missing row excluded) -> mean 120.
    assert out["daily_sales"] == pytest.approx(120.0)
    assert out["tx_per_checkout"] == pytest.approx(((4 + 2) + 3) / 2 / 3)
