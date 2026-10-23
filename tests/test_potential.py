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
