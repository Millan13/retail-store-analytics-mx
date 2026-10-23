"""Los comandos de promociones y potencial reproducen las cifras reportadas."""

import pytest

from retail_analytics.analyses import potential_results, promotion_results
from retail_analytics.cleaning import build_clean_panel
from retail_analytics.io import load_raw

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def panel():
    raw = load_raw()
    return build_clean_panel(raw.transactions, raw.stores, raw.calendar)


def test_promotion_effect_on_amount_is_near_zero(panel):
    effects, segments = promotion_results(panel)
    amount = effects.set_index("metrica").loc["monto_vendido"]
    assert amount["efecto"] == pytest.approx(-0.0022, abs=0.0005)
    assert amount["ic_inf"] < 0 < amount["ic_sup"] + 0.0005
    assert {"p_ajustado"} <= set(segments.columns)


def test_potential_matches_reported_opportunity(panel):
    stores = potential_results(panel)
    assert len(stores) == 80
    assert stores["gap_p50"].sum() / 1e6 == pytest.approx(1_707, abs=1)
    assert stores["gap_p75"].sum() / 1e6 == pytest.approx(4_420, abs=1)
    assert (stores["index"] < 0.85).sum() == 17
