"""Integridad de los datos reales: checksum + contratos crudos + limpieza completa."""

import pytest

from retail_analytics.cleaning import build_clean_panel
from retail_analytics.contracts import TRANSACTIONS_CLEAN, validate_raw
from retail_analytics.io import load_raw, verify_manifest

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_raw_files_match_manifest():
    assert all(verify_manifest().values())


def test_raw_contracts(raw):
    validate_raw(raw.transactions, raw.stores, raw.calendar)


def test_clean_panel_full_grid_and_contract(raw):
    panel = build_clean_panel(raw.transactions, raw.stores, raw.calendar)
    n_expected = raw.stores.shape[0] * raw.calendar.shape[0] * 6
    assert len(panel) == n_expected
    # La limpieza no altera ninguna fila real: solo agrega las faltantes.
    assert (~panel["is_missing_row"]).sum() == len(raw.transactions)
    assert panel["amount_total"].sum() == pytest.approx(raw.transactions["amount_total"].sum())
    TRANSACTIONS_CLEAN.validate(panel, lazy=True)
