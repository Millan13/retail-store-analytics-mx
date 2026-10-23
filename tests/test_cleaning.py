import pandas as pd

from retail_analytics.cleaning import complete_grid, reconstruct_payments
from retail_analytics.contracts import TRANSACTIONS_CLEAN


def test_complete_grid_adds_missing_day_flagged(tx_small, calendar_small):
    out = complete_grid(tx_small, calendar_small)
    assert len(out) == 4
    assert out["is_missing_row"].tolist() == [False, False, False, True]
    # La fila agregada no se rellena con ceros: queda nula.
    assert out.loc[3, ["total_transactions", "amount_total", "units_sold"]].isna().all()


def test_complete_grid_does_not_duplicate_or_change_existing(tx_small, calendar_small):
    out = complete_grid(tx_small, calendar_small)
    assert not out.duplicated(["date", "store_id", "category"]).any()
    pd.testing.assert_series_equal(
        out.loc[:2, "amount_total"].reset_index(drop=True),
        tx_small["amount_total"],
        check_names=False,
    )


def test_cash_reconstruction_by_identity_and_ticket(tx_small, calendar_small):
    out = reconstruct_payments(complete_grid(tx_small, calendar_small))
    # día 2: sin promo -> 120 - 70
    assert out.loc[1, "cash_transactions"] == 50
    assert out.loc[1, "cash_source"] == "identidad"
    # día 3: promo -> 11000 / 100 - 50 = 60 (NO 130 - 50 = 80, el total viene inflado)
    assert out.loc[2, "cash_transactions"] == 60
    assert out.loc[2, "cash_source"] == "ticket"
    assert out.loc[2, "phantom_transactions"] == 20
    assert out.loc[3, "cash_source"] == "sin_dato"


def test_amount_cash_and_ticket_filled_consistently(tx_small, calendar_small):
    out = reconstruct_payments(complete_grid(tx_small, calendar_small))
    assert out.loc[1, "amount_cash"] == 5_000.0
    assert abs(out.loc[1, "avg_ticket"] - 12_000 / 120) < 1e-9


def test_estimated_cash_when_promo_without_ticket(tx_small, calendar_small):
    tx = tx_small.copy()
    tx.loc[2, "avg_ticket"] = None
    out = reconstruct_payments(complete_grid(tx, calendar_small))
    # efectivo estimado = monto_efectivo / ticket_tarjeta = 6000 / (5000/50) = 60
    assert out.loc[2, "cash_transactions"] == 60
    assert out.loc[2, "cash_source"] == "estimado"


def test_units_sold_is_not_imputed(tx_small, calendar_small):
    # Imputar unidades aquí filtraría información entre train y test.
    out = reconstruct_payments(complete_grid(tx_small, calendar_small))
    assert pd.isna(out.loc[1, "units_sold"])


def test_clean_contract_passes(tx_small, calendar_small):
    out = reconstruct_payments(complete_grid(tx_small, calendar_small))
    TRANSACTIONS_CLEAN.validate(out, lazy=True)
