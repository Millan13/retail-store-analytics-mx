import numpy as np
import pandas as pd

from retail_analytics.quality import (
    accounting_identities,
    missing_summary,
    phantom_transactions,
    replenishment_window_check,
    within_cell_lift,
)


def test_missing_summary_only_reports_columns_with_nulls():
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": [1, 2, 3, 4]})
    out = missing_summary(df)
    assert list(out.index) == ["a"]
    assert out.loc["a", "pct"] == 25.0


def test_identities_detect_promo_inflation(tx_small):
    out = accounting_identities(tx_small)
    # El único día con promo y efectivo reportado no existe en el fixture -> se agrega uno.
    extra = tx_small.iloc[[0]].assign(
        date=pd.Timestamp("2023-01-05"),
        total_transactions=pd.array([120], dtype="Int64"),
        has_promotion=pd.array([1], dtype="Int64"),
    )
    out = accounting_identities(pd.concat([tx_small, extra], ignore_index=True))
    assert out.loc["total_tx = efectivo + tarjeta", "sin promo (%)"] == 100.0
    assert out.loc["total_tx = efectivo + tarjeta", "con promo (%)"] == 0.0


def test_phantom_transactions():
    df = pd.DataFrame(
        {
            "total_transactions": pd.array([10, 12], dtype="Int64"),
            "cash_transactions": pd.array([4, 5], dtype="Int64"),
            "card_transactions": pd.array([6, 5], dtype="Int64"),
        }
    )
    assert phantom_transactions(df).tolist() == [0, 2]


def test_replenishment_check_identifies_future_window():
    # Señal construida a propósito con una ventana que termina en t+1.
    dates = pd.date_range("2023-01-01", periods=30)
    rng = np.random.default_rng(0)
    tx = pd.DataFrame(
        {
            "date": dates,
            "store_id": "S1",
            "category": "C1",
            "total_transactions": rng.integers(50, 150, size=30),
        }
    )
    tx["replenishment_signal"] = (
        tx["total_transactions"].astype(float).shift(-1).rolling(7, min_periods=1).mean()
    )
    out = replenishment_window_check(tx).set_index("termina_en")
    assert out.loc[1, "coincidencia_exacta_%"] == 100.0
    assert out.loc[0, "coincidencia_exacta_%"] < 50


def test_within_cell_lift_controls_for_level():
    # Dos tiendas con niveles muy distintos y efecto real de +10% en ambas.
    df = pd.DataFrame(
        {
            "store": ["A"] * 4 + ["B"] * 4,
            "promo": [0, 0, 1, 1] * 2,
            "y": [100, 100, 110, 110, 1000, 1000, 1100, 1100],
        }
    )
    lift = within_cell_lift(df, "y", "promo", ["store"])
    assert np.allclose(lift.to_numpy(), 0.10)
