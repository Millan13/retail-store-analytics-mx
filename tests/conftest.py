import pandas as pd
import pytest


@pytest.fixture
def calendar_small() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2023-01-01", periods=4, freq="D")})


@pytest.fixture
def tx_small() -> pd.DataFrame:
    """Mini-panel con los casos que la limpieza debe resolver.

    Una tienda, una categoría, 4 días; el día 4 no existe en el archivo.
    - día 1: sin promo, todo reportado.
    - día 2: sin promo, efectivo nulo -> se deduce por identidad.
    - día 3: con promo (total inflado), efectivo nulo, ticket disponible -> por ticket.
    """
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "store_id": pd.array(["STR_001"] * 3, dtype="string"),
            "category": pd.array(["Bebidas"] * 3, dtype="string"),
            "total_transactions": pd.array([100, 120, 130], dtype="Int64"),
            "cash_transactions": pd.array([40, None, None], dtype="Int64"),
            "card_transactions": pd.array([60, 70, 50], dtype="Int64"),
            "amount_total": [10_000.0, 12_000.0, 11_000.0],
            "amount_cash": [4_000.0, None, None],
            "amount_card": [6_000.0, 7_000.0, 5_000.0],
            "units_sold": [200.0, None, 210.0],
            # día 3: transacciones reales = 110 (60 efectivo + 50 tarjeta), total reportado 130
            "avg_ticket": [100.0, None, 100.0],
            "has_promotion": pd.array([0, 0, 1], dtype="Int64"),
            "replenishment_signal": [None, None, 116.67],
        }
    )
