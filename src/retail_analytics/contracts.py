"""Data contracts (pandera) for each stage of the pipeline.

A contract documents what we *expect* from a table and fails loudly when it does not
hold. Raw contracts describe the files as received (nulls included); the clean
contract describes what ``cleaning.build_clean_panel`` guarantees.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

CATEGORIES = ["Abarrotes", "Bebidas", "Cuidado_Personal", "Hogar", "Electronica", "Ropa"]
PERIOD_START, PERIOD_END = "2023-01-01", "2024-02-29"

_positive = Check.gt(0)

TRANSACTIONS_RAW = DataFrameSchema(
    {
        "date": Column("datetime64[ns]", Check.in_range(PERIOD_START, PERIOD_END)),
        "store_id": Column("string", Check.str_matches(r"^STR_\d{3}$")),
        "category": Column("string", Check.isin(CATEGORIES)),
        "total_transactions": Column("Int64", _positive),
        "cash_transactions": Column("Int64", _positive, nullable=True),
        "card_transactions": Column("Int64", _positive),
        "amount_total": Column(float, _positive),
        "amount_cash": Column(float, _positive, nullable=True),
        "amount_card": Column(float, _positive),
        "units_sold": Column(float, _positive, nullable=True),
        "avg_ticket": Column(float, _positive, nullable=True),
        "has_promotion": Column("Int64", Check.isin([0, 1])),
        "replenishment_signal": Column(float, _positive, nullable=True),
    },
    unique=["date", "store_id", "category"],
    strict=True,
    name="transactions_raw",
)

STORES_RAW = DataFrameSchema(
    {
        "store_id": Column("string", Check.str_matches(r"^STR_\d{3}$"), unique=True),
        "store_format": Column("category", Check.isin(["Express", "Bodega", "Supercenter"])),
        "region": Column(
            "category", Check.isin(["Norte", "Centro", "Sur", "Occidente", "Oriente"])
        ),
        "size_sqm": Column(int, _positive),
        "num_checkouts": Column(int, _positive),
        "opening_year": Column(int, Check.in_range(1950, 2024)),
        "socioeconomic_level": Column("category", Check.isin(["C", "C+", "B", "A/B"])),
        "has_pharmacy": Column(bool),
        "has_fuel_station": Column(bool),
    },
    strict=True,
    name="stores_raw",
)

CALENDAR_RAW = DataFrameSchema(
    {
        "date": Column("datetime64[ns]", unique=True),
        "day_of_week": Column(int, Check.in_range(0, 6)),
        "day_name": Column(str),
        "week_of_year": Column(int, Check.in_range(1, 53)),
        "month": Column(int, Check.in_range(1, 12)),
        "year": Column(int),
        "quarter": Column(int, Check.in_range(1, 4)),
        "season": Column(str, Check.isin(["Invierno", "Primavera", "Verano", "Otoño"])),
        "is_holiday": Column(bool),
        "holiday_name": Column(str, nullable=True),
        "is_payday": Column(bool),
        "is_weekend": Column(bool),
        "is_navidad_season": Column(bool),
        "is_buen_fin": Column(bool),
        "is_semana_santa": Column(bool),
    },
    checks=[
        # Flags derived from the date must agree with the date itself.
        Check(lambda d: (d["date"].dt.dayofweek == d["day_of_week"]).all(), name="dow_coherente"),
        Check(
            lambda d: ((d["day_of_week"] >= 5) == d["is_weekend"]).all(), name="weekend_coherente"
        ),
        Check(
            lambda d: (
                ((d["date"].dt.day == 15) | d["date"].dt.is_month_end) == d["is_payday"]
            ).all(),
            name="quincena_coherente",
        ),
    ],
    strict=True,
    name="calendar_raw",
)

# What cleaning guarantees: no gaps in the grid, accounting identities hold, and nulls
# only where we chose to keep them (missing rows and units_sold).
TRANSACTIONS_CLEAN = DataFrameSchema(
    {
        "date": Column("datetime64[ns]"),
        "store_id": Column("string"),
        "category": Column("string", Check.isin(CATEGORIES)),
        "is_missing_row": Column(bool),
        "transactions_clean": Column("Int64", _positive, nullable=True),
        "cash_transactions": Column("Int64", _positive, nullable=True),
        "card_transactions": Column("Int64", _positive, nullable=True),
        "amount_total": Column(float, _positive, nullable=True),
        "amount_cash": Column(float, _positive, nullable=True),
        "amount_card": Column(float, _positive, nullable=True),
        "avg_ticket": Column(float, _positive, nullable=True),
        "units_sold": Column(float, _positive, nullable=True),
        "has_promotion": Column("Int64", Check.isin([0, 1]), nullable=True),
        "cash_source": Column(
            str, Check.isin(["reported", "identity", "ticket", "estimated", "missing"])
        ),
    },
    checks=[
        Check(
            lambda d: (
                (d["transactions_clean"] == d["cash_transactions"] + d["card_transactions"])
                .dropna()
                .all()
            ),
            name="transactions_add_up",
        ),
        Check(
            lambda d: (
                (d["amount_total"] - d["amount_cash"] - d["amount_card"]).dropna().abs() < 0.05
            ).all(),
            name="amounts_add_up",
        ),
        # Only rows that did not exist in the file may lack transactions.
        Check(
            lambda d: d.loc[~d["is_missing_row"], "transactions_clean"].notna().all(),
            name="no_nulls_in_real_rows",
        ),
    ],
    unique=["date", "store_id", "category"],
    strict=False,
    name="transactions_clean",
)


def validate_raw(transactions, stores, calendar) -> None:
    """Validate the three raw datasets; raise ``SchemaErrors`` with every failure at once.

    Parameters
    ----------
    transactions, stores, calendar : pandas.DataFrame
        Output of ``io.load_raw``.
    """
    TRANSACTIONS_RAW.validate(transactions, lazy=True)
    STORES_RAW.validate(stores, lazy=True)
    CALENDAR_RAW.validate(calendar, lazy=True)


__all__ = [
    "CALENDAR_RAW",
    "CATEGORIES",
    "STORES_RAW",
    "TRANSACTIONS_CLEAN",
    "TRANSACTIONS_RAW",
    "pa",
    "validate_raw",
]
