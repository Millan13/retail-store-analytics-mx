"""Loading of the three raw files with explicit dtypes.

Dtypes are fixed here instead of left to pandas inference: ``cash_transactions``
arrives as float because of its nulls, calendar booleans arrive as text, etc.
A single entry point keeps every notebook from "fixing" dtypes its own way.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import fsspec
import pandas as pd

from retail_analytics.config import load_config, raw_data_location

CALENDAR_BOOL_COLS = [
    "is_holiday",
    "is_payday",
    "is_weekend",
    "is_navidad_season",
    "is_buen_fin",
    "is_semana_santa",
]
STORE_BOOL_COLS = ["has_pharmacy", "has_fuel_station"]


@dataclass(frozen=True)
class RawData:
    """Container for the three raw datasets."""

    transactions: pd.DataFrame
    stores: pd.DataFrame
    calendar: pd.DataFrame


def _uri(name: str, location: str | None) -> str:
    base = location or raw_data_location()
    return f"{base}/{load_config()['files'][name]}"


def load_transactions(location: str | None = None) -> pd.DataFrame:
    """Read ``transactions.csv``.

    Parameters
    ----------
    location : str, optional
        Local folder or URI (``gs://...``). Defaults to the configured location.

    Returns
    -------
    pandas.DataFrame
        One row per date x store x category. Counts with nulls are ``Int64``
        (nullable integer) so integers are not mixed with floats.
    """
    df = pd.read_csv(
        _uri("transactions", location),
        parse_dates=["date"],
        dtype={"store_id": "string", "category": "string"},
    )
    for col in ["total_transactions", "cash_transactions", "card_transactions", "has_promotion"]:
        df[col] = df[col].astype("Int64")
    return df


def load_stores(location: str | None = None) -> pd.DataFrame:
    """Read ``stores.csv``, with ordered categoricals where the order is meaningful.

    Parameters
    ----------
    location : str, optional
        Local folder or URI. Defaults to the configured location.

    Returns
    -------
    pandas.DataFrame
        One row per store.
    """
    df = pd.read_csv(_uri("stores", location), dtype={"store_id": "string"})
    df["store_format"] = pd.Categorical(
        df["store_format"], categories=["Express", "Bodega", "Supercenter"], ordered=True
    )
    df["socioeconomic_level"] = pd.Categorical(
        df["socioeconomic_level"], categories=["C", "C+", "B", "A/B"], ordered=True
    )
    df["region"] = df["region"].astype("category")
    for col in STORE_BOOL_COLS:
        df[col] = df[col].astype(bool)
    return df


def load_calendar(location: str | None = None) -> pd.DataFrame:
    """Read ``calendar.csv``.

    Parameters
    ----------
    location : str, optional
        Local folder or URI. Defaults to the configured location.

    Returns
    -------
    pandas.DataFrame
        One row per day of the period.
    """
    df = pd.read_csv(_uri("calendar", location), parse_dates=["date"])
    for col in CALENDAR_BOOL_COLS:
        df[col] = df[col].astype(bool)
    return df


def load_raw(location: str | None = None) -> RawData:
    """Load the three datasets.

    Parameters
    ----------
    location : str, optional
        Local folder or URI. Defaults to the configured location.

    Returns
    -------
    RawData
    """
    return RawData(
        transactions=load_transactions(location),
        stores=load_stores(location),
        calendar=load_calendar(location),
    )


def verify_manifest(location: str | Path | None = None) -> dict[str, bool]:
    """Compare the SHA-256 of each CSV against ``MANIFEST.sha256``.

    This is the data-versioning check: if someone replaces a CSV, results are no
    longer comparable and this check catches it.

    Parameters
    ----------
    location : str or Path, optional
        Folder with the CSVs and the manifest.

    Returns
    -------
    dict
        ``{file_name: matches}`` for every file listed in the manifest.
    """
    base = str(location or raw_data_location())
    with fsspec.open(f"{base}/{load_config()['files']['manifest']}", "r") as fh:
        expected = dict(reversed(line.split()) for line in fh if line.strip())
    result = {}
    for fname, digest in expected.items():
        with fsspec.open(f"{base}/{fname}", "rb") as fh:
            result[fname] = hashlib.sha256(fh.read()).hexdigest() == digest
    return result
