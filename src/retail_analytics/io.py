"""Carga de los tres archivos crudos con tipos explícitos.

Los tipos se fijan aquí y no se dejan a la inferencia de pandas: ``cash_transactions``
llega como float por sus nulos, los booleanos del calendario llegan como texto, etc.
Tener un solo punto de entrada evita que cada notebook "arregle" los tipos a su manera.
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
    """Contenedor de los tres datasets crudos."""

    transactions: pd.DataFrame
    stores: pd.DataFrame
    calendar: pd.DataFrame


def _uri(name: str, location: str | None) -> str:
    base = location or raw_data_location()
    return f"{base}/{load_config()['files'][name]}"


def load_transactions(location: str | None = None) -> pd.DataFrame:
    """Lee ``transactions.csv``.

    Parameters
    ----------
    location : str, optional
        Carpeta local o URI (``gs://...``). Por defecto la de la configuración.

    Returns
    -------
    pandas.DataFrame
        Una fila por fecha × tienda × categoría. Los conteos con nulos quedan como
        ``Int64`` (entero que admite nulos) para no mezclar enteros con floats.
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
    """Lee ``stores.csv`` con categóricas ordenadas donde el orden tiene sentido.

    Parameters
    ----------
    location : str, optional
        Carpeta local o URI. Por defecto la de la configuración.

    Returns
    -------
    pandas.DataFrame
        Una fila por tienda.
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
    """Lee ``calendar.csv``.

    Parameters
    ----------
    location : str, optional
        Carpeta local o URI. Por defecto la de la configuración.

    Returns
    -------
    pandas.DataFrame
        Una fila por día del periodo.
    """
    df = pd.read_csv(_uri("calendar", location), parse_dates=["date"])
    for col in CALENDAR_BOOL_COLS:
        df[col] = df[col].astype(bool)
    return df


def load_raw(location: str | None = None) -> RawData:
    """Carga los tres datasets.

    Parameters
    ----------
    location : str, optional
        Carpeta local o URI. Por defecto la de la configuración.

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
    """Compara el SHA-256 de cada CSV contra ``MANIFEST.sha256``.

    Es el control de versión de los datos: si alguien reemplaza un CSV, los
    resultados ya no son comparables y este chequeo lo detecta.

    Parameters
    ----------
    location : str or Path, optional
        Carpeta con los CSV y el manifiesto.

    Returns
    -------
    dict
        ``{archivo: coincide}`` para cada archivo listado en el manifiesto.
    """
    base = str(location or raw_data_location())
    with fsspec.open(f"{base}/{load_config()['files']['manifest']}", "r") as fh:
        expected = dict(reversed(line.split()) for line in fh if line.strip())
    result = {}
    for fname, digest in expected.items():
        with fsspec.open(f"{base}/{fname}", "rb") as fh:
            result[fname] = hashlib.sha256(fh.read()).hexdigest() == digest
    return result
