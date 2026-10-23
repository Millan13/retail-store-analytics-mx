"""Lectura de la configuración central (configs/config.yaml)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Carga el YAML de configuración.

    Parameters
    ----------
    path : str or Path, optional
        Ruta al archivo de configuración. Por defecto ``configs/config.yaml``.

    Returns
    -------
    dict
        Configuración como diccionario anidado.
    """
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def raw_data_location() -> str:
    """Devuelve dónde están los datos crudos: local o en la nube.

    La variable de entorno ``RAW_DATA_URI`` (por ejemplo ``gs://bucket/retail/raw``)
    tiene prioridad sobre ``paths.raw`` del YAML. Así el mismo código corre en la
    laptop y en un job que lee de GCS sin tocar nada.

    Returns
    -------
    str
        Ruta local absoluta o URI remota, sin diagonal final.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    uri = os.getenv("RAW_DATA_URI")
    if uri:
        return uri.rstrip("/")
    return str(PROJECT_ROOT / load_config()["paths"]["raw"])


def project_path(key: str) -> Path:
    """Ruta absoluta de una carpeta declarada en ``paths`` del YAML.

    Parameters
    ----------
    key : str
        Llave dentro de ``paths`` (``interim``, ``processed``, ``figures``...).

    Returns
    -------
    Path
        Carpeta absoluta; se crea si no existe.
    """
    path = PROJECT_ROOT / load_config()["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path
