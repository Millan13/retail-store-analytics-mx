"""Access to the central configuration (configs/config.yaml)."""

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
    """Load the YAML configuration.

    Parameters
    ----------
    path : str or Path, optional
        Path to the configuration file. Defaults to ``configs/config.yaml``.

    Returns
    -------
    dict
        Configuration as a nested dictionary.
    """
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def raw_data_location() -> str:
    """Return where the raw data lives: a local folder or a cloud URI.

    The ``RAW_DATA_URI`` environment variable (e.g. ``gs://bucket/retail/raw``)
    takes precedence over ``paths.raw`` in the YAML, so the same code runs on a
    laptop and in a job that reads from GCS without changes.

    Returns
    -------
    str
        Absolute local path or remote URI, without a trailing slash.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    uri = os.getenv("RAW_DATA_URI")
    if uri:
        return uri.rstrip("/")
    return str(PROJECT_ROOT / load_config()["paths"]["raw"])


def project_path(key: str) -> Path:
    """Return the absolute path of a folder declared under ``paths`` in the YAML.

    Parameters
    ----------
    key : str
        Key inside ``paths`` (``processed``, ``figures``...).

    Returns
    -------
    Path
        Absolute folder; created if it does not exist.
    """
    path = PROJECT_ROOT / load_config()["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path
