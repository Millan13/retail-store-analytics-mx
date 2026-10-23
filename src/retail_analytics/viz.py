"""Estilo único para todas las gráficas del proyecto.

Paleta categórica en orden fijo (validada para daltonismo en pares adyacentes);
los colores siguen a la entidad, nunca a su posición en un ranking.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = {"neg": "#e34948", "mid": "#f0efec", "pos": "#2a78d6"}
CRITICAL = "#d03b3b"

CATEGORY_ORDER = ["Abarrotes", "Bebidas", "Cuidado_Personal", "Hogar", "Electronica", "Ropa"]
FORMAT_ORDER = ["Express", "Bodega", "Supercenter"]
REGION_ORDER = ["Norte", "Occidente", "Centro", "Oriente", "Sur"]

CATEGORY_COLORS = dict(zip(CATEGORY_ORDER, SERIES, strict=False))
FORMAT_COLORS = dict(zip(FORMAT_ORDER, SERIES, strict=False))
REGION_COLORS = dict(zip(REGION_ORDER, SERIES, strict=False))


def set_style() -> None:
    """Aplica el tema de seaborn con la paleta del proyecto (fondo, tipografía y rejilla discretos)."""
    sns.set_theme(context="notebook", style="whitegrid", palette=SERIES, font_scale=0.95)
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "axes.prop_cycle": plt.cycler(color=SERIES),
        }
    )


def fmt_millions(ax, axis: str = "y", decimals: int = 0) -> None:
    """Formatea un eje en millones (``12 M``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : {"x", "y"}
    decimals : int
    """
    f = FuncFormatter(lambda v, _: f"{v / 1e6:,.{decimals}f} M")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def fmt_thousands(ax, axis: str = "y") -> None:
    """Formatea un eje con separador de miles.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : {"x", "y"}
    """
    f = FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def fmt_pct(ax, axis: str = "y", decimals: int = 0) -> None:
    """Formatea un eje como porcentaje a partir de fracciones (0.1 → 10%).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : {"x", "y"}
    decimals : int
    """
    f = FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def subtitle(ax, text: str) -> None:
    """Agrega un subtítulo gris bajo el título (la "lectura" de la gráfica).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    text : str
    """
    # Sube el título para dejar espacio a la línea gris debajo de él.
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=20)
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def save(fig, name: str) -> None:
    """Guarda la figura en ``reports/figures/<name>.png``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Nombre sin extensión.
    """
    from retail_analytics.config import project_path

    fig.savefig(project_path("figures") / f"{name}.png", bbox_inches="tight", dpi=150)
