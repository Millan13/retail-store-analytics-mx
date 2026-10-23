"""Single plotting style for every chart in the project.

Colors follow the retail visual kit: white surface, dark-blue text, True Blue as the
main data color and yellow only as an accent. Each store format keeps one color in
every chart. All series colors were checked with the dataviz palette validator
(lightness band, chroma floor, color-vision-deficiency and normal-vision separation);
the kit's two greens for Bodega and Express fail the normal-vision floor, so Bodega
uses a dark tint of its own yellow, which the kit allows as a functional adaptation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Surfaces and ink
SURFACE = "#FFFFFF"
INK = "#001E60"  # Bentonville: titles and primary text
INK_2 = "#62718A"  # secondary text, subtitles, tick labels
MUTED = "#B7C0CE"  # de-emphasized marks ("the rest")
GRID = "#D8E3F0"
AXIS = "#D4D4D4"

# Data colors
TRUE_BLUE = "#0053E2"  # main series / chosen model
NAVY = "#001E60"  # reference series (current system) next to True Blue
SPARK_YELLOW = "#FFC220"  # accent only (highlight bands, callouts), never a series
SERIES = [TRUE_BLUE, NAVY]
SEQ_BLUE = ["#E8EFFB", "#C3D4F5", "#8FB0EE", "#5A8BE8", "#0053E2", "#003A9E", "#001E60"]
DIVERGING = {"neg": "#D7263D", "mid": "#EEF1F6", "pos": TRUE_BLUE}

CATEGORY_ORDER = ["Abarrotes", "Bebidas", "Cuidado_Personal", "Hogar", "Electronica", "Ropa"]
FORMAT_ORDER = ["Express", "Bodega", "Supercenter"]
REGION_ORDER = ["Norte", "Occidente", "Centro", "Oriente", "Sur"]

# One color per format, everywhere (validated all-pairs: CVD 13.1, normal vision 26.3).
FORMAT_COLORS = {"Express": "#008B30", "Bodega": "#E0A100", "Supercenter": TRUE_BLUE}


def set_style() -> None:
    """Apply the seaborn theme with the project palette (quiet surface, type and grid)."""
    sns.set_theme(context="notebook", style="whitegrid", palette=SERIES, font_scale=0.95)
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
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
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "axes.prop_cycle": plt.cycler(color=SERIES),
        }
    )


def fmt_millions(ax, axis: str = "y", decimals: int = 0) -> None:
    """Format an axis in millions (``12 M``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : {"x", "y"}
    decimals : int
    """
    f = FuncFormatter(lambda v, _: f"{v / 1e6:,.{decimals}f} M")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def fmt_thousands(ax, axis: str = "y") -> None:
    """Format an axis with thousands separators.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : {"x", "y"}
    """
    f = FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def fmt_pct(ax, axis: str = "y", decimals: int = 0) -> None:
    """Format an axis as a percentage from fractions (0.1 -> 10%).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : {"x", "y"}
    decimals : int
    """
    f = FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def subtitle(ax, text: str) -> None:
    """Add a gray subtitle under the title (how to read the chart).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    text : str
    """
    # Push the title up to make room for the gray line below it.
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=20)
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def save(fig, name: str) -> None:
    """Save the figure to ``reports/figures/<name>.png``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        File name without extension.
    """
    from retail_analytics.config import project_path

    fig.savefig(project_path("figures") / f"{name}.png", bbox_inches="tight", dpi=150)
