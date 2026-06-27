"""Shared Tufte-style matplotlib helpers for all study figures.

Principles applied: maximize data-ink ratio (no gridlines, no boxes, despined),
range-frames (spines span only the data), direct labeling over legends, a muted
palette with colour used only to carry information, light typography, and small
multiples with shared scales. Import and call ``use()`` at the top of a script.
"""
from __future__ import annotations
import matplotlib as mpl
import numpy as np

INK   = "#1a1a1a"   # near-black for data/text
MUTE  = "#8a8a8a"   # grey for reference / context
PALE  = "#d9d9d9"
ACCENT  = "#2166ac"  # one blue   (the result)
ACCENT2 = "#b2182b"  # one red    (null / warning)
GOOD    = "#2a7f4f"  # one green  (recovered signal)
PALETTE = [ACCENT, ACCENT2, GOOD, "#762a83", "#b8860b", MUTE]


def use():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": INK, "axes.linewidth": 0.7,
        "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "axes.titlepad": 7,
        "axes.labelsize": 10, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
        "text.color": INK,
        "legend.frameon": False, "legend.fontsize": 9, "legend.handlelength": 1.4,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
        "figure.dpi": 120, "savefig.dpi": 130,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def rangeframe(ax, x, y):
    """Limit the left/bottom spines to the data range (Tufte range-frame)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x): ax.spines["bottom"].set_bounds(x.min(), x.max())
    if len(y): ax.spines["left"].set_bounds(y.min(), y.max())


def label_end(ax, x, y, text, color, dx=0.0, dy=0.0, **kw):
    """Direct label at the end of a line/point instead of a legend entry."""
    ax.annotate(text, (x, y), xytext=(x + dx, y + dy), color=color,
                fontsize=9, va="center", **kw)


def minimal(ax):
    """Strip to the essentials: thin spines, few ticks."""
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.5, width=0.7)
    ax.locator_params(nbins=5)


# muted sequential colormap for 2-D density (light grey -> accent blue)
def density_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("tufte_density", ["white", "#cfd8e6", ACCENT, "#0b2c52"])
