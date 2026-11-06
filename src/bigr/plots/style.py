"""Publication style, colour mapping and figure export shared by the figure scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

PUB_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
}

# Colour mapping shared by the figures.
COLOR_MAP = "#e45756"        # red: guided MAP / selected particle
COLOR_PARTICLE = "#4c78a8"   # blue: particle cloud
COLOR_BASELINE = "#7c3aed"   # purple: deterministic TRM
COLOR_ORACLE = "#111111"     # black: oracle particle

# rcParams of the main-result figure (Figure 1).
MAIN_RESULT_RC: dict = {
    **PUB_RC,
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.6,
    "savefig.dpi": 300,
}


def save_figure(fig: "matplotlib.figure.Figure", path: Path, *, dpi: int = 150) -> None:
    """Save a figure as both .pdf and .png.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    path:
        Destination path; the suffix is replaced, so
        ``Path('assets/figures/fig_main_result.pdf')`` writes both
        ``fig_main_result.pdf`` and ``fig_main_result.png``.
    dpi:
        DPI of the PNG. The PDF is vector.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
