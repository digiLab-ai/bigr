"""fig_token_entropy — Figure 4: token-entropy heatmaps and calibration (Sudoku-Extreme).

Three panels: a solved and an unsolved test-set puzzle coloured by the
trajectory-averaged token entropy, and a fold-aggregated calibration curve
(token accuracy vs final-step and trajectory-averaged entropy) for the test
and DF splits.

Reads  <test-dir>/fold_*/entropy_data.npz and <df-dir>/fold_*/entropy_data.npz, written
       by scripts/exps/run_entropy.py (bash scripts/run_experiments.sh --token-entropy).
Writes <out>/fig_token_entropy.{pdf,png}; --out defaults to assets/figures.

python scripts/figs/fig_token_entropy.py --test-dir assets/runs/sudoku/entropy_test_k5 \\
    --df-dir assets/runs/sudoku/entropy_impossible_k5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from bigr.plots.kfold import ci_bounds, fold_dirs, is_kfold_run

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while not (REPO_ROOT / "src").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DIGIT_OFFSET = 2  # token 3 → digit 1, …, token 11 → digit 9

RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.4,
    "hatch.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}

CELL_CMAP = plt.cm.YlOrRd
OUT_NAME = "fig_token_entropy"
# Index (within the first test fold) of the solved puzzle shown in the paper's
# Figure 4. When it is in range it overrides the automatic selection.
PREFERRED_SOLVED_EXAMPLE_IDX = 1810


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="fig_token_entropy — token entropy heatmap + calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="K-fold run dir with fold_*/entropy_data.npz for the test split.",
    )
    p.add_argument(
        "--df-dir",
        type=Path,
        default=None,
        help="K-fold run dir with fold_*/entropy_data.npz for the DF split.",
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "figures")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class SplitData:
    """Entropy npz data for one split with derived per-cell and per-puzzle stats."""

    def __init__(self, run_dir: Path, label: str) -> None:
        self.label = label
        cache = np.load(run_dir / "entropy_data.npz")
        self.inputs = cache["inputs"]  # (N, 81)
        self.labels = cache["labels"]  # (N, 81)
        self.preds = cache["preds"]  # (N, S, 81)
        self.weights = cache["weights"]  # (N, S)
        # Per-cell trajectory-averaged normalised entropy, computed during inference.
        self._path_uncert: np.ndarray | None = (
            cache["path_auc_uncert"].astype(np.float32)
            if "path_auc_uncert" in cache
            else None
        )
        N, S, L = self.preds.shape
        self.N, self.S, self.L = N, S, L
        self.task = "sudoku" if int(L) == 81 else "maze"
        n_classes = int(self.preds.max() + 1)

        w = self.weights.astype(np.float32)
        w /= np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)

        one_hot = (
            self.preds[:, :, :, None]
            == np.arange(n_classes, dtype=self.preds.dtype)[None, None, None, :]
        ).astype(np.float32)
        probs = (w[:, :, None, None] * one_hot).sum(axis=1)
        probs /= np.clip(probs.sum(axis=-1, keepdims=True), 1e-12, None)
        self.probs = probs  # (N, 81, C)

        if self.task == "sudoku":
            # Sudoku convention: token 2 is "empty to be predicted".
            self.mask = self.inputs < 3
        else:
            # Maze convention: only SPACE cells (token 2) are predicted.
            self.mask = self.inputs == 2
        self.map_pred = probs.argmax(axis=-1)  # (N, 81)
        tok_correct = self.map_pred == self.labels
        self.puzzle_exact = (tok_correct | ~self.mask).all(axis=1)  # (N,) bool

        # Final-step normalised entropy $H_\ell$ of the weighted terminal token marginal.
        safe = np.clip(probs, 1e-12, 1.0)
        log_C = np.log(float(n_classes))
        self.ent_final = -(safe * np.log(safe)).sum(axis=-1) / log_C  # (N, 81)

        # Trajectory-averaged normalised entropy $\bar H_\ell$, stored by run_entropy
        # as `path_auc_uncert`; falls back to the final-step entropy when absent.
        self.ent_path: np.ndarray = (
            self._path_uncert if self._path_uncert is not None else self.ent_final
        )

        # Heatmaps colour cells by the trajectory-averaged entropy so they match
        # the $\bar H_\ell$ calibration curve; the final-step entropy stays
        # available as `ent_final` for the primary calibration curve and AUROC.
        self.ent_display: np.ndarray = self.ent_path

        # Per-puzzle mean display uncertainty (over non-given cells); drives the
        # automatic example-puzzle selection.
        self.mean_uncert = np.where(self.mask, self.ent_display, 0.0).sum(
            axis=1
        ) / np.maximum(self.mask.sum(axis=1), 1)

    def cal_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Flat (final-step normalised token entropy, correct) arrays over non-given tokens."""
        u = self.ent_final[self.mask]
        c = (self.map_pred == self.labels)[self.mask].astype(np.float32)
        return u, c

    def ent_path_cal_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Flat (trajectory-averaged token entropy, correct) arrays over non-given tokens."""
        u = self.ent_path[self.mask]
        c = (self.map_pred == self.labels)[self.mask].astype(np.float32)
        return u, c


def _select_puzzles(data: SplitData) -> tuple[int, int]:
    """Return (solved_idx, unsolved_idx).

    Unsolved: prefer puzzles with 0 live-particle solves (weight > 1/(2S)).
    Solved: highest mean uncertainty strictly below unsolved's H * 0.9.
    """
    S = data.preds.shape[1]
    w_thresh = 1.0 / (2 * S)
    live = data.weights > w_thresh  # (N, S)
    ok_all = ((data.preds == data.labels[:, None, :]) | ~data.mask[:, None, :]).all(
        axis=2
    )  # (N, S)
    live_ok = (ok_all & live).sum(axis=1)  # (N,)

    unsolved_all = np.where(~data.puzzle_exact)[0]
    if len(unsolved_all) == 0:
        ui = 1
    else:
        zero_mask = live_ok[unsolved_all] == 0
        cands_u = unsolved_all[zero_mask] if zero_mask.any() else unsolved_all
        ui = int(cands_u[np.argmax(data.mean_uncert[cands_u])])

    solved_all = np.where(data.puzzle_exact)[0]
    if len(solved_all) == 0:
        si = 0
    else:
        h_cap = data.mean_uncert[ui] * 0.9
        capped = solved_all[data.mean_uncert[solved_all] < h_cap]
        pool = capped if len(capped) > 0 else solved_all
        si = int(pool[np.argmax(data.mean_uncert[pool])])

    return si, ui


def _preferred_solved_index(data: SplitData, fallback_idx: int) -> int:
    """Use the paper's solved example when its index is in range, else the fallback."""
    if 0 <= PREFERRED_SOLVED_EXAMPLE_IDX < data.N:
        return PREFERRED_SOLVED_EXAMPLE_IDX
    return fallback_idx


def _example_title(status: str, solved_particles: int, total_particles: int) -> str:
    """Consistent panel titles for solved/unsolved example plots.

    ``status`` is either "Correct" or "Incorrect" — the guided-MAP outcome on
    this puzzle. The second line reports how many of the ``S`` particles
    independently solved the puzzle.
    """
    return (
        rf"$\mathbf{{{status}}}$"
        "\n"
        rf"({solved_particles}/{total_particles} particles solved)"
    )


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _binned_calibration_fixed(
    uncertainty: np.ndarray,
    correct: np.ndarray,
    edges: np.ndarray,
) -> pd.DataFrame:
    x = np.asarray(uncertainty, dtype=float)
    y = np.asarray(correct, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    bins = np.digitize(x, edges[1:-1], right=True)
    rows = []
    for b in range(len(edges) - 1):
        mask = bins == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "x": float(x[mask].mean()),
                "y": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def _auroc(scores: np.ndarray, labels01: np.ndarray) -> float:
    """AUROC via Mann-Whitney U (high score → positive label)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels01, dtype=np.int8)
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sx = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sx[j] == sx[i]:
            j += 1
        r = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = r
        i = j
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _summarise_fold_calibration(
    fold_arrays: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_bins: int,
) -> dict:
    """Fold-aggregate a token-level calibration curve and its AUROC.

    Bin edges are entropy quantiles pooled over folds; each fold is binned on
    the shared edges and the per-bin mean accuracy is averaged across folds
    with a Student-t confidence interval.
    """
    all_uncert = np.concatenate([arrays[0] for arrays in fold_arrays])
    edges = np.quantile(all_uncert, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(
            float(all_uncert.min()), float(all_uncert.max()) + 1e-9, n_bins + 1
        )

    bin_rows: list[dict] = []
    aurocs: list[float] = []
    for fold_index, (u, c) in enumerate(fold_arrays, start=1):
        cal_df = _binned_calibration_fixed(u, c, edges)
        for _, row in cal_df.iterrows():
            bin_rows.append(
                {
                    "fold": fold_index,
                    "bin": int(row["bin"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
            )
        aurocs.append(_auroc(u, (1.0 - c).astype(np.int8)))

    cal_summary = (
        pd.DataFrame(bin_rows)
        .groupby("bin")[["x", "y"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    cal_summary.columns = [
        f"{base}_{stat}" if stat else str(base) for base, stat in cal_summary.columns
    ]
    y_half, y_lo, y_hi = ci_bounds(
        cal_summary["y_mean"].to_numpy(),
        cal_summary["y_std"].fillna(0.0).to_numpy(),
        len(fold_arrays),
    )
    cal_summary["x"] = cal_summary["x_mean"]
    cal_summary["y"] = cal_summary["y_mean"]
    cal_summary["y_ci"] = y_half
    cal_summary["y_lo"] = y_lo
    cal_summary["y_hi"] = y_hi

    auroc_arr = np.asarray(aurocs, dtype=float)
    auroc_half, _, _ = ci_bounds(
        auroc_arr.mean(), auroc_arr.std(ddof=1), len(fold_arrays)
    )
    return {
        "calibration": cal_summary,
        "auroc_mean": float(auroc_arr.mean()),
        "auroc_ci": float(auroc_half),
    }


def summarise_entropy_folds(run_dir: Path, *, label: str, n_bins: int = 20) -> dict:
    """Fold-aggregated calibration for one split: final-step (primary) and trajectory-averaged."""
    folds = [SplitData(fold_dir, label) for fold_dir in fold_dirs(run_dir)]
    primary = _summarise_fold_calibration(
        [fold.cal_arrays() for fold in folds], n_bins=n_bins
    )
    ent_path = _summarise_fold_calibration(
        [fold.ent_path_cal_arrays() for fold in folds], n_bins=n_bins
    )
    n_solved = np.array([int(fold.puzzle_exact.sum()) for fold in folds], dtype=float)
    solved_half, _, _ = ci_bounds(n_solved.mean(), n_solved.std(ddof=1), len(folds))
    return {
        "label": label,
        "calibration": primary["calibration"],
        "auroc_mean": primary["auroc_mean"],
        "auroc_ci": primary["auroc_ci"],
        "ent_path": ent_path,
        "solved_per_fold_mean": float(n_solved.mean()),
        "solved_per_fold_ci": float(solved_half),
    }


# ---------------------------------------------------------------------------
# Panel drawers
# ---------------------------------------------------------------------------


def _draw_sudoku_entropy(
    ax: plt.Axes,
    data: SplitData,
    idx: int,
    *,
    vmax: float,
    title: str = "",
    show_givens: bool = True,
    mark_incorrect: bool = False,
) -> None:
    """9×9 entropy heatmap with digit labels.

    Given clues: white background, dark digit.
    Predicted cells: entropy-coloured, dark digit.
    Optional hash marks incorrect non-given tiles.
    """
    inp = data.inputs[idx].reshape(9, 9)
    lb = data.labels[idx].reshape(9, 9)
    en = data.ent_display[idx].reshape(9, 9)
    pr = data.map_pred[idx].reshape(9, 9)
    is_given = inp >= 3

    # ── background colours ──────────────────────────────────────────────────
    bg = np.ones((9, 9, 3))
    for i in range(9):
        for j in range(9):
            if is_given[i, j]:
                bg[i, j] = [1.0, 1.0, 1.0] if show_givens else [0.0, 0.0, 0.0]
            else:
                t = float(np.clip(en[i, j] / vmax, 0.0, 1.0))
                bg[i, j] = CELL_CMAP(t)[:3]

    ax.imshow(
        bg, extent=(0, 9, 0, 9), origin="upper", aspect="equal", interpolation="nearest"
    )

    # ── Hashing for wrong predictions ────────────────────────────────────────
    if mark_incorrect:
        for i in range(9):
            for j in range(9):
                if not is_given[i, j] and int(pr[i, j]) != int(lb[i, j]):
                    rect = mpatches.Rectangle(
                        (j, 8 - i), 1, 1,
                        fill=False,
                        hatch="//",
                        edgecolor="black",
                        linewidth=0.0,
                        alpha=0.9,
                        zorder=6,
                    )
                    ax.add_patch(rect)
        # Hatch-pattern key sized to match a sudoku cell (1/9 of axes), placed
        # next to the "Incorrect" word in the title with a solid black border.
        swatch_size = 0.075
        swatch_xy = (0.74, 1.13)
        hatch_swatch = mpatches.Rectangle(
            swatch_xy, swatch_size, swatch_size,
            transform=ax.transAxes,
            fill=False,
            hatch="//",
            edgecolor="black",
            linewidth=0.0,
            alpha=0.9,
            clip_on=False,
            zorder=10,
        )
        ax.add_patch(hatch_swatch)
        border_swatch = mpatches.Rectangle(
            swatch_xy, swatch_size, swatch_size,
            transform=ax.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=0.9,
            clip_on=False,
            zorder=11,
        )
        ax.add_patch(border_swatch)

    # ── digit labels ─────────────────────────────────────────────────────────
    for i in range(9):
        for j in range(9):
            cx = j + 0.5
            cy = 8 - i + 0.5  # row 0 at top (origin="upper")
            if is_given[i, j] and show_givens:
                digit = int(inp[i, j]) - DIGIT_OFFSET
                if 1 <= digit <= 9:
                    ax.text(
                        cx,
                        cy,
                        str(digit),
                        ha="center",
                        va="center",
                        fontsize=16.5,
                        fontweight="normal",
                        color="#222222",
                        zorder=7,
                    )
            else:
                pred_digit = int(pr[i, j]) - DIGIT_OFFSET
                if 1 <= pred_digit <= 9:
                    ax.text(
                        cx,
                        cy,
                        str(pred_digit),
                        ha="center",
                        va="center",
                        fontsize=16.5,
                        fontweight="normal",
                        color="#1a1a1a",
                        zorder=8,
                    )

    # ── grid lines ─────────────────────────────────────────────────────────
    for k in range(10):
        lw = 1.2 if k % 3 == 0 else 0.25
        ax.axhline(k, color="k", lw=lw, zorder=3)
        ax.axvline(k, color="k", lw=lw, zorder=3)

    ax.set(xlim=(0, 9), ylim=(0, 9), xticks=[], yticks=[])
    if title:
        ax.set_title(title, fontsize=16.0, pad=9)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _draw_grid_entropy(
    ax: plt.Axes,
    data: SplitData,
    idx: int,
    *,
    vmax: float,
    title: str,
) -> None:
    """Generic square-grid entropy view (e.g. maze)."""
    L = int(data.L)
    side = int(round(L**0.5))
    if side * side != L:
        raise ValueError(f"Cannot render non-square token grid with L={L}")

    inp = data.inputs[idx].reshape(side, side)
    lb = data.labels[idx].reshape(side, side)
    en = data.ent_display[idx].reshape(side, side)
    pr = data.map_pred[idx].reshape(side, side)
    mask = data.mask[idx].reshape(side, side)

    # Maze palette from evaluator semantics:
    # 1 wall, 2 space, 3 start, 4 goal, 5 predicted path.
    WALL_COLOR = np.array([0.29, 0.29, 0.29])
    SPACE_COLOR = np.array([1.0, 1.0, 1.0])
    SE_COLOR = np.array([0.94, 0.27, 0.27])
    PATH_OK = np.array([0.13, 0.77, 0.37])
    PATH_BAD = np.array([0.98, 0.45, 0.14])

    bg = np.ones((side, side, 3))
    for i in range(side):
        for j in range(side):
            token = int(inp[i, j])
            if token == 1:
                bg[i, j] = WALL_COLOR
            elif token in (3, 4):
                bg[i, j] = SE_COLOR
            elif not mask[i, j]:
                bg[i, j] = SPACE_COLOR
            else:
                t = float(np.clip(en[i, j] / vmax, 0.0, 1.0))
                bg[i, j] = CELL_CMAP(t)[:3]

    ax.imshow(
        bg,
        extent=(0, side, 0, side),
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )

    # Draw predicted/target path as circles to make solved routes visible.
    for i in range(side):
        for j in range(side):
            if not mask[i, j]:
                continue
            pred_path = int(pr[i, j]) == 5
            true_path = int(lb[i, j]) == 5
            if pred_path:
                color = PATH_OK if true_path else PATH_BAD
                ax.add_patch(
                    mpatches.Circle(
                        (j + 0.5, side - i - 0.5),
                        0.30,
                        facecolor=color,
                        edgecolor="none",
                        zorder=8,
                    )
                )
            elif true_path:
                # Missing path token: hollow orange marker.
                ax.add_patch(
                    mpatches.Circle(
                        (j + 0.5, side - i - 0.5),
                        0.30,
                        fill=False,
                        edgecolor=PATH_BAD,
                        linewidth=1.2,
                        zorder=8,
                    )
                )
            if int(pr[i, j]) != int(lb[i, j]):
                ax.add_patch(
                    mpatches.Rectangle(
                        (j, side - 1 - i),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.6,
                        zorder=7,
                    )
                )

    # Mark maze start/end tokens.
    for i in range(side):
        for j in range(side):
            token = int(inp[i, j])
            if token in (3, 4):
                ax.text(
                    j + 0.5,
                    side - i - 0.5,
                    "S" if token == 3 else "E",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                    color="#ffffff",
                    zorder=8,
                )

    for k in range(side + 1):
        ax.axhline(k, color="#e0e0e0", lw=0.2, zorder=3)
        ax.axvline(k, color="#e0e0e0", lw=0.2, zorder=3)

    ax.set(xlim=(0, side), ylim=(0, side), xticks=[], yticks=[], title=title)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _draw_calibration_summary(ax: plt.Axes, summaries: list[dict]) -> None:
    """Token accuracy vs entropy score, one colour per split, with fold CI bands.

    Solid line with circles: final-step entropy $H_\\ell$ (primary score).
    Dashed line with squares: trajectory-averaged entropy $\\bar H_\\ell$.
    """
    palette = ["#2171b5", "#cb4335", "#33a02c", "#ff7f00"]

    for i, summary in enumerate(summaries):
        cal_df = summary["calibration"]
        color = palette[i % len(palette)]
        if cal_df.empty:
            continue
        ax.fill_between(
            cal_df["x"],
            cal_df["y_lo"],
            cal_df["y_hi"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            cal_df["x"],
            cal_df["y"],
            color=color,
            lw=1.8,
            marker="o",
            ms=3.5,
            label=rf"{summary['label']} ($H_\ell$)",
        )

    for i, summary in enumerate(summaries):
        baseline_df = summary["ent_path"]["calibration"]
        if baseline_df.empty:
            continue
        color = palette[i % len(palette)]
        ax.fill_between(
            baseline_df["x"],
            baseline_df["y_lo"],
            baseline_df["y_hi"],
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(
            baseline_df["x"],
            baseline_df["y"],
            color=color,
            lw=1.4,
            ls="--",
            marker="s",
            ms=3.5,
            alpha=0.85,
            label=rf"{summary['label']} ($\bar H_\ell$)",
        )

    x_all = np.concatenate(
        [
            np.asarray(line.get_xdata(), dtype=float)
            for line in ax.lines
            if len(line.get_xdata())
        ]
    )
    y_lo = min(
        float(collection.get_paths()[0].vertices[:, 1].min())
        for collection in ax.collections
        if collection.get_paths()
    )
    y_hi = max(
        float(collection.get_paths()[0].vertices[:, 1].max())
        for collection in ax.collections
        if collection.get_paths()
    )
    x_pad = max(0.005, 0.06 * float(x_all.max() - x_all.min() + 1e-9))
    y_pad = max(0.015, 0.08 * float(y_hi - y_lo + 1e-9))

    ax.set_xlim(max(0.0, float(x_all.min()) - x_pad), float(x_all.max()) + x_pad)
    ax.set_ylim(max(0.5, y_lo - y_pad), min(1.01, y_hi + y_pad))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_xlabel("Token entropy score")
    ax.set_ylabel("Token accuracy")
    ax.set_title(
        "Token accuracy decreases\nwith higher entropy",
        fontweight="bold", pad=9,
    )
    ax.legend(framealpha=0.9, loc="lower left", fontsize=15.5)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


def make_kfold_figure(
    cal_summaries: list[dict],
    example_data: SplitData,
) -> plt.Figure:
    """Two example heatmaps from ``example_data`` plus the fold-aggregated calibration panel."""
    solved_idx, unsolved_idx = _select_puzzles(example_data)
    solved_idx = _preferred_solved_index(example_data, solved_idx)
    draw_panel = (
        _draw_sudoku_entropy if example_data.task == "sudoku" else _draw_grid_entropy
    )
    preds_s = example_data.preds[solved_idx]
    labels_s = example_data.labels[solved_idx]
    mask_s = example_data.mask[solved_idx]
    solved_particles = int(
        ((preds_s == labels_s[None, :]) | ~mask_s[None, :]).all(axis=1).sum()
    )
    preds_u = example_data.preds[unsolved_idx]
    labels_u = example_data.labels[unsolved_idx]
    mask_u = example_data.mask[unsolved_idx]
    unsolved_particles = int(
        ((preds_u == labels_u[None, :]) | ~mask_u[None, :]).all(axis=1).sum()
    )
    vmax = (
        max(
            float(
                example_data.ent_display[solved_idx][
                    example_data.mask[solved_idx]
                ].max()
            ),
            float(
                example_data.ent_display[unsolved_idx][
                    example_data.mask[unsolved_idx]
                ].max()
            ),
        )
        + 1e-6
    )

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(13.8, 5.2), layout="constrained")
        gs = gridspec.GridSpec(
            2,
            3,
            figure=fig,
            width_ratios=[1.0, 1.0, 1.55],
            height_ratios=[1.0, 0.12],
            wspace=0.08,
        )
        ax_s_ex = fig.add_subplot(gs[0, 0])
        ax_u_ex = fig.add_subplot(gs[0, 1])
        ax_cal = fig.add_subplot(gs[:, 2])
        draw_panel(
            ax_s_ex,
            example_data,
            solved_idx,
            vmax=vmax,
            title=_example_title("Correct", solved_particles, example_data.S),
            **({"show_givens": True} if example_data.task == "sudoku" else {}),
        )
        draw_panel(
            ax_u_ex,
            example_data,
            unsolved_idx,
            vmax=vmax,
            title=_example_title("Incorrect", unsolved_particles, example_data.S),
            **(
                {"show_givens": True, "mark_incorrect": True}
                if example_data.task == "sudoku"
                else {}
            ),
        )
        _draw_calibration_summary(ax_cal, cal_summaries)

        sm = plt.cm.ScalarMappable(cmap=CELL_CMAP, norm=mcolors.Normalize(0.0, vmax))
        sm.set_array([])
        cax = fig.add_subplot(gs[1, 0:2])
        fig.colorbar(
            sm,
            cax=cax,
            orientation="horizontal",
            label=r"Path-integrated token entropy $\bar H_\ell$",
        )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    runs = REPO_ROOT / "assets" / "runs" / "sudoku"
    test_dir = args.test_dir
    if test_dir is None and (runs / "entropy_test_k5").exists():
        test_dir = runs / "entropy_test_k5"
    df_dir = args.df_dir
    if df_dir is None and (runs / "entropy_impossible_k5").exists():
        df_dir = runs / "entropy_impossible_k5"

    if test_dir is None:
        raise FileNotFoundError(
            "No test-split entropy run found. Pass --test-dir, or run "
            "`bash scripts/run_experiments.sh --token-entropy` first."
        )
    for run_dir in (test_dir, df_dir):
        if run_dir is not None and not is_kfold_run(run_dir):
            raise FileNotFoundError(
                f"{run_dir} is not a k-fold run directory (no fold_manifest.json)."
            )

    print(f"Test-set dir: {test_dir}")
    cal_summaries = [summarise_entropy_folds(test_dir, label="Test set")]
    example_data = SplitData(next(iter(fold_dirs(test_dir))), "Test")
    if df_dir is not None:
        print(f"DF-set dir:   {df_dir}")
        cal_summaries.append(summarise_entropy_folds(df_dir, label="DF set"))
    fig = make_kfold_figure(cal_summaries, example_data)

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"{OUT_NAME}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}/{OUT_NAME}.pdf + .png")

    for summary in cal_summaries:
        print(
            f"{summary['label']} solved puzzles per fold: "
            f"{summary['solved_per_fold_mean']:.1f} ± {summary['solved_per_fold_ci']:.1f}"
        )
        print(
            f"  final-step token entropy  H_l    AUROC: "
            f"{summary['auroc_mean']:.3f} ± {summary['auroc_ci']:.3f}"
        )
        print(
            f"  trajectory-averaged       Hbar   AUROC: "
            f"{summary['ent_path']['auroc_mean']:.3f} ± {summary['ent_path']['auroc_ci']:.3f}"
        )


if __name__ == "__main__":
    main()
