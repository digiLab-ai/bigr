"""fig_token_contraction — Figure 7 (appendix): trajectory contraction of the token marginal.

Per non-given token, the contraction C_l = Hbar_l - H_l is the trajectory-averaged
normalised entropy (cached by run_entropy as `path_auc_uncert`) minus the final-step
entropy (recomputed from the saved terminal preds and weights); C_l > 0 means the
cloud sharpened towards the end of the trajectory. The figure plots the DF split
only: a histogram of the per-puzzle mean contraction split by guided exact solve,
and a scatter of per-puzzle mean Hbar_l against H_l. Summary statistics (with a
Welch t) are printed for both the test and DF splits.

Reads  <test-dir>/fold_*/entropy_data.npz and <df-dir>/fold_*/entropy_data.npz
       (or a single entropy_data.npz per dir), from scripts/exps/run_entropy.py.
Writes <out>/fig_token_contraction.{pdf,png}; --out defaults to assets/figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while not (REPO_ROOT / "src").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent

SOLVED_COLOUR = "#2ca02c"
UNSOLVED_COLOUR = "#d62728"

RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--df-dir", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "assets" / "figures",
        help="Output directory for the figure.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fold loading
# ---------------------------------------------------------------------------


def _fold_dirs(run_dir: Path) -> list[Path]:
    folds = sorted(run_dir.glob("fold_*"))
    folds = [f for f in folds if (f / "entropy_data.npz").exists()]
    if folds:
        return folds
    if (run_dir / "entropy_data.npz").exists():
        return [run_dir]
    raise FileNotFoundError(f"No entropy_data.npz under {run_dir}")


def _fold_arrays(npz_path: Path) -> dict:
    """Load one fold and return per-puzzle and per-token contraction arrays."""
    cache = np.load(npz_path)
    inputs = cache["inputs"]  # (N, L)
    labels = cache["labels"]  # (N, L)
    preds = cache["preds"]  # (N, S, L)
    weights = cache["weights"].astype(np.float32)
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)

    if "path_auc_uncert" not in cache.files:
        raise KeyError(
            f"path_auc_uncert not present in {npz_path}; this diagnostic "
            f"requires the trajectory-averaged entropy to be cached."
        )
    ent_path = cache["path_auc_uncert"].astype(np.float32)  # (N, L)

    L = inputs.shape[1]
    task_sudoku = L == 81
    mask = inputs < 3 if task_sudoku else inputs == 2

    n_classes = int(preds.max() + 1)
    one_hot = (
        preds[:, :, :, None]
        == np.arange(n_classes, dtype=preds.dtype)[None, None, None, :]
    ).astype(np.float32)
    probs = (weights[:, :, None, None] * one_hot).sum(axis=1)
    probs /= np.clip(probs.sum(axis=-1, keepdims=True), 1e-12, None)
    map_pred = probs.argmax(axis=-1)

    safe = np.clip(probs, 1e-12, 1.0)
    log_C = np.log(float(n_classes))
    ent_final = -(safe * np.log(safe)).sum(axis=-1) / log_C  # (N, L)

    token_correct = map_pred == labels
    puzzle_exact = (token_correct | ~mask).all(axis=1)  # (N,) bool

    # Per-puzzle mean entropies over non-given tokens
    mask_counts = np.maximum(mask.sum(axis=1), 1)
    mean_ent_final = (np.where(mask, ent_final, 0.0).sum(axis=1) / mask_counts).astype(
        np.float32
    )
    mean_ent_path = (np.where(mask, ent_path, 0.0).sum(axis=1) / mask_counts).astype(
        np.float32
    )
    mean_contract = mean_ent_path - mean_ent_final

    return {
        "puzzle_exact": puzzle_exact,
        "mean_ent_final": mean_ent_final,
        "mean_ent_path": mean_ent_path,
        "mean_contract": mean_contract,
    }


def _load_split(run_dir: Path) -> dict:
    folds = _fold_dirs(run_dir)
    arrays = [_fold_arrays(f / "entropy_data.npz") for f in folds]
    return {
        "n_folds": len(folds),
        "puzzle_exact": np.concatenate([a["puzzle_exact"] for a in arrays]),
        "mean_ent_final": np.concatenate([a["mean_ent_final"] for a in arrays]),
        "mean_ent_path": np.concatenate([a["mean_ent_path"] for a in arrays]),
        "mean_contract": np.concatenate([a["mean_contract"] for a in arrays]),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _draw_contraction_hist(ax: plt.Axes, split: dict) -> None:
    solved = split["puzzle_exact"].astype(bool)
    mean_c = split["mean_contract"]
    c_solved = mean_c[solved]
    c_unsolved = mean_c[~solved]

    bins = np.linspace(
        min(c_solved.min(), c_unsolved.min(), 0.0) - 0.01,
        max(c_solved.max(), c_unsolved.max()) + 0.01,
        40,
    )
    ax.hist(
        c_solved,
        bins=bins,
        density=True,
        color=SOLVED_COLOUR,
        alpha=0.55,
        label=f"Solved ({len(c_solved)})",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.hist(
        c_unsolved,
        bins=bins,
        density=True,
        color=UNSOLVED_COLOUR,
        alpha=0.55,
        label=f"Unsolved ({len(c_unsolved)})",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
    # Means
    for arr, colour in ((c_solved, SOLVED_COLOUR), (c_unsolved, UNSOLVED_COLOUR)):
        if len(arr):
            ax.axvline(float(arr.mean()), color=colour, linestyle="--", linewidth=1.3)
    ax.set_xlabel(r"Per-puzzle mean contraction $\bar C = \bar H_\ell - H_\ell$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", frameon=False)


def _draw_scatter(ax: plt.Axes, split: dict) -> None:
    solved = split["puzzle_exact"].astype(bool)
    x = split["mean_ent_final"]
    y = split["mean_ent_path"]
    lim = float(max(x.max(), y.max())) + 0.02

    ax.scatter(
        x[~solved],
        y[~solved],
        color=UNSOLVED_COLOUR,
        alpha=0.45,
        s=9,
        label=f"Unsolved ({int((~solved).sum())})",
        edgecolor="none",
    )
    ax.scatter(
        x[solved],
        y[solved],
        color=SOLVED_COLOUR,
        alpha=0.35,
        s=9,
        label=f"Solved ({int(solved.sum())})",
        edgecolor="none",
    )
    ax.plot(
        [0.0, lim], [0.0, lim], color="gray", linestyle=":", linewidth=1.0,
        label=r"$\bar H_\ell = H_\ell$ (no contraction)",
    )
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Per-puzzle mean $H_\ell$ (terminal)")
    ax.set_ylabel(r"Per-puzzle mean $\bar H_\ell$ (trajectory average)")
    ax.legend(loc="upper left", frameon=False)


def make_figure(split: dict) -> plt.Figure:
    """Contraction histogram (left) and trajectory-vs-terminal entropy scatter (right) for one split."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), layout="constrained")
        _draw_contraction_hist(axes[0], split)
        _draw_scatter(axes[1], split)
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _summary_stats(split: dict, name: str) -> None:
    solved = split["puzzle_exact"].astype(bool)
    mean_c = split["mean_contract"]
    c_solved = mean_c[solved]
    c_unsolved = mean_c[~solved]
    print(
        f"{name}: puzzles={len(mean_c)} solved={int(solved.sum())} "
        f"unsolved={int((~solved).sum())}"
    )
    print(
        f"  mean contraction | solved:   {c_solved.mean():.4f}  "
        f"median {np.median(c_solved):.4f}  std {c_solved.std(ddof=1):.4f}"
    )
    print(
        f"  mean contraction | unsolved: {c_unsolved.mean():.4f}  "
        f"median {np.median(c_unsolved):.4f}  std {c_unsolved.std(ddof=1):.4f}"
    )
    # Welch t-stat as a quick effect-size hint (not the primary claim)
    n1, n2 = len(c_solved), len(c_unsolved)
    if n1 > 1 and n2 > 1:
        v1, v2 = c_solved.var(ddof=1), c_unsolved.var(ddof=1)
        se = np.sqrt(v1 / n1 + v2 / n2)
        t = (c_solved.mean() - c_unsolved.mean()) / max(se, 1e-12)
        print(f"  Welch t (solved - unsolved): t ≈ {t:.2f}")


def main() -> None:
    args = parse_args()

    split_test = _load_split(args.test_dir)
    split_df = _load_split(args.df_dir)
    print(f"Test folds: {split_test['n_folds']}  DF folds: {split_df['n_folds']}")
    _summary_stats(split_test, "TEST")
    _summary_stats(split_df, "DF  ")

    args.out.mkdir(parents=True, exist_ok=True)
    fig = make_figure(split_df)
    pdf_path = args.out / "fig_token_contraction.pdf"
    png_path = args.out / "fig_token_contraction.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved: {pdf_path} + {png_path.suffix}")


if __name__ == "__main__":
    main()
