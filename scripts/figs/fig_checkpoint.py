"""Figure 8: guided inference across the TRM training checkpoints (Sudoku-Extreme).

Two panels:

* Left: terminal exact-solve rate vs training epoch on the test split,
  overlaying the deterministic TRM baseline, unguided PF, and guided PF
  (β=0.25, S=16, σ=0.3 — the paper operating point).
* Right: TRM token cross-entropy curves (train and validation) vs epoch,
  taken from ``training_progress.csv`` (the ``{train,val}/lm_loss`` columns).

Guided inference reaches the final checkpoint's deterministic accuracy after
roughly a quarter of the training budget, before the loss curve has saturated.

Reads from ``assets/runs/sudoku/pf_checkpoint_sweep/``:

* ``checkpoint_condition_summary.csv`` — per-(epoch, condition) aggregated PF
  metrics across folds.
* ``epoch_*/<condition>/fold_*/trm_depth_profile.csv`` — per-fold deterministic
  exact-solve trajectories (used to compute the TRM baseline per epoch).
* ``training_progress.csv`` — TRM training-progress trace with train/val token
  cross-entropy in the ``{train,val}/lm_loss`` columns.

Outputs ``fig_checkpoint.pdf`` / ``.png``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while not (REPO_ROOT / "src").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bigr.plots.style import (  # noqa: E402
    COLOR_BASELINE,
    COLOR_MAP,
    COLOR_PARTICLE,
    PUB_RC,
    save_figure,
)

PLOT_COLOR_BASELINE = COLOR_BASELINE
PLOT_COLOR_GUIDED = COLOR_MAP
PLOT_COLOR_PARTICLE = COLOR_PARTICLE

COLOR_TRAIN_LOSS = "#0d9488"
COLOR_VAL_LOSS = "#f59e0b"

LW_MAIN = 1.8
MARKER_SIZE = 4.4
BAND_ALPHA = 0.12
FIGSIZE = (8.6, 3.4)

LABEL_BASELINE = "Deterministic TRM baseline"
LABEL_UNGUIDED = "Single rollout (unguided)"
LABEL_GUIDED = "Guided MAP (our solution)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render checkpoint progression benchmark figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "runs" / "sudoku" / "pf_checkpoint_sweep",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "figures")
    parser.add_argument(
        "--figure-name",
        default="fig_checkpoint",
        help="Output filename stem for the rendered PDF/PNG pair.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------


def _load_pf_summary(run_dir: Path) -> pd.DataFrame:
    """Load per-(epoch, condition) PF summary."""
    path = run_dir / "checkpoint_condition_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_csv(path)
    df["split"] = df["condition"].str.split("_").str[0]
    df["mode"] = df["condition"].str.split("_").str[1]
    df = df.sort_values(["split", "mode", "checkpoint_epoch"]).reset_index(drop=True)
    return df


def _load_training_progress(run_dir: Path) -> pd.DataFrame:
    """Load TRM training-progress trace (train/val LM loss vs epoch)."""
    path = run_dir / "training_progress.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).sort_values("epoch").reset_index(drop=True)


def _load_trm_per_epoch(run_dir: Path) -> pd.DataFrame:
    """Aggregate deterministic-TRM exact-solve at terminal step per (split, epoch)
    from the per-fold ``trm_depth_profile.csv`` files. TRM is deterministic, so
    we read it from any one condition and average across folds."""
    records = []
    for split, cond in (("test", "test_guided"), ("impossible", "impossible_guided")):
        by_epoch: dict[int, list[float]] = defaultdict(list)
        for fold_csv in run_dir.glob(f"epoch_*/{cond}/fold_*/trm_depth_profile.csv"):
            m = re.search(r"epoch_(\d+)", str(fold_csv))
            if m is None:
                continue
            epoch = int(m.group(1))
            df = pd.read_csv(fold_csv)
            row = df.loc[df["step"] == 48]
            if row.empty:
                continue
            by_epoch[epoch].append(float(row["exact_solved"].iloc[0]))
        for epoch, vals in by_epoch.items():
            arr = np.asarray(vals, dtype=float)
            records.append(
                {
                    "split": split,
                    "epoch": epoch,
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "n_folds": int(arr.size),
                }
            )
    return pd.DataFrame(records).sort_values(["split", "epoch"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------


def _plot_curve(
    ax: plt.Axes,
    epochs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    marker: str = "o",
    linewidth: float = LW_MAIN,
    zorder: int = 3,
) -> None:
    lo = np.clip(mean - std, 0.0, 1.0)
    hi = np.clip(mean + std, 0.0, 1.0)
    ax.fill_between(epochs, lo, hi, color=color, alpha=BAND_ALPHA, linewidth=0,
                    zorder=zorder - 1)
    ax.plot(
        epochs, mean,
        color=color, linewidth=linewidth, linestyle=linestyle,
        marker=marker, markersize=MARKER_SIZE, label=label, zorder=zorder,
    )


def _first_crossing(epochs: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    """Linear-interpolated first epoch at which `values >= threshold`."""
    if len(epochs) == 0:
        return None
    if values[0] >= threshold:
        return float(epochs[0])
    for i in range(1, len(epochs)):
        if values[i] >= threshold:
            x0, x1 = float(epochs[i - 1]), float(epochs[i])
            y0, y1 = float(values[i - 1]), float(values[i])
            if y1 == y0:
                return x1
            frac = (threshold - y0) / (y1 - y0)
            return x0 + frac * (x1 - x0)
    return None


def _draw_panel(
    ax: plt.Axes,
    pf_df: pd.DataFrame,
    trm_df: pd.DataFrame,
    *,
    split: str,
    title: str,
    show_ylabel: bool,
    annotate_speedup: bool,
) -> dict:
    pf_split = pf_df[pf_df["split"] == split]
    trm_split = trm_df[trm_df["split"] == split].sort_values("epoch")

    guided = pf_split[pf_split["mode"] == "guided"].sort_values("checkpoint_epoch")
    unguided = pf_split[pf_split["mode"] == "unguided"].sort_values("checkpoint_epoch")

    g_x = guided["checkpoint_epoch"].to_numpy(dtype=float)
    g_y = guided["exact_accuracy_mean"].to_numpy(dtype=float)
    g_s = guided["exact_accuracy_std"].to_numpy(dtype=float)

    u_x = unguided["checkpoint_epoch"].to_numpy(dtype=float)
    u_y = unguided["exact_accuracy_mean"].to_numpy(dtype=float)
    u_s = unguided["exact_accuracy_std"].to_numpy(dtype=float)

    t_x = trm_split["epoch"].to_numpy(dtype=float)
    t_y = trm_split["mean"].to_numpy(dtype=float)
    t_s = trm_split["std"].to_numpy(dtype=float)

    _plot_curve(ax, t_x, t_y, t_s,
                color=PLOT_COLOR_BASELINE, label=LABEL_BASELINE,
                linestyle="-", marker="o", zorder=5)
    _plot_curve(ax, u_x, u_y, u_s,
                color=PLOT_COLOR_PARTICLE, label=LABEL_UNGUIDED,
                linestyle="--", marker="o", zorder=4)
    _plot_curve(ax, g_x, g_y, g_s,
                color=PLOT_COLOR_GUIDED, label=LABEL_GUIDED,
                linestyle="-", marker="o", zorder=6)

    info: dict = {"split": split}

    if annotate_speedup and len(t_y) > 0:
        trm_terminal = float(t_y[-1])
        info["trm_terminal"] = trm_terminal
        guided_cross = _first_crossing(g_x, g_y, trm_terminal)
        trm_cross = _first_crossing(t_x, t_y, trm_terminal)
        info["guided_crossing_epoch"] = guided_cross
        info["trm_terminal_epoch"] = trm_cross

        ax.axhline(trm_terminal,
                   color=PLOT_COLOR_BASELINE, linestyle=":", linewidth=1.0,
                   alpha=0.85, zorder=2)

        if guided_cross is not None:
            ax.axvline(guided_cross,
                       color=PLOT_COLOR_GUIDED, linestyle=":", linewidth=1.0,
                       alpha=0.85, zorder=2)
            if trm_cross is not None and trm_cross > guided_cross:
                ratio = trm_cross / guided_cross
                info["speedup_ratio"] = ratio

    ax.set_title(title, pad=6, fontweight="bold")
    ax.set_xlabel("Training epoch")
    if show_ylabel:
        ax.set_ylabel("Terminal exact-solve rate")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    x_min = float(min(g_x[0], u_x[0], t_x[0]))
    ax.set_xlim(left=x_min, right=max(g_x[-1], t_x[-1]) * 1.02)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    info["x_min"] = x_min
    return info


def _draw_loss_panel(ax: plt.Axes, training: pd.DataFrame, *, x_min: float) -> None:
    if training.empty or "epoch" not in training.columns:
        ax.text(
            0.5, 0.5, "no training_progress.csv found",
            transform=ax.transAxes, ha="center", va="center", color="#4b5563",
        )
        ax.set_xlabel("Training epoch")
        ax.set_title("TRM training loss", pad=6, fontweight="bold")
        ax.set_ylabel("Cross-entropy loss")
        return

    train = training.dropna(subset=["train/lm_loss"]) if "train/lm_loss" in training.columns else training.iloc[0:0]
    val = training.dropna(subset=["val/lm_loss"]) if "val/lm_loss" in training.columns else training.iloc[0:0]

    train = train[train["epoch"] >= x_min]
    val = val[val["epoch"] >= x_min]

    if not train.empty:
        ax.plot(
            train["epoch"], train["train/lm_loss"],
            color=COLOR_TRAIN_LOSS, linewidth=LW_MAIN, alpha=0.8,
            label="Training set", zorder=4,
        )
    if not val.empty:
        ax.plot(
            val["epoch"], val["val/lm_loss"],
            color=COLOR_VAL_LOSS, linewidth=LW_MAIN, linestyle="--",
            label="Validation set", zorder=5,
        )

    ax.set_title("TRM training loss", pad=6, fontweight="bold")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_ylim(bottom=0.0)
    if not training.empty:
        ax.set_xlim(left=x_min, right=float(training["epoch"].max()) * 1.02)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", framealpha=0.92, ncol=1)


def make_figure(
    pf_df: pd.DataFrame,
    trm_df: pd.DataFrame,
    training: pd.DataFrame,
    *,
    rc: dict,
) -> tuple[plt.Figure, list[dict]]:
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            1, 2, figsize=FIGSIZE,
            gridspec_kw={"wspace": 0.22}, layout="constrained",
        )
        info_test = _draw_panel(
            axes[0], pf_df, trm_df,
            split="test",
            title="Sudoku Extreme — Test split",
            show_ylabel=True,
            annotate_speedup=True,
        )
        axes[0].legend(loc="lower right", framealpha=0.92, ncol=1, fontsize=7.5)
        _draw_loss_panel(axes[1], training, x_min=info_test.get("x_min", 0.0))
    return fig, [info_test]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not (args.run_dir / "checkpoint_condition_summary.csv").exists():
        raise FileNotFoundError(
            f"Expected {args.run_dir / 'checkpoint_condition_summary.csv'}; "
            "run `bash scripts/run_experiments.sh --checkpoint-sweep` before rendering."
        )

    pf_df = _load_pf_summary(args.run_dir)
    trm_df = _load_trm_per_epoch(args.run_dir)
    training = _load_training_progress(args.run_dir)

    rc = {
        **PUB_RC,
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.dpi": 300,
    }
    fig, infos = make_figure(pf_df, trm_df, training, rc=rc)
    save_figure(fig, args.out / f"{args.figure_name}.pdf", dpi=300)
    plt.close(fig)

    for info in infos:
        split = info["split"]
        terminal = info.get("trm_terminal")
        cross = info.get("guided_crossing_epoch")
        trm_cross = info.get("trm_terminal_epoch")
        if terminal is not None:
            msg = f"  {split}: TRM terminal = {terminal*100:.1f}%"
            if cross is not None:
                msg += f"; guided crosses at epoch ~{cross:.0f}"
            if trm_cross is not None:
                msg += f"; TRM reaches its terminal at epoch ~{trm_cross:.0f}"
                if cross is not None and cross > 0:
                    msg += f" ({trm_cross/cross:.2f}x later)"
            print(msg)
    print(f"Saved: {args.out}/{args.figure_name}.pdf + .png")


if __name__ == "__main__":
    main()
