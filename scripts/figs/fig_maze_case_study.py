"""fig_maze_case_study.py — Figure 5: Maze-Hard tube stability and Q-head training curves.

Mosaic [["in_tube", "q_head"], ["survival", "q_head"]]:
  in_tube  (top left): E[||e_m||² 1_{τ_r>m}] / r² vs inner step m on Maze-Hard for
           one tube radius; one curve per σ with its analytical bound (dotted).
  survival (bottom left): Pr(τ_r > M) vs σ for each tube radius α on Maze-Hard.
  q_head   (right, full height): validation Q-halt accuracy and BCE loss over
           training epochs for the Sudoku-Extreme and Maze-Hard checkpoints.

Reads  <inner-dev-dir>/{deviation_masked,survival_at_M}[_summary].csv and
       <q-training-dir>/manifest.csv plus the Lightning metrics CSVs it lists.
Writes <out>/fig_maze_case_study.{pdf,png}; --out defaults to assets/figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while not (REPO_ROOT / "src").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bigr.plots.kfold import (  # noqa: E402
    ci_bounds,
    normalise_summary_columns,
    num_folds_from_run,
)

COLOR_ACCURACY = "#2563eb"
COLOR_BCE = "#dc2626"
COLOR_TASK_INDEX = "#6b7280"
LINEWIDTH = 2.0

TASK_LINESTYLES = {"maze_hard": "-", "sudoku_extreme": (0, (1.0, 1.8))}

SIGMA_COLORS: dict[float, str] = {
    0.20: "#17becf",
    0.25: "#e05c2a",
    0.30: "#17becf",
    0.50: "#d62728",
    0.75: "#9467bd",
    0.80: "#9467bd",
    1.00: "#7f7f7f",
}

ALPHA_COLORS: dict[float, str] = {
    0.5: "#08519c",
    1.0: "#4292c6",
    1.5: "#9ecae1",
    2.0: "#c6dbef",
}
ALPHA_MARKERS: dict[float, str] = {0.5: "o", 1.0: "s", 1.5: "^", 2.0: "D"}

LEFT_PANEL_SIGMAS_EXCLUDED = {0.10, 1.30}
PLOT_SIGMAS_EXCLUDED: set[float] = {0.20}
SURVIVAL_ALPHAS_SHOWN: set[float] = {0.5, 1.0, 1.5}
SIGMA_OPERATING_POINT_MAZE = 0.3

PUB_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--inner-dev-dir",
        type=Path,
        default=REPO_ROOT
        / "assets"
        / "runs"
        / "maze"
        / "inner_deviation_maze_hard_k5",
        help="Maze-Hard inner-deviation run dir (deviation_masked / survival_at_M).",
    )
    p.add_argument(
        "--q-training-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "runs" / "sudoku" / "q_training",
    )
    p.add_argument(
        "--left-alpha",
        type=float,
        default=1.5,
        help="Tube alpha (radius multiple) shown in the in-tube deviation panel.",
    )
    p.add_argument(
        "--left-ylim",
        type=float,
        default=1.0,
        help=(
            "Upper y-limit for the in-tube panel when no σ curve is plotted; "
            "otherwise the limit follows the largest plotted bound."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "assets" / "figures",
    )
    return p.parse_args()


def _sigma_colour(sigma: float) -> str:
    return SIGMA_COLORS.get(round(sigma, 2), "#888888")


def _alpha_colour(alpha: float) -> str:
    return ALPHA_COLORS.get(round(alpha, 1), "#888888")


def _load_inner_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    masked_summary = run_dir / "deviation_masked_summary.csv"
    surv_summary = run_dir / "survival_at_M_summary.csv"
    if masked_summary.exists() and surv_summary.exists():
        num_folds = num_folds_from_run(run_dir)
        return (
            normalise_summary_columns(pd.read_csv(masked_summary), num_folds=num_folds),
            normalise_summary_columns(pd.read_csv(surv_summary), num_folds=num_folds),
        )
    return (
        pd.read_csv(run_dir / "deviation_masked.csv"),
        pd.read_csv(run_dir / "survival_at_M.csv"),
    )


# ---------------------------------------------------------------------------
# In-tube panel (top left): stability bound
# ---------------------------------------------------------------------------


def _draw_in_tube(
    ax: plt.Axes,
    masked_df: pd.DataFrame,
    *,
    left_alpha: float,
    left_ylim: float,
) -> None:
    available_alphas = sorted(masked_df["tube_alpha"].unique())
    if left_alpha not in available_alphas:
        left_alpha = available_alphas[0]
    left_df = masked_df[masked_df["tube_alpha"] == left_alpha]
    r_sq = left_df["tube_radius"].iloc[0] ** 2
    sigmas = sorted(masked_df["sigma"].unique())
    plot_sigmas = [
        sigma
        for sigma in sigmas
        if round(float(sigma), 2) not in PLOT_SIGMAS_EXCLUDED
    ]

    plotted: list[tuple[float, pd.DataFrame]] = []
    for sigma in plot_sigmas:
        if round(float(sigma), 2) in LEFT_PANEL_SIGMAS_EXCLUDED:
            continue
        sub = left_df[left_df["sigma"] == sigma].sort_values("inner_step")
        if sub.empty or sub["survival_prob"].iloc[0] == 0.0:
            continue
        plotted.append((sigma, sub))

    for sigma, sub in plotted:
        ms = sub["inner_step"].values
        surv = sub["survival_prob"].values
        in_tube = surv > 0.01
        devs = sub["mean_masked_dev"].values / r_sq
        bound = np.where(in_tube, sub["partial_bound"].values / r_sq, np.nan)
        col = _sigma_colour(sigma)
        ax.fill_between(ms, devs, bound, color=col, alpha=0.18, linewidth=0)
        ax.plot(
            ms, devs, "-o", color=col, markersize=5.5, label=rf"$\sigma={sigma}$"
        )
        ax.plot(ms, bound, ":", color=col, linewidth=1.1, alpha=0.85)

    ax.plot([], [], "k:", linewidth=1.1, alpha=0.7, label="Theory")
    ax.set_xlabel(r"Inner step $m$")
    ax.set_ylabel(r"$\mathbb{E}[\|e_m\|^2\,\mathbf{1}_{\{\tau_r>m\}}]\;/\;r^2$")
    ax.set_title("Maze in-tube stability", pad=4, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        ncol=len(handles),
        loc="center",
        fontsize=7.5,
        handlelength=1.6,
        borderpad=0.3,
        columnspacing=1.0,
        framealpha=0.92,
    )
    ax.set_xlim(left=0)
    if plotted:
        max_y = max(
            float(np.nanmax(sub["partial_bound"].values / r_sq)) for _, sub in plotted
        )
        ax.set_ylim(0, max_y * 1.08)
    else:
        ax.set_ylim(0, left_ylim)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


# ---------------------------------------------------------------------------
# Survival panel (bottom left): tube exit probability
# ---------------------------------------------------------------------------


def _draw_survival(ax: plt.Axes, surv_df: pd.DataFrame) -> None:
    sigmas = sorted(surv_df["sigma"].unique())
    plot_sigmas = [
        sigma for sigma in sigmas if round(float(sigma), 2) not in PLOT_SIGMAS_EXCLUDED
    ]
    alphas = [
        alpha
        for alpha in sorted(surv_df["tube_alpha"].unique())
        if round(float(alpha), 1) in SURVIVAL_ALPHAS_SHOWN
    ]
    first_row = surv_df[surv_df["tube_alpha"] == alphas[0]].iloc[0]
    r0 = first_row["tube_radius"] / first_row["tube_alpha"]

    for alpha in alphas:
        sub = surv_df[surv_df["tube_alpha"] == alpha].sort_values("sigma")
        sub = sub[
            ~sub["sigma"].round(2).isin(
                {round(sigma, 2) for sigma in PLOT_SIGMAS_EXCLUDED}
            )
        ]
        col = _alpha_colour(alpha)
        mk = ALPHA_MARKERS.get(round(alpha, 1), "o")
        r_val = alpha * r0
        ax.plot(
            sub["sigma"],
            sub["survival_prob_at_M"].values,
            linestyle="-",
            marker=mk,
            color=col,
            markersize=4,
            label=rf"$\alpha={alpha:.1f}$  ($r={r_val:.0f}$)",
        )
        if (
            "survival_prob_at_M_lo" in sub.columns
            and "survival_prob_at_M_hi" in sub.columns
        ):
            ax.fill_between(
                sub["sigma"].values,
                np.clip(sub["survival_prob_at_M_lo"].values, 0.0, 1.0),
                np.clip(sub["survival_prob_at_M_hi"].values, 0.0, 1.0),
                color=col,
                alpha=0.18,
                linewidth=0,
            )

    ax.axvline(
        SIGMA_OPERATING_POINT_MAZE,
        color="#888888",
        linewidth=0.9,
        linestyle="--",
        zorder=3,
    )
    ax.set_xlabel(r"Noise scale $\sigma$")
    ax.set_ylabel(r"$\Pr(\tau_r > M)$")
    ax.set_title("Maze tube exit probability", pad=4, fontweight="bold")
    ax.legend(
        loc="center right",
        bbox_to_anchor=(1.0, 0.38),
        fontsize=7.5,
        handlelength=1.6,
        borderpad=0.3,
        framealpha=0.92,
    )
    ax.set_xlim(left=0, right=max(plot_sigmas) * 1.05)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks(plot_sigmas)


# ---------------------------------------------------------------------------
# Q-head panel (right): training quality
# ---------------------------------------------------------------------------


def _load_q_training(run_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(run_dir / "manifest.csv")
    rows: list[pd.DataFrame] = []
    for entry in manifest.itertuples(index=False):
        frame = pd.read_csv(run_dir / entry.path)
        split = "val"
        for metric in ("q_halt_accuracy", "q_halt_loss"):
            col = f"{split}/{metric}"
            if col not in frame.columns:
                continue
            sub = frame.loc[
                frame[col].notna(), ["epoch", col]
            ].rename(columns={col: "value"}).copy()
            if sub.empty:
                continue
            sub["task"] = entry.task
            sub["split"] = split
            sub["metric"] = metric
            rows.append(sub)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["epoch", "value"])


def _summarise_q(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["task", "metric", "epoch"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["lo"] = grouped["mean"]
    grouped["hi"] = grouped["mean"]
    for count, idx in grouped.groupby("count").groups.items():
        if int(count) <= 1:
            continue
        _, lo, hi = ci_bounds(
            grouped.loc[idx, "mean"].to_numpy(),
            grouped.loc[idx, "std"].fillna(0.0).to_numpy(),
            int(count),
        )
        grouped.loc[idx, "lo"] = lo
        grouped.loc[idx, "hi"] = hi
    return grouped


def _draw_q_training(ax: plt.Axes, run_dir: Path) -> None:
    metrics = _load_q_training(run_dir)
    if metrics.empty:
        ax.text(
            0.5,
            0.5,
            "no q_training metrics found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#6b7280",
        )
        return

    summary = _summarise_q(metrics)
    ax_loss = ax.twinx()
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(True)

    for task, ls in TASK_LINESTYLES.items():
        acc = summary[
            (summary["task"] == task) & (summary["metric"] == "q_halt_accuracy")
        ].sort_values("epoch")
        loss = summary[
            (summary["task"] == task) & (summary["metric"] == "q_halt_loss")
        ].sort_values("epoch")
        if not acc.empty:
            ax.plot(
                acc["epoch"],
                acc["mean"] * 100.0,
                color=COLOR_ACCURACY,
                linestyle=ls,
                linewidth=LINEWIDTH,
            )
            ax.fill_between(
                acc["epoch"],
                acc["lo"] * 100.0,
                acc["hi"] * 100.0,
                color=COLOR_ACCURACY,
                alpha=0.10,
                linewidth=0,
            )
        if not loss.empty:
            ax_loss.plot(
                loss["epoch"],
                loss["mean"],
                color=COLOR_BCE,
                linestyle=ls,
                linewidth=LINEWIDTH,
            )
            ax_loss.fill_between(
                loss["epoch"],
                loss["lo"],
                loss["hi"],
                color=COLOR_BCE,
                alpha=0.08,
                linewidth=0,
            )

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Accuracy", color="black")
    ax_loss.set_ylabel("BCE loss", color="black")
    ax.tick_params(axis="y", labelcolor=COLOR_ACCURACY)
    ax_loss.tick_params(axis="y", labelcolor=COLOR_BCE)
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 102.0)
    loss_max = float(summary[summary["metric"] == "q_halt_loss"]["mean"].max() or 1.0)
    ax_loss.set_ylim(0.0, max(1.0, loss_max * 2.05))
    ax.set_title("Q-head training quality", pad=4, fontweight="bold")

    handles = [
        Line2D([0], [0], color=COLOR_ACCURACY, linewidth=LINEWIDTH, label="Accuracy"),
        Line2D([0], [0], color=COLOR_BCE, linewidth=LINEWIDTH, label="BCE loss"),
        Line2D(
            [0],
            [0],
            color=COLOR_TASK_INDEX,
            linewidth=LINEWIDTH,
            linestyle=TASK_LINESTYLES["maze_hard"],
            label="Maze Hard",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_TASK_INDEX,
            linewidth=LINEWIDTH,
            linestyle=TASK_LINESTYLES["sudoku_extreme"],
            label="Sudoku Extreme",
        ),
    ]
    ax.legend(handles=handles, loc="center", framealpha=0.92)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    masked_df, surv_df = _load_inner_tables(args.inner_dev_dir)

    with plt.rc_context(PUB_RC):
        fig, ax_dict = plt.subplot_mosaic(
            [["in_tube", "q_head"],
             ["survival", "q_head"]],
            figsize=(11.5, 5.2),
            width_ratios=[1.0, 1.15],
            layout="constrained",
        )
        _draw_in_tube(
            ax_dict["in_tube"],
            masked_df,
            left_alpha=args.left_alpha,
            left_ylim=args.left_ylim,
        )
        _draw_survival(ax_dict["survival"], surv_df)
        _draw_q_training(ax_dict["q_head"], args.q_training_dir)

        for ext in ("pdf", "png"):
            p = args.out / f"fig_maze_case_study.{ext}"
            fig.savefig(p, bbox_inches="tight")
            print(f"Saved {p}")
        plt.close(fig)


if __name__ == "__main__":
    main()
