"""fig_ablation_sensitivity.py — Figure 6: 2x3 sensitivity sweeps (Sudoku-Extreme top, Maze-Hard bottom).

Each row shows the terminal exact-solve rate against the tempering β, the
particle count S and the resampling threshold τ_ESS (the last with mean ESS/S
on a second axis). All panels draw 95% confidence bands from the *_summary.csv schema.

Reads  <runs>/ablation_sensitivity*/resample_stress_results_summary.csv (β and S sweeps)
       and <runs>/ablation_sensitivity*/resample/summary_summary.csv (τ_ESS sweep),
       under assets/runs/sudoku and assets/runs/maze.
Writes <out>/fig_ablation_sensitivity.{pdf,png}; --out defaults to assets/figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from bigr.plots.kfold import (
    has_summary_schema,
    normalise_summary_columns,
    num_folds_from_run,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while not (REPO_ROOT / "src").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent

MAP_COLORS = {
    "no_resample": "#c0392b",
    "tau_0.30": "#1f77b4",
}
ORACLE_COLORS = {
    "no_resample": "#e6a19a",
    "tau_0.30": "#9cc2eb",
}
ESS_COLOR = "#1f9d55"
TAU_DEFAULT = 0.30

PUB_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
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
    parser = argparse.ArgumentParser(
        description="2x3 sensitivity figure: sudoku (top) + maze (bottom)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sudoku_root = REPO_ROOT / "assets" / "runs" / "sudoku" / "ablation_sensitivity_k5"
    maze_root = REPO_ROOT / "assets" / "runs" / "maze" / "ablation_sensitivity_maze_hard_k5"
    parser.add_argument(
        "--sudoku-stress-csv",
        type=Path,
        default=sudoku_root / "resample_stress_results_summary.csv",
    )
    parser.add_argument(
        "--sudoku-tau-csv",
        type=Path,
        default=sudoku_root / "resample" / "summary_summary.csv",
    )
    parser.add_argument(
        "--maze-stress-csv",
        type=Path,
        default=maze_root / "resample_stress_results_summary.csv",
    )
    parser.add_argument(
        "--maze-tau-csv",
        type=Path,
        default=maze_root / "resample" / "summary_summary.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "assets" / "figures",
    )
    return parser.parse_args()


def _label(label: str) -> str:
    if label == "no_resample":
        return "No resampling"
    if label == "tau_0.30":
        return "Resampling\n" + r"($\tau_{\mathrm{ESS}}=0.3$)"
    return label.replace("_", " ")


def _load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if has_summary_schema(frame):
        run_dir = path.parent.parent if path.parent.name == "resample" else path.parent
        return normalise_summary_columns(frame, num_folds=num_folds_from_run(run_dir))
    return frame


def _plot_sweep(ax, df: pd.DataFrame, x_col: str, xlabel: str) -> None:
    """Terminal exact-solve rate vs ``x_col`` per resampling label, with the oracle particle dotted."""
    for label, sub in df.groupby("label", sort=True):
        colour = MAP_COLORS.get(label, "#666666")
        oracle_colour = ORACLE_COLORS.get(label, "#bbbbbb")
        ax.plot(
            sub[x_col],
            sub["final_acc"],
            color=colour,
            marker="o",
            label=_label(str(label)),
        )
        if "final_acc_lo" in sub.columns:
            ax.fill_between(
                sub[x_col],
                sub["final_acc_lo"],
                sub["final_acc_hi"],
                color=colour,
                alpha=0.10,
                linewidth=0,
            )
        ax.plot(
            sub[x_col],
            sub["final_oracle"],
            color=oracle_colour,
            linestyle=":",
            linewidth=1.1,
            alpha=0.95,
        )
        if "final_oracle_lo" in sub.columns:
            ax.fill_between(
                sub[x_col],
                sub["final_oracle_lo"],
                sub["final_oracle_hi"],
                color=oracle_colour,
                alpha=0.08,
                linewidth=0,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Terminal exact-solve rate")


def _plot_tau(ax, tau_df: pd.DataFrame, *, show_legend: bool = True) -> None:
    ax2 = ax.twinx()
    ax.plot(
        tau_df["ess_threshold"],
        tau_df["final_accuracy"],
        color="#1f77b4",
        marker="o",
        label="MAP accuracy",
    )
    if "final_accuracy_lo" in tau_df.columns:
        ax.fill_between(
            tau_df["ess_threshold"],
            tau_df["final_accuracy_lo"],
            tau_df["final_accuracy_hi"],
            color="#1f77b4",
            alpha=0.10,
            linewidth=0,
        )
    ax2.plot(
        tau_df["ess_threshold"],
        tau_df["final_ess"],
        color=ESS_COLOR,
        marker="s",
        linestyle="--",
        label=r"Mean $\mathrm{ESS} / S$",
    )
    if "final_ess_lo" in tau_df.columns:
        ax2.fill_between(
            tau_df["ess_threshold"],
            tau_df["final_ess_lo"],
            tau_df["final_ess_hi"],
            color=ESS_COLOR,
            alpha=0.08,
            linewidth=0,
        )
    ax.axvline(
        TAU_DEFAULT,
        color="#636363",
        linewidth=0.9,
        linestyle=":",
        alpha=0.75,
    )
    ax.set_xlabel(r"Resampling threshold $\tau_{\mathrm{ESS}}$")
    ax.set_ylabel("Terminal exact-solve rate", color="black")
    ax2.set_ylabel(r"Mean $\mathrm{ESS} / S$", color="black")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor=ESS_COLOR)
    ax2.set_ylim(0.10, 1.02)
    if show_legend:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="lower right",
            framealpha=0.92,
        )


def _plot_row(
    axes_row,
    stress_df: pd.DataFrame,
    tau_df: pd.DataFrame,
    *,
    row_title: str,
    show_side_legends: bool = True,
) -> None:
    beta_df = stress_df[stress_df["exp"] == "beta_sweep"].sort_values(["label", "beta"])
    s_df = stress_df[stress_df["exp"] == "S_sweep"].sort_values(["label", "S"])
    # The paper's τ_ESS panel uses these rows only; the *_rejuv rows in the same
    # CSV (rejuvenation variants) are not part of the figure.
    tau_df = tau_df[
        tau_df["tag"].isin(
            ["no_resample", "tau_0.30", "tau_0.50", "tau_0.70", "tau_1.00"]
        )
    ].sort_values("ess_threshold")

    ax_beta, ax_S, ax_tau = axes_row
    _plot_sweep(ax_beta, beta_df, "beta", r"Tempering $\beta$")
    _plot_sweep(ax_S, s_df, "S", r"Particles $S$")
    ax_S.set_xticks(sorted(s_df["S"].unique()))
    _plot_tau(ax_tau, tau_df, show_legend=show_side_legends)
    if show_side_legends:
        handles, labels = ax_beta.get_legend_handles_labels()
        oracle_handle = Line2D(
            [], [], color="#888888", linestyle=":", linewidth=1.1, label="Oracle particle"
        )
        ax_beta.legend(
            handles + [oracle_handle],
            labels + ["Oracle particle"],
            loc="lower right",
            framealpha=0.92,
        )
    ax_beta.set_title(f"{row_title} — tempering", pad=4)
    ax_S.set_title(f"{row_title} — particles", pad=4)
    ax_tau.set_title(f"{row_title} — ESS threshold", pad=4)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sudoku_stress = _load_table(args.sudoku_stress_csv)
    sudoku_tau = _load_table(args.sudoku_tau_csv)
    maze_stress = _load_table(args.maze_stress_csv)
    maze_tau = _load_table(args.maze_tau_csv)

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), layout="constrained")
        _plot_row(axes[0], sudoku_stress, sudoku_tau, row_title="Sudoku Extreme")
        _plot_row(
            axes[1],
            maze_stress,
            maze_tau,
            row_title="Maze Hard",
            show_side_legends=False,
        )

        for ext in ("pdf", "png"):
            path = args.out / f"fig_ablation_sensitivity.{ext}"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
