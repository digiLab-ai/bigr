"""fig_inner_deviation.py — Figure 2: inner-rollout deviation and the σ sweep (Sudoku-Extreme).

Three-panel mosaic when --sigma-guided-csv is supplied (the paper figure,
saved as fig_inner_deviation_sudoku by scripts/regenerate_figures.sh):
  accuracy  (left, full height): terminal exact-solve rate vs σ for unguided
            and guided inference at S=4 and S=16, plus the TRM baseline.
  survival  (top right): Pr(τ_r > M) vs σ for each tube radius α.
  stability (bottom right): E[||e_m||² 1_{τ_r>m}] / r² vs inner step m for one
            tube radius, one curve per σ, each with its analytical bound (dotted).
Without the CSV only the stability and survival panels are drawn, side by side.

python scripts/figs/fig_inner_deviation.py --run-dir assets/runs/sudoku/inner_deviation_k5 \\
    --sigma-guided-csv assets/runs/sudoku/sigma_guided_results_k5/sigma_guided_results.csv
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

from bigr.plots.kfold import (
    has_summary_schema,
    normalise_summary_columns,
    num_folds_from_run,
)


def _normalise_condition(value: object) -> str:
    value_str = str(value)
    if value_str in {"unguided", "guided", "resampled"}:
        return value_str
    if value_str.lower() in {"0", "0.0", "false"}:
        return "unguided"
    return "guided"


def _load_sigma_table(csv_path: Path) -> pd.DataFrame:
    """Sigma-sweep table with a `condition` column; expands the k-fold summary schema with CIs."""
    df = pd.read_csv(csv_path)
    if has_summary_schema(df):
        df = normalise_summary_columns(df, num_folds=num_folds_from_run(csv_path.parent))
    df["condition"] = df["guided"].map(_normalise_condition)
    return df

# Accuracy-panel colours.
C_UNGUIDED = "#c0392b"
C_GUIDED = "#4c78a8"

PUB_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
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

# Colours for the σ curves in the stability panel.
SIGMA_COLORS: dict[float, str] = {
    0.20: "#17becf",
    0.25: "#e05c2a",
    0.30: "#17becf",
    0.50: "#9467bd",
    0.75: "#9467bd",
    0.80: "#9467bd",
    1.00: "#7f7f7f",
}

# Colours and markers for the α curves in the survival panel — fading blues, α=0.5 darkest.
ALPHA_COLORS: dict[float, str] = {
    0.5: "#08519c",
    1.0: "#4292c6",
    1.5: "#9ecae1",
    2.0: "#c6dbef",
}
ALPHA_MARKERS: dict[float, str] = {0.5: "o", 1.0: "s", 1.5: "^", 2.0: "D"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot inner-rollout deviation (Figure 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "runs" / "sudoku" / "inner_deviation_k5",
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "figures")
    p.add_argument(
        "--left-alpha",
        type=float,
        default=1.5,
        help="Tube alpha (radius multiple) to show in the in-tube stability panel.",
    )
    p.add_argument(
        "--extra-run-dirs",
        type=Path,
        nargs="+",
        default=[],
        help="Additional run directories whose (α, σ) rows are concatenated onto the main run.",
    )
    p.add_argument(
        "--sigma-guided-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to the sigma-guided sweep CSV "
            "(e.g. assets/runs/sudoku/sigma_guided_results_k5/sigma_guided_results.csv). "
            "When supplied, a third panel shows terminal MAP accuracy vs σ."
        ),
    )
    p.add_argument(
        "--right-panel-exclude-alphas",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Override the default tube_alpha exclusion set for the survival "
            "panel. Pass an empty list (`--right-panel-exclude-alphas`) to show "
            "every available alpha."
        ),
    )
    p.add_argument(
        "--left-ylim",
        type=float,
        default=0.4,
        help="Upper y-limit for the in-tube stability panel.",
    )
    p.add_argument(
        "--figure-name",
        type=str,
        default="fig_inner_deviation",
        help="Basename for the saved figure (without extension).",
    )
    p.add_argument(
        "--accuracy-panel-title",
        type=str,
        default="Sudoku-Extreme test-set accuracy",
        help="Title for the sigma-accuracy panel.",
    )
    return p.parse_args()


SIGMA_DEFAULT = 0.3  # paper operating point — marked in the survival and accuracy panels
# The σ and α subsets drawn are chosen for legibility; the run directories hold more.
LEFT_PANEL_SIGMAS_EXCLUDED = {0.10, 0.80}
PLOT_SIGMAS_EXCLUDED: set[float] = {0.20}
RIGHT_PANEL_ALPHAS_EXCLUDED = {1.25, 2.0}


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


def _condition_colour(condition: str) -> str:
    return C_UNGUIDED if condition == "unguided" else C_GUIDED


def _particle_style(particles: int) -> dict:
    """Linestyle per particle count S; markers are uniform small dots."""
    if particles == 4:
        return {"linestyle": (0, (6, 2.5)), "marker": "o"}  # long-dashed
    if particles == 16:
        return {"linestyle": (0, (1.2, 1.5)), "marker": "o"}  # dotted
    return {"linestyle": "-", "marker": "o"}


TRM_BASELINE_COLOR = "#444444"


def _draw_sigma_accuracy_panel(ax: plt.Axes, csv_path: Path) -> None:
    """Render the terminal MAP-accuracy vs σ panel (unguided + guided-fixed only).

    The caller sets the σ ticks and limits so they match the survival panel.
    """
    df = _load_sigma_table(csv_path)
    conditions = ["unguided", "guided"]

    for particles in (16, 4):
        pstyle = _particle_style(particles)
        style = {
            **pstyle,
            "markersize": 2.5,
            "linewidth": 1.2,
            "alpha": 0.9,
        }
        ci_alpha = 0.10 if particles == 16 else 0.07
        for condition in conditions:
            sub = df[(df["S"] == particles) & (df["condition"] == condition)].sort_values("sigma")
            if sub.empty:
                continue
            x = sub["sigma"].to_numpy()
            y = sub["final_acc"].to_numpy()
            colour = _condition_colour(condition)
            if "final_acc_lo" in sub.columns and "final_acc_hi" in sub.columns:
                ax.fill_between(
                    x,
                    np.clip(sub["final_acc_lo"].to_numpy(), 0.0, 1.0),
                    np.clip(sub["final_acc_hi"].to_numpy(), 0.0, 1.0),
                    color=colour,
                    alpha=ci_alpha,
                    linewidth=0,
                )
            ax.plot(x, y, color=colour, **style)

    det_rows = df[np.isclose(df["sigma"], 0.0)]
    if not det_rows.empty:
        ax.axhline(
            float(det_rows["final_acc"].iloc[0]),
            color=TRM_BASELINE_COLOR,
            linewidth=1.0,
            linestyle="-",
            alpha=0.85,
        )

    ax.set_xlabel(r"Noise scale $\sigma$")
    ax.set_ylabel("Exact-solve rate")
    ax.set_ylim(0.2, 1.0)

    ls_s4 = (0, (6, 2.5))
    ls_s16 = (0, (1.2, 1.5))
    legend_handles = [
        Line2D([0], [0], color=C_GUIDED, linestyle=ls_s16, linewidth=1.4,
               label=r"Guided ($S=16$)"),
        Line2D([0], [0], color=C_GUIDED, linestyle=ls_s4, linewidth=1.4,
               label=r"Guided ($S=4$)"),
        Line2D([0], [0], color=C_UNGUIDED, linestyle=ls_s16, linewidth=1.4,
               label=r"Unguided ($S=16$)"),
        Line2D([0], [0], color=C_UNGUIDED, linestyle=ls_s4, linewidth=1.4,
               label=r"Unguided ($S=4$)"),
        Line2D([0], [0], color=TRM_BASELINE_COLOR, linestyle="-", linewidth=1.0,
               alpha=0.85, label="TRM baseline"),
    ]
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(0.08, 0.35),
              ncol=1, fontsize=6.5,
              handlelength=1.8, handletextpad=0.4,
              borderpad=0.3, framealpha=0.92)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    masked_df, surv_df = _load_inner_tables(args.run_dir)

    for extra_dir in args.extra_run_dirs:
        extra_masked, extra_surv = _load_inner_tables(extra_dir)
        masked_df = pd.concat([masked_df, extra_masked], ignore_index=True)
        surv_df = pd.concat([surv_df, extra_surv], ignore_index=True)

    sigmas = sorted(masked_df["sigma"].unique())
    plot_sigmas = [
        sigma for sigma in sigmas if round(float(sigma), 2) not in PLOT_SIGMAS_EXCLUDED
    ]
    alphas = sorted(surv_df["tube_alpha"].unique())

    if args.right_panel_exclude_alphas is None:
        right_panel_alphas_excluded = RIGHT_PANEL_ALPHAS_EXCLUDED
    else:
        right_panel_alphas_excluded = {round(a, 2) for a in args.right_panel_exclude_alphas}

    # ── Select tube for the stability panel ───────────────────────────────────
    available_alphas = sorted(masked_df["tube_alpha"].unique())
    left_alpha = args.left_alpha
    if left_alpha not in available_alphas:
        left_alpha = available_alphas[0]

    include_sigma_panel = args.sigma_guided_csv is not None
    if include_sigma_panel and not args.sigma_guided_csv.exists():
        raise FileNotFoundError(f"--sigma-guided-csv not found: {args.sigma_guided_csv}")

    with plt.rc_context(PUB_RC):
        if include_sigma_panel:
            fig, ax_dict = plt.subplot_mosaic(
                [["acc", "surv"],
                 ["acc", "left"]],
                figsize=(7.9, 3.2),
                width_ratios=[1.0, 1.15],
                layout="constrained",
            )
            fig.get_layout_engine().set(wspace=0.03, hspace=0.04)
            ax_left = ax_dict["left"]
            ax_surv = ax_dict["surv"]
            ax_acc = ax_dict["acc"]
        else:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(7.4, 2.75),
                gridspec_kw={"width_ratios": [1.15, 1.0]},
                layout="constrained",
            )
            ax_left = axes[0]
            ax_surv = axes[1]
            ax_acc = None

        # ── Stability panel: E[||e_m||² 1_{τ>m}] / r²  vs inner step m ──────
        ax = ax_left
        left_df = masked_df[masked_df["tube_alpha"] == left_alpha]
        r_sq = left_df["tube_radius"].iloc[0] ** 2
        plotted_left_series: list[tuple[float, pd.DataFrame]] = []
        for sigma in plot_sigmas:
            if round(float(sigma), 2) in LEFT_PANEL_SIGMAS_EXCLUDED:
                continue
            sub = left_df[left_df["sigma"] == sigma].sort_values("inner_step")
            if sub["survival_prob"].iloc[0] == 0.0:
                continue  # all rollouts exit immediately — skip
            plotted_left_series.append((sigma, sub))

        for sigma, sub in plotted_left_series:
            ms = sub["inner_step"].values
            surv = sub["survival_prob"].values
            in_tube = surv > 0.01
            devs = sub["mean_masked_dev"].values / r_sq
            # Mask bound to NaN where no particles remain in the tube.
            bound = np.where(in_tube, sub["partial_bound"].values / r_sq, np.nan)
            col = _sigma_colour(sigma)
            # Shade the headroom between the empirical curve and the bound.
            ax.fill_between(ms, devs, bound, color=col, alpha=0.18, linewidth=0)
            ax.plot(
                ms, devs, "-o", color=col, markersize=5.5, label=rf"$\sigma={sigma}$"
            )
            ax.plot(ms, bound, ":", color=col, linewidth=1.1, alpha=0.85)

        # Tube boundary reference and bound style dummy entry
        ax.plot([], [], "k:", linewidth=1.1, alpha=0.7, label="Theory")

        ax.set_xlabel(r"Inner step $m$")
        ax.set_ylabel(r"$\mathbb{E}[\|e_m\|^2\,\mathbf{1}_{\{\tau_r>m\}}]\;/\;r^2$")
        ax.set_title("In-tube stability bound", pad=6, fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            ncol=len(handles),
            loc="upper center",
            handlelength=1.6,
            columnspacing=1.0,
            borderpad=0.3,
            framealpha=0.92,
        )
        ax.set_xlim(left=0)
        ax.set_ylim(0, args.left_ylim)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # ── Survival panel: Pr(τ_r > M) vs σ ─────────────────────────────────
        ax = ax_surv
        # Build r₀ from the first entry in survival data (r = alpha * r0, so r0 = r/alpha)
        first_row = surv_df[surv_df["tube_alpha"] == alphas[0]].iloc[0]
        r0 = first_row["tube_radius"] / first_row["tube_alpha"]

        for alpha in alphas:
            if round(float(alpha), 2) in right_panel_alphas_excluded:
                continue
            sub = surv_df[surv_df["tube_alpha"] == alpha].sort_values("sigma")
            sub = sub[
                ~sub["sigma"].round(2).isin(
                    {round(sigma, 2) for sigma in PLOT_SIGMAS_EXCLUDED}
                )
            ]
            col = _alpha_colour(alpha)
            mk = ALPHA_MARKERS.get(round(alpha, 1), "o")
            r_val = alpha * r0
            p = sub["survival_prob_at_M"].values
            ax.plot(
                sub["sigma"],
                p,
                linestyle="-",
                marker=mk,
                color=col,
                markersize=4,
                label=rf"$\alpha={alpha:.1f}$  ($r={r_val:.0f}$)",
            )
            if "survival_prob_at_M_lo" in sub.columns and "survival_prob_at_M_hi" in sub.columns:
                ax.fill_between(
                    sub["sigma"].values,
                    np.clip(sub["survival_prob_at_M_lo"].values, 0.0, 1.0),
                    np.clip(sub["survival_prob_at_M_hi"].values, 0.0, 1.0),
                    color=col,
                    alpha=0.18,
                    linewidth=0,
                )

        # Mark the paper's σ operating point (dashed grey).
        op_line_kw = dict(color="#888888", linewidth=0.9, linestyle="--", zorder=3)
        ax.axvline(SIGMA_DEFAULT, **op_line_kw)

        ax.set_ylabel(r"$\Pr(\tau_r > M)$")
        ax.set_title("Tube Exit Probability", pad=6, fontweight="bold")
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.38),
                  fontsize=7.5, handlelength=1.6, borderpad=0.3,
                  framealpha=0.92)
        ax.set_xlim(left=0, right=max(plot_sigmas) * 1.05)
        ax.set_ylim(-0.03, 1.08)
        ax.set_xticks(plot_sigmas)
        ax.set_xlabel(r"Noise scale $\sigma$")

        # ── Accuracy panel (optional): terminal MAP accuracy vs σ ────────────
        if include_sigma_panel:
            _draw_sigma_accuracy_panel(ax_acc, args.sigma_guided_csv)
            ax_acc.set_title(args.accuracy_panel_title, pad=6, fontweight="bold")
            # Same σ ticks and limits as the survival panel.
            ax_acc.set_xticks(plot_sigmas)
            ax_acc.set_xlim(left=0, right=max(plot_sigmas) * 1.05)
            ax_acc.axvline(SIGMA_DEFAULT, **op_line_kw)

        out_pdf = args.out / f"{args.figure_name}.pdf"
        out_png = args.out / f"{args.figure_name}.png"
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
