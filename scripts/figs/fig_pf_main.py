"""Figure 1: exact-solve rate against outer step on Sudoku-Extreme.

Two panels sharing a y-axis: the Test split (left) and the deterministic-failure
split (right). Each shows the deterministic TRM, a single unguided rollout, the
guided highest-weight particle, the oracle (any correct particle) and the
Q-ranked particle trajectories, with fold-seed confidence bands.

Reads the seed-fold run directories under assets/runs/sudoku/ (``--quick`` in
scripts/regenerate_figures.sh substitutes the single-seed ``pf*_k5`` runs) and
writes assets/figures/fig_main_result.{pdf,png}.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
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
    COLOR_ORACLE,
    COLOR_PARTICLE,
    MAIN_RESULT_RC,
    save_figure,
)
from bigr.plots.kfold import (  # noqa: E402
    aggregate_trm_depth_profiles,
    normalise_summary_columns,
    num_folds_from_run,
)


# ----------------------------------------------------------------------
# Visual constants
# ----------------------------------------------------------------------

PLOT_COLOR_BASELINE = COLOR_BASELINE
PLOT_COLOR_GUIDED = COLOR_MAP
PLOT_COLOR_ORACLE = COLOR_ORACLE

LW_PARTICLE = 0.85
LW_MAIN = 1.8
BASELINE_MARKER_SIZE = 4.4
ORACLE_MARKER_SIZE = 5.2
MAX_BASELINE_MARKERS = 48

LABEL_BASELINE = "Deterministic TRM baseline"
LABEL_PARTICLES = "Weight-ranked particles"
LABEL_UNGUIDED = "Single rollout (unguided)"
LABEL_GUIDED = "Guided MAP (our solution)"
LABEL_ORACLE = "Oracle particles"

ANNOT_FONTSIZE = 13


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-panel main-result figure (test + DF recovery)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sudoku = REPO_ROOT / "assets" / "runs" / "sudoku"
    p.add_argument(
        "--guided-test-dir", type=Path,
        default=sudoku / "pf_seed_fold_test",
        help="Guided test-split run dir (5x5 by default; 5-fold also accepted).",
    )
    p.add_argument(
        "--unguided-test-dir", type=Path,
        default=sudoku / "pf_seed_fold_test_unguided",
    )
    p.add_argument(
        "--guided-df-dir", type=Path,
        default=sudoku / "pf_seed_fold_impossible",
    )
    p.add_argument(
        "--unguided-df-dir", type=Path,
        default=sudoku / "pf_seed_fold_impossible_unguided",
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "figures")
    p.add_argument("--particles", type=int, default=16, help="Number of Q-ranked particle trajectories drawn.")
    return p.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _has_ci(df: pd.DataFrame, column: str) -> bool:
    return f"{column}_lo" in df.columns and f"{column}_hi" in df.columns


def _fill_ci(ax, x, df, column, *, color, alpha, zorder):
    if not _has_ci(df, column):
        return
    ax.fill_between(
        x,
        df[f"{column}_lo"].to_numpy(),
        df[f"{column}_hi"].to_numpy(),
        color=color, alpha=alpha, linewidth=0, zorder=zorder,
    )


def _load_run(run_dir: Path):
    summary_step = run_dir / "step_metrics_summary.csv"
    if summary_step.exists():
        num_folds = num_folds_from_run(run_dir)
        step_df = normalise_summary_columns(pd.read_csv(summary_step), num_folds=num_folds)
        rank_path = run_dir / "rank_profiles_summary.csv"
        rank_df = (
            normalise_summary_columns(pd.read_csv(rank_path), num_folds=num_folds)
            if rank_path.exists() else None
        )
        trm_df = aggregate_trm_depth_profiles(run_dir)
        return step_df, rank_df, trm_df

    rank_path = run_dir / "rank_profiles.csv"
    rank_df = pd.read_csv(rank_path) if rank_path.exists() else None
    return (
        pd.read_csv(run_dir / "step_metrics.csv"),
        rank_df,
        pd.read_csv(run_dir / "trm_depth_profile.csv"),
    )


def _first_crossing(s_arr, v_arr, threshold):
    s = np.asarray(s_arr, dtype=float)
    v = np.asarray(v_arr, dtype=float)
    if s.size == 0:
        return None, None
    if v[0] >= threshold:
        return float(s[0]), float(v[0])
    for idx in range(1, len(s)):
        prev_s, curr_s = float(s[idx - 1]), float(s[idx])
        prev_v, curr_v = float(v[idx - 1]), float(v[idx])
        if curr_v < threshold:
            continue
        if prev_v == curr_v:
            return curr_s, curr_v
        frac = (threshold - prev_v) / (curr_v - prev_v)
        frac = float(np.clip(frac, 0.0, 1.0))
        step = prev_s + frac * (curr_s - prev_s)
        return step, float(threshold)
    return None, None


# ----------------------------------------------------------------------
# Per-panel plotting
# ----------------------------------------------------------------------


def _plot_particles(ax, rank_df, *, n_particles_cap):
    """Faint per-particle (Q-rank) trajectories."""
    if rank_df is None or rank_df.empty:
        return None
    rank_vals = sorted(rank_df["rank"].unique())
    n = min(len(rank_vals), n_particles_cap)
    palette = plt.cm.Blues(np.linspace(0.82, 0.22, n))
    for ri, rank in enumerate(rank_vals[:n]):
        sub = rank_df[rank_df["rank"] == rank].sort_values("step")
        ax.plot(
            sub["step"].values, sub["exact_rate"].values,
            color=palette[ri], linewidth=LW_PARTICLE, alpha=0.65, zorder=2,
        )
    return palette


def _plot_traces(ax, guided_df, unguided_df, trm_df):
    """Common trace overlay: TRM markers, unguided dashed, guided solid, oracle."""
    steps = guided_df["step"].to_numpy()
    guided_map = guided_df["exact_accuracy"].to_numpy()
    unguided_map = unguided_df["exact_accuracy"].to_numpy()
    oracle = guided_df["any_exact_rate"].to_numpy()

    # CI fills
    _fill_ci(ax, trm_df["step"].to_numpy(), trm_df, "exact_solved",
             color=PLOT_COLOR_BASELINE, alpha=0.10, zorder=1)
    _fill_ci(ax, steps, unguided_df, "exact_accuracy",
             color=COLOR_PARTICLE, alpha=0.10, zorder=2)
    _fill_ci(ax, steps, guided_df, "exact_accuracy",
             color=PLOT_COLOR_GUIDED, alpha=0.12, zorder=3)
    _fill_ci(ax, steps, guided_df, "any_exact_rate",
             color=PLOT_COLOR_ORACLE, alpha=0.10, zorder=3)

    # TRM baseline markers — sub-sample if dense
    trm_steps_all = trm_df["step"].to_numpy()
    trm_exact_all = trm_df["exact_solved"].to_numpy()
    n_marks = min(len(trm_steps_all), MAX_BASELINE_MARKERS)
    idx = np.unique(np.linspace(0, len(trm_steps_all) - 1, num=n_marks, dtype=int))
    baseline_handle, = ax.plot(
        trm_steps_all[idx], trm_exact_all[idx],
        color=PLOT_COLOR_BASELINE, linestyle="None", marker="o",
        markersize=BASELINE_MARKER_SIZE, label=LABEL_BASELINE, zorder=5,
    )

    unguided_handle, = ax.plot(
        steps, unguided_map, color=COLOR_PARTICLE,
        linewidth=LW_MAIN, linestyle="--", label=LABEL_UNGUIDED, zorder=4,
    )
    guided_handle, = ax.plot(
        steps, guided_map, color=PLOT_COLOR_GUIDED,
        linewidth=LW_MAIN, linestyle="-", label=LABEL_GUIDED, zorder=6,
    )
    oracle_handle, = ax.plot(
        steps, oracle, color=PLOT_COLOR_ORACLE,
        linestyle="None", marker="o", markersize=ORACLE_MARKER_SIZE,
        markevery=3, markerfacecolor="none",
        markeredgecolor=PLOT_COLOR_ORACLE, markeredgewidth=1.5,
        label=LABEL_ORACLE, zorder=5,
    )

    return {
        "steps": steps,
        "guided_map": guided_map,
        "unguided_map": unguided_map,
        "oracle": oracle,
        "baseline": baseline_handle,
        "unguided": unguided_handle,
        "guided": guided_handle,
        "oracle_h": oracle_handle,
    }


def _draw_test_panel(ax, guided_df, unguided_df, rank_df, trm_df, *, particles):
    palette = _plot_particles(ax, rank_df, n_particles_cap=particles)
    handles = _plot_traces(ax, guided_df, unguided_df, trm_df)
    final_trm = float(trm_df["exact_solved"].iloc[-1])

    # Reference dashes: terminal-TRM horizontal and TRM-match vertical.
    ax.axhline(
        final_trm, color=PLOT_COLOR_BASELINE,
        linestyle="--", linewidth=1.2, alpha=0.85, zorder=3,
    )
    map_cx, _ = _first_crossing(handles["steps"], handles["guided_map"], final_trm)
    if map_cx is not None:
        ax.axvline(
            map_cx, color=PLOT_COLOR_GUIDED,
            linestyle="--", linewidth=1.2, alpha=0.9, zorder=3,
        )

    particle_color = palette[0] if palette is not None else COLOR_PARTICLE
    particle_proxy = mlines.Line2D(
        [], [], color=particle_color, linewidth=LW_PARTICLE, alpha=0.65,
        label=LABEL_PARTICLES,
    )
    ax.legend(
        [handles["baseline"], particle_proxy, handles["unguided"],
         handles["guided"], handles["oracle_h"]],
        [LABEL_BASELINE, LABEL_PARTICLES, LABEL_UNGUIDED, LABEL_GUIDED, LABEL_ORACLE],
        loc="lower right", framealpha=0.92,
    )


def _draw_df_panel(ax, guided_df, unguided_df, rank_df, trm_df, *, particles):
    _plot_particles(ax, rank_df, n_particles_cap=particles)
    handles = _plot_traces(ax, guided_df, unguided_df, trm_df)
    steps = handles["steps"]
    max_step = int(steps[-1])

    # Exploration / Exploitation arrows (gap arrows between TRM, unguided, guided)
    deterministic = trm_df["exact_solved"].to_numpy()
    arrow_x = int(max_step * 0.58)
    trm_at_x = float(np.interp(arrow_x, trm_df["step"].values, deterministic))
    unguided_at_x = float(np.interp(arrow_x, steps, handles["unguided_map"]))
    guided_at_x = float(np.interp(arrow_x, steps, handles["guided_map"]))

    ax.annotate(
        "", xy=(arrow_x, trm_at_x), xytext=(arrow_x, unguided_at_x),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2), zorder=7,
    )
    ax.text(
        arrow_x + max_step * 0.015, 0.5 * (trm_at_x + unguided_at_x),
        "Exploration", fontsize=ANNOT_FONTSIZE, color="#555555",
        ha="left", va="center", fontweight="semibold",
    )

    ax.annotate(
        "", xy=(arrow_x, unguided_at_x), xytext=(arrow_x, guided_at_x),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2), zorder=7,
    )
    ax.text(
        arrow_x + max_step * 0.015, 0.5 * (unguided_at_x + guided_at_x),
        "Exploitation", fontsize=ANNOT_FONTSIZE, color="#555555",
        ha="left", va="center", fontweight="semibold",
    )


def _style_axis(ax, *, title, show_ylabel):
    ax.set_title(title, pad=14, fontweight="bold")
    ax.set_xlabel(r"Outer step, $n$")
    if show_ylabel:
        ax.set_ylabel("Exact-solve rate", fontweight="bold")
    ax.set_xlim(0.0, 48.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _print_summary(label, guided_df, unguided_df, trm_df=None):
    print(f"=== {label} ===")
    final_guided = float(guided_df["exact_accuracy"].iloc[-1])
    final_unguided = float(unguided_df["exact_accuracy"].iloc[-1])
    final_oracle = float(guided_df["any_exact_rate"].iloc[-1])
    print(f"  Guided   terminal: {final_guided:.4f}")
    print(f"  Unguided terminal: {final_unguided:.4f}")
    print(f"  Oracle   terminal: {final_oracle:.4f}")
    if trm_df is not None:
        final_trm = float(trm_df["exact_solved"].iloc[-1])
        delta_pp = (final_guided - final_trm) * 100
        match_step, _ = _first_crossing(
            guided_df["step"].to_numpy(),
            guided_df["exact_accuracy"].to_numpy(),
            final_trm,
        )
        print(f"  TRM      terminal: {final_trm:.4f}")
        if match_step is not None:
            print(f"  TRM match: step {math.ceil(match_step)}")
        print(f"  Gap:       {delta_pp:+.1f} pp")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Test guided dir:    {args.guided_test_dir}")
    print(f"Test unguided dir:  {args.unguided_test_dir}")
    print(f"DF   guided dir:    {args.guided_df_dir}")
    print(f"DF   unguided dir:  {args.unguided_df_dir}")

    guided_test, rank_test, trm_test = _load_run(args.guided_test_dir)
    unguided_test, _, _ = _load_run(args.unguided_test_dir)
    guided_df_, rank_df_, trm_df_ = _load_run(args.guided_df_dir)
    unguided_df_, _, _ = _load_run(args.unguided_df_dir)

    with plt.rc_context(MAIN_RESULT_RC):
        fig, axes = plt.subplots(
            1, 2,
            figsize=(11.0, 4.4),
            sharey=True,
            gridspec_kw={"wspace": 0.06},
            layout="constrained",
        )
        _draw_test_panel(
            axes[0], guided_test, unguided_test, rank_test, trm_test,
            particles=args.particles,
        )
        _draw_df_panel(
            axes[1], guided_df_, unguided_df_, rank_df_, trm_df_,
            particles=args.particles,
        )
        _style_axis(axes[0], title="Sudoku Extreme Test Set", show_ylabel=True)
        _style_axis(axes[1], title="Recovery of baseline failure cases", show_ylabel=False)

        # `sharey=True` hides the right panel's tick labels by default; turn
        # them back on so both panels show the percentage scale.
        axes[1].tick_params(axis="y", which="both", labelleft=True)
        for tick in axes[1].get_yticklabels():
            tick.set_visible(True)

        save_figure(fig, args.out / "fig_main_result.pdf", dpi=300)
        plt.close(fig)
    print(f"Saved: {args.out}/fig_main_result.pdf + .png")

    _print_summary("Test split", guided_test, unguided_test, trm_test)
    _print_summary("DF split",  guided_df_, unguided_df_, trm_df_)


if __name__ == "__main__":
    main()
