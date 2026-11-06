"""fig_guide_alignment.py — Figure 3: guide-value alignment on Sudoku-Extreme and Maze-Hard.

Mosaic [[A, B], [C, C]]. A, B: split violins of raw Q-head logits at selected
outer steps, A_n (deterministic continuation solves) vs A_n^c, Sudoku left and
Maze right. C: per outer step, the Q-spread sup_{t∈[0,β_op]} std_s[q_θ] (left
axis, log) and the class-conditional gap Δ_n = E[q_θ|A_n] − E[q_θ|A_n^c] (right
axis); grey bands mark spreads too small for the tilt to move the solve
probability by the stated pp at each benchmark's β.

q_θ = log σ(q_raw) is the space in which the filter tilts (log_w += β·q_θ); the sup
is a discrete max over a 101-point t grid after reweighting the β₀ = 0.25 cloud to π_{n,t}.

Reads  <dir>/fold_*/continuation_labels.npz (scripts/exps/run_continuation_labels.py).
Writes <out>/fig_guide_alignment.{pdf,png}; --out defaults to assets/figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bigr.plots.kfold import fold_dirs, is_kfold_run

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while not (REPO_ROOT / "src").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent

# --- benchmark colours ------------------------------------------------------
C_SUDOKU   = "#1f77b4"   # blue   — Sudoku Extreme
C_MAZE     = "#9467bd"   # purple — Maze-Hard
C_THRESH   = "#7a7a7a"   # grey   — threshold guide line

# β_op is the operating-point upper limit of the sup in eq.(20); it sets the
# right-axis pp envelope and the std-floor for the necessary-condition line.
BETA_OP      = 1.0
# 2-pp gap threshold (§4.2 / eq.(20)). The threshold line is at
# std̂_s = CRITERION_PP / (50·β_op); at β_op = 1 this is 0.04.
CRITERION_PP = 2.0
# Second threshold band, for the Sudoku-Extreme operating point β = 0.25: a
# spread of 1.0 bounds the solve-probability change by 100·(β/2)·1.0 = 12.5 pp.
BETA_SUDOKU_OP = 0.25
STD_CRITERION_SUDOKU = 1.0
VIOLIN_STEPS = [3, 12, 24, 48]

# Particle clouds are drawn from a guided filter run at β₀ = BETA_PROPOSAL
# (see scripts/exps/run_continuation_labels.py: --beta default 0.25), so the
# saved cloud at step n approximates π_{n, β₀}. Importance-reweighting by
# exp((t − β₀)·q_θ) maps it to π_{n,t} for any t, used in `_tilted_std`.
BETA_PROPOSAL = 0.25
T_GRID_SUP    = np.linspace(0.0, BETA_OP, 101)

PUB_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
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
        description="Guide-value alignment figure (Figure 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sudoku-cont-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "runs" / "sudoku" / "continuation_labels_k5",
        help="Sudoku k-fold dir with fold_*/continuation_labels.npz",
    )
    p.add_argument(
        "--maze-cont-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "runs" / "maze" / "continuation_labels_maze_hard_k5",
        help="Maze k-fold dir with fold_*/continuation_labels.npz",
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "figures")
    p.add_argument(
        "--beta-op",
        type=float,
        default=BETA_OP,
        help="Operating-point β: upper limit of sup_t in eq.(20) and right-axis pp scale.",
    )
    p.add_argument(
        "--criterion-pp",
        type=float,
        default=CRITERION_PP,
        help="Necessary-condition threshold for |p_n(β_op) − p_n(0)|, in percentage points.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _continuation_fold_paths(run_dir: Path) -> list[Path]:
    """Return ordered list of fold paths each containing continuation_labels.npz."""
    if not run_dir.exists():
        return []
    if is_kfold_run(run_dir):
        return [p for p in fold_dirs(run_dir) if (p / "continuation_labels.npz").exists()]
    if (run_dir / "continuation_labels.npz").exists():
        return [run_dir]
    return []


def _tilted_std(q: np.ndarray, t: float, beta0: float = BETA_PROPOSAL) -> np.ndarray:
    """IS-reweighted std of q_θ under π_{n,t} from a cloud drawn at π_{n,β₀}.

    Weights ``W^{(s)} ∝ exp((t − β₀)·q^{(s)})`` are normalised across the
    trailing particle axis; for ``q`` of shape ``(..., S)`` the result has
    shape ``(...,)``.
    """
    log_w = (t - beta0) * q
    log_w = log_w - log_w.max(axis=-1, keepdims=True)  # softmax stability
    w = np.exp(log_w)
    w = w / w.sum(axis=-1, keepdims=True)
    mu = (w * q).sum(axis=-1)
    var = (w * (q - mu[..., None]) ** 2).sum(axis=-1)
    return np.sqrt(np.clip(var, 0.0, None))


def _curves_from_continuation(run_dir: Path) -> Optional[dict]:
    """Per-step Q-spread and Δ_n curves from a continuation run dir (None if no folds).

    Statistics are computed in the guide space q_θ = log σ(q_raw), in which the
    filter applies its tilt. Returns ``steps``, ``sup_t_std`` (discrete-grid
    supremum over t ∈ [0, β_op] of the mean-over-puzzles IS-reweighted std) and
    ``gap`` (post-hoc Δ̂_n on the cloud as sampled).
    """
    fold_paths = _continuation_fold_paths(run_dir)
    if not fold_paths:
        return None

    q_per_fold: list[np.ndarray] = []
    g_per_fold: list[np.ndarray] = []
    steps_ref: Optional[np.ndarray] = None
    for fold in fold_paths:
        npz = np.load(fold / "continuation_labels.npz")
        steps = np.asarray(npz["steps"])
        if steps_ref is None:
            steps_ref = steps
        elif not np.array_equal(steps, steps_ref):
            raise ValueError(f"step grid mismatch at {fold}")
        if "q_score" in npz.files:
            q_per_fold.append(np.asarray(npz["q_score"]))
        else:
            q_raw = np.asarray(npz["q_raw"])
            q_per_fold.append(-np.logaddexp(0.0, -q_raw))
        g_per_fold.append(np.asarray(npz["g_labels"]).astype(bool))

    q_pooled = np.concatenate(q_per_fold, axis=1)  # (K, B_total, S)
    g_pooled = np.concatenate(g_per_fold, axis=1)
    K = q_pooled.shape[0]

    gap = np.full(K, np.nan)
    sup_t_std = np.full(K, np.nan)
    for ki in range(K):
        q_flat = q_pooled[ki].ravel()
        g_flat = g_pooled[ki].ravel()
        if g_flat.any() and (~g_flat).any():
            gap[ki] = float(q_flat[g_flat].mean() - q_flat[~g_flat].mean())

        # Sup over t ∈ [0, β_OP] of mean-over-puzzles of std_{π_{n,t}}.
        std_curve_t = np.array(
            [float(_tilted_std(q_pooled[ki], float(t)).mean()) for t in T_GRID_SUP]
        )
        sup_t_std[ki] = float(std_curve_t.max())

    return dict(steps=steps_ref, sup_t_std=sup_t_std, gap=gap)


def _violin_data_from_continuation(
    run_dir: Path, violin_steps: list[int]
) -> Optional[dict[int, dict[str, np.ndarray]]]:
    """Pooled (over folds) raw-logit arrays per violin step, split by A_n / A_n^c."""
    fold_paths = _continuation_fold_paths(run_dir)
    if not fold_paths:
        return None

    pooled: dict[int, dict[str, list[np.ndarray]]] = {
        step: {"success": [], "failure": []} for step in violin_steps
    }
    for fold in fold_paths:
        npz = np.load(fold / "continuation_labels.npz")
        steps = np.asarray(npz["steps"])
        q = np.asarray(npz["q_raw"])
        g = np.asarray(npz["g_labels"]).astype(bool)
        for step in violin_steps:
            ki = int(np.argmin(np.abs(steps - step)))
            q_flat = q[ki].ravel()
            g_flat = g[ki].ravel()
            pooled[step]["success"].append(q_flat[g_flat])
            pooled[step]["failure"].append(q_flat[~g_flat])

    return {
        step: {
            grp: np.concatenate(parts) if parts else np.array([], dtype=float)
            for grp, parts in pooled[step].items()
        }
        for step in violin_steps
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _select_steps(steps: np.ndarray, desired: list[int]) -> list[int]:
    selected: list[int] = []
    for d in desired:
        idx = int(np.argmin(np.abs(steps - d)))
        if int(steps[idx]) not in selected:
            selected.append(int(steps[idx]))
    return sorted(selected)


def _draw_violin_panel(
    ax: plt.Axes,
    pooled: dict[int, dict[str, np.ndarray]],
    violin_steps: list[int],
    *,
    title: str,
    show_ylabel: bool = True,
    show_legend: bool = True,
    legend_loc: str = "center",
    success_color: str = C_SUDOKU,
    failure_edge_color: str = C_SUDOKU,
) -> None:
    """Split-violin of guide logits, A_n vs A_n^c, per outer step.

    ``success_color`` selects the colour for the A_n (success) violins so the
    panel can match its benchmark (Sudoku blue, Maze purple).  The A_n^c
    (failure) violins are drawn as white-filled bodies with ``failure_edge_color``
    outlines, so the eye reads the success distribution as the foreground.
    ``legend_loc`` is a matplotlib legend location.
    """
    pos_s, pos_f, x_ticks, x_labels = [], [], [], []
    data_s, data_f = [], []
    for xi, step in enumerate(violin_steps):
        data_s.append(pooled[step]["success"])
        data_f.append(pooled[step]["failure"])
        pos_s.append(xi * 2.5 - 0.42)
        pos_f.append(xi * 2.5 + 0.42)
        x_ticks.append(xi * 2.5)
        x_labels.append(str(step))

    def _violin(data_list, positions, fill_color, edge_color, alpha, edge_lw):
        non_empty = [
            (np.asarray(d), p) for d, p in zip(data_list, positions) if len(d) > 0
        ]
        if not non_empty:
            return
        vals, pos = zip(*non_empty)
        parts = ax.violinplot(
            list(vals), positions=list(pos), widths=0.75,
            showmedians=True, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(fill_color)
            pc.set_edgecolor(edge_color)
            pc.set_linewidth(edge_lw)
            pc.set_alpha(alpha)
        parts["cmedians"].set_color(edge_color)
        parts["cmedians"].set_linewidth(2.0)

    _violin(data_s, pos_s, fill_color=success_color, edge_color=success_color,
            alpha=0.55, edge_lw=0.6)
    _violin(data_f, pos_f, fill_color="white", edge_color=failure_edge_color,
            alpha=1.0, edge_lw=1.4)

    legend_handles = [
        Patch(facecolor=success_color, edgecolor=success_color, alpha=0.55,
              label=r"$A_n$ (solves)"),
        Patch(facecolor="white", edgecolor=failure_edge_color, linewidth=1.4,
              label=r"$A_n^c$ (fails)"),
    ]

    ax.axhline(0, color="#636363", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"Outer step, $n$")
    if show_ylabel:
        # Bold only the non-math words; commas and math expressions stay
        # regular weight by wrapping the words in `\mathbf{}` rather than
        # passing `fontweight="bold"` on the whole label.
        ax.set_ylabel(
            r"$\mathbf{Q\!-\!head\ logits},\ \tilde q_\theta(x, h_{n}^{(s)})$",
        )
    ax.set_title(title, pad=4, fontweight="bold")

    if show_legend:
        ax.legend(handles=legend_handles, loc=legend_loc, framealpha=0.85)


# Log-axis lower bound. At β_op = 1 the eq.(20) envelope is (β_op/2)·std̂_s,
# so 5e-3 corresponds to a 0.25-pp envelope — well below the 2-pp criterion.
STD_AXIS_FLOOR = 5e-3


def _draw_right_panel(
    ax: plt.Axes,
    sudoku: Optional[dict],
    maze: Optional[dict],
    *,
    beta_op: float,
    criterion_pp: float,
) -> None:
    """Bottom panel: Q-spread (left, log) and Δ_n (right, linear).

    Sudoku blue / Maze purple; std solid+circle, Δ_n dashed+square. Log std
    axis lets Maze's collapse and Sudoku's saturated value share scale. The
    grey band/line marks the std̂_s threshold below which eq.(20) cannot
    induce a ``criterion_pp``-pp gap on the success set at ``beta_op``; a
    lighter band does the same for the Sudoku operating point (module constants).
    """
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    factor_pp = 100.0 * 0.5 * beta_op
    std_criterion = criterion_pp / factor_pp
    threshold_alpha = 0.18

    criterion_pp_sudoku = 100.0 * 0.5 * BETA_SUDOKU_OP * STD_CRITERION_SUDOKU  # 12.5
    threshold_alpha_high = 0.07              # lighter than threshold_alpha=0.18

    ax.axhspan(
        STD_AXIS_FLOOR, std_criterion,
        color=C_THRESH, alpha=threshold_alpha, lw=0, zorder=0,
    )
    ax.axhspan(
        std_criterion, STD_CRITERION_SUDOKU,
        color=C_THRESH, alpha=threshold_alpha_high, lw=0, zorder=0,
    )
    ax.axhline(
        std_criterion, color=C_THRESH, lw=1.2, linestyle="-", alpha=0.9, zorder=1,
    )
    ax.axhline(
        STD_CRITERION_SUDOKU, color=C_THRESH, lw=1.2, linestyle="-", alpha=0.9, zorder=1,
    )

    benchmarks = [
        ("Sudoku Extreme", sudoku, C_SUDOKU),
        ("Maze-Hard",      maze,   C_MAZE),
    ]

    std_max = 0.0
    delta_max = 0.0
    for _label, res, col in benchmarks:
        if res is None:
            continue
        std_curve = res["sup_t_std"]
        ax.plot(
            res["steps"], std_curve,
            color=col, linestyle="-", linewidth=1.8,
            marker="o", markersize=4.0,
            zorder=4,
        )
        std_max = max(std_max, float(np.nanmax(std_curve)))
        valid = np.isfinite(res["gap"])
        if valid.any():
            ax2.plot(
                res["steps"][valid], res["gap"][valid],
                color=col, linestyle="--", linewidth=1.8,
                marker="s", markersize=3.6,
                zorder=3,
            )
            delta_max = max(delta_max, float(np.nanmax(res["gap"][valid])))

    ax.set_yscale("log")
    ax.set_ylim(STD_AXIS_FLOOR, max(3.0, std_max * 1.15))
    ax2.set_ylim(0.0, max(1.0, delta_max * 1.10))

    ax.set_xlim(3, 48)
    ax.set_xticks([3, 12, 24, 36, 48])

    ax.set_xlabel(r"Outer step, $n$")
    ax.set_ylabel(
        r"$\mathbf{Q\!-\!spread},\ \widehat{\mathrm{std}}_s[\,q_\theta(h_{n}^{(s)})\,]$",
        labelpad=10,
    )
    ax2.set_ylabel(
        r"$\mathbf{Class\!-\!conditional\ gap},\ \Delta_n$",
        labelpad=10,
    )
    ax.set_title("Two guide-alignment diagnostics", pad=4, fontweight="bold")

    # Three-column layout (column-major, nrows=2). Both threshold patches stack
    # in col 3; explicit RGBA edge keeps the borders visible while the fills
    # stay light.
    edge_rgba = mcolors.to_rgba(C_THRESH, alpha=0.9)
    threshold_patch_low = Patch(
        facecolor=mcolors.to_rgba(C_THRESH, alpha=threshold_alpha),
        edgecolor=edge_rgba, linewidth=0.9,
        label=rf"$\widehat{{\mathrm{{std}}}}_s[q_\theta]\leq{std_criterion:.2f}$"
              rf"$\;\Rightarrow\;|\Delta p_n|\leq {criterion_pp:g}\%\,"
              rf"(\beta\!=\!{beta_op:.1f},\,\mathrm{{Maze}})$",
    )
    threshold_patch_high = Patch(
        facecolor=mcolors.to_rgba(C_THRESH, alpha=threshold_alpha_high),
        edgecolor=edge_rgba, linewidth=0.9,
        label=rf"$\widehat{{\mathrm{{std}}}}_s[q_\theta]\leq{STD_CRITERION_SUDOKU:.2f}$"
              rf"$\;\Rightarrow\;|\Delta p_n|\leq {criterion_pp_sudoku:g}\%\,"
              rf"(\beta\!=\!{BETA_SUDOKU_OP:g},\,\mathrm{{Sudoku}})$",
    )
    legend_handles = [
        Patch(facecolor=C_SUDOKU, edgecolor=C_SUDOKU, label="Sudoku Extreme"),
        Patch(facecolor=C_MAZE,   edgecolor=C_MAZE,   label="Maze-Hard"),
        Line2D([0], [0], color="#3a3a3a", linestyle="--", linewidth=1.8,
               label=r"$\Delta_n$ (class-conditional gap)"),
        Line2D([0], [0], color="#3a3a3a", linestyle="-",  linewidth=1.8,
               label=r"$\sup\,\widehat{\mathrm{std}}_s[q_\theta]$ ($Q$-spread)"),
        threshold_patch_low,
        threshold_patch_high,
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.68),
        ncol=3,
        frameon=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Determine the violin step grid from the sudoku run (assumed canonical).
    sudoku_paths = _continuation_fold_paths(args.sudoku_cont_dir)
    if not sudoku_paths:
        raise FileNotFoundError(
            f"No continuation_labels.npz under {args.sudoku_cont_dir} — "
            f"run scripts/exps/run_continuation_labels.py first."
        )
    first_npz = np.load(sudoku_paths[0] / "continuation_labels.npz")
    violin_steps = _select_steps(np.asarray(first_npz["steps"]), VIOLIN_STEPS)

    sudoku_pooled = _violin_data_from_continuation(args.sudoku_cont_dir, violin_steps)
    maze_pooled = _violin_data_from_continuation(args.maze_cont_dir, violin_steps)
    sudoku_curves = _curves_from_continuation(args.sudoku_cont_dir)
    maze_curves = _curves_from_continuation(args.maze_cont_dir)

    with plt.rc_context(PUB_RC):
        fig, ax_dict = plt.subplot_mosaic(
            [["violin_s", "violin_m"],
             ["right",    "right"]],
            figsize=(11.0, 6.0),
            gridspec_kw={
                "width_ratios": [1.0, 1.0],
                "height_ratios": [1.0, 1.0],
                "wspace": 0.02,
            },
            layout="constrained",
        )

        if sudoku_pooled is not None:
            _draw_violin_panel(
                ax_dict["violin_s"], sudoku_pooled, violin_steps,
                title="Sudoku-Extreme", show_ylabel=True,
                show_legend=True, legend_loc="center",
                success_color=C_SUDOKU,
                failure_edge_color=C_SUDOKU,
            )
        if maze_pooled is not None:
            _draw_violin_panel(
                ax_dict["violin_m"], maze_pooled, violin_steps,
                title="Maze-Hard", show_ylabel=False,
                show_legend=True, legend_loc="lower center",
                success_color=C_MAZE,
                failure_edge_color=C_MAZE,
            )
        else:
            ax_dict["violin_m"].text(
                0.5, 0.5, f"no continuation data at\n{args.maze_cont_dir}",
                ha="center", va="center", color="#6b7280",
                transform=ax_dict["violin_m"].transAxes,
            )

        # Share y-axis between the violin panels so the dashed y=0 line in
        # each appears at the same height; remove the left axis on the right
        # panel so the two dashed lines visually extend through to meet.
        if sudoku_pooled is not None and maze_pooled is not None:
            y0 = min(ax_dict["violin_s"].get_ylim()[0],
                     ax_dict["violin_m"].get_ylim()[0])
            y1 = max(ax_dict["violin_s"].get_ylim()[1],
                     ax_dict["violin_m"].get_ylim()[1])
            ax_dict["violin_s"].set_ylim(y0, y1)
            ax_dict["violin_m"].set_ylim(y0, y1)
            ax_dict["violin_m"].sharey(ax_dict["violin_s"])
            ax_dict["violin_m"].spines["left"].set_visible(False)
            ax_dict["violin_m"].tick_params(
                axis="y", which="both", left=False, labelleft=False,
            )

        _draw_right_panel(
            ax_dict["right"], sudoku_curves, maze_curves,
            beta_op=args.beta_op, criterion_pp=args.criterion_pp,
        )

        for ext in ("pdf", "png"):
            out_path = args.out / f"fig_guide_alignment.{ext}"
            fig.savefig(out_path, bbox_inches="tight")
            print(f"Saved {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
