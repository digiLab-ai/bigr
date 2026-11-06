"""run_early_stopping — per-instance adaptive-halting simulation from saved q-traces (Table 14, compute side).

Uses per-particle Q-logit checkpoints (every 3rd outer step) saved with the headline
runs. Halting rule: stop at the first checkpointed step where the cloud's maximum
guide confidence max_s sigmoid(q_s) crosses a threshold tau.

Reports the stopping-time distribution and outer-step compute saved. The accuracy
join (was the selected particle correct at the stop step?) needs per-particle
correctness traces: run_trace_analyses.py computes it from the traces saved by
`run_pf.py --save-particle-trace` (the ea7.csv rows of Table 14). Marginal top-Q
accuracy per step is available in rank_profiles for context.

Outputs
-------
    assets/runs/analysis/early_stopping/stopping.csv
    assets/runs/analysis/early_stopping/REPORT.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_data import OUT_ROOT, RUNS

ARMS = {
    ("sudoku", "guided"): "sudoku/pf_k5",
    ("sudoku", "unguided"): "sudoku/pf_unguided_k5",
    ("maze", "guided"): "maze/pf_maze_hard_k5",
    ("maze", "unguided"): "maze/pf_maze_hard_unguided_k5",
}
THRESHOLDS = [0.90, 0.95, 0.99]
HORIZON = 48


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fold_qmax(fold_dir: Path) -> tuple[np.ndarray, list[int]]:
    """(B, T) max-confidence matrix over checkpointed steps, plus the step list."""
    steps = sorted(
        int(re.search(r"step(\d+)", p.name).group(1))
        for p in fold_dir.glob("ckpt_qlogits_step*.npy")
    )
    cols = [sigmoid(np.load(fold_dir / f"ckpt_qlogits_step{n}.npy")).max(axis=1) for n in steps]
    return np.stack(cols, axis=1), steps


def stop_steps(qmax: np.ndarray, steps: list[int], tau: float) -> np.ndarray:
    hit = qmax >= tau
    first = np.argmax(hit, axis=1)
    stopped = hit.any(axis=1)
    out = np.full(qmax.shape[0], HORIZON, dtype=float)
    out[stopped] = np.asarray(steps, dtype=float)[first[stopped]]
    return out


def main() -> None:
    out_dir = OUT_ROOT / "early_stopping"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (bench, arm), rel in ARMS.items():
        fold_dirs = sorted((RUNS / rel).glob("fold_*"))
        for fold_dir in tqdm(fold_dirs, desc=f"{bench}/{arm}"):
            if not list(fold_dir.glob("ckpt_qlogits_step*.npy")):
                continue
            qmax, steps = fold_qmax(fold_dir)
            for tau in THRESHOLDS:
                ss = stop_steps(qmax, steps, tau)
                rows.append(
                    {
                        "benchmark": bench,
                        "arm": arm,
                        "fold": fold_dir.name,
                        "tau": tau,
                        "n_puzzles": len(ss),
                        "mean_stop_step": ss.mean(),
                        "median_stop_step": float(np.median(ss)),
                        "frac_stopped_early": float((ss < HORIZON).mean()),
                        "mean_compute_frac": ss.mean() / HORIZON,
                    }
                )
    if not rows:
        raise SystemExit(
            "run_early_stopping: no ckpt_qlogits_step*.npy arrays found under the *_k5 run directories. "
            "These per-step arrays are not committed; regenerate them with "
            "`bash scripts/run_experiments.sh --profile {sudoku_extreme,maze_hard} --main-result-quick`. "
            "Nothing was written (the committed stopping.csv is left untouched)."
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stopping.csv", index=False)

    summary = (
        df.groupby(["benchmark", "arm", "tau"])[
            ["mean_stop_step", "frac_stopped_early", "mean_compute_frac"]
        ]
        .mean()
        .reset_index()
    )
    lines = [
        "# Adaptive halting from the online guide signal (compute side; Table 14)",
        "",
        "Halt when max_s sigmoid(q_s) >= tau, checked every 3rd outer step (trace",
        "granularity). The accuracy join comes from the per-particle traces (run_trace_analyses.py).",
        "",
        "| Benchmark | Arm | tau | mean stop step | stopped early | compute vs N=48 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.benchmark} | {r.arm} | {r.tau:.2f} | {r.mean_stop_step:.1f} "
            f"| {r.frac_stopped_early * 100:.1f}% | {r.mean_compute_frac * 100:.1f}% |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print((out_dir / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
