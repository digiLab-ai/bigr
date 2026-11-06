"""run_calibration_ladder — rank-binned guide-quality ladder from saved rank profiles (Table 10 prose).

Three rungs, per benchmark and arm, from rank_profiles.csv cells (fold x step x rank):

    calibration     rank-binned reliability points (sigmoid(mean q logit) vs exact rate)
                    and an ECE-style proxy over 10 confidence bins
    discrimination  rank-1 minus rank-16 exact-rate gap, and Spearman monotonicity of
                    exact rate across ranks
    flatness check  terminal mean confidence vs per-particle base success rate
                    (the "calibrated-but-flat" test for Maze)

Rank-binned quantities are approximations (logits averaged within rank before the
sigmoid); the per-particle ECE of Table 10 comes from the particle traces saved by
`run_pf.py --save-particle-trace` and analysed by run_trace_analyses.py.

Outputs
-------
    assets/runs/analysis/calibration_ladder/reliability_points.csv
    assets/runs/analysis/calibration_ladder/per_step.csv
    assets/runs/analysis/calibration_ladder/REPORT.md
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_data import OUT_ROOT, RUNS, iter_cells

ARMS = {
    ("sudoku", "guided"): "sudoku/pf_k5",
    ("sudoku", "unguided"): "sudoku/pf_unguided_k5",
    ("maze", "guided"): "maze/pf_maze_hard_k5",
    ("maze", "unguided"): "maze/pf_maze_hard_unguided_k5",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def collect_points() -> pd.DataFrame:
    frames = []
    for (bench, arm), rel in ARMS.items():
        for cell in iter_cells(RUNS / rel):
            rp = cell.rank_profiles[["step", "rank", "q_logit", "exact_rate"]].copy()
            rp["confidence"] = sigmoid(rp["q_logit"].to_numpy())
            rp["benchmark"] = bench
            rp["arm"] = arm
            rp["fold"] = cell.fold
            frames.append(rp)
    return pd.concat(frames, ignore_index=True)


def ece_proxy(grp: pd.DataFrame, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(grp["confidence"], edges) - 1, 0, bins - 1)
    total = len(grp)
    ece = 0.0
    for b in range(bins):
        sel = grp[idx == b]
        if len(sel):
            ece += len(sel) / total * abs(sel["confidence"].mean() - sel["exact_rate"].mean())
    return ece


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def per_step_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (bench, arm, step), grp in points.groupby(["benchmark", "arm", "step"]):
        by_rank = grp.groupby("rank")[["confidence", "exact_rate"]].mean()
        rows.append(
            {
                "benchmark": bench,
                "arm": arm,
                "step": step,
                "ece_proxy": ece_proxy(grp),
                "rank_gap": by_rank["exact_rate"].iloc[0] - by_rank["exact_rate"].iloc[-1],
                "rank_spearman": spearman(
                    -by_rank.index.to_numpy(), by_rank["exact_rate"].to_numpy()
                ),
                "mean_confidence": grp["confidence"].mean(),
                "base_success": grp["exact_rate"].mean(),
            }
        )
    return pd.DataFrame(rows)


def write_report(per_step: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Rank-binned guide-quality ladder (Table 10 prose)",
        "",
        "Terminal step (n = 48), per benchmark and arm. `rank_gap` = exact-rate of the",
        "highest-Q particle minus the lowest; `spearman` = monotonicity of success in Q-rank;",
        "`ece_proxy` = rank-binned expected calibration error (approximation, see docstring).",
        "",
        "| Benchmark | Arm | ECE proxy | rank-1 − rank-16 gap | rank Spearman | mean conf | base success |",
        "|---|---|---|---|---|---|---|",
    ]
    term = per_step[per_step.step == 48].sort_values(["benchmark", "arm"])
    for _, r in term.iterrows():
        lines.append(
            f"| {r.benchmark} | {r.arm} | {r.ece_proxy:.3f} | {r.rank_gap * 100:.1f} pts "
            f"| {r.rank_spearman:.2f} | {r.mean_confidence:.3f} | {r.base_success:.3f} |"
        )
    lines += [
        "",
        "Calibrated-but-flat check: a guide with `mean conf ≈ base success` but a small",
        "rank gap has acceptable aggregate calibration yet little discriminative power.",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    out_dir = OUT_ROOT / "calibration_ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    points = collect_points()
    points.to_csv(out_dir / "reliability_points.csv", index=False)
    per_step = per_step_summary(points)
    per_step.to_csv(out_dir / "per_step.csv", index=False)
    write_report(per_step, out_dir / "REPORT.md")
    print(f"{len(points)} reliability points -> {out_dir}")
    print((out_dir / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
