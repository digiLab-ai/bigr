"""analysis_data — shared loaders for the offline analyses over assets/runs (evidence register inputs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "assets" / "runs"
OUT_ROOT = REPO_ROOT / "assets" / "runs" / "analysis"

ARM_PAIRS = {
    ("sudoku", "test", "k5"): ("sudoku/pf_k5", "sudoku/pf_unguided_k5"),
    ("sudoku", "df", "k5"): ("sudoku/pf_impossible_k5", "sudoku/pf_impossible_unguided_k5"),
    ("maze", "test", "k5"): ("maze/pf_maze_hard_k5", "maze/pf_maze_hard_unguided_k5"),
    ("maze", "df", "k5"): (
        "maze/pf_maze_hard_impossible_k5",
        "maze/pf_maze_hard_impossible_unguided_k5",
    ),
    ("sudoku", "test", "seed_fold"): (
        "sudoku/pf_seed_fold_test",
        "sudoku/pf_seed_fold_test_unguided",
    ),
    ("sudoku", "df", "seed_fold"): (
        "sudoku/pf_seed_fold_impossible",
        "sudoku/pf_seed_fold_impossible_unguided",
    ),
    ("maze", "test", "seed_fold"): (
        "maze/pf_seed_fold_maze_hard_k5",
        "maze/pf_seed_fold_maze_hard_unguided_k5",
    ),
    ("maze", "df", "seed_fold"): (
        "maze/pf_seed_fold_maze_hard_impossible_k5",
        "maze/pf_seed_fold_maze_hard_impossible_unguided_k5",
    ),
}


@dataclass(frozen=True)
class Cell:
    seed: str
    fold: str
    step_metrics: pd.DataFrame
    rank_profiles: pd.DataFrame


def iter_cells(run_dir: Path):
    """Yield one Cell per (seed, fold); fold-only designs yield seed='-'."""
    if not run_dir.exists():
        return
    seed_dirs = sorted(run_dir.glob("seed_*")) or [run_dir]
    for seed_dir in seed_dirs:
        seed = seed_dir.name if seed_dir != run_dir else "-"
        for fold_dir in sorted(seed_dir.glob("fold_*")):
            sm = fold_dir / "step_metrics.csv"
            rp = fold_dir / "rank_profiles.csv"
            if not (sm.exists() and rp.exists()):
                continue
            yield Cell(seed, fold_dir.name, pd.read_csv(sm), pd.read_csv(rp))


def cell_metrics(cell: Cell, steps: list[int]) -> dict[int, dict[str, float]]:
    """Extract MAP, oracle, and rank-1 terminal-Q rates at the given outer steps."""
    out: dict[int, dict[str, float]] = {}
    sm = cell.step_metrics.set_index("step")
    rp = cell.rank_profiles
    for n in steps:
        if n not in sm.index:
            continue
        r1 = rp[(rp["step"] == n) & (rp["rank"] == 1)]
        out[n] = {
            "map": float(sm.loc[n, "exact_accuracy"]),
            "oracle": float(sm.loc[n, "any_exact_rate"]),
            "top_q": float(r1["exact_rate"].iloc[0]) if len(r1) else float("nan"),
        }
    return out
