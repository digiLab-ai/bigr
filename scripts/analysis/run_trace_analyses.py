"""run_trace_analyses — selection rules, per-step calibration, guide corruption and anytime
halting from --save-particle-trace arrays (original particle order, all steps).

Feeds the rescoring rows of Table 1 and Tables 8, 10 and 14 (Sudoku-Extreme and Maze-Hard
via `run_experiments.sh --traces`, PPBench via `--ppbench`). The output file names keep the
codes used while the analyses were run: rules.csv, e4b.csv (calibration), e10.csv
(corruption), ea7.csv (anytime halting).

Inputs per fold (produced by run_pf.py on an unguided beta=0 run, so particles are
independent rollouts):

    particle_qlogits.npy  (N, T, S)   particle_correct.npy  (N, T, S)
    particle_weights.npy  (N, T, S)   particle_preds.npy    (N, T, S, L)

Analyses
--------
    rules   terminal-Q argmax, path-sum argmax (sum_n log sigmoid(q_n) — the
            Feynman-Kac integrated score without resampling), plurality vote,
            oracle — exact, at the terminal step (Table 1 rescoring rows, Table 8)
    e4b     per-particle ECE and AUROC per outer step (Table 10)
    e10     guide-corruption stress test: q + eps, eps ~ N(0, s^2) iid per
            (particle, step); terminal vs path-sum selection accuracy vs s
            (Table 1, corrupted-guide column)
    ea7     adaptive halting accuracy join: stop when max_s sigmoid(q) >= tau,
            answer = argmax-q particle at the stop step (Table 14)

Usage
-----
    python scripts/analysis/run_trace_analyses.py \
        --trace-dir assets/runs/analysis/ablations/calibration_corruption/raw/sudoku_test/trace \
        --name sudoku_test --out assets/runs/analysis/ablations/calibration_corruption/analyses/sudoku_test_full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_data import OUT_ROOT

CORRUPTION_SCALES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]
CORRUPTION_REPS = 8
TAUS = [0.90, 0.95, 0.99]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label == 1], score[label == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    idx = np.clip((conf * bins).astype(int), 0, bins - 1)
    total = len(conf)
    return float(
        sum(
            (idx == b).sum() / total * abs(conf[idx == b].mean() - correct[idx == b].mean())
            for b in range(bins)
            if (idx == b).any()
        )
    )


def select_acc(scores: np.ndarray, correct_T: np.ndarray) -> float:
    """Accuracy of picking the argmax-score particle; scores/correct_T are (N, S)."""
    return float(correct_T[np.arange(len(scores)), scores.argmax(1)].mean())


def plurality_acc(preds_T: np.ndarray, correct_T: np.ndarray) -> float:
    picks = []
    for b in range(len(preds_T)):
        groups: dict[bytes, tuple[int, int]] = {}
        for s in range(preds_T.shape[1]):
            key = preds_T[b, s].tobytes()
            cnt, first = groups.get(key, (0, s))
            groups[key] = (cnt + 1, first)
        picks.append(max(groups.values(), key=lambda t: t[0])[1])
    return float(correct_T[np.arange(len(preds_T)), picks].mean())


def analyse_fold(fold_dir: Path, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    q = np.load(fold_dir / "particle_qlogits.npy")  # (N, T, S)
    correct = np.load(fold_dir / "particle_correct.npy")  # (N, T, S)
    preds = np.load(fold_dir / "particle_preds.npy", mmap_mode="r")
    N, T, S = q.shape
    q_T, correct_T = q[:, -1, :], correct[:, -1, :]
    logsig = np.log(sigmoid(q))
    pathsum = logsig.sum(axis=1)  # (N, S)

    rules = pd.DataFrame(
        [
            {
                "fold": fold_dir.name,
                "terminal_q": select_acc(q_T, correct_T),
                "path_sum": select_acc(pathsum, correct_T),
                "plurality": plurality_acc(np.asarray(preds[:, -1]), correct_T),
                "oracle": float(correct_T.any(1).mean()),
                "base": float(correct_T.mean()),
                "n_puzzles": N,
            }
        ]
    )

    e4b = pd.DataFrame(
        [
            {
                "fold": fold_dir.name,
                "step": t + 1,
                "ece": ece(sigmoid(q[:, t, :]).ravel(), correct[:, t, :].ravel().astype(int)),
                "auroc": auroc(q[:, t, :].ravel(), correct[:, t, :].ravel().astype(int)),
            }
            for t in range(2, T, 3)
        ]
    )

    e10_rows = []
    for s in CORRUPTION_SCALES:
        reps = 1 if s == 0.0 else CORRUPTION_REPS
        term_accs, path_accs = [], []
        for _ in range(reps):
            noise = rng.normal(0.0, s, size=q.shape) if s > 0 else 0.0
            qc = q + noise
            term_accs.append(select_acc(qc[:, -1, :], correct_T))
            path_accs.append(select_acc(np.log(sigmoid(qc)).sum(axis=1), correct_T))
        e10_rows.append(
            {
                "fold": fold_dir.name,
                "scale": s,
                "terminal_q": np.mean(term_accs),
                "path_sum": np.mean(path_accs),
            }
        )
    e10 = pd.DataFrame(e10_rows)

    conf_max = sigmoid(q).max(axis=2)  # (N, T)
    ea7_rows = []
    for tau in TAUS:
        hit = conf_max >= tau
        first = np.where(hit.any(1), hit.argmax(1), T - 1)
        sel = q[np.arange(N), first, :].argmax(1)
        ea7_rows.append(
            {
                "fold": fold_dir.name,
                "tau": tau,
                "accuracy": float(correct[np.arange(N), first, sel].mean()),
                "terminal_acc_of_stopped": float(correct_T[np.arange(N), q_T.argmax(1)].mean()),
                "mean_stop_step": float((first + 1).mean()),
                "compute_frac": float((first + 1).mean() / T),
            }
        )
    ea7 = pd.DataFrame(ea7_rows)
    return {"rules": rules, "e4b": e4b, "e10": e10, "ea7": ea7}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output directory (default: assets/runs/analysis/trace_analyses/<name>). The register "
        "reads the full-set Sudoku/Maze analyses from ablations/calibration_corruption/analyses/<name>_full.",
    )
    args = ap.parse_args()

    out_dir = args.out or (OUT_ROOT / "trace_analyses" / args.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    fold_dirs = sorted(args.trace_dir.glob("fold_*")) or [args.trace_dir]
    parts: dict[str, list[pd.DataFrame]] = {}
    for fold_dir in tqdm(fold_dirs, desc=args.name):
        if not (fold_dir / "particle_qlogits.npy").exists():
            continue
        for key, df in analyse_fold(fold_dir, rng).items():
            parts.setdefault(key, []).append(df)

    lines = [f"# Trace analyses — {args.name}", ""]
    for key, frames in parts.items():
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out_dir / f"{key}.csv", index=False)
        group_cols = [c for c in ("step", "scale", "tau") if c in df.columns]
        numeric = df.drop(columns=["fold"])
        if group_cols:
            agg = numeric.groupby(group_cols).agg(["mean", "std"])
        else:
            agg = numeric.agg(["mean", "std"]).T
        lines += [f"## {key}", "", "```", agg.round(4).to_string(), "```", ""]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print((out_dir / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
