"""collect_training_logs — gather the Lightning CSV logs behind Figures 5 and 8.

Copies every ``csv/version_*/metrics.csv`` of a Lightning run directory, byte for byte, to
``<out>/<task>/version_N_metrics.csv`` and records it in ``<out>/manifest.csv`` (columns
task, run, path, source_path; the source is recorded at its place in the release's data
layout, ``data/<task>/csv/version_N/metrics.csv``). With --training-progress it also derives
the checkpoint sweep's training_progress.csv from the run's latest log (or --version): one
row per epoch over the train/val metric columns, keeping the last logged value of each
column (Lightning writes the train and val metrics of an epoch in separate rows).

Usage
-----
    python scripts/data/collect_training_logs.py --task sudoku_extreme --run-dir data/sudoku_extreme \
        --training-progress assets/runs/sudoku/pf_checkpoint_sweep/training_progress.csv
    python scripts/data/collect_training_logs.py --task maze_hard --run-dir data/maze_hard
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("sudoku_extreme", "maze_hard")
METRIC_COLUMNS = [
    "train/accuracy",
    "train/exact_accuracy",
    "train/lm_loss",
    "train/q_halt_loss",
    "train/steps",
    "val/accuracy",
    "val/exact_accuracy",
    "val/lm_loss",
    "val/loss",
    "val/q_halt_loss",
    "val/steps",
]


def version_logs(run_dir: Path) -> dict[int, Path]:
    """Map version number to ``csv/version_N/metrics.csv`` for every version of the run."""
    logs = {}
    for path in (run_dir / "csv").glob("version_*/metrics.csv"):
        match = re.fullmatch(r"version_(\d+)", path.parent.name)
        if match:
            logs[int(match.group(1))] = path
    if not logs:
        raise FileNotFoundError(f"no csv/version_*/metrics.csv under {run_dir}")
    return dict(sorted(logs.items()))


def collect_q_training(logs: dict[int, Path], task: str, out: Path) -> None:
    """Copy the logs to ``<out>/<task>/`` and update ``<out>/manifest.csv``."""
    (out / task).mkdir(parents=True, exist_ok=True)
    rows = []
    for version, src in tqdm(logs.items(), desc=f"Copying {task} logs"):
        rel = Path(task) / f"version_{version}_metrics.csv"
        shutil.copyfile(src, out / rel)
        rows.append(
            {
                "task": task,
                "run": f"version_{version}",
                "path": rel.as_posix(),
                "source_path": f"data/{task}/csv/version_{version}/metrics.csv",
            }
        )
    manifest_path = out / "manifest.csv"
    manifest = pd.DataFrame(rows)
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        manifest = pd.concat([old[old["task"] != task], manifest], ignore_index=True)
    manifest.sort_values(["task", "run"]).to_csv(manifest_path, index=False)


def training_progress(metrics_csv: Path) -> pd.DataFrame:
    """One row per epoch over the metric columns, keeping the last logged value of each."""
    raw = pd.read_csv(metrics_csv)
    columns = ["epoch", *[c for c in METRIC_COLUMNS if c in raw.columns]]
    return raw[columns].groupby("epoch", sort=True).last().reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect the Lightning CSV logs of a backbone training run (Figures 5 and 8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument(
        "--run-dir", type=Path, default=None, help="Lightning run directory (default: data/<task>)."
    )
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "assets" / "runs" / "sudoku" / "q_training",
        help="q_training directory to write <task>/version_N_metrics.csv and manifest.csv into.",
    )
    ap.add_argument(
        "--training-progress", type=Path, default=None,
        help="Also derive the checkpoint sweep's training_progress.csv at this path.",
    )
    ap.add_argument(
        "--version", type=int, default=None,
        help="Log version to derive --training-progress from (default: the latest).",
    )
    args = ap.parse_args()
    run_dir = args.run_dir or REPO_ROOT / "data" / args.task

    logs = version_logs(run_dir)
    collect_q_training(logs, args.task, args.out)
    print(f"{args.task}: copied {len(logs)} log(s) to {args.out / args.task}; manifest at {args.out / 'manifest.csv'}")

    if args.training_progress is not None:
        version = max(logs) if args.version is None else args.version
        if version not in logs:
            raise ValueError(f"version {version} not among {sorted(logs)}")
        progress = training_progress(logs[version])
        args.training_progress.parent.mkdir(parents=True, exist_ok=True)
        progress.to_csv(args.training_progress, index=False)
        print(f"training progress: {len(progress)} epochs from version_{version} -> {args.training_progress}")


if __name__ == "__main__":
    main()
