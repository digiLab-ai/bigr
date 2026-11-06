"""Fetch the released checkpoints and (optionally) the raw particle traces from the Hub.

The model repo mirrors this repository's data/ tree, so files land where the experiment
scripts expect them:

    data/<backbone>/checkpoints/last.ckpt          the frozen TRM backbone
    data/sudoku_extreme/checkpoints/epoch_*.ckpt   the intermediate checkpoints for Figure 8 (--all-checkpoints, ~1.5 GB)

with <backbone> in sudoku_extreme | maze_hard | ppbench. The evaluation splits are not
hosted: `python scripts/data/build_eval_cohorts.py` rebuilds them from the HRM releases.
The raw particle traces (3.9 GB) behind the rescoring, corruption, calibration and anytime
tables live in a dataset repo that mirrors assets/runs/analysis/ (--traces).

    python scripts/download_checkpoints.py                      # the three checkpoints
    python scripts/download_checkpoints.py maze_hard            # one backbone only
    python scripts/download_checkpoints.py --all-checkpoints    # plus the intermediate checkpoints
    python scripts/download_checkpoints.py --traces             # the raw traces

Set --repo-id / --traces-repo-id (or BIGR_CHECKPOINT_REPO / BIGR_TRACES_REPO) if the assets move.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = os.environ.get("BIGR_CHECKPOINT_REPO", "andrw3000/bigr-checkpoints")
DEFAULT_TRACES_REPO = os.environ.get("BIGR_TRACES_REPO", "andrw3000/bigr-traces")
BACKBONES = ("sudoku_extreme", "maze_hard", "ppbench")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backbones", nargs="*", choices=BACKBONES, help="backbones to fetch (default: all three)")
    ap.add_argument("--repo-id", default=DEFAULT_REPO, help="Hugging Face model repo mirroring data/")
    ap.add_argument("--all-checkpoints", action="store_true", help="also fetch every intermediate epoch_*.ckpt (Figure 8)")
    ap.add_argument("--traces", action="store_true", help="fetch the raw particle traces into assets/runs/analysis/")
    ap.add_argument("--traces-repo-id", default=DEFAULT_TRACES_REPO, help="Hugging Face dataset repo mirroring assets/runs/analysis/")
    args = ap.parse_args()

    backbones = args.backbones or list(BACKBONES)
    patterns = [f"{name}/checkpoints/last.ckpt" for name in backbones]
    if args.all_checkpoints:
        patterns += [f"{name}/checkpoints/*.ckpt" for name in backbones]
    print(f"{args.repo_id} -> data/  ({', '.join(patterns)})")
    snapshot_download(repo_id=args.repo_id, repo_type="model", allow_patterns=patterns, local_dir=REPO_ROOT / "data")
    for name in backbones:
        ckpt = REPO_ROOT / "data" / name / "checkpoints" / "last.ckpt"
        print(f"  {'ok' if ckpt.exists() else 'MISSING'}: {ckpt}")

    if args.traces:
        dest = REPO_ROOT / "assets" / "runs" / "analysis"
        print(f"{args.traces_repo_id} -> {dest}")
        snapshot_download(repo_id=args.traces_repo_id, repo_type="dataset", local_dir=dest)
        print("  traces fetched; the trace analyses can now be re-derived (scripts/analysis/run_trace_analyses.py)")


if __name__ == "__main__":
    main()
