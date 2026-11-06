#!/usr/bin/env bash
# reproduce_all.sh — end-to-end reproduction: every experiment for both benchmarks and
# PPBench, the timing benchmark, the offline analyses, the evidence register, and all
# figures. Needs one CUDA GPU (paper: a single H100, 80 GB), the checkpoints (including
# the intermediate Sudoku-Extreme checkpoints for Figure 8) and the datasets under data/
# (README: Data and checkpoints). Budget: roughly 30 GPU-hours for the headline
# experiments, plus the full-set traces, the PPBench runs and the timing benchmark.
# For a cheap sanity pass add --smoke.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

for path in data/sudoku_extreme/checkpoints/last.ckpt data/sudoku_extreme/dataset/sudoku_extreme/test \
            data/sudoku_extreme/dataset/sudoku_extreme/impossible data/maze_hard/checkpoints/last.ckpt \
            data/maze_hard/dataset/maze-30x30-hard-1k/test data/ppbench/checkpoints/last.ckpt; do
    [[ -e "$path" ]] || { echo "Missing: $path  (README: Data and checkpoints; splits: python scripts/data/build_eval_cohorts.py)" >&2; exit 1; }
done
ls data/sudoku_extreme/checkpoints/epoch_*.ckpt >/dev/null 2>&1 \
    || { echo "Missing: data/sudoku_extreme/checkpoints/epoch_*.ckpt for Figure 8  (python scripts/download_checkpoints.py --all-checkpoints)" >&2; exit 1; }

# Every Sudoku-Extreme experiment except --analysis, which runs once both benchmarks are done.
bash scripts/run_experiments.sh --profile sudoku_extreme --main-result --df-recovery --main-result-quick --df-recovery-quick \
     --inner-deviation --guide-alignment --sigma-guided --ablation-sensitivity --token-entropy --checkpoint-sweep \
     --traces --inner-depth --noise-families --frontier --depth-baseline --timing --ppbench --ppbench-depth "$@"
bash scripts/run_experiments.sh --profile maze_hard --main-result --df-recovery --main-result-quick --df-recovery-quick \
     --inner-deviation --guide-alignment --ablation-sensitivity --traces --noise-families "$@"
bash scripts/run_experiments.sh --analysis
bash scripts/regenerate_figures.sh
echo "Done: assets/runs/analysis/register/REPORT.md and assets/figures/ are regenerated."
