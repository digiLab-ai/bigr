#!/usr/bin/env bash
# regenerate_figures.sh — render every figure in the paper from the cached run
# artefacts under assets/runs (no GPU, about 30 s). Outputs go to assets/figures/.
# Each section names the paper figure it renders and the runs it reads.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python}"

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/regenerate_figures.sh            # Figures 1-8, paper-faithful (5 seeds x 5 folds for Fig 1)
  bash scripts/regenerate_figures.sh --quick    # Fig 1 from the single-seed 5-fold runs (pf*_k5) instead
  bash scripts/regenerate_figures.sh --out DIR  # write somewhere other than assets/figures/
USAGE
}

OUT_DIR="assets/figures"
QUICK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1 ;;
        --out) shift; OUT_DIR="${1:?--out needs a value}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
    shift
done
mkdir -p "$OUT_DIR"

banner() { echo; echo "══════════════════════════════════════════════════════════════════════"; printf "  %s\n" "$*"; echo "══════════════════════════════════════════════════════════════════════"; }

SUDOKU_RUNS="assets/runs/sudoku"
MAZE_RUNS="assets/runs/maze"

if [[ $QUICK -eq 1 ]]; then
    PF_GUIDED_DIR="${SUDOKU_RUNS}/pf_k5";             PF_UNGUIDED_DIR="${SUDOKU_RUNS}/pf_unguided_k5"
    PF_IMP_GUIDED_DIR="${SUDOKU_RUNS}/pf_impossible_k5"; PF_IMP_UNGUIDED_DIR="${SUDOKU_RUNS}/pf_impossible_unguided_k5"
    banner "Figure 1 — fig_main_result (test + DF, single seed x 5 folds: --quick)"
else
    PF_GUIDED_DIR="${SUDOKU_RUNS}/pf_seed_fold_test";       PF_UNGUIDED_DIR="${SUDOKU_RUNS}/pf_seed_fold_test_unguided"
    PF_IMP_GUIDED_DIR="${SUDOKU_RUNS}/pf_seed_fold_impossible"; PF_IMP_UNGUIDED_DIR="${SUDOKU_RUNS}/pf_seed_fold_impossible_unguided"
    banner "Figure 1 — fig_main_result (test + DF, 5 seeds x 5 folds)"
fi
$PY scripts/figs/fig_pf_main.py \
    --guided-test-dir "${PF_GUIDED_DIR}" --unguided-test-dir "${PF_UNGUIDED_DIR}" \
    --guided-df-dir "${PF_IMP_GUIDED_DIR}" --unguided-df-dir "${PF_IMP_UNGUIDED_DIR}" \
    --particles 16 --out "$OUT_DIR"

banner "Figure 2 — fig_inner_deviation_sudoku (tube stability + sigma sweep)"
$PY scripts/figs/fig_inner_deviation.py \
    --run-dir "${SUDOKU_RUNS}/inner_deviation_k5" \
    --sigma-guided-csv "${SUDOKU_RUNS}/sigma_guided_results_k5/sigma_guided_results.csv" \
    --left-alpha 1.5 --figure-name fig_inner_deviation_sudoku \
    --accuracy-panel-title "Sudoku-Extreme test-set accuracy" --out "$OUT_DIR"

banner "Figure 3 — fig_guide_alignment (Sudoku + Maze continuation labels)"
$PY scripts/figs/fig_guide_alignment.py \
    --sudoku-cont-dir "${SUDOKU_RUNS}/continuation_labels_k5" \
    --maze-cont-dir "${MAZE_RUNS}/continuation_labels_maze_hard_k5" \
    --beta-op 1.0 --out "$OUT_DIR"

banner "Figure 4 — fig_token_entropy (Sudoku test + DF)"
$PY scripts/figs/fig_token_entropy.py \
    --test-dir "${SUDOKU_RUNS}/entropy_test_k5" --df-dir "${SUDOKU_RUNS}/entropy_impossible_k5" --out "$OUT_DIR"

banner "Figure 5 — fig_maze_case_study (Maze tube stability + Q-head training curves)"
$PY scripts/figs/fig_maze_case_study.py \
    --inner-dev-dir "${MAZE_RUNS}/inner_deviation_maze_hard_k5" \
    --q-training-dir "${SUDOKU_RUNS}/q_training" \
    --left-alpha 1.5 --left-ylim 1.0 --out "$OUT_DIR"

banner "Figure 6 — fig_ablation_sensitivity (Sudoku top, Maze bottom)"
$PY scripts/figs/fig_ablation_sensitivity.py \
    --sudoku-stress-csv "${SUDOKU_RUNS}/ablation_sensitivity_k5/resample_stress_results_summary.csv" \
    --sudoku-tau-csv    "${SUDOKU_RUNS}/ablation_sensitivity_k5/resample/summary_summary.csv" \
    --maze-stress-csv   "${MAZE_RUNS}/ablation_sensitivity_maze_hard_k5/resample_stress_results_summary.csv" \
    --maze-tau-csv      "${MAZE_RUNS}/ablation_sensitivity_maze_hard_k5/resample/summary_summary.csv" \
    --out "$OUT_DIR"

banner "Figure 7 — fig_token_contraction (appendix)"
$PY scripts/figs/fig_token_contraction.py \
    --test-dir "${SUDOKU_RUNS}/entropy_test_k5" --df-dir "${SUDOKU_RUNS}/entropy_impossible_k5" --out "$OUT_DIR"

banner "Figure 8 — fig_checkpoint (appendix: training-checkpoint sweep)"
$PY scripts/figs/fig_checkpoint.py \
    --run-dir "${SUDOKU_RUNS}/pf_checkpoint_sweep" --figure-name fig_checkpoint --out "$OUT_DIR"

echo; echo "Figures written to ${OUT_DIR}/"; echo "Done."
