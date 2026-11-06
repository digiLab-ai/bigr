#!/usr/bin/env bash
# run_experiments.sh — regenerate the run artefacts behind every table and figure
# in "Boosting Inference with Guided Reasoning" (BIGR). Figures are rendered by
# scripts/regenerate_figures.sh; tables by scripts/build_tables.sh.
#
# Requires the TRM checkpoints and datasets under data/ (README: Data and
# checkpoints). Paper results were produced on one NVIDIA H100 (80 GB).
#
# Output layout (identical to the cached artefacts shipped under assets/runs):
#   assets/runs/sudoku/…, assets/runs/maze/…      per-benchmark runs (Figs 1–8, Table 1)
#   assets/runs/analysis/ablations/…              appendix ablations (Tables 5–8, 10–14)
#   assets/runs/analysis/register/                the evidence register (--analysis)
#
# Every flag below names the paper item it feeds. Hyperparameters are the paper's
# (Table 4): S=16, N=48, tau_ESS=0.3; Sudoku sigma=0.3, beta=0.25; Maze sigma=0.5,
# beta=1.0; PPBench sigma=0.3, beta=0.25, tau_ESS=0.5. Five folds (fold-seed 31)
# everywhere unless --num-folds says otherwise.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python}"

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/run_experiments.sh --profile {sudoku_extreme|maze_hard} [experiments...] [options]

Experiments (per --profile unless noted):
  --main-result        Table 1 / Fig 1: guided + unguided, test split, 5 seeds x 5 folds
                       -> pf_seed_fold_test{,_unguided} (sudoku) / pf_seed_fold_maze_hard{,_unguided}_k5 (maze)
  --df-recovery        Table 1 / Fig 1: same on the deterministic-failure (DF) split
  --main-result-quick  one seed x 5 folds on the test split -> pf_k5{,_unguided} (also feeds the
                       calibration ladder, early-stopping and cohort rows of the register)
  --df-recovery-quick  one seed x 5 folds on the DF split -> pf_impossible{,_unguided}_k5
  --inner-deviation    Fig 2 (sudoku) / Fig 5 (maze): tube-stability diagnostic
  --guide-alignment    Fig 3: deterministic continuation labels
  --sigma-guided       Fig 2 right panel: sigma sweep for S in {4,16} (sudoku)
  --ablation-sensitivity  Fig 6: beta / S stress sweeps + tau_ESS sweep
  --token-entropy      Figs 4 and 7: per-token entropy on test + DF (sudoku)
  --checkpoint-sweep   Fig 8: guided vs deterministic across training checkpoints (sudoku only)
  --traces             Appendix (selection rules, corruption, calibration, anytime): full-set
                       unguided traces on test + DF, then the trace analyses
  --inner-depth        Table 5: inference-time inner depth M in {2,4,8,12} (sudoku only)
  --noise-families     Table 6: Laplace / Student-t proposal noise (both profiles)
  --frontier           Table 13: accuracy vs particle count S in {1,2,4,8,16} (sudoku only)
  --depth-baseline     Table 11: deterministic recursion to N=768 (sudoku only)
  --timing             Table 3: wall-clock / memory vs S with bench_device.py, tabulated by
                       build_timing_table.py (h100_timing.md when --device is cuda, else <device>_timing.md)
  --ppbench            Table 1 PPBench column, Tables 9-10: 5 seeds, guided + unguided, on the
                       validation and golden splits, plus their evaluations (profile-independent)
  --ppbench-depth      Table 11 PPBench rows: deterministic recursion to N=768 on the validation and
                       golden splits, per puzzle type -> analysis/ppbench_depth/ppbench_depth_{val,test}
                       (profile-independent)
  --analysis           Offline analysis chain + evidence register (no GPU). Always reads and writes
                       the shipped layout under assets/runs (it ignores --runs-root).
  --all                Every experiment above for the profile (+ --ppbench, --ppbench-depth, --analysis)

Options:
  --profile NAME       sudoku_extreme (default) | maze_hard
  --device DEV         cuda | mps | cpu (default: auto)
  --num-folds N        Folds for the k-fold experiments (default 5, as in the paper)
  --smoke              Tiny settings everywhere (sanity check; numbers will not match the paper).
                       Writes under tmp/runs_smoke/ unless --runs-root is given.
  --runs-root DIR      Root for all run outputs (default assets/runs, the shipped artefact layout)
  -h, --help
USAGE
}

banner() { echo; echo "══════════════════════════════════════════════════════════════════════"; printf "  %s\n" "$*"; echo "══════════════════════════════════════════════════════════════════════"; }

PROFILE="sudoku_extreme"; DEVICE=""; NUM_FOLDS=5; SMOKE=0; RUNS_BASE=""
# One variable per experiment (plain variables: macOS ships bash 3.2, which has no associative arrays).
R_MAIN=0; R_DF=0; R_MAIN_QUICK=0; R_DF_QUICK=0; R_INNER=0; R_GUIDE=0; R_SIGMA=0; R_ABLATION=0; R_ENTROPY=0
R_CKPT=0; R_TRACES=0; R_INNER_DEPTH=0; R_NOISE=0; R_FRONTIER=0; R_DEPTH=0; R_TIMING=0; R_PPBENCH=0; R_PPBENCH_DEPTH=0; R_ANALYSIS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) shift; PROFILE="${1:?--profile needs a value}" ;;
        --device) shift; DEVICE="${1:?--device needs a value}" ;;
        --num-folds) shift; NUM_FOLDS="${1:?--num-folds needs a value}" ;;
        --runs-root) shift; RUNS_BASE="${1:?--runs-root needs a value}" ;;
        --smoke) SMOKE=1 ;;
        --main-result) R_MAIN=1 ;;
        --df-recovery) R_DF=1 ;;
        --main-result-quick) R_MAIN_QUICK=1 ;;
        --df-recovery-quick) R_DF_QUICK=1 ;;
        --inner-deviation) R_INNER=1 ;;
        --guide-alignment) R_GUIDE=1 ;;
        --sigma-guided) R_SIGMA=1 ;;
        --ablation-sensitivity) R_ABLATION=1 ;;
        --token-entropy) R_ENTROPY=1 ;;
        --checkpoint-sweep) R_CKPT=1 ;;
        --traces) R_TRACES=1 ;;
        --inner-depth) R_INNER_DEPTH=1 ;;
        --noise-families) R_NOISE=1 ;;
        --frontier) R_FRONTIER=1 ;;
        --depth-baseline) R_DEPTH=1 ;;
        --timing) R_TIMING=1 ;;
        --ppbench) R_PPBENCH=1 ;;
        --ppbench-depth) R_PPBENCH_DEPTH=1 ;;
        --analysis) R_ANALYSIS=1 ;;
        --all) R_MAIN=1; R_DF=1; R_MAIN_QUICK=1; R_DF_QUICK=1; R_INNER=1; R_GUIDE=1; R_SIGMA=1; R_ABLATION=1; R_ENTROPY=1
               R_CKPT=1; R_TRACES=1; R_INNER_DEPTH=1; R_NOISE=1; R_FRONTIER=1; R_DEPTH=1; R_TIMING=1; R_PPBENCH=1; R_PPBENCH_DEPTH=1; R_ANALYSIS=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
    shift
done
total=$((R_MAIN + R_DF + R_MAIN_QUICK + R_DF_QUICK + R_INNER + R_GUIDE + R_SIGMA + R_ABLATION + R_ENTROPY + R_CKPT + R_TRACES + R_INNER_DEPTH + R_NOISE + R_FRONTIER + R_DEPTH + R_TIMING + R_PPBENCH + R_PPBENCH_DEPTH + R_ANALYSIS))
if [[ $total -eq 0 ]]; then usage; exit 1; fi
if [[ "$PROFILE" != "sudoku_extreme" && "$PROFILE" != "maze_hard" ]]; then echo "Invalid --profile: $PROFILE" >&2; exit 1; fi
if [[ -z "$RUNS_BASE" ]]; then
    if [[ $SMOKE -eq 1 ]]; then RUNS_BASE="tmp/runs_smoke"; else RUNS_BASE="assets/runs"; fi
fi

DEVICE_ARGS=(); [[ -n "$DEVICE" ]] && DEVICE_ARGS=(--device "$DEVICE")
SMOKE_ARGS=(); [[ $SMOKE -eq 1 ]] && SMOKE_ARGS=(--smoke)
KFOLD_ARGS=(--num-folds "$NUM_FOLDS")
K_SUFFIX=""; [[ $NUM_FOLDS -gt 1 ]] && K_SUFFIX="_k${NUM_FOLDS}"
PROFILE_ARGS=(--profile "$PROFILE")

# --- Paper hyperparameters (Table 4) ------------------------------------------------
PF_PARTICLES=16; PF_STEPS=48; ESS_THRESHOLD=0.3
if [[ "$PROFILE" == "sudoku_extreme" ]]; then
    PF_SIGMA=0.3; PF_BETA=0.25
    PF_BATCH_SIZE=64; PF_N_BATCHES_MAIN=160; PF_N_BATCHES_DF=160   # whole batches of 64: 9,984 of the 10,020 test puzzles, 1,408 of the 1,427 DF puzzles (Table 2)
    PF_SEEDS=(31 32 33 34 35)
    INNER_DEV_PUZZLES=2000; INNER_DEV_SIGMAS=(0.0 0.1 0.2 0.3 0.5 0.8); INNER_DEV_BATCH=64; INNER_DEV_LOADED="2,048"   # whole batches: 32 x 64
    SUFFIX=""; RUNS_ROOT="${RUNS_BASE}/sudoku"
    SEED_FOLD_TEST="${RUNS_ROOT}/pf_seed_fold_test";             SEED_FOLD_TEST_UNG="${RUNS_ROOT}/pf_seed_fold_test_unguided"
    SEED_FOLD_DF="${RUNS_ROOT}/pf_seed_fold_impossible";         SEED_FOLD_DF_UNG="${RUNS_ROOT}/pf_seed_fold_impossible_unguided"
    K5_TEST="${RUNS_ROOT}/pf${K_SUFFIX}";                        K5_TEST_UNG="${RUNS_ROOT}/pf_unguided${K_SUFFIX}"
    K5_DF="${RUNS_ROOT}/pf_impossible${K_SUFFIX}";               K5_DF_UNG="${RUNS_ROOT}/pf_impossible_unguided${K_SUFFIX}"
    BENCH="sudoku"
else
    PF_SIGMA=0.5; PF_BETA=1.0
    PF_BATCH_SIZE=16; PF_N_BATCHES_MAIN=64; PF_N_BATCHES_DF=10     # whole batches of 16: 992 of the 1,000 test mazes, 128 of the 134 DF mazes (Table 2)
    PF_SEEDS=(31 41 59 83 109)
    INNER_DEV_PUZZLES=1000; INNER_DEV_SIGMAS=(0.0 0.1 0.2 0.3 0.5 0.8 1.3); INNER_DEV_BATCH=16; INNER_DEV_LOADED="992"   # 62 full batches of 16
    SUFFIX="_maze_hard"; RUNS_ROOT="${RUNS_BASE}/maze"
    SEED_FOLD_TEST="${RUNS_ROOT}/pf_seed_fold_maze_hard${K_SUFFIX}";            SEED_FOLD_TEST_UNG="${RUNS_ROOT}/pf_seed_fold_maze_hard_unguided${K_SUFFIX}"
    SEED_FOLD_DF="${RUNS_ROOT}/pf_seed_fold_maze_hard_impossible${K_SUFFIX}";   SEED_FOLD_DF_UNG="${RUNS_ROOT}/pf_seed_fold_maze_hard_impossible_unguided${K_SUFFIX}"
    K5_TEST="${RUNS_ROOT}/pf_maze_hard${K_SUFFIX}";              K5_TEST_UNG="${RUNS_ROOT}/pf_maze_hard_unguided${K_SUFFIX}"
    K5_DF="${RUNS_ROOT}/pf_maze_hard_impossible${K_SUFFIX}";     K5_DF_UNG="${RUNS_ROOT}/pf_maze_hard_impossible_unguided${K_SUFFIX}"
    BENCH="maze"
fi
ANALYSIS="${RUNS_BASE}/analysis"
if [[ $SMOKE -eq 1 ]]; then
    PF_PARTICLES=4; PF_STEPS=12; PF_BATCH_SIZE=4; PF_N_BATCHES_MAIN=2; PF_N_BATCHES_DF=2
    INNER_DEV_PUZZLES=8; INNER_DEV_BATCH=4; INNER_DEV_LOADED="8"
fi

pf_common=(${SMOKE_ARGS[@]+${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"}} --particles "$PF_PARTICLES" --total-steps "$PF_STEPS" --sigma "$PF_SIGMA" --ess-threshold "$ESS_THRESHOLD" "${KFOLD_ARGS[@]}" --fold-seed 31 ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"})

run_pf_pair() {  # <split> <n-batches> <guided-out> <unguided-out> [extra args...]
    local split="$1" nb="$2" out_g="$3" out_u="$4"; shift 4
    banner "run_pf  $PROFILE  split=$split  guided (beta=$PF_BETA)  S=$PF_PARTICLES N=$PF_STEPS sigma=$PF_SIGMA tau=$ESS_THRESHOLD  batch=${PF_BATCH_SIZE}x${nb}  -> $out_g"
    $PY scripts/exps/run_pf.py "${PROFILE_ARGS[@]}" --split "$split" --beta "$PF_BETA" --batch-size "$PF_BATCH_SIZE" --n-batches "$nb" "${pf_common[@]}" "$@" --out "$out_g"
    banner "run_pf  $PROFILE  split=$split  unguided (beta=0)  -> $out_u"
    $PY scripts/exps/run_pf.py "${PROFILE_ARGS[@]}" --split "$split" --beta 0.0 --batch-size "$PF_BATCH_SIZE" --n-batches "$nb" "${pf_common[@]}" "$@" --out "$out_u"
}

# --- Table 1 / Figure 1: 5 seeds x 5 folds ---------------------------------------------
if [[ $R_MAIN -eq 1 ]]; then
    run_pf_pair test "$PF_N_BATCHES_MAIN" "$SEED_FOLD_TEST" "$SEED_FOLD_TEST_UNG" --seeds "${PF_SEEDS[@]}"
fi
if [[ $R_DF -eq 1 ]]; then
    run_pf_pair impossible "$PF_N_BATCHES_DF" "$SEED_FOLD_DF" "$SEED_FOLD_DF_UNG" --seeds "${PF_SEEDS[@]}"
fi
if [[ $R_MAIN_QUICK -eq 1 ]]; then
    run_pf_pair test "$PF_N_BATCHES_MAIN" "$K5_TEST" "$K5_TEST_UNG" --seed 31
fi
if [[ $R_DF_QUICK -eq 1 ]]; then
    run_pf_pair impossible "$PF_N_BATCHES_DF" "$K5_DF" "$K5_DF_UNG" --seed 31
fi

# --- Figure 2 (sudoku) / Figure 5 (maze): tube stability --------------------------------
if [[ $R_INNER -eq 1 ]]; then
    banner "run_inner_deviation  $PROFILE  ${INNER_DEV_LOADED} puzzles (--n-puzzles ${INNER_DEV_PUZZLES}, whole batches of ${INNER_DEV_BATCH})  sigma in {${INNER_DEV_SIGMAS[*]}}  alpha in {0.5 1.0 1.25 1.5 2.0}"
    $PY scripts/exps/run_inner_deviation.py "${PROFILE_ARGS[@]}" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
        --n-puzzles "$INNER_DEV_PUZZLES" --outer-steps 48 --n-samples 16 \
        --sigma-values "${INNER_DEV_SIGMAS[@]}" --tube-alphas 0.5 1.0 1.25 1.5 2.0 \
        --n-probes 16 --batch-size "$INNER_DEV_BATCH" --seed 31 "${KFOLD_ARGS[@]}" --fold-seed 31 \
        ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} --out "${RUNS_ROOT}/inner_deviation${SUFFIX}${K_SUFFIX}"
fi

# --- Figure 3: guide alignment (continuation labels) ------------------------------------
if [[ $R_GUIDE -eq 1 ]]; then
    banner "run_continuation_labels  $PROFILE  S=$PF_PARTICLES sigma=$PF_SIGMA beta=$PF_BETA tau=$ESS_THRESHOLD  batch=${PF_BATCH_SIZE}x${PF_N_BATCHES_MAIN}"
    $PY scripts/exps/run_continuation_labels.py "${PROFILE_ARGS[@]}" \
        --particles "$PF_PARTICLES" --sigma "$PF_SIGMA" --beta "$PF_BETA" --ess-threshold "$ESS_THRESHOLD" \
        --total-steps "$PF_STEPS" --batch-size "$PF_BATCH_SIZE" --n-batches "$PF_N_BATCHES_MAIN" --seed 31 \
        "${KFOLD_ARGS[@]}" --fold-seed 31 ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} \
        --out "${RUNS_ROOT}/continuation_labels${SUFFIX}${K_SUFFIX}"
fi

# --- Figure 2 right panel: sigma sweep ---------------------------------------------------
if [[ $R_SIGMA -eq 1 ]]; then
    banner "run_sigma_guided  $PROFILE  S in {4,16}  sigma in {0,0.1,0.2,0.3,0.5,0.8}  batch=${PF_BATCH_SIZE}x${PF_N_BATCHES_MAIN}"
    $PY scripts/exps/run_sigma_guided.py "${PROFILE_ARGS[@]}" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
        --total-steps 48 --batch-size "$PF_BATCH_SIZE" --n-batches "$PF_N_BATCHES_MAIN" --seed 31 \
        "${KFOLD_ARGS[@]}" --fold-seed 31 ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} \
        --out "${RUNS_ROOT}/sigma_guided_results${SUFFIX}${K_SUFFIX}"
fi

# --- Figure 6: sensitivity sweeps ---------------------------------------------------------
if [[ $R_ABLATION -eq 1 ]]; then
    banner "run_ablation_sensitivity  $PROFILE  beta / S stress sweeps + tau_ESS sweep (profile defaults: see the script's docstring)"
    $PY scripts/exps/run_ablation_sensitivity.py "${PROFILE_ARGS[@]}" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
        --seed 31 "${KFOLD_ARGS[@]}" --fold-seed 31 ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} \
        --out-dir "${RUNS_ROOT}/ablation_sensitivity${SUFFIX}${K_SUFFIX}"
fi

# --- Figures 4 and 7: token entropy ------------------------------------------------------
if [[ $R_ENTROPY -eq 1 ]]; then
    for split in test impossible; do
        nb="$PF_N_BATCHES_MAIN"; [[ "$split" == "impossible" ]] && nb="$PF_N_BATCHES_DF"
        banner "run_entropy  $PROFILE  split=$split  S=$PF_PARTICLES sigma=$PF_SIGMA beta=$PF_BETA tau=$ESS_THRESHOLD  stride=3  batch=${PF_BATCH_SIZE}x${nb}"
        $PY scripts/exps/run_entropy.py "${PROFILE_ARGS[@]}" --split "$split" \
            --particles "$PF_PARTICLES" --sigma "$PF_SIGMA" --beta "$PF_BETA" --ess-threshold "$ESS_THRESHOLD" \
            --total-steps "$PF_STEPS" --batch-size "$PF_BATCH_SIZE" --n-batches "$nb" --temporal-stride 3 --seed 31 \
            "${KFOLD_ARGS[@]}" --fold-seed 31 ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} \
            --out "${RUNS_ROOT}/entropy${SUFFIX}_${split}${K_SUFFIX}"
    done
fi

# --- Figure 8: training-checkpoint sweep (sudoku) ---------------------------------------
if [[ $R_CKPT -eq 1 ]]; then
    if [[ "$PROFILE" != "sudoku_extreme" ]]; then echo "--checkpoint-sweep is Sudoku-only (Figure 8); skipping for $PROFILE"; else
    banner "run_checkpoint_sweep  every epoch_*.ckpt in data/sudoku_extreme/checkpoints  4 conditions x ${NUM_FOLDS} folds  batch=64x32"
    $PY scripts/exps/run_checkpoint_sweep.py ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
        --checkpoint-dir data/sudoku_extreme/checkpoints --particles 16 --total-steps 48 --sigma 0.3 --ess-threshold 0.3 \
        --batch-size 64 --n-batches 32 --seed 31 "${KFOLD_ARGS[@]}" --fold-seed 31 ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} \
        --out "${RUNS_ROOT}/pf_checkpoint_sweep"
    if [[ -d data/sudoku_extreme/csv ]]; then
        banner "collect_training_logs  sudoku_extreme  Lightning CSV logs -> ${RUNS_BASE}/sudoku/q_training, training_progress.csv"
        $PY scripts/data/collect_training_logs.py --task sudoku_extreme --run-dir data/sudoku_extreme --out "${RUNS_BASE}/sudoku/q_training" \
            --training-progress "${RUNS_ROOT}/pf_checkpoint_sweep/training_progress.csv"
    else
        echo "Figure 8 also needs ${RUNS_ROOT}/pf_checkpoint_sweep/training_progress.csv, derived from the Lightning CSV log"
        echo "(csv/version_*/metrics.csv) of the training run that produced the checkpoints, which is not under data/sudoku_extreme:"
        echo "  python scripts/data/collect_training_logs.py --task sudoku_extreme --run-dir <run dir> --training-progress ${RUNS_ROOT}/pf_checkpoint_sweep/training_progress.csv"
    fi
    fi
fi

# --- Appendix: full-set unguided traces + trace analyses (selection rules, corruption, ...)
if [[ $R_TRACES -eq 1 ]]; then
    CORR="${ANALYSIS}/ablations/calibration_corruption"
    for split in test impossible; do
        name="${BENCH}_test"; [[ "$split" == "impossible" ]] && name="${BENCH}_df"
        raw="${CORR}/raw/${name}"
        banner "run_pf  $PROFILE  split=$split  unguided full-set trace (beta=0, sigma=$PF_SIGMA, tau=$ESS_THRESHOLD, batch 16, ${NUM_FOLDS} folds)  -> $raw"
        $PY scripts/exps/run_pf.py "${PROFILE_ARGS[@]}" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} --split "$split" --beta 0.0 --sigma "$PF_SIGMA" --ess-threshold "$ESS_THRESHOLD" \
            --particles 16 --total-steps 48 --batch-size 16 --n-batches 100000 --seed 31 "${KFOLD_ARGS[@]}" --fold-seed 31 \
            ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} --save-particle-trace "${raw}/trace" --out "$raw"
        banner "run_trace_analyses  $name  -> ${CORR}/analyses/${name}_full"
        $PY scripts/analysis/run_trace_analyses.py --trace-dir "${raw}/trace" --name "$name" --out "${CORR}/analyses/${name}_full"
    done
fi

# --- Table 5: inner depth (sudoku) ---------------------------------------------------------
if [[ $R_INNER_DEPTH -eq 1 ]]; then
    if [[ "$PROFILE" != "sudoku_extreme" ]]; then echo "--inner-depth is Sudoku-only (Table 5); skipping"; else
    for M in 2 4 8 12; do
        banner "run_pf  sudoku  inner depth M=$M  guided  1,008-puzzle subset (16x63), ${NUM_FOLDS} folds"
        $PY scripts/exps/run_pf.py --profile sudoku_extreme ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} --split test --particles 16 --total-steps 48 --sigma 0.3 --beta 0.25 \
            --ess-threshold 0.3 --inner-depth "$M" --batch-size 16 --n-batches 63 "${KFOLD_ARGS[@]}" --fold-seed 31 --seed 31 \
            ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} --out "${ANALYSIS}/ablations/inner_depth/e6_M${M}_sudoku"
    done
    fi
fi

# --- Table 6: proposal noise families ---------------------------------------------------
if [[ $R_NOISE -eq 1 ]]; then
    fams=(laplace student_t); [[ "$PROFILE" == "maze_hard" ]] && fams=(gaussian laplace student_t)
    for fam in "${fams[@]}"; do
        banner "run_pf  $PROFILE  noise family=$fam  guided (beta=0.25, as in the paper ablation)  1,008-puzzle subset (16x63), ${NUM_FOLDS} folds"
        $PY scripts/exps/run_pf.py "${PROFILE_ARGS[@]}" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} --split test --particles 16 --total-steps 48 --sigma "$PF_SIGMA" --beta 0.25 \
            --ess-threshold 0.3 --noise-family "$fam" --batch-size 16 --n-batches 63 "${KFOLD_ARGS[@]}" --fold-seed 31 --seed 31 \
            ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} --out "${ANALYSIS}/ablations/noise_families/e7_${fam}_${BENCH}"
    done
fi

# --- Table 13: particle-count frontier (sudoku) -------------------------------------------
if [[ $R_FRONTIER -eq 1 ]]; then
    if [[ "$PROFILE" != "sudoku_extreme" ]]; then echo "--frontier is Sudoku-only (Table 13); skipping"; else
    banner "run_test_time_compute  sudoku  S in {1,2,4,8,16}  guided + unguided  16x32 puzzles, 3 folds"
    $PY scripts/exps/run_test_time_compute.py --profile sudoku_extreme ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} --split test --particles 1 2 4 8 16 --total-steps 48 \
        --sigma 0.3 --beta 0.25 --ess-threshold 0.3 --batch-size 16 --n-batches 32 --num-folds 3 --seed 31 \
        ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} --out "${ANALYSIS}/ablations/baselines/e1c_frontier_sudoku"
    fi
fi

# --- Table 11: deterministic recursion depth (sudoku) ----------------------------------------
if [[ $R_DEPTH -eq 1 ]]; then
    if [[ "$PROFILE" != "sudoku_extreme" ]]; then echo "--depth-baseline is Sudoku-only (Table 11); skipping"; else
    banner "run_depth_baseline  sudoku  deterministic TRM to N=768  256 puzzles"
    $PY scripts/analysis/run_depth_baseline.py --profile sudoku_extreme --split test --total-steps 768 --batch-size 16 --n-batches 16 \
        ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} --out "${ANALYSIS}/ablations/baselines/e2_depth_sudoku"
    fi
fi

# --- Table 3: cost vs S -------------------------------------------------------------------
if [[ $R_TIMING -eq 1 ]]; then
    dev="${DEVICE:-cuda}"; TIMING="${ANALYSIS}/ablations/timing"; mkdir -p "$TIMING"
    for S in 1 2 4 8 16; do
        banner "bench_device  $PROFILE  S=$S  N=48  4 batches of 16  device=$dev"
        $PY scripts/exps/bench_device.py "${PROFILE_ARGS[@]}" --devices "$dev" --particles "$S" --steps 48 --n-batches 4 --batch-size 16 \
            --json-out "${TIMING}/bench_${BENCH}_${dev}_S${S}.json" | tee "${TIMING}/bench_${BENCH}_${dev}_S${S}.log"
    done
    table="${TIMING}/${dev}_timing.md"; [[ "$dev" == "cuda" ]] && table="${TIMING}/h100_timing.md"   # the register reads h100_timing.md
    banner "build_timing_table  $PROFILE  device=$dev  -> $table"
    $PY scripts/analysis/build_timing_table.py --device "$dev" "${TIMING}/bench_${BENCH}_${dev}_S*.json" --out "$table"
fi

# --- PPBench: Table 1 column, Tables 9-10, and the golden-split trace analyses -----------------
if [[ $R_PPBENCH -eq 1 ]]; then
    PPB="${ANALYSIS}/ablations/ppbench"; mkdir -p "$PPB/suite"
    PPB_SEEDS=(31 32 33 34 35); [[ $SMOKE -eq 1 ]] && PPB_SEEDS=(31 32)
    for split in val test; do
        sub="val"; [[ "$split" == "test" ]] && sub="golden"
        for s in "${PPB_SEEDS[@]}"; do
            for arm in guided unguided; do
                beta=0.25; name="pf_ppbench_seed${s}"; [[ "$arm" == "unguided" ]] && { beta=0.0; name="pf_ppbench_unguided_seed${s}"; }
                banner "run_pf  ppbench  split=$split ($sub)  seed=$s  $arm  S=16 N=48 sigma=0.3 beta=$beta tau=0.5  batch 14, one fold, with trace"
                $PY scripts/exps/run_pf.py --profile ppbench ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} --split "$split" --seed "$s" --fold-seed "$s" --beta "$beta" --sigma 0.3 \
                    --ess-threshold 0.5 --particles 16 --total-steps 48 --batch-size 14 --n-batches 9999 --num-folds 1 \
                    ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} --save-particle-trace "$PPB/$sub/$name" --out "$PPB/$sub/$name"
            done
        done
        banner "eval_ppbench + eval_ppbench_suite  $sub"
        gout="$PPB/$sub/ppbench_metrics_guided_seeds.json"; [[ "$sub" == "val" ]] && gout="$PPB/val/ppbench_metrics_guided_val.json"
        $PY scripts/exps/eval_ppbench.py --run-glob "$PPB/$sub/pf_ppbench_seed*" --split "$split" --out "$gout"
        $PY scripts/exps/eval_ppbench.py --run-glob "$PPB/$sub/pf_ppbench_unguided_seed*" --split "$split" --out "$PPB/$sub/ppbench_metrics_seeds.json"
        $PY scripts/analysis/eval_ppbench_suite.py --run-glob "$PPB/$sub/pf_ppbench_unguided_seed*" --split "$split" --out "$PPB/suite/ppbench_suite_by_type_${sub}.json"
    done
    cp data/ppbench/metadata.json "$PPB/metadata.json"
    for s in "${PPB_SEEDS[@]}"; do
        $PY scripts/analysis/run_trace_analyses.py --trace-dir "$PPB/golden/pf_ppbench_unguided_seed${s}" --name "ppbench_unguided_seed${s}" \
            --out "${ANALYSIS}/trace_analyses/ppbench_unguided_seed${s}"
    done
fi

# --- Table 11 (PPBench rows): deterministic recursion depth on both PPBench splits ------------
if [[ $R_PPBENCH_DEPTH -eq 1 ]]; then
    for split in val test; do
        sub="val"; [[ "$split" == "test" ]] && sub="golden"
        banner "run_depth_baseline  ppbench  split=$split ($sub)  deterministic TRM to N=768  whole split in batches of 14, per type"
        $PY scripts/analysis/run_depth_baseline.py --profile ppbench --split "$split" --total-steps 768 --batch-size 14 --n-batches 9999 --per-type \
            ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} --out "${ANALYSIS}/ppbench_depth/ppbench_depth_${split}"
    done
fi

# --- Offline analysis chain + evidence register (no GPU) --------------------------------------
if [[ $R_ANALYSIS -eq 1 ]]; then
    banner "analysis: paired statistics, calibration ladder, early stopping, evidence register"
    $PY scripts/analysis/run_paired_stats.py
    $PY scripts/analysis/run_calibration_ladder.py
    $PY scripts/analysis/run_early_stopping.py
    $PY scripts/analysis/build_evidence_register.py
fi

echo; echo "Done."
