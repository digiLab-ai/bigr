# Cached run artefacts

Everything the paper prints is derived from the files in this directory. Nothing here
needs a GPU to use: `scripts/regenerate_figures.sh` renders Figures 1-8 from
`sudoku/` and `maze/`, and `scripts/build_tables.sh` rebuilds the evidence register in
`analysis/register/` that every table transcribes. `../experiments_log.json` lists the
same provenance in machine-readable form.

Large per-step arrays (`*.npy`, the raw particle traces) are not committed; the
summaries that depend on them are, and the arrays themselves are in the Hub dataset repo
`andrw3000/bigr-traces` (`scripts/download_checkpoints.py --traces`). The `.npz` inputs of Figures 3, 4 and 7 are stored with
`numpy.savez_compressed`; the continuation labels are bit-exact copies of the originals,
and the entropy arrays store the integer predictions as `uint8` (exact) with the floating
point arrays unchanged.

## `sudoku/` and `maze/` — per-benchmark runs

| Directory | Paper item | Regenerate with | Protocol |
|---|---|---|---|
| `sudoku/pf_seed_fold_test{,_unguided}` | Table 1, Figure 1 | `run_experiments.sh --profile sudoku_extreme --main-result` | 5 seeds (31-35) x 5 folds, S=16, N=48, sigma 0.3, beta 0.25 / 0, tau 0.3, batch 64 x 160 |
| `sudoku/pf_seed_fold_impossible{,_unguided}` | Table 1, Figure 1 (DF split) | `--df-recovery` | same, DF split |
| `maze/pf_seed_fold_maze_hard{,_unguided}_k5` | Table 1 | `--profile maze_hard --main-result` | seeds 31, 41, 59, 83, 109; sigma 0.5, beta 1.0 / 0; batch 16 x 64 |
| `maze/pf_seed_fold_maze_hard_impossible{,_unguided}_k5` | Table 12 | `--profile maze_hard --df-recovery` | same, DF split (134 mazes) |
| `sudoku/pf{,_unguided,_impossible,_impossible_unguided}_k5`, `maze/pf_maze_hard{,...}_k5` | puzzle counts, calibration ladder, early stopping, `regenerate_figures.sh --quick` | `--main-result-quick --df-recovery-quick` | one seed (31) x 5 folds |
| `sudoku/inner_deviation_k5` | Figure 2 | `--inner-deviation` | 2,048 puzzles (`--n-puzzles 2000` loads whole batches of 64), sigma in {0, .1, .2, .3, .5, .8}, alpha in {.5, 1, 1.25, 1.5, 2}, 16 probes |
| `sudoku/sigma_guided_results_k5` | Figure 2 (right) | `--sigma-guided` | S in {4, 16}, same sigma grid |
| `sudoku/continuation_labels_k5`, `maze/continuation_labels_maze_hard_k5` | Figure 3 | `--guide-alignment` | S=16, deterministic continuations from every saved particle state |
| `sudoku/entropy_{test,impossible}_k5` | Figures 4, 7 | `--token-entropy` | S=16, temporal stride 3 |
| `maze/inner_deviation_maze_hard_k5` | Figure 5 (left) | `--profile maze_hard --inner-deviation` | 992 mazes (62 full batches of 16), sigma grid plus 1.3 |
| `sudoku/q_training` | Figure 5 (right) | `scripts/data/collect_training_logs.py --task {sudoku_extreme,maze_hard}` | byte copies of the Lightning CSV logs (`csv/version_*/metrics.csv`) of the backbone training runs, listed in `manifest.csv` |
| `sudoku/ablation_sensitivity_k5`, `maze/ablation_sensitivity_maze_hard_k5` | Figure 6 | `--ablation-sensitivity` | beta and S stress sweeps at the profile's sigma (0.3 / 0.5) and the tau_ESS sweep (`resample/`). The Maze tau_ESS sweep was run with the Sudoku-Extreme values sigma 0.3 / beta 0.25 (see its `resample_config.json`), and the script's Maze defaults reproduce that. |
| `sudoku/pf_checkpoint_sweep` | Figure 8 | `--checkpoint-sweep` | every intermediate checkpoint of the Sudoku-Extreme run, 4 conditions x 5 folds, batch 64 x 32; `training_progress.csv` is derived from the run's Lightning log (`q_training/sudoku_extreme/version_1_metrics.csv`) by `scripts/data/collect_training_logs.py --training-progress`: one row per epoch, the last logged value of each train/val column |

Each run directory keeps its `config.json` (exact arguments; paths are those of the machine
that produced it), per-fold `step_metrics.csv`, `rank_profiles.csv` and
`trm_depth_profile.csv` (the deterministic baseline on the same puzzles), and the
`*_summary.csv` / `fold_manifest.json` / `seed_manifest.json` aggregates.

## `analysis/` — the evidence register and its inputs

Directory and file names keep the experiment codes used while the analyses were run:
`e1c` = particle-count frontier, `e2` = deterministic depth, `e6` = inner depth,
`e7` = proposal noise families; inside a trace analysis, `rules.csv` = selection rules
at the terminal step, `e4b.csv` = per-step calibration (ECE, AUROC), `e10.csv` = guide
corruption, `ea7.csv` = anytime halting.

| Directory | Paper item | Produced by |
|---|---|---|
| `register/metrics.csv`, `register/REPORT.md` | every table (680 numbers, one row each, with source and design) | `scripts/build_tables.sh` |
| `paired_stats/` | Table 1 core runs, Table 7 | `scripts/analysis/run_paired_stats.py` over the seed-fold and `_k5` runs |
| `ablations/calibration_corruption/raw/{sudoku,maze}_{test,df}` | run summaries of the full-set unguided traces | `run_experiments.sh --traces` (the `trace/` arrays themselves are hosted externally) |
| `ablations/calibration_corruption/analyses/*_full` | Tables 1 (rescoring rows), 8, 10, 14 | `scripts/analysis/run_trace_analyses.py` on those traces |
| `ablations/ppbench/{val,golden}` | Table 1 PPBench column, Tables 9, 10, 14 | `run_experiments.sh --ppbench` (five seeds, guided and unguided, with `eval_ppbench.py`) |
| `ablations/ppbench/suite` | Table 9 by type | `scripts/analysis/eval_ppbench_suite.py` |
| `trace_analyses/ppbench_unguided_seed*` | Tables 8, 10, 14 (PPBench) | `run_trace_analyses.py` on the golden-split traces |
| `ablations/inner_depth/e6_M*_sudoku` | Table 5 | `--inner-depth` |
| `ablations/noise_families/e7_*` | Table 6 | `--noise-families` (beta = 0.25 on both benchmarks, the Sudoku-Extreme value) |
| `ablations/baselines/e2_depth_sudoku` | Table 11 | `--depth-baseline` |
| `ablations/baselines/e1c_frontier_sudoku` | Table 13 | `--frontier` |
| `ablations/timing/h100_timing.md` | Table 3 | `--timing`: `bench_device.py --json-out` records (S in {1, 2, 4, 8, 16}, N=48, 4 batches of 16), tabulated by `scripts/analysis/build_timing_table.py`; the shipped table is the paper's run on one H100 |
| `calibration_ladder/`, `early_stopping/` | Table 10 (prose), Table 14 | `run_calibration_ladder.py`, `run_early_stopping.py` over the `_k5` runs (early stopping needs their per-step `ckpt_qlogits_step*.npy`, regenerated by `--main-result-quick`) |
| `ppbench_depth/ppbench_depth_{val,test}` | Table 11 (PPBench rows) | `--ppbench-depth` (`run_depth_baseline.py --profile ppbench --per-type`, deterministic TRM to N=768 on the whole validation and golden splits; `test` = golden) |
