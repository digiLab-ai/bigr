# Evaluation cohorts

The exact puzzles the paper evaluates (Table 2), as small index files into the HRM releases
on the Hugging Face Hub, in the paper's row order. `scripts/data/build_eval_cohorts.py` turns
them into the TRM on-disk splits that the experiment scripts read.

| File | Lines | Cohort | One line per puzzle |
|---|---|---|---|
| `sudoku_extreme_test.txt` | 10,020 | Sudoku-Extreme Test | the 81-character puzzle string (`question` column of `sapientinc/sudoku-extreme` `test.csv`, `.` for a blank) |
| `sudoku_extreme_df.txt` | 1,427 | Sudoku-Extreme DF | as above |
| `maze_hard_test.txt` | 1,000 | Maze-Hard Test | the 0-based row of `sapientinc/maze-30x30-hard-1k` `test.csv` |
| `maze_hard_df.txt` | 134 | Maze-Hard DF | as above |

The Sudoku-Extreme Test cohort is a 10,020-puzzle subsample of the 422,786 puzzles in the HRM
`test.csv`, listed in the order the paper's dataset held them (not the CSV order); the Sudoku
puzzle strings are unique in `test.csv`, so they identify rows. The Maze-Hard Test cohort is
the whole 1,000-maze release in CSV order (rows 0 to 999). Each DF (deterministic-failure)
cohort is the subset of its Test cohort that the frozen backbone fails to solve, in the same
relative order.

**Derivation.** The files were written from the puzzle inputs recorded in the paper's particle
traces (`assets/runs/analysis/ablations/calibration_corruption/raw/<cohort>/trace/fold_0*/inputs.npy`,
fetched by `scripts/download_checkpoints.py --traces`). The traces hold each cohort in the
five folds of `scripts/exps/fold_utils.py` (fold seed 31); inverting that permutation restores
the original row order, and decoding the tokens gives the puzzle string, which matches exactly
one row of the HRM `test.csv`. The same derivation runs with

```bash
python scripts/data/build_eval_cohorts.py --from-traces assets/runs/analysis/ablations/calibration_corruption/raw
```

**Rebuild.** Without the traces, the cohorts are rebuilt from the HRM releases alone:

```bash
python scripts/data/build_eval_cohorts.py            # -> data/{sudoku_extreme,maze_hard}/dataset/<name>/{test,impossible}/
```

These are the `test/` and `impossible/` splits that the experiment scripts read.
