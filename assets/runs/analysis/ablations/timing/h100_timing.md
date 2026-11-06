# H100 timing and memory against particle count S (paper Table 3)

`bench_device.py` on an H100, N=48 steps, batch 16 × 4 batches = 64 puzzles per run.

| S | run time (s) | steps/s | peak GPU MB | s/puzzle |
|---|---|---|---|---|
| 1 | 0.89 | 53.9 | 84 | 0.014 |
| 2 | 0.86 | 55.6 | 111 | 0.013 |
| 4 | 1.13 | 42.3 | 185 | 0.018 |
| 8 | 1.98 | 24.2 | 307 | 0.031 |
| 16 | 3.70 | 13.0 | 554 | 0.058 |

- **Wall-clock:** 16× particles costs only ~4× time (0.89 → 3.70 s) — the H100 folds particles into the batch axis, so per-particle steps parallelize well.
- **Memory:** scales ~linearly in S (84 → 554 MB), as predicted (state is 2·L·D floats per particle).
- Apple-silicon (MPS) reference, not in the paper: a 16-puzzle batch at S=16 completes in about 16 s (about 1 s per puzzle).
