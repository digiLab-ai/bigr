# Paired baseline statistics (terminal step, per design; Table 1 core runs, Table 7)

All values in percentage points. `Δ ± CI95` is the paired per-cell difference.

| Benchmark | Split | Design | n | guided MAP | majority | terminal-Q | termQ−guided | guided−majority | oracle diff (g−u) |
|---|---|---|---|---|---|---|---|---|---|
| maze | df | k5 | 5 | 12.4 ± 5.0 | 9.5 ± 5.4 | 16.7 ± 7.7 | +4.36 ± 9.02 | +2.88 ± 6.82 | -5.10 ± 8.88 |
| maze | df | seed_fold | 25 | 12.5 ± 5.5 | 13.1 ± 6.5 | 18.1 ± 8.3 | +5.56 ± 2.79 | -0.58 ± 2.58 | -0.34 ± 3.10 |
| maze | test | k5 | 5 | 85.3 ± 2.5 | 85.3 ± 3.0 | 87.3 ± 2.9 | +2.02 ± 1.60 | -0.00 ± 1.03 | -0.19 ± 0.91 |
| maze | test | seed_fold | 25 | 85.6 ± 2.8 | 85.2 ± 2.5 | 86.8 ± 2.5 | +1.21 ± 0.33 | +0.39 ± 0.64 | -0.45 ± 0.41 |
| sudoku | df | k5 | 5 | 85.7 ± 2.0 | 34.2 ± 2.2 | 85.9 ± 1.4 | +0.16 ± 2.34 | +51.54 ± 4.18 | -0.16 ± 2.34 |
| sudoku | df | seed_fold | 25 | 85.9 ± 1.5 | 35.7 ± 2.8 | 86.8 ± 1.8 | +0.88 ± 0.67 | +50.24 ± 1.03 | -0.84 ± 0.67 |
| sudoku | test | k5 | 5 | 97.9 ± 0.1 | 84.5 ± 1.3 | 98.2 ± 0.3 | +0.30 ± 0.44 | +13.44 ± 1.66 | -0.29 ± 0.43 |
| sudoku | test | seed_fold | 25 | 98.0 ± 0.2 | 84.3 ± 0.7 | 98.2 ± 0.3 | +0.20 ± 0.12 | +13.67 ± 0.29 | -0.20 ± 0.12 |
