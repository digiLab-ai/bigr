# Rank-binned guide-quality ladder (Table 10 prose)

Terminal step (n = 48), per benchmark and arm. `rank_gap` = exact-rate of the
highest-Q particle minus the lowest; `spearman` = monotonicity of success in Q-rank;
`ece_proxy` = rank-binned expected calibration error (approximation, see docstring).

| Benchmark | Arm | ECE proxy | rank-1 − rank-16 gap | rank Spearman | mean conf | base success |
|---|---|---|---|---|---|---|
| maze | guided | 0.137 | 3.6 pts | 0.84 | 0.995 | 0.858 |
| maze | unguided | 0.138 | 4.5 pts | 0.93 | 0.995 | 0.856 |
| sudoku | guided | 0.058 | 25.8 pts | 0.99 | 0.993 | 0.934 |
| sudoku | unguided | 0.117 | 35.2 pts | 1.00 | 0.961 | 0.844 |

Calibrated-but-flat check: a guide with `mean conf ≈ base success` but a small
rank gap has acceptable aggregate calibration yet little discriminative power.
