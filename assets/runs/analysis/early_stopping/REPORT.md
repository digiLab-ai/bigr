# Adaptive halting from the online guide signal (compute side; Table 14)

Halt when max_s sigmoid(q_s) >= tau, checked every 3rd outer step (trace
granularity). The accuracy join comes from the per-particle traces (run_trace_analyses.py).

| Benchmark | Arm | tau | mean stop step | stopped early | compute vs N=48 |
|---|---|---|---|---|---|
| maze | guided | 0.90 | 3.2 | 100.0% | 6.7% |
| maze | guided | 0.95 | 3.2 | 100.0% | 6.8% |
| maze | guided | 0.99 | 3.6 | 99.9% | 7.4% |
| maze | unguided | 0.90 | 3.2 | 100.0% | 6.7% |
| maze | unguided | 0.95 | 3.3 | 100.0% | 6.8% |
| maze | unguided | 0.99 | 3.5 | 100.0% | 7.2% |
| sudoku | guided | 0.90 | 8.2 | 97.8% | 17.1% |
| sudoku | guided | 0.95 | 8.2 | 97.8% | 17.1% |
| sudoku | guided | 0.99 | 8.3 | 97.8% | 17.4% |
| sudoku | unguided | 0.90 | 8.1 | 98.0% | 16.8% |
| sudoku | unguided | 0.95 | 8.1 | 98.0% | 16.9% |
| sudoku | unguided | 0.99 | 8.2 | 98.0% | 17.2% |
