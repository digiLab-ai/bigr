#!/usr/bin/env bash
# smoke_test.sh — no-GPU sanity check (about 10 s): imports, cached runs, two paper figures.
# Figures are written to tmp/smoke_figures/ (or the directory given as the first argument),
# leaving the shipped assets/figures/ untouched.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python}"
OUT="${1:-tmp/smoke_figures}"

echo "==> Import check"
$PY - <<'PY'
import importlib
for mod in ("bigr", "bigr.filter", "bigr.stochastic", "bigr.backbone", "bigr.plots.kfold", "bigr.plots.style",
            "trm.nn.models.trm", "trm.nn.data.sudoku_datamodule"):
    importlib.import_module(mod)
    print(f"  ok: {mod}")
PY

echo; echo "==> Cached-run check"
for f in assets/runs/sudoku/pf_seed_fold_test/step_metrics_summary.csv \
         assets/runs/sudoku/pf_seed_fold_test_unguided/step_metrics_summary.csv \
         assets/runs/sudoku/pf_seed_fold_impossible/step_metrics_summary.csv \
         assets/runs/sudoku/pf_seed_fold_impossible_unguided/step_metrics_summary.csv \
         assets/runs/sudoku/inner_deviation_k5/deviation_masked_summary.csv \
         assets/runs/sudoku/sigma_guided_results_k5/sigma_guided_results.csv \
         assets/runs/analysis/register/metrics.csv; do
    [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
    echo "  ok: $f"
done

echo; echo "==> Render Figure 1 (fig_main_result) and Figure 2 (fig_inner_deviation_sudoku) into $OUT"
$PY scripts/figs/fig_pf_main.py \
    --guided-test-dir assets/runs/sudoku/pf_seed_fold_test --unguided-test-dir assets/runs/sudoku/pf_seed_fold_test_unguided \
    --guided-df-dir assets/runs/sudoku/pf_seed_fold_impossible --unguided-df-dir assets/runs/sudoku/pf_seed_fold_impossible_unguided \
    --particles 16 --out "$OUT"
$PY scripts/figs/fig_inner_deviation.py \
    --run-dir assets/runs/sudoku/inner_deviation_k5 \
    --sigma-guided-csv assets/runs/sudoku/sigma_guided_results_k5/sigma_guided_results.csv \
    --left-alpha 1.5 --figure-name fig_inner_deviation_sudoku \
    --accuracy-panel-title "Sudoku-Extreme test-set accuracy" --out "$OUT"
for f in "$OUT/fig_main_result.pdf" "$OUT/fig_inner_deviation_sudoku.pdf"; do
    [[ -f "$f" ]] || { echo "Smoke test FAILED: $f not produced." >&2; exit 1; }
done
echo; echo "Smoke test passed (figures in $OUT/)."
