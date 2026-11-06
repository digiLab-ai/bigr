#!/usr/bin/env bash
# build_tables.sh — rebuild the evidence register that every table in the paper
# transcribes, from the cached artefacts under assets/runs (no GPU, seconds).
#
#   bash scripts/build_tables.sh            # -> assets/runs/analysis/register/{metrics.csv,REPORT.md}
#   bash scripts/build_tables.sh --out DIR  # write elsewhere (e.g. to diff against the shipped copy)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
"${PYTHON:-python}" scripts/analysis/build_evidence_register.py "$@"
