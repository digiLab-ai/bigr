"""The evidence register must rebuild byte-for-byte from the artefacts shipped in the repo.

Every table in the paper transcribes assets/runs/analysis/register/metrics.csv. This test
rebuilds that file with scripts/analysis/build_evidence_register.py into a scratch
directory and compares it with the committed copy, so a stale or missing artefact under
assets/runs is caught without a GPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED = REPO_ROOT / "assets" / "runs" / "analysis" / "register" / "metrics.csv"


@pytest.mark.integration
def test_register_rebuilds_identically(tmp_path: Path) -> None:
    out = tmp_path / "register"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "analysis" / "build_evidence_register.py"), "--out", str(out)],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    rebuilt = (out / "metrics.csv").read_text()
    assert rebuilt == SHIPPED.read_text(), "rebuilt register differs from the shipped metrics.csv"
    assert (out / "REPORT.md").read_text() == (SHIPPED.parent / "REPORT.md").read_text()


def test_register_headline_numbers_match_the_paper() -> None:
    """Spot-check Table 1: the headline exact-solve rates the abstract quotes."""
    df = pd.read_csv(SHIPPED)
    head = df[(df.table == "headline") & (df.benchmark == "sudoku") & (df.split == "test")]
    guided = head[(head.method == "sequential guiding (ours)") & head.design.str.startswith("5 seeds")]
    assert len(guided) == 1
    assert abs(float(guided.value.iloc[0]) - 98.0) < 0.05
    det = head[head.method.str.startswith("deterministic")]
    assert any(abs(v - 85.9) < 0.05 for v in det.value.astype(float))
