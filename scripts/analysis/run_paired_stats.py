"""run_paired_stats — paired per-cell differences for the inference-strategy suite (Table 1 core runs, Table 7).

For each benchmark/split/design, join guided and unguided arms on (seed, fold) and
report paired differences at selected outer steps:

    termq_minus_guided   terminal-Q (unguided cloud) − guided MAP
    guided_minus_major   guided MAP − majority vote (unguided MAP)
    oracle_diff          guided-cloud oracle − unguided-cloud oracle

Outputs
-------
    assets/runs/analysis/paired_stats/cells.csv
    assets/runs/analysis/paired_stats/summary.csv
    assets/runs/analysis/paired_stats/REPORT.md
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_data import ARM_PAIRS, OUT_ROOT, RUNS, cell_metrics, iter_cells

STEPS = [6, 12, 24, 36, 48]
T95 = {4: 2.776, 24: 2.064}  # two-sided 95% Student-t critical values for the k5 (5 cells) and seed-fold (25 cells) designs


def collect_cells() -> pd.DataFrame:
    rows = []
    for (bench, split, design), (g_rel, u_rel) in ARM_PAIRS.items():
        guided = {(c.seed, c.fold): cell_metrics(c, STEPS) for c in iter_cells(RUNS / g_rel)}
        unguided = {(c.seed, c.fold): cell_metrics(c, STEPS) for c in iter_cells(RUNS / u_rel)}
        for key in sorted(set(guided) & set(unguided)):
            for n in STEPS:
                g, u = guided[key].get(n), unguided[key].get(n)
                if g is None or u is None:
                    continue
                rows.append(
                    {
                        "benchmark": bench,
                        "split": split,
                        "design": design,
                        "seed": key[0],
                        "fold": key[1],
                        "step": n,
                        "guided_map": g["map"],
                        "guided_oracle": g["oracle"],
                        "guided_top_q": g["top_q"],
                        "majority": u["map"],
                        "unguided_oracle": u["oracle"],
                        "terminal_q": u["top_q"],
                    }
                )
    return pd.DataFrame(rows)


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    cells = cells.assign(
        termq_minus_guided=cells.terminal_q - cells.guided_map,
        guided_minus_major=cells.guided_map - cells.majority,
        oracle_diff=cells.guided_oracle - cells.unguided_oracle,
    )
    diffs = ["termq_minus_guided", "guided_minus_major", "oracle_diff"]
    rows = []
    for (bench, split, design, n), grp in cells.groupby(["benchmark", "split", "design", "step"]):
        row = {"benchmark": bench, "split": split, "design": design, "step": n, "n_cells": len(grp)}
        dof = len(grp) - 1
        if dof not in T95:
            raise ValueError(
                f"{bench}/{split}/{design} step {n}: {len(grp)} cells, but T95 only tabulates "
                f"df in {sorted(T95)}; add the critical value for df={dof} before summarising"
            )
        tcrit = T95[dof]
        for d in diffs:
            mean, sd = grp[d].mean(), grp[d].std(ddof=1)
            sem = sd / math.sqrt(len(grp))
            row[f"{d}_mean"] = mean
            row[f"{d}_sd"] = sd
            row[f"{d}_ci95"] = tcrit * sem
        for col in ["guided_map", "majority", "terminal_q", "guided_oracle", "unguided_oracle"]:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_sd"] = grp[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Paired baseline statistics (terminal step, per design; Table 1 core runs, Table 7)",
        "",
        "All values in percentage points. `Δ ± CI95` is the paired per-cell difference.",
        "",
        "| Benchmark | Split | Design | n | guided MAP | majority | terminal-Q | termQ−guided | guided−majority | oracle diff (g−u) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    term = summary[summary.step == 48].sort_values(["benchmark", "split", "design"])
    for _, r in term.iterrows():
        lines.append(
            f"| {r.benchmark} | {r.split} | {r.design} | {r.n_cells} "
            f"| {r.guided_map_mean * 100:.1f} ± {r.guided_map_sd * 100:.1f} "
            f"| {r.majority_mean * 100:.1f} ± {r.majority_sd * 100:.1f} "
            f"| {r.terminal_q_mean * 100:.1f} ± {r.terminal_q_sd * 100:.1f} "
            f"| {r.termq_minus_guided_mean * 100:+.2f} ± {r.termq_minus_guided_ci95 * 100:.2f} "
            f"| {r.guided_minus_major_mean * 100:+.2f} ± {r.guided_minus_major_ci95 * 100:.2f} "
            f"| {r.oracle_diff_mean * 100:+.2f} ± {r.oracle_diff_ci95 * 100:.2f} |"
        )
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    out_dir = OUT_ROOT / "paired_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = collect_cells()
    cells.to_csv(out_dir / "cells.csv", index=False)
    summary = summarise(cells)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_report(summary, out_dir / "REPORT.md")
    print(f"{len(cells)} cell-rows -> {out_dir}")
    print((out_dir / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
