"""build_evidence_register — the machine-generated evidence register behind every table in the paper.

Reads only saved artefacts under assets/runs and writes

    assets/runs/analysis/register/metrics.csv
    assets/runs/analysis/register/REPORT.md

metrics.csv is long format: one row per number the paper may print, with the
artefact it came from, the design (what was averaged), and the cohort size.
REPORT.md renders the same rows as the tables the paper transcribes, headed by
the compute-matched main table (Table 1) assembled from them.

Percentages are in points; ECE and AUROC are raw. Spreads are the sample
standard deviation over cells (seed x fold or fold) unless the row says ci95.

Usage
-----
    python scripts/analysis/build_evidence_register.py            # -> assets/runs/analysis/register/
    python scripts/analysis/build_evidence_register.py --out DIR  # e.g. to diff against the shipped copy
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "assets" / "runs"
ANALYSIS = RUNS / "analysis"
OUT = ANALYSIS / "register"
TERMINAL = 48
DEPTHS = [48, 96, 192, 384, 768]
GUIDED_STEPS = [6, 12, 24, 36, 48]

SEED_FOLD = {
    ("sudoku", "test"): "sudoku/pf_seed_fold_test",
    ("sudoku", "df"): "sudoku/pf_seed_fold_impossible",
    ("maze", "test"): "maze/pf_seed_fold_maze_hard_k5",
    ("maze", "df"): "maze/pf_seed_fold_maze_hard_impossible_k5",
}
K5 = {
    ("sudoku", "test"): "sudoku/pf_k5",
    ("sudoku", "df"): "sudoku/pf_impossible_k5",
    ("maze", "test"): "maze/pf_maze_hard_k5",
    ("maze", "df"): "maze/pf_maze_hard_impossible_k5",
}
FULL_TRACES = {
    ("sudoku", "test"): "sudoku_test",
    ("sudoku", "df"): "sudoku_df",
    ("maze", "test"): "maze_test",
    ("maze", "df"): "maze_df",
}
CORR = ANALYSIS / "ablations" / "calibration_corruption"
PPB = ANALYSIS / "ablations" / "ppbench"
INNER_DEPTH = {"sudoku": 6, "maze": 4, "ppbench": 4}  # L_cycles of each checkpoint

RULE_LABELS = {
    "base": "single rollout (mean per particle)",
    "plurality": "plurality vote",
    "terminal_q": "terminal-Q (unguided cloud)",
    "path_sum": "path-sum tilt (no resampling)",
    "oracle": "oracle (unguided cloud)",
}
CELL_LABELS = {
    "guided_map": "sequential guiding (ours)",
    "guided_top_q": "terminal-Q (guided cloud)",
    "guided_oracle": "oracle (guided cloud)",
    "majority": "single rollout (particle 0)",
    "unguided_oracle": "oracle (unguided cloud)",
    "terminal_q": "terminal-Q (unguided cloud)",
}

rows: list[dict] = []


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def add(table, benchmark, split, method, metric, value, spread=None, spread_kind="",
        n_cells=None, design="", cohort_n=None, source="", note=""):
    rows.append(
        dict(table=table, benchmark=benchmark, split=split, method=method, metric=metric,
             value=float(value), spread=None if spread is None or pd.isna(spread) else float(spread),
             spread_kind=spread_kind, n_cells=n_cells, design=design, cohort_n=cohort_n,
             source=source, note=note)
    )


def pct(x):
    return 100.0 * float(x)


def sd(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def cohort_from_manifest(run_dir: Path) -> int | None:
    for cand in [run_dir / "fold_manifest.json", *sorted(run_dir.glob("seed_*/fold_manifest.json"))]:
        if cand.exists():
            return int(json.loads(cand.read_text())["total_examples"])
    return None


def cell_dirs(run_dir: Path) -> list[Path]:
    bases = sorted(run_dir.glob("seed_*")) or [run_dir]
    return [f for b in bases for f in sorted(b.glob("fold_*")) if (f / "step_metrics.csv").exists()]


def deterministic_per_cell(run_dir: Path) -> np.ndarray:
    vals = []
    for f in cell_dirs(run_dir):
        p = f / "trm_depth_profile.csv"
        if p.exists():
            d = pd.read_csv(p).set_index("step")
            if TERMINAL in d.index:
                vals.append(float(d.loc[TERMINAL, "exact_solved"]))
    return np.array(vals)


# --------------------------------------------------------------------------- headline designs
def headline() -> None:
    src = ANALYSIS / "paired_stats" / "cells.csv"
    cells = pd.read_csv(src)
    cells = cells[cells["step"] == TERMINAL]
    for (bench, split, design), g in cells.groupby(["benchmark", "split", "design"]):
        run_rel = (SEED_FOLD if design == "seed_fold" else K5)[(bench, split)]
        cohort = cohort_from_manifest(RUNS / run_rel)
        dlabel = "5 seeds x 5 folds" if design == "seed_fold" else "5 folds, one seed"
        for col, name in CELL_LABELS.items():
            add("headline", bench, split, name, "exact", pct(g[col].mean()), pct(sd(g[col])),
                "sd over cells", len(g), dlabel, cohort, rel(src))
        det = deterministic_per_cell(RUNS / run_rel)
        if len(det):
            add("headline", bench, split, "deterministic TRM", "exact", pct(det.mean()), pct(sd(det)),
                "sd over cells", len(det), dlabel, cohort, run_rel + "/**/trm_depth_profile.csv")
    summ = pd.read_csv(ANALYSIS / "paired_stats" / "summary.csv")
    summ = summ[summ["step"] == TERMINAL]
    for _, r in summ.iterrows():
        dlabel = "5 seeds x 5 folds" if r["design"] == "seed_fold" else "5 folds, one seed"
        cohort = cohort_from_manifest(RUNS / (SEED_FOLD if r["design"] == "seed_fold" else K5)[(r["benchmark"], r["split"])])
        for key, name in [("termq_minus_guided", "terminal-Q (unguided) minus ours"),
                          ("guided_minus_major", "ours minus single rollout (particle 0)"),
                          ("oracle_diff", "oracle guided minus oracle unguided")]:
            add("paired_differences", r["benchmark"], r["split"], name, "paired difference",
                pct(r[f"{key}_mean"]), pct(r[f"{key}_ci95"]), "ci95 over cells", int(r["n_cells"]),
                dlabel, cohort, rel(ANALYSIS / "paired_stats" / "summary.csv"))


# --------------------------------------------------------------------------- full-set traces
def full_traces() -> None:
    for (bench, split), name in FULL_TRACES.items():
        d = CORR / "analyses" / f"{name}_full"
        rules = pd.read_csv(d / "rules.csv")
        cohort = int(rules["n_puzzles"].sum())
        design = "5 folds, one seed, unguided traces"
        for col, label in RULE_LABELS.items():
            add("selection_rules", bench, split, label, "exact", pct(rules[col].mean()), pct(sd(rules[col])),
                "sd over folds", len(rules), design, cohort, rel(d / "rules.csv"))
        e10 = pd.read_csv(d / "e10.csv")
        for scale, g in e10.groupby("scale"):
            for col in ["terminal_q", "path_sum"]:
                add("corruption", bench, split, RULE_LABELS[col], f"exact@s={scale:g}", pct(g[col].mean()),
                    pct(sd(g[col])), "sd over folds", len(g), design, cohort, rel(d / "e10.csv"),
                    "iid Gaussian noise of sd s added to every raw logit, 8 draws per fold, stored clouds rescored")
        e4b = pd.read_csv(d / "e4b.csv")
        for step, g in e4b.groupby("step"):
            for col in ["ece", "auroc"]:
                add("calibration", bench, split, "per-particle, unguided cloud", f"{col}@n={step}",
                    g[col].mean(), sd(g[col]), "sd over folds", len(g), design, cohort, rel(d / "e4b.csv"),
                    "10-bin ECE of sigmoid(q) against per-particle exact correctness; AUROC pools particles over puzzles")
        ea7 = pd.read_csv(d / "ea7.csv")
        for tau, g in ea7.groupby("tau"):
            for col, metric in [("accuracy", "exact at stop"), ("mean_stop_step", "mean stop step"),
                                ("compute_frac", "fraction of N=48 budget")]:
                v = g[col].mean()
                add("anytime", bench, split, f"halt when max_s sigmoid(q) >= {tau:g}", metric,
                    pct(v) if col != "mean_stop_step" else v, pct(sd(g[col])) if col != "mean_stop_step" else sd(g[col]),
                    "sd over folds", len(g), design, cohort, rel(d / "ea7.csv"),
                    "checked every third outer step; answer is the argmax-q particle at the stop step")


# --------------------------------------------------------------------------- PPBench
def ppbench() -> None:
    meta = json.loads((PPB / "metadata.json").read_text())
    cfg = json.loads((PPB / "golden" / "pf_ppbench_seed31" / "config.json").read_text())
    add("protocol", "ppbench", "all", "particle filter", "S", cfg["particles"], source=rel(PPB / "golden" / "pf_ppbench_seed31" / "config.json"))
    add("protocol", "ppbench", "all", "particle filter", "N", cfg["total_steps"], source="config.json")
    add("protocol", "ppbench", "all", "particle filter", "sigma", cfg["sigma"], source="config.json")
    add("protocol", "ppbench", "all", "particle filter", "beta", cfg["beta"], source="config.json")
    add("protocol", "ppbench", "all", "particle filter", "tau_ESS", cfg["ess_threshold"], source="config.json",
        note="differs from the 0.3 used on Sudoku-Extreme and Maze-Hard")
    add("protocol", "ppbench", "all", "backbone", "M (L_cycles)", INNER_DEPTH["ppbench"], source="src/trm/nn/configs/experiment/trm_ppbench.yaml")
    add("protocol", "ppbench", "train", "dataset", "examples", meta["splits"]["train"]["examples"], source=rel(PPB / "metadata.json"))
    splits = {"val": "val", "golden": "test"}
    for split, key in splits.items():
        add("cohorts", "ppbench", split, "dataset", "puzzles", meta["splits"][key]["examples"], source="metadata.json")
        for t, n in meta["splits"][key]["per_type"].items():
            add("cohorts", "ppbench", f"{split}/{t}", "dataset", "puzzles", n, source="metadata.json")
    for split, key in splits.items():
        suite = json.loads((PPB / "suite" / f"ppbench_suite_by_type_{split}.json").read_text())
        n_all = meta["splits"][key]["examples"]
        design = "5 seeds, one fold, unguided traces"
        for rule in suite["rules"]:
            m, s = suite["aggregate"][rule]
            add("ppbench_rules", "ppbench", split, RULE_LABELS[rule], "exact", pct(m), pct(s), "sd over seeds", 5,
                design, n_all, rel(PPB / "suite" / f"ppbench_suite_by_type_{split}.json"))
            for t, tv in suite["types"].items():
                m, s = tv[rule]
                add("ppbench_rules", "ppbench", f"{split}/{t}", RULE_LABELS[rule], "exact", pct(m), pct(s),
                    "sd over seeds", 5, design, meta["splits"][key]["per_type"][t], "ppbench_suite_by_type")
        for diag in suite["diagnostics"]:
            m, s = suite["aggregate"][diag]
            add("ppbench_diagnostics", "ppbench", split, "unguided cloud", diag, m, s, "sd over seeds", 5, design, n_all,
                "ppbench_suite_by_type", "q_spread is the sd of the terminal raw logit over particles; delta_n is the raw-logit class gap")
            for t, tv in suite["types"].items():
                m, s = tv[diag]
                add("ppbench_diagnostics", "ppbench", f"{split}/{t}", "unguided cloud", diag, m, s, "sd over seeds", 5,
                    design, meta["splits"][key]["per_type"][t], "ppbench_suite_by_type")
    # guided (ours) and deterministic TRM, validation split
    vg = json.loads((PPB / "val" / "ppbench_metrics_guided_val.json").read_text())
    add("ppbench_rules", "ppbench", "val", "sequential guiding (ours)", "exact", pct(vg["guided_exact_agg"]["mean"]),
        pct(vg["guided_exact_agg"]["std"]), "sd over seeds", 5, "5 seeds, one fold, guided runs", 253,
        rel(PPB / "val" / "ppbench_metrics_guided_val.json"))
    add("ppbench_rules", "ppbench", "val", "deterministic TRM", "exact", pct(vg["trm_exact_agg"]["mean"]), None, "", 1,
        "deterministic", 253, "ppbench_metrics_guided_val.json")
    for t, tv in vg["per_type"].items():
        add("ppbench_rules", "ppbench", f"val/{t}", "sequential guiding (ours)", "exact", pct(tv["guided"]["mean"]),
            pct(tv["guided"]["std"]), "sd over seeds", 5, "5 seeds, one fold, guided runs",
            meta["splits"]["val"]["per_type"][t], "ppbench_metrics_guided_val.json")
        add("ppbench_rules", "ppbench", f"val/{t}", "deterministic TRM", "exact", pct(tv["trm"]), None, "", 1,
            "deterministic", meta["splits"]["val"]["per_type"][t], "ppbench_metrics_guided_val.json")
    # guided and deterministic, golden split (per-seed file)
    gg = json.loads((PPB / "golden" / "ppbench_metrics_guided_seeds.json").read_text())
    per_seed_agg = [s["guided_exact_agg"] for s in gg["seeds"]]
    add("ppbench_rules", "ppbench", "golden", "sequential guiding (ours)", "exact", pct(np.mean(per_seed_agg)),
        pct(sd(per_seed_agg)), "sd over seeds", len(per_seed_agg), "5 seeds, one fold, guided runs", 42,
        rel(PPB / "golden" / "ppbench_metrics_guided_seeds.json"))
    add("ppbench_rules", "ppbench", "golden", "deterministic TRM", "exact", pct(gg["seeds"][0]["trm_exact_agg"]), None, "",
        1, "deterministic", 42, "ppbench_metrics_guided_seeds.json")
    for t in gg["seeds"][0]["per_type"]:
        vals = [s["per_type"][t]["guided_exact"] for s in gg["seeds"]]
        add("ppbench_rules", "ppbench", f"golden/{t}", "sequential guiding (ours)", "exact", pct(np.mean(vals)), pct(sd(vals)),
            "sd over seeds", len(vals), "5 seeds, one fold, guided runs", gg["seeds"][0]["per_type"][t]["n"],
            "ppbench_metrics_guided_seeds.json")
        add("ppbench_rules", "ppbench", f"golden/{t}", "deterministic TRM", "exact", pct(gg["seeds"][0]["per_type"][t]["trm_exact"]),
            None, "", 1, "deterministic", gg["seeds"][0]["per_type"][t]["n"], "ppbench_metrics_guided_seeds.json")
    # single rollout as particle 0 (unguided MAP) from the unguided runs
    for split, sub in [("val", "val"), ("golden", "golden")]:
        ug = json.loads((PPB / sub / "ppbench_metrics_seeds.json").read_text())
        vals = [s["guided_exact_agg"] for s in ug["seeds"]]
        add("ppbench_rules", "ppbench", split, "single rollout (particle 0)", "exact", pct(np.mean(vals)), pct(sd(vals)),
            "sd over seeds", len(vals), "5 seeds, one fold, unguided runs", ug["seeds"][0]["num_examples"],
            rel(PPB / sub / "ppbench_metrics_seeds.json"), "exact_accuracy of the beta=0 run, i.e. the first particle")
        for t in ug["seeds"][0]["per_type"]:
            vals = [s["per_type"][t]["guided_exact"] for s in ug["seeds"]]
            add("ppbench_rules", "ppbench", f"{split}/{t}", "single rollout (particle 0)", "exact", pct(np.mean(vals)),
                pct(sd(vals)), "sd over seeds", len(vals), "5 seeds, one fold, unguided runs",
                ug["seeds"][0]["per_type"][t]["n"], "ppbench_metrics_seeds.json")
    # depth profiles
    for split, d in [("val", "ppbench_depth_val"), ("golden", "ppbench_depth_test")]:
        a = pd.read_csv(ANALYSIS / "ppbench_depth" / d / "aggregate.csv").set_index("depth")
        for N in DEPTHS:
            add("depth", "ppbench", split, "deterministic TRM", f"exact@N={N}", pct(a.loc[N, "exact"]), None, "", 1,
                "deterministic", meta["splits"][splits[split]]["examples"], rel(ANALYSIS / "ppbench_depth" / d / "aggregate.csv"))
    # golden-split trace analyses, aggregated over the five seeds
    seed_dirs = sorted((ANALYSIS / "trace_analyses").glob("ppbench_unguided_seed*"))
    e10 = pd.concat([pd.read_csv(s / "e10.csv").assign(seed=s.name) for s in seed_dirs])
    for scale, g in e10.groupby("scale"):
        for col in ["terminal_q", "path_sum"]:
            add("corruption", "ppbench", "golden", RULE_LABELS[col], f"exact@s={scale:g}", pct(g[col].mean()), pct(sd(g[col])),
                "sd over seeds", len(g), "5 seeds, one fold, unguided traces", 42, "trace_analyses/ppbench_unguided_seed*/e10.csv")
    e4b = pd.concat([pd.read_csv(s / "e4b.csv").assign(seed=s.name) for s in seed_dirs])
    for step, g in e4b.groupby("step"):
        for col in ["ece", "auroc"]:
            add("calibration", "ppbench", "golden", "per-particle, unguided cloud", f"{col}@n={step}", g[col].mean(), sd(g[col]),
                "sd over seeds", len(g), "5 seeds, one fold, unguided traces", 42, "trace_analyses/ppbench_unguided_seed*/e4b.csv")
    ea7 = pd.concat([pd.read_csv(s / "ea7.csv").assign(seed=s.name) for s in seed_dirs])
    for tau, g in ea7.groupby("tau"):
        for col, metric in [("accuracy", "exact at stop"), ("mean_stop_step", "mean stop step"), ("compute_frac", "fraction of N=48 budget")]:
            v = g[col].mean()
            add("anytime", "ppbench", "golden", f"halt when max_s sigmoid(q) >= {tau:g}", metric,
                pct(v) if col != "mean_stop_step" else v, pct(sd(g[col])) if col != "mean_stop_step" else sd(g[col]),
                "sd over seeds", len(g), "5 seeds, one fold, unguided traces", 42, "trace_analyses/ppbench_unguided_seed*/ea7.csv")


# --------------------------------------------------------------------------- depth, particles, M, noise, cost
def depth_and_particles() -> None:
    d = ANALYSIS / "ablations" / "baselines" / "e2_depth_sudoku"
    cfg = json.loads((d / "config.json").read_text())
    cohort = int(cfg["batch_size"]) * int(cfg["n_batches"])
    prof = pd.read_csv(d / "trm_depth_profile.csv").set_index("step")
    for N in DEPTHS + [576]:
        add("depth", "sudoku", f"test ({cohort}-puzzle subset)", "deterministic TRM", f"exact@N={N}", pct(prof.loc[N, "exact_solved"]),
            None, "", 1, "deterministic, one run", cohort, rel(d / "trm_depth_profile.csv"))
    for (bench, split), run_rel in SEED_FOLD.items():
        sm = pd.read_csv(RUNS / run_rel / "step_metrics_summary.csv").set_index("step")
        cohort = cohort_from_manifest(RUNS / run_rel)
        for N in GUIDED_STEPS:
            add("depth", bench, split, "sequential guiding (ours)", f"exact@N={N}", pct(sm.loc[N, "exact_accuracy_mean"]),
                pct(sm.loc[N, "exact_accuracy_std"]), "sd over cells", 25, "5 seeds x 5 folds", cohort,
                run_rel + "/step_metrics_summary.csv")
    # crossover of guided Sudoku Test over the deterministic terminal value
    sm = pd.read_csv(RUNS / SEED_FOLD[("sudoku", "test")] / "step_metrics_summary.csv").set_index("step")
    det = deterministic_per_cell(RUNS / SEED_FOLD[("sudoku", "test")]).mean()
    above = sm.index[sm["exact_accuracy_mean"] > det]
    if len(above) == 0:
        raise RuntimeError(
            f"guided Sudoku test ({SEED_FOLD[('sudoku', 'test')]}) never exceeds the deterministic terminal "
            f"mean {pct(det):.1f} at any outer step, so the crossover row cannot be computed"
        )
    cross = int(above[0])
    add("depth", "sudoku", "test", "sequential guiding (ours)", "first step above deterministic terminal", cross, None, "", 25,
        "5 seeds x 5 folds", cohort_from_manifest(RUNS / SEED_FOLD[("sudoku", "test")]), "step_metrics_summary.csv",
        f"deterministic terminal mean {pct(det):.1f}")
    # small-S frontier
    d = ANALYSIS / "ablations" / "baselines" / "e1c_frontier_sudoku"
    man = json.loads((d / "fold_manifest.json").read_text())
    cohort = sum(f["num_examples"] for f in man["folds"])
    fr = pd.read_csv(d / "compute_summary.csv")
    for _, r in fr.iterrows():
        add("frontier", "sudoku", f"test ({cohort}-puzzle subset)", r["method"], "exact", pct(r["terminal_exact_accuracy"]),
            pct(r["terminal_exact_accuracy_ci"]), "ci95 over folds", len(man["folds"]), "3 folds, one seed", cohort,
            rel(d / "compute_summary.csv"), f"S x N backbone calls = {int(r['terminal_f_theta_calls'])}")
        add("frontier", "sudoku", f"test ({cohort}-puzzle subset)", r["method"], "oracle", pct(r["terminal_oracle_accuracy"]),
            pct(r["terminal_oracle_accuracy_ci"]), "ci95 over folds", len(man["folds"]), "3 folds, one seed", cohort,
            rel(d / "compute_summary.csv"))
    # inner depth sweep
    for M in [2, 4, 8, 12]:
        d = ANALYSIS / "ablations" / "inner_depth" / f"e6_M{M}_sudoku"
        sm = pd.read_csv(d / "step_metrics_summary.csv").set_index("step")
        cohort = cohort_from_manifest(d)
        for col, label in [("exact_accuracy", "sequential guiding (ours)"), ("any_exact_rate", "oracle (guided cloud)")]:
            add("inner_depth", "sudoku", f"test ({cohort}-puzzle subset)", label, f"exact@M={M}", pct(sm.loc[TERMINAL, f"{col}_mean"]),
                pct(sm.loc[TERMINAL, f"{col}_std"]), "sd over folds", 5, "5 folds, one seed", cohort, rel(d / "step_metrics_summary.csv"),
                "inner_depth overrides the checkpoint's trained M=6")
    # noise families
    for fam in ["gaussian", "laplace", "student_t"]:
        for bench in ["sudoku", "maze"]:
            d = ANALYSIS / "ablations" / "noise_families" / f"e7_{fam}_{bench}"
            if not d.exists():
                continue
            sm = pd.read_csv(d / "step_metrics_summary.csv").set_index("step")
            cohort = cohort_from_manifest(d)
            cfg = json.loads((d / "config.json").read_text())
            for col, label, metric in [("exact_accuracy", "sequential guiding (ours)", "exact"),
                                       ("any_exact_rate", "oracle (guided cloud)", "exact"),
                                       ("q_diversity", "guided cloud", "q_spread (raw logit sd)")]:
                add("noise_families", bench, f"test ({cohort}-puzzle subset)", label, f"{metric}@{fam}",
                    pct(sm.loc[TERMINAL, f"{col}_mean"]) if metric == "exact" else sm.loc[TERMINAL, f"{col}_mean"],
                    pct(sm.loc[TERMINAL, f"{col}_std"]) if metric == "exact" else sm.loc[TERMINAL, f"{col}_std"],
                    "sd over folds", 5, "5 folds, one seed", cohort, rel(d / "step_metrics_summary.csv"),
                    f"sigma={cfg['sigma']}, variance-matched; laplace b=1/sqrt(2); student_t df=3 scaled by 1/sqrt(3)")
    # cost
    md = (ANALYSIS / "ablations" / "timing" / "h100_timing.md").read_text()
    for m in re.finditer(r"^\| (\d+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \| ([\d.]+) \|", md, re.M):
        S, run_s, steps_s, mem, per_puzzle = m.groups()
        add("cost", "unrecorded task", "64 puzzles", f"S={S}", "seconds per puzzle", float(per_puzzle), None, "", 1,
            "one H100, 4 batches of 16, N=48", 64, rel(ANALYSIS / "ablations" / "timing" / "h100_timing.md"),
            "amortised batched throughput, not single-query latency; task profile not recorded")
        add("cost", "unrecorded task", "64 puzzles", f"S={S}", "peak GPU memory (MB)", float(mem), None, "", 1,
            "one H100, 4 batches of 16, N=48", 64, "h100_timing.md")
        add("cost", "unrecorded task", "64 puzzles", f"S={S}", "run time (s)", float(run_s), None, "", 1,
            "one H100, 4 batches of 16, N=48", 64, "h100_timing.md")
    # calibration ladder: confidence versus success at the terminal step
    lad = pd.read_csv(ANALYSIS / "calibration_ladder" / "per_step.csv")
    lad = lad[lad["step"] == TERMINAL]
    for _, r in lad.iterrows():
        for col, metric in [("mean_confidence", "mean terminal confidence"), ("base_success", "per-particle success"),
                            ("rank_gap", "rank-1 minus rank-16 exact rate"), ("rank_spearman", "rank Spearman")]:
            add("guide_flatness", r["benchmark"], "test", f"{r['arm']} cloud", metric, r[col], None, "", 5, "5 folds, one seed",
                cohort_from_manifest(RUNS / K5[(r["benchmark"], "test")]), rel(ANALYSIS / "calibration_ladder" / "per_step.csv"),
                "rank-binned; the ece_proxy column of this file is not an ECE and is not exported")
    # early stopping, compute side, guided and unguided arms
    st = pd.read_csv(ANALYSIS / "early_stopping" / "stopping.csv")
    for (bench, arm, tau), g in st.groupby(["benchmark", "arm", "tau"]):
        add("anytime_compute", bench, "test", f"{arm} cloud, halt when max_s sigmoid(q) >= {tau:g}", "fraction of N=48 budget",
            pct(g["mean_compute_frac"].mean()), pct(sd(g["mean_compute_frac"])), "sd over folds", len(g), "5 folds, one seed",
            int(g["n_puzzles"].sum()), rel(ANALYSIS / "early_stopping" / "stopping.csv"), "compute side only; no accuracy join on these arms")
        add("anytime_compute", bench, "test", f"{arm} cloud, halt when max_s sigmoid(q) >= {tau:g}", "mean stop step",
            g["mean_stop_step"].mean(), sd(g["mean_stop_step"]), "sd over folds", len(g), "5 folds, one seed",
            int(g["n_puzzles"].sum()), "stopping.csv")


# --------------------------------------------------------------------------- cohorts and budgets
def cohorts_and_budgets() -> None:
    for (bench, split), run_rel in SEED_FOLD.items():
        add("cohorts", bench, split, "seed-fold runs", "puzzles", cohort_from_manifest(RUNS / run_rel), source=run_rel)
    for (bench, split), run_rel in K5.items():
        add("cohorts", bench, split, "k5 runs", "puzzles", cohort_from_manifest(RUNS / run_rel), source=run_rel)
    for (bench, split), name in FULL_TRACES.items():
        add("cohorts", bench, split, "full-set unguided traces", "puzzles", cohort_from_manifest(CORR / "raw" / name),
            source=rel(CORR / "raw" / name))
    for bench, (S, N, M) in CALLS_SNM().items():
        add("budget", bench, "all", "particle filter", "backbone calls per puzzle, S x N x (M+1)", S * N * (M + 1), source="config.json + trm yaml",
            note=f"S={S}, N={N}, M={M}")
        add("budget", bench, "all", "deterministic TRM", "backbone calls per puzzle at N=48", N * (M + 1))
        add("budget", bench, "all", "deterministic TRM", "depth N matching the particle budget", S * N)


def CALLS_SNM() -> dict[str, tuple[int, int, int]]:
    return {b: (16, 48, M) for b, M in INNER_DEPTH.items()}


# --------------------------------------------------------------------------- report
def _round_half_up(v, digits: int) -> str:
    """Decimal half-up rounding, so 0.25 prints as 0.3 (the paper's convention), not Python's 0.2."""
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP))


def fmt(v, s, metric):
    digits = 3 if metric.startswith(("ece", "auroc")) or "spread" in metric or "Spearman" in metric else (1 if "confidence" not in metric and "success" not in metric else 3)
    if s is None or pd.isna(s):
        return _round_half_up(v, digits)
    return f"{_round_half_up(v, digits)} ± {_round_half_up(s, digits)}"


def draft_main_table(df: pd.DataFrame) -> str:
    def pick(table, bench, split, method, metric="exact", design_startswith=None):
        q = df[(df.table == table) & (df.benchmark == bench) & (df.split == split) & (df.method == method) & (df.metric == metric)]
        if design_startswith:
            q = q[q.design.str.startswith(design_startswith)]
        if q.empty:
            return "–"
        r = q.iloc[0]
        return fmt(r.value, r.spread, metric)

    cols = [("Sudoku Test", "sudoku", "test"), ("Sudoku DF", "sudoku", "df"), ("Maze Test", "maze", "test"), ("PPBench val", "ppbench", "val")]
    lines = ["| Method | " + " | ".join(c[0] for c in cols) + " | Sudoku DF, corrupted guide s=16 |", "|---|" + "---|" * (len(cols) + 1)]

    def row(label, getter, corr=None):
        cells = [getter(b, s) for _, b, s in cols]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {corr or '–'} |")

    row("Deterministic TRM (fold s.d.)", lambda b, s: pick("headline", b, s, "deterministic TRM", design_startswith="5 folds") if b != "ppbench" else pick("ppbench_rules", b, s, "deterministic TRM"))
    row("Single rollout (first particle of the beta=0 run)", lambda b, s: pick("headline", b, s, "single rollout (particle 0)", design_startswith="5 seeds") if b != "ppbench" else pick("ppbench_rules", b, s, "single rollout (particle 0)"))
    row("Plurality vote", lambda b, s: pick("selection_rules", b, s, RULE_LABELS["plurality"]) if b != "ppbench" else pick("ppbench_rules", b, s, RULE_LABELS["plurality"]))
    row("Terminal-Q (PTRM-style best-of-S)", lambda b, s: pick("selection_rules", b, s, RULE_LABELS["terminal_q"]) if b != "ppbench" else pick("ppbench_rules", b, s, RULE_LABELS["terminal_q"]),
        pick("corruption", "sudoku", "df", RULE_LABELS["terminal_q"], "exact@s=16"))
    row("Path-sum tilt (no resampling)", lambda b, s: pick("selection_rules", b, s, RULE_LABELS["path_sum"]) if b != "ppbench" else pick("ppbench_rules", b, s, RULE_LABELS["path_sum"]),
        pick("corruption", "sudoku", "df", RULE_LABELS["path_sum"], "exact@s=16"))
    row("Sequential guiding (ours)", lambda b, s: pick("headline", b, s, "sequential guiding (ours)", design_startswith="5 seeds") if b != "ppbench" else pick("ppbench_rules", b, s, "sequential guiding (ours)"))
    row("Oracle (unguided cloud)", lambda b, s: pick("selection_rules", b, s, RULE_LABELS["oracle"]) if b != "ppbench" else pick("ppbench_rules", b, s, RULE_LABELS["oracle"]))
    lines.append("")
    lines.append("Designs: single rollout and sequential guiding are the 5 seeds x 5 folds runs (Sudoku Test 9,984; DF 1,408; Maze Test 992); the deterministic TRM has no seed variation, so its spread is the s.d. over the 5 folds; "
                 "the rescoring rules and the oracle are the full-set unguided traces, 5 folds of one seed (Sudoku Test 10,020; DF 1,427; Maze Test 1,000); "
                 "PPBench is 5 seeds on the 253-puzzle validation split, no tuning. The corrupted column rescores the DF traces with iid Gaussian noise of sd 16 on the raw logits.")
    return "\n".join(lines)


def render(df: pd.DataFrame) -> str:
    out = ["# Evidence register", "",
           "Generated by `scripts/analysis/build_evidence_register.py` from saved artefacts only. "
           "Percentages in points; spreads are sample s.d. over the cells named in `design` unless marked ci95.", "",
           "## Compute-matched main table (paper Table 1)", "", draft_main_table(df), ""]
    for table, g in df.groupby("table", sort=False):
        out += [f"## {table}", "", "| benchmark | split | method | metric | value | n_cells | design | cohort_n | source | note |", "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            out.append(f"| {r.benchmark} | {r.split} | {r.method} | {r.metric} | {fmt(r.value, r.spread, r.metric)} | "
                       f"{'' if pd.isna(r.n_cells) else int(r.n_cells)} | {r.design} | {'' if pd.isna(r.cohort_n) else int(r.cohort_n)} | `{r.source}` | {r.note} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild the evidence register from the cached artefacts.")
    ap.add_argument("--out", type=Path, default=OUT, help="output directory for metrics.csv and REPORT.md")
    out = ap.parse_args().out
    headline()
    full_traces()
    ppbench()
    depth_and_particles()
    cohorts_and_budgets()
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics.csv", index=False)
    (out / "REPORT.md").write_text(render(df))
    print(f"{len(df)} rows -> {out / 'metrics.csv'}")
    print(draft_main_table(df))


if __name__ == "__main__":
    main()
