"""Figure renderers on synthetic k-fold runs, and the wiring of scripts/regenerate_figures.sh."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from bigr.plots.kfold import has_summary_schema, student_t_multiplier


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = REPO_ROOT / "scripts" / "figs"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _fold_manifest(num_folds: int) -> dict:
    return {
        "num_folds": num_folds,
        "fold_seed": 31,
        "folds": [
            {
                "fold_index": idx,
                "fold_name": f"fold_{idx + 1:02d}",
                "num_examples": 8,
                "num_batches": 1,
                "path": f"fold_{idx + 1:02d}",
            }
            for idx in range(num_folds)
        ],
    }


def _summary_from_folds(frames: list[pd.DataFrame], group_cols: list[str]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    value_cols = [col for col in df.columns if col not in group_cols and col != "fold"]
    summary = df.groupby(group_cols)[value_cols].agg(["mean", "std"]).reset_index()
    flat: list[str] = []
    for column in summary.columns:
        if isinstance(column, tuple):
            base, stat = column
            flat.append(f"{base}_{stat}" if stat else str(base))
        else:
            flat.append(str(column))
    summary.columns = flat
    return summary


def _create_pf_kfold(root: Path, name: str, *, guided_offset: float) -> Path:
    run_dir = root / f"{name}_k5"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "fold_manifest.json", _fold_manifest(2))
    _write_json(
        run_dir / "config.json",
        {
            "split": "test",
            "particles": 2,
            "sigma": 0.3,
            "beta": guided_offset,
            "total_steps": 48,
            "ess_threshold": 0.5,
            "batch_size": 4,
            "n_batches": 2,
            "seed": 31,
            "num_folds": 2,
            "fold_seed": 31,
        },
    )

    step_frames = []
    rank_frames = []
    for fold_index, delta in enumerate((0.0, 0.05), start=1):
        fold_dir = run_dir / f"fold_{fold_index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        step_df = pd.DataFrame(
            {
                "step": [3, 48],
                "exact_accuracy": [0.55 + guided_offset + delta, 0.75 + guided_offset + delta],
                "any_exact_rate": [0.65 + guided_offset + delta, 0.88 + guided_offset + delta],
            }
        )
        rank_df = pd.DataFrame(
            {
                "step": [3, 3, 48, 48],
                "rank": [1, 2, 1, 2],
                "exact_rate": [0.60 + delta, 0.40 + delta, 0.82 + delta, 0.58 + delta],
            }
        )
        trm_df = pd.DataFrame({"step": [3, 48], "exact_solved": [0.45 + delta, 0.70 + delta]})

        step_df.to_csv(fold_dir / "step_metrics.csv", index=False)
        rank_df.to_csv(fold_dir / "rank_profiles.csv", index=False)
        trm_df.to_csv(fold_dir / "trm_depth_profile.csv", index=False)
        np.save(fold_dir / "ckpt_weights_step3.npy", np.array([[0.8, 0.2], [0.7, 0.3]], dtype=np.float32))
        np.save(fold_dir / "ckpt_weights_step48.npy", np.array([[0.9, 0.1], [0.85, 0.15]], dtype=np.float32))

        step_frames.append(step_df.assign(fold=fold_index))
        rank_frames.append(rank_df.assign(fold=fold_index))

    _summary_from_folds(step_frames, ["step"]).to_csv(run_dir / "step_metrics_summary.csv", index=False)
    _summary_from_folds(rank_frames, ["step", "rank"]).to_csv(run_dir / "rank_profiles_summary.csv", index=False)
    return run_dir


def _create_continuation_kfold(root: Path) -> Path:
    run_dir = root / "continuation_labels_k5"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "fold_manifest.json", _fold_manifest(2))
    steps = np.array([3, 12, 24, 48], dtype=np.int64)
    for fold_index, offset in enumerate((0.0, 0.25), start=1):
        fold_dir = run_dir / f"fold_{fold_index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            fold_dir / "config.json",
            {
                "particles": 3,
                "sigma": 0.3,
                "beta": 0.25,
                "ess_threshold": 0.5,
                "batch_size": 4,
                "n_batches": 2,
                "total_steps": 48,
                "seed": 31,
                "num_folds": 2,
                "fold_seed": 31,
            },
        )
        q_raw = np.zeros((len(steps), 4, 3), dtype=np.float32)
        g_labels = np.zeros((len(steps), 4, 3), dtype=np.float32)
        for step_index in range(len(steps)):
            q_raw[step_index] = np.array(
                [
                    [2.2 + offset + step_index * 0.1, 1.2 + offset, -0.3 + offset],
                    [2.0 + offset, 1.0 + offset, -0.5 + offset],
                    [1.8 + offset, 0.9 + offset, -0.7 + offset],
                    [1.7 + offset, 0.8 + offset, -0.9 + offset],
                ],
                dtype=np.float32,
            )
            g_labels[step_index] = np.array(
                [
                    [1, 1, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 0, 0],
                ],
                dtype=np.float32,
            )
        np.savez(fold_dir / "continuation_labels.npz", steps=steps, q_raw=q_raw, g_labels=g_labels)
    return run_dir


def _create_entropy_fold(run_dir: Path, split: str, offset: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "fold_manifest.json", _fold_manifest(2))
    for fold_index, delta in enumerate((0.0, 0.05), start=1):
        fold_dir = run_dir / f"fold_{fold_index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            fold_dir / "config.json",
            {
                "split": split,
                "particles": 1,
                "sigma": 0.3,
                "beta": 0.25,
                "ess_threshold": 0.5,
                "batch_size": 2,
                "n_batches": 1,
                "temporal_stride": 3,
                "total_steps": 48,
                "seed": 31,
                "num_folds": 2,
                "fold_seed": 31,
            },
        )
        inputs = np.zeros((2, 81), dtype=np.int64)
        labels = np.full((2, 81), 3, dtype=np.int64)
        preds = np.zeros((2, 1, 81), dtype=np.int64)
        preds[0, 0] = 3
        preds[1, 0] = 4
        weights = np.ones((2, 1), dtype=np.float32)
        path_auc = np.stack(
            [
                np.full(81, 0.10 + offset + delta, dtype=np.float32),
                np.full(81, 0.55 + offset + delta, dtype=np.float32),
            ],
            axis=0,
        )
        np.savez(
            fold_dir / "entropy_data.npz",
            inputs=inputs,
            labels=labels,
            preds=preds,
            weights=weights,
            path_auc_uncert=path_auc,
        )


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FIGS_DIR / script), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_kfold_helpers_basic() -> None:
    assert student_t_multiplier(2) > 0.0
    assert has_summary_schema(pd.DataFrame({"metric_mean": [1.0], "metric_std": [0.1]}))


def test_pf_main_kfold_smoke_render(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    guided_test = _create_pf_kfold(runs, "pf", guided_offset=0.08)
    unguided_test = _create_pf_kfold(runs, "pf_unguided", guided_offset=0.0)
    guided_df = _create_pf_kfold(runs, "pf_impossible", guided_offset=0.05)
    unguided_df = _create_pf_kfold(runs, "pf_impossible_unguided", guided_offset=-0.2)
    out_dir = tmp_path / "figures"

    _run_script(
        "fig_pf_main.py",
        "--guided-test-dir", str(guided_test),
        "--unguided-test-dir", str(unguided_test),
        "--guided-df-dir", str(guided_df),
        "--unguided-df-dir", str(unguided_df),
        "--out", str(out_dir),
    )

    assert (out_dir / "fig_main_result.pdf").exists()
    assert (out_dir / "fig_main_result.png").exists()


def test_guide_alignment_kfold_smoke_render(tmp_path: Path) -> None:
    run_dir = _create_continuation_kfold(tmp_path / "runs")
    out_dir = tmp_path / "figures"

    # The figure reads a Sudoku and a Maze continuation run; the synthetic run stands in for both.
    _run_script(
        "fig_guide_alignment.py",
        "--sudoku-cont-dir",
        str(run_dir),
        "--maze-cont-dir",
        str(run_dir),
        "--beta-op",
        "1.0",
        "--out",
        str(out_dir),
    )

    assert (out_dir / "fig_guide_alignment.pdf").exists()


def test_token_entropy_kfold_smoke_render(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _create_entropy_fold(runs / "entropy_test_k5", "test", 0.0)
    _create_entropy_fold(runs / "entropy_impossible_k5", "impossible", 0.1)
    out_dir = tmp_path / "figures"

    _run_script(
        "fig_token_entropy.py",
        "--test-dir",
        str(runs / "entropy_test_k5"),
        "--df-dir",
        str(runs / "entropy_impossible_k5"),
        "--out",
        str(out_dir),
    )

    assert (out_dir / "fig_token_entropy.pdf").exists()


def _run_regenerate_figures(tmp_path: Path, *flags: str) -> str:
    """Run scripts/regenerate_figures.sh in a scratch tree with a `python` shim on PATH
    that only logs its arguments, and return the log (one tab-separated line per call)."""
    workdir = tmp_path / "repo"
    (workdir / "scripts").mkdir(parents=True)
    script = workdir / "scripts" / "regenerate_figures.sh"
    shutil.copy(REPO_ROOT / "scripts" / "regenerate_figures.sh", script)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "python.log"
    shim = bin_dir / "python"
    shim.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            {{
                printf 'INVOKE'
                for a in "$@"; do printf '\\t%s' "$a"; done
                printf '\\n'
            }} >> {log}
            exit 0
            """
        )
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env.pop("PYTHON", None)
    subprocess.run(
        ["bash", str(script), *flags],
        cwd=workdir, env=env, check=True, capture_output=True, text=True,
    )
    return log.read_text()


SEED_FOLD_PATHS = [
    "assets/runs/sudoku/pf_seed_fold_test",
    "assets/runs/sudoku/pf_seed_fold_test_unguided",
    "assets/runs/sudoku/pf_seed_fold_impossible",
    "assets/runs/sudoku/pf_seed_fold_impossible_unguided",
]
QUICK_PATHS = [
    "assets/runs/sudoku/pf_k5",
    "assets/runs/sudoku/pf_unguided_k5",
    "assets/runs/sudoku/pf_impossible_k5",
    "assets/runs/sudoku/pf_impossible_unguided_k5",
]
EXPECTED_INVOCATIONS = [
    ("scripts/figs/fig_inner_deviation.py", "assets/runs/sudoku/inner_deviation_k5"),
    ("scripts/figs/fig_inner_deviation.py",
     "assets/runs/sudoku/sigma_guided_results_k5/sigma_guided_results.csv"),
    ("scripts/figs/fig_guide_alignment.py", "assets/runs/sudoku/continuation_labels_k5"),
    ("scripts/figs/fig_guide_alignment.py", "assets/runs/maze/continuation_labels_maze_hard_k5"),
    ("scripts/figs/fig_token_entropy.py", "assets/runs/sudoku/entropy_test_k5"),
    ("scripts/figs/fig_maze_case_study.py", "assets/runs/maze/inner_deviation_maze_hard_k5"),
    ("scripts/figs/fig_maze_case_study.py", "assets/runs/sudoku/q_training"),
    ("scripts/figs/fig_ablation_sensitivity.py",
     "assets/runs/sudoku/ablation_sensitivity_k5/resample_stress_results_summary.csv"),
    ("scripts/figs/fig_ablation_sensitivity.py",
     "assets/runs/maze/ablation_sensitivity_maze_hard_k5/resample/summary_summary.csv"),
    ("scripts/figs/fig_token_contraction.py", "assets/runs/sudoku/entropy_test_k5"),
    ("scripts/figs/fig_checkpoint.py", "assets/runs/sudoku/pf_checkpoint_sweep"),
]


def test_regenerate_figures_default_is_paper_faithful(tmp_path: Path) -> None:
    """With no flags the driver renders Figures 1-8 from the paper's run directories,
    Figure 1 from the 5-seed x 5-fold runs."""
    log_text = _run_regenerate_figures(tmp_path)

    for path in SEED_FOLD_PATHS:
        assert path in log_text, f"Fig 1 should read {path}\n--- log ---\n{log_text}"
    for path in QUICK_PATHS:
        assert path not in log_text, f"default run must not read {path}\n--- log ---\n{log_text}"
    for script_path, run_dir in EXPECTED_INVOCATIONS:
        assert script_path in log_text, f"missing invocation: {script_path}\n--- log ---\n{log_text}"
        assert run_dir in log_text, f"missing path {run_dir!r} for {script_path}\n--- log ---\n{log_text}"
    assert log_text.count("INVOKE") == 8, "expected exactly eight figure renders"


def test_regenerate_figures_quick_uses_single_seed_runs(tmp_path: Path) -> None:
    """`--quick` swaps Figure 1 to the single-seed 5-fold runs and leaves the rest alone."""
    log_text = _run_regenerate_figures(tmp_path, "--quick")

    for path in QUICK_PATHS:
        assert path in log_text, f"--quick should read {path}\n--- log ---\n{log_text}"
    for path in SEED_FOLD_PATHS:
        assert path not in log_text, f"--quick must not read {path}\n--- log ---\n{log_text}"
    for script_path, run_dir in EXPECTED_INVOCATIONS:
        assert script_path in log_text and run_dir in log_text
