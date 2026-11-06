"""Helpers for reading k-fold and seed-fold run directories and their fold statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def load_json(path: Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def is_kfold_run(run_dir: Path) -> bool:
    run_dir = Path(run_dir)
    return (run_dir / "fold_manifest.json").exists()


def is_seed_fold_run(run_dir: Path) -> bool:
    run_dir = Path(run_dir)
    return (run_dir / "seed_manifest.json").exists()


def load_fold_manifest(run_dir: Path) -> dict:
    manifest_path = Path(run_dir) / "fold_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing fold manifest: {manifest_path}")
    return load_json(manifest_path)


def load_seed_manifest(run_dir: Path) -> dict:
    manifest_path = Path(run_dir) / "seed_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing seed manifest: {manifest_path}")
    return load_json(manifest_path)


def fold_dirs(run_dir: Path) -> list[Path]:
    manifest = load_fold_manifest(run_dir)
    run_dir = Path(run_dir)
    folds = manifest.get("folds")
    if folds is None:
        folds = manifest.get("stress_folds")
    if folds is None:
        raise ValueError(f"Manifest in {run_dir} does not contain fold entries")

    fold_paths: list[Path] = []
    for fold in folds:
        rel = fold.get("path")
        if rel:
            fold_paths.append(run_dir / rel)
            continue
        idx = int(fold["fold_index"]) + 1
        fold_paths.append(run_dir / f"fold_{idx:02d}")
    return fold_paths


def seed_fold_dirs(run_dir: Path) -> list[Path]:
    """Return all (seed, fold) leaf directories for a seed-fold run."""
    manifest = load_seed_manifest(run_dir)
    run_dir = Path(run_dir)
    leaves: list[Path] = []
    for entry in manifest.get("seed_entries", []):
        seed_path = run_dir / entry["path"]
        for f_idx in range(int(entry.get("num_folds", 0))):
            leaves.append(seed_path / f"fold_{f_idx + 1:02d}")
    return leaves


def num_folds_from_run(run_dir: Path) -> int:
    """Effective sample size for student-t CI bounds.

    For seed-fold runs (5 seeds × 5 folds), returns the total ``num_runs``
    from ``seed_manifest.json`` so the t-distribution gets the correct
    degrees of freedom. For fold-only runs, returns the number of folds.
    Returns 1 for any directory without a recognised manifest.
    """
    if is_seed_fold_run(run_dir):
        manifest = load_seed_manifest(run_dir)
        if "num_runs" in manifest:
            return int(manifest["num_runs"])
        return len(seed_fold_dirs(run_dir))
    if not is_kfold_run(run_dir):
        return 1
    manifest = load_fold_manifest(run_dir)
    return int(manifest.get("num_folds", len(fold_dirs(run_dir))))


def has_summary_schema(df: pd.DataFrame) -> bool:
    columns = set(df.columns)
    return any(col.endswith("_mean") and f"{col[:-5]}_std" in columns for col in columns)


def student_t_multiplier(num_folds: int, confidence: float = 0.95) -> float:
    if num_folds <= 1:
        return 0.0
    alpha = 0.5 * (1.0 + confidence)
    return float(student_t.ppf(alpha, df=num_folds - 1))


def ci_half_width(std: np.ndarray | pd.Series | float, num_folds: int) -> np.ndarray:
    arr = np.asarray(std, dtype=float)
    if num_folds <= 1:
        return np.zeros_like(arr, dtype=float)
    return student_t_multiplier(num_folds) * arr / np.sqrt(float(num_folds))


def ci_bounds(
    mean: np.ndarray | pd.Series | float,
    std: np.ndarray | pd.Series | float,
    num_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_arr = np.asarray(mean, dtype=float)
    half = ci_half_width(std, num_folds)
    return half, mean_arr - half, mean_arr + half


def normalise_summary_columns(df: pd.DataFrame, num_folds: int) -> pd.DataFrame:
    """Expand `<metric>_mean/<metric>_std` columns with CI bounds."""
    out = df.copy()
    seen: list[str] = []
    for column in df.columns:
        if not column.endswith("_mean"):
            continue
        base = column[:-5]
        std_col = f"{base}_std"
        if std_col not in df.columns or base in seen:
            continue
        seen.append(base)
        out[base] = out[column]
        half, lo, hi = ci_bounds(out[column].to_numpy(), out[std_col].to_numpy(), num_folds)
        out[f"{base}_ci"] = half
        out[f"{base}_lo"] = lo
        out[f"{base}_hi"] = hi
    return out


def aggregate_numeric_frames(
    frames: list[pd.DataFrame],
    *,
    group_cols: list[str],
) -> pd.DataFrame:
    if not frames:
        raise ValueError("No frames supplied for aggregation")
    merged = pd.concat(frames, ignore_index=True)
    value_cols = [
        column
        for column in merged.columns
        if column not in group_cols and pd.api.types.is_numeric_dtype(merged[column])
    ]
    grouped = merged.groupby(group_cols)[value_cols].agg(["mean", "std"]).reset_index()
    flat_columns: list[str] = []
    for column in grouped.columns:
        if isinstance(column, tuple):
            base, stat = column
            flat_columns.append(f"{base}_{stat}" if stat else str(base))
        else:
            flat_columns.append(str(column))
    grouped.columns = flat_columns
    return grouped


def aggregate_fold_csvs(
    run_dir: Path,
    relative_path: str,
    *,
    group_cols: list[str],
    derived: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold_dir in fold_dirs(run_dir):
        frame = pd.read_csv(fold_dir / relative_path)
        if derived is not None:
            frame = derived(frame.copy())
        frames.append(frame)
    return aggregate_numeric_frames(frames, group_cols=group_cols).sort_values(group_cols)


def aggregate_trm_depth_profiles(run_dir: Path) -> pd.DataFrame:
    """Aggregate per-fold deterministic depth profiles with fold CIs.

    Supports both fold-only (``fold_*/trm_depth_profile.csv``) and seed-fold
    (``seed_*/fold_*/trm_depth_profile.csv``) layouts. The TRM trace is
    deterministic, so seeds contribute identical curves; pooling them just
    matches the t-distribution df to the seed-fold pf summary.
    """
    leaves = seed_fold_dirs(run_dir) if is_seed_fold_run(run_dir) else fold_dirs(run_dir)
    frames: list[pd.DataFrame] = []
    for index, leaf_dir in enumerate(leaves, start=1):
        csv_path = leaf_dir / "trm_depth_profile.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path).copy()
        if "global_cycle" in frame.columns and "step" not in frame.columns:
            frame = frame.rename(columns={"global_cycle": "step"})
        if "exact_accuracy" in frame.columns and "exact_solved" not in frame.columns:
            frame = frame.rename(columns={"exact_accuracy": "exact_solved"})
        frame["fold"] = index
        frames.append(frame)
    summary = aggregate_numeric_frames(frames, group_cols=["fold", "step"]).sort_values("step")
    return normalise_summary_columns(summary, num_folds=len(frames))


def markdown_table(df: pd.DataFrame, *, digits: int = 4) -> str:
    """Render a compact deterministic Markdown table."""
    table = df.copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: f"{value:.{digits}f}")
    headers = [str(column) for column in table.columns]
    rows = [
        [str(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body_lines])
