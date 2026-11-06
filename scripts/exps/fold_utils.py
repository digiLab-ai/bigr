"""Shared k-fold helpers for the experiment scripts: flatten loaded batches into examples,
shuffle them deterministically into folds, repack them, and aggregate per-fold frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import pandas as pd


@dataclass(frozen=True)
class FoldBatchInfo:
    fold_index: int
    num_examples: int
    num_batches: int


def flatten_batches(batches: list[dict]) -> list[dict]:
    """Flatten a batch list into individual example dictionaries."""
    examples: list[dict] = []
    for batch in batches:
        inputs = batch["inputs"]
        labels = batch["labels"]
        puzzle_identifiers = batch.get(
            "puzzle_identifiers",
            torch.zeros(inputs.shape[0], dtype=torch.long, device=inputs.device),
        )

        for i in range(inputs.shape[0]):
            examples.append(
                {
                    "inputs": inputs[i].detach().cpu(),
                    "labels": labels[i].detach().cpu(),
                    "puzzle_identifiers": puzzle_identifiers[i].detach().cpu(),
                }
            )
    return examples


def split_examples_into_folds(
    examples: list[dict], num_folds: int, seed: int
) -> list[list[dict]]:
    """Shuffle examples deterministically and split them into ``num_folds`` parts."""
    if num_folds < 1:
        raise ValueError(f"num_folds must be >= 1, got {num_folds}")
    if not examples:
        return [[] for _ in range(num_folds)]
    if len(examples) < num_folds:
        raise ValueError(
            f"Cannot split {len(examples)} examples into {num_folds} non-empty folds"
        )

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(examples))
    fold_indices = np.array_split(order, num_folds)
    folds: list[list[dict]] = []
    for fold_ids in fold_indices:
        folds.append([examples[int(idx)] for idx in fold_ids.tolist()])
    return folds


def examples_to_batches(examples: list[dict], batch_size: int) -> list[dict]:
    """Pack example dictionaries back into model batches."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    batches: list[dict] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        if not chunk:
            continue
        batches.append(
            {
                "inputs": torch.stack([ex["inputs"] for ex in chunk], dim=0),
                "labels": torch.stack([ex["labels"] for ex in chunk], dim=0),
                "puzzle_identifiers": torch.stack(
                    [ex["puzzle_identifiers"] for ex in chunk], dim=0
                ).long(),
            }
        )
    return batches


def build_example_folds(
    batches: list[dict], num_folds: int, seed: int, batch_size: int
) -> tuple[list[list[dict]], list[FoldBatchInfo]]:
    """Split a loaded batch list into fold-aligned mini-batch lists."""
    examples = flatten_batches(batches)
    fold_examples = split_examples_into_folds(examples, num_folds=num_folds, seed=seed)

    fold_batches: list[list[dict]] = []
    fold_info: list[FoldBatchInfo] = []
    for fold_index, fold in enumerate(fold_examples):
        packed = examples_to_batches(fold, batch_size=batch_size)
        fold_batches.append(packed)
        fold_info.append(
            FoldBatchInfo(
                fold_index=fold_index,
                num_examples=len(fold),
                num_batches=len(packed),
            )
        )
    return fold_batches, fold_info


def summarize_numeric_df(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate numeric columns with mean/std and flatten the resulting schema."""
    value_cols = [
        col
        for col in df.columns
        if col not in group_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    grouped = df.groupby(group_cols)[value_cols].agg(["mean", "std"]).reset_index()
    flat_columns: list[str] = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            base, stat = col
            flat_columns.append(f"{base}_{stat}" if stat else str(base))
        else:
            flat_columns.append(str(col))
    grouped.columns = flat_columns
    return grouped
