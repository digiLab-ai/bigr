# Third-party notices

The code written for this repository is released under the Apache License 2.0 (`LICENSE`,
`NOTICE`). The components below are included under their own licences.

## TRM backbone (`src/trm/`)

`src/trm/` is a trimmed fork of **nano-trm** (https://github.com/olivkoch/nano-trm,
commit `e5659b1f5b309883e936b8b829ccaa09b3df5eb0`), a compact reimplementation of the
Tiny Recursive Model (Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny
Networks*, 2025), itself derived from the Hierarchical Reasoning Model code (Wang et al.,
2025). It is retained under its upstream **MIT licence**: the original licence text is in
`src/trm/UPSTREAM_LICENSE` and the original README in `src/trm/UPSTREAM_README.md`.

Modifications relative to upstream:

* Removed everything not used by the paper: the ARC-AGI pipeline (solver, evaluator,
  datamodule, viewer), the MLP model, the 4x4/6x6/9x9 Sudoku, XOR and dummy datamodules
  and configs, W&B sweep tooling, and the vendored `adam_atan2` and `stable_max_loss`
  modules (the model imports the `adam-atan2` package when `use_adam_atan2` is set).
* Kept the Hydra training entry point (`trm.nn.train`), the evaluation entry point
  (`trm.nn.evaluate`, which writes the deterministic-failure split), the Sudoku/Maze
  evaluators, the EMA and PPBench callbacks, and the three experiment configs used in
  the paper (`trm_sudoku_extreme_1k_aug_1k`, `trm_maze`, `trm_ppbench`).
* Added `nn/configs/data/ppbench.yaml`; cleared the private W&B entity in
  `nn/configs/logger/wandb.yaml`; the validation step stashes the greedy prediction for
  the metric callbacks.

### Hierarchical Reasoning Model code (Apache License 2.0)

nano-trm, and therefore `src/trm/`, descends from the HRM reference code
(https://github.com/sapientinc/HRM, Wang et al., 2025), released under the Apache
License 2.0. The model layers (`nn/modules/trm_block.py`, `nn/modules/utils.py`), the
sparse puzzle embedding (`nn/modules/sparse_embeddings.py`) and the dataset builders
`scripts/data/build_sudoku_extreme_dataset.py` and `scripts/data/build_maze_dataset.py`
contain code adapted from HRM's `models/` and `dataset/` directories; each has been
modified for this repository. The Apache License 2.0 text is the repository's `LICENSE`.

## Datasets

* **Sudoku-Extreme** (https://huggingface.co/datasets/sapientinc/sudoku-extreme) and
  **Maze-Hard** (https://huggingface.co/datasets/sapientinc/maze-30x30-hard-1k) were released
  by the authors of the Hierarchical Reasoning Model (Wang et al., 2025). Neither dataset
  card states a licence (checked November 2026); the data are not redistributed here, and
  `scripts/data/build_sudoku_extreme_dataset.py` / `build_maze_dataset.py` download and
  package them from the Hub. The evaluation cohorts the paper used are not redistributed
  either: `scripts/data/build_eval_cohorts.py` rebuilds them from these releases using the
  index files in `assets/cohorts/`.
* **PPBench (Pencil Puzzle Bench)** — Waugh et al., 2026, MIT licence,
  https://github.com/approximatelabs/pencil-puzzle-bench. The four-type training,
  validation and golden splits used in the paper are redistributed under `data/ppbench/`
  in the TRM on-disk format, under the same licence.

## Optimisers

* `src/trm/nn/optimizers/muon.py` — Muon, originally released by Keller Jordan under MIT.
* `adam-atan2` (PyPI, Apache 2.0) — optional, only for training the backbones.

## Everything else

All other Python dependencies are listed in `pyproject.toml` and pinned in
`requirements.txt`; each is used under its own licence.
