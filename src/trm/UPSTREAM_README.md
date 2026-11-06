# Fast, easy-to-play-with Tiny Recursive Models

This is an implementation of the [Tiny Recursive Model (TRM)](https://arxiv.org/pdf/2510.04871v1). 

Train a TRM in a few minutes on an A10. Reproduce the official TRM results. Push the envelope.

# Motivation

Recently, recursive models made a big comeback, notably with Tiny Recursive Models, which [won 1st paper award at the ARC-AGI 2 contest](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.youtube.com/watch%3Fv%3DP9zzUM0PrBM&ved=2ahUKEwiO86mcjfWRAxW7BfsDHeyvHsUQwqsBegQIFhAB&usg=AOvVaw12B_Wsm15MK-AVqrZVbGAS) and maybe more importantly, reached an impressive level of performance on several benchmarks such as ARC-AGI 2 and Sudoku Extreme. TRM brings a lot of simplicity to its ancestor, HRM. However, the codebase inherits much of HRM's legacy.

We propose a clean implementation of TRM. We call it "nano" because it is easy to experiment with, yet incorporates all important implementation details of TRM. The project uses hydra, torch lightning and uv to make experimentation easy. We propose an [in-code introductory video](https://youtu.be/8Gzv5tGmJ1M) and small datasets (Sudoku 4x4 and 6x6) that lets you train a TRM on an A10 in a few minutes! 

This repo reproduces the results on Sudoku Extreme and Maze Hard (87% and 75% exact accuracy on validation, respectively). We hope you will find this repo useful for your own experimentation on TRM.

# Installation (BIGR integration)

In this repository nano-trm is vendored under `src/trm` and installed with the
`bigr` package (`pip install -e .`). Training and evaluation commands run as:

```
python -m trm.nn.train ...
```

The dataset builders live under `scripts/data/`. The rest of this file is the
upstream README, kept for reference; some sections describe files that were
removed from this fork (see NOTICES.md).


# Sudoku Extreme

Generate data: 

`python scripts/data/build_sudoku_extreme_dataset.py --output-dir ./data/sudoku_extreme_1k_aug_1k --subsample-size 1000 --num-aug 1000 --eval-ratio 0.01`

Run a training: 

`python -m trm.nn.train experiment=trm_sudoku_extreme_1k_aug_1k`

Training time ~1h on an H100 SXM5. You should get to ~87% exact accuracy on validation (same as the reference implementation)

# Maze Hard

Generate data: 

`python scripts/data/build_maze_dataset.py --output-dir ./data/maze-30x30-hard-1k --num-aug 0 --eval-ratio 1.0`

Run a training: 

`python -m trm.nn.train experiment=trm_maze`

Training time ~2h on an H100 SXM5. You should get to ~75% exact accuracy on validation (same as the reference implementation)

# Small Sudoku datasets

Generate data:

`python scripts/data/generate_sudoku_data.py --grid-size 4 --num-train 10000 --num-val 1000 --num-test 1000 --output-dir ./data/sudoku_4x4_small` -> choose which dataset you want to generate

Run a training:

`python -m trm.nn.train experiment=trm_sudoku_4x4`

This take a few minutes on a A10!

# ARC-AGI

Download the data from the [kaggle challenge page](https://www.kaggle.com/competitions/arc-prize-2025).

# Visualizations

Sudoku:
- Evaluate and generate a gif: `python -m trm.nn.evaluate checkpoint=./checkpoints/smooth-sunset-204.ckpt data_dir=./data/sudoku-extreme-1k-aug-1k visualize=true save_gif=true min_steps=9`

Maze:

- Generate a dataset with test data (not just val): `python scripts/data/build_maze_dataset.py --output-dir ./data/maze-30x30-hard-1k --num-aug 0 --eval-ratio 0.5`
- Evaluate and generate a gif: `python -m trm.nn.evaluate checkpoint=./checkpoints/stellar-shape-194.ckpt data_dir=./data/maze-30x30-hard-1k visualize=true save_gif=true min_steps=9 task=maze`

# Technical Notes

## Codebase structure

- `src/trm/nn/train.py`: main training script
- `src/trm/nn/models/trm.py`: TRM model
- `src/trm/nn/configs/experiments`: main experimentation configurations.

## Installing AdamATan2

This project uses the vanilla AdamW optimizer. If you want AdamATan2 and struggle to install it, here is how to install it from source:

```
# Clone the repo
cd /tmp
git clone https://github.com/imoneoi/adam-atan2.git

cd adam-atan2/

pip install --verbose --no-cache-dir --no-build-isolation -e .

# Test
cd /home/ubuntu/nano-trm
python
from adam_atan2 import AdamATan2 -> this should work
```

AdamAtan2 is not needed to reproduce the official results on Sudoku Extreme and Maze Hard. It might be needed on harder problems / deeper models (e.g. ARC AGI 2)

# Contact

Follow me on [X](https://x.com/olivkoch)

My [homepage](https://olivkoch.github.io)
