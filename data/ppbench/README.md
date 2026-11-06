# PPBench splits used in the paper

Four puzzle types of Pencil Puzzle Bench (Waugh et al., 2026; MIT) merged into one task in
the TRM on-disk format. These files are the asset of record for every PPBench number in
the paper; the backbone in `ppbench/checkpoints/last.ckpt` was trained on `train/`.

| Split | Directory | Puzzles | sudoku | lightup | nurikabe | tapa | Source |
|---|---|---|---|---|---|---|---|
| train | `train/` | 33,460 (3,346 puzzles x 10 examples) | 7,110 | 8,660 | 14,380 | 3,310 | `full_dataset.jsonl` |
| validation | `val/` | 253 | 95 | 63 | 67 | 28 | `full_dataset.jsonl` |
| golden | `test/` | 42 | 15 | 8 | 9 | 10 | `golden_300.jsonl` |

The golden benchmark is stored as `test/` so that the loaders and `run_pf.py --split test`
work unchanged.

**Derivation.** From the PPBench release (`bluecoconut/pencil-puzzle-bench` on the Hugging
Face Hub), keeping classic 9x9 sudoku and 10x10 lightup, nurikabe and tapa. Golden: the
puzzles of those types and sizes in `golden_300.jsonl`. Validation: 100 puzzles per type (50
for tapa) sampled from `full_dataset.jsonl` with seed 42 before the size filter, following
the PTRM protocol. Train: the remaining puzzles, each as 10 examples: the clue grid plus nine
random combinations of a partial-solution prefix and one of the eight dihedral transforms.
Puzzles are de-duplicated across splits on their puzz.link URL. Clue and solved grids were
rendered with the pzpr.js engine through the `ppbench` package.

**Encoding.** Each example is a 10x10 grid flattened row-major to `seq_len = 100` (sudoku
occupies the top-left 9x9; the 19 off-grid cells are PAD = 0 in the input and -100 in the
label). Tokens are namespaced per type, so the token stream identifies the puzzle type:
`metadata.json["token_map"]` lists all 114 `type|symbol` tokens (`vocab_size = 115` with PAD).
`all__types.npy` gives the type index of every row (`metadata.json["type_index"]`); every
row has puzzle identifier 0.

The build script that produced these arrays is not part of this release; the recipe above and
`metadata.json` record its settings.
