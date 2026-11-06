"""Build the paper's evaluation cohorts in the TRM on-disk format from the HRM releases.

The cohorts the paper evaluates (Sudoku-Extreme Test 10,020 and DF 1,427 puzzles; Maze-Hard
Test 1,000 and DF 134 mazes) are identified by the index files in assets/cohorts/ and written,
in the paper's row order, to <output-root>/sudoku_extreme/dataset/sudoku_extreme/{test,impossible}/
and <output-root>/maze_hard/dataset/maze-30x30-hard-1k/{test,impossible}/, the layout the
experiment scripts read:
    python scripts/data/build_eval_cohorts.py --backbone all
Provenance route: --from-traces DIR first regenerates the index files from the puzzle inputs
recorded in the paper's particle traces (DIR/{sudoku_test,sudoku_df,maze_test,maze_df}/trace/, as
fetched by download_checkpoints.py --traces into assets/runs/analysis/ablations/calibration_corruption/raw),
inverting the fold permutation of scripts/exps/fold_utils.py to restore the original order."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import csv
import json
import numpy as np

import click
from tqdm import tqdm
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
MAZE_CHARSET = "# SGo"  # wall, space, start, goal, path; token = index + 1, 0 = PAD


def sudoku_encode(s: str) -> np.ndarray:
    # 0 = PAD, 1 = separator, 2 = blank, 3-11 = digits 1-9
    return np.frombuffer(s.replace(".", "0").encode(), dtype=np.uint8) - ord("0") + 2


def sudoku_decode(tokens: np.ndarray) -> str:
    return "".join("." if t == 2 else chr(ord("0") + int(t) - 2) for t in tokens)


_maze_char2id = np.zeros(256, dtype=np.uint8)
for _i, _c in enumerate(MAZE_CHARSET):
    _maze_char2id[ord(_c)] = _i + 1


def maze_encode(s: str) -> np.ndarray:
    return _maze_char2id[np.frombuffer(s.encode(), dtype=np.uint8)]


def maze_decode(tokens: np.ndarray) -> str:
    return "".join(MAZE_CHARSET[int(t) - 1] for t in tokens)


@dataclass(frozen=True)
class Cohort:
    source_repo: str
    dataset: str
    grid_size: int
    vocab_size: int
    encode: Callable[[str], np.ndarray]
    decode: Callable[[np.ndarray], str]
    index_files: Dict[str, str]  # split -> index file stem in --cohorts-dir
    trace_dirs: Dict[str, str]  # split -> trace directory name under --from-traces
    lines_are_puzzles: bool  # index lines are puzzle strings (else 0-based test.csv rows)
    givens: bool  # dataset.json carries min_givens / max_givens


BACKBONES: Dict[str, Cohort] = {
    "sudoku_extreme": Cohort(
        source_repo="sapientinc/sudoku-extreme",
        dataset="sudoku_extreme",
        grid_size=9,
        vocab_size=12,  # PAD + separator + blank + digits 1-9
        encode=sudoku_encode,
        decode=sudoku_decode,
        index_files={"test": "sudoku_extreme_test", "impossible": "sudoku_extreme_df"},
        trace_dirs={"test": "sudoku_test", "impossible": "sudoku_df"},
        lines_are_puzzles=True,
        givens=True,
    ),
    "maze_hard": Cohort(
        source_repo="sapientinc/maze-30x30-hard-1k",
        dataset="maze-30x30-hard-1k",
        grid_size=30,
        vocab_size=len(MAZE_CHARSET) + 1,
        encode=maze_encode,
        decode=maze_decode,
        index_files={"test": "maze_hard_test", "impossible": "maze_hard_df"},
        trace_dirs={"test": "maze_test", "impossible": "maze_df"},
        lines_are_puzzles=False,
        givens=False,
    ),
}


def load_release(cohort: Cohort) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Read test.csv of the HRM release: puzzles, solutions and the puzzle -> row map."""
    questions, answers = [], []
    with open(hf_hub_download(cohort.source_repo, "test.csv", repo_type="dataset"), newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for _source, q, a, _rating in reader:
            assert len(q) == len(a) == cohort.grid_size**2
            questions.append(q)
            answers.append(a)
    by_puzzle = {q: i for i, q in enumerate(questions)}
    assert len(by_puzzle) == len(questions), f"duplicate puzzles in {cohort.source_repo}/test.csv"
    return questions, answers, by_puzzle


def recover_trace_inputs(trace_dir: Path) -> np.ndarray:
    """Undo the fold shuffle of scripts/exps/fold_utils.py and return the inputs in dataset order."""
    manifest = json.loads((trace_dir / "fold_manifest.json").read_text())
    folds = [
        np.load(trace_dir / fold["fold_name"] / "inputs.npy")
        for fold in sorted(manifest["folds"], key=lambda fold: fold["fold_index"])
    ]
    n = sum(len(fold) for fold in folds)
    order = np.random.RandomState(manifest["fold_seed"]).permutation(n)
    inputs = np.zeros((n, folds[0].shape[1]), dtype=folds[0].dtype)
    seen = np.zeros(n, dtype=bool)
    for ids, fold in zip(np.array_split(order, manifest["num_folds"]), folds):
        assert len(ids) == len(fold), f"fold sizes in {trace_dir} do not match seed {manifest['fold_seed']}"
        inputs[ids] = fold
        seen[ids] = True
    assert seen.all()
    return inputs


def write_index_from_traces(cohort: Cohort, split: str, traces_root: Path, cohorts_dir: Path,
                            by_puzzle: Dict[str, int]) -> None:
    trace_dir = traces_root / cohort.trace_dirs[split] / "trace"
    inputs = recover_trace_inputs(trace_dir)
    puzzles = [cohort.decode(row) for row in tqdm(inputs, desc=f"Decoding {trace_dir.parent.name}")]
    missing = [p for p in puzzles if p not in by_puzzle]
    assert not missing, f"{len(missing)} trace puzzles of {trace_dir} are not in {cohort.source_repo}/test.csv"
    assert len(set(puzzles)) == len(puzzles), f"repeated puzzles in {trace_dir}"
    lines = puzzles if cohort.lines_are_puzzles else [str(by_puzzle[p]) for p in puzzles]
    cohorts_dir.mkdir(parents=True, exist_ok=True)
    path = cohorts_dir / f"{cohort.index_files[split]}.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"  {path}: {len(lines)} lines from {trace_dir}")


def read_index(cohort: Cohort, split: str, cohorts_dir: Path, by_puzzle: Dict[str, int]) -> List[int]:
    """Resolve an index file to 0-based rows of test.csv, in cohort order."""
    lines = (cohorts_dir / f"{cohort.index_files[split]}.txt").read_text().split()
    if cohort.lines_are_puzzles:
        rows = [by_puzzle[line] for line in lines]
    else:
        rows = [int(line) for line in lines]
        assert all(0 <= r < len(by_puzzle) for r in rows)
    assert len(set(rows)) == len(rows), f"repeated rows in {cohort.index_files[split]}.txt"
    return rows


def check_labels(cohort: Cohort, inputs: np.ndarray, labels: np.ndarray) -> None:
    if cohort.givens:
        given = inputs != 2
        assert np.array_equal(labels[given], inputs[given])
        assert np.all((labels >= 3) & (labels <= 11))
    else:
        changed = inputs != labels
        assert np.all(inputs[changed] == 2) and np.all(labels[changed] == 5)


def write_split(save_dir: Path, inputs: np.ndarray, labels: np.ndarray, metadata: dict) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    n = len(inputs)
    with open(save_dir / "dataset.json", "w") as f:
        json.dump(metadata, f, indent=2)
    np.save(save_dir / "all__inputs.npy", inputs)
    np.save(save_dir / "all__labels.npy", labels)
    np.save(save_dir / "all__puzzle_identifiers.npy", np.zeros(n, dtype=np.int32))
    np.save(save_dir / "all__puzzle_indices.npy", np.arange(n + 1, dtype=np.int32))
    np.save(save_dir / "all__group_indices.npy", np.arange(n + 1, dtype=np.int32))


def build_backbone(name: str, output_root: Path, cohorts_dir: Path, traces_root: Path | None) -> None:
    cohort = BACKBONES[name]
    questions, answers, by_puzzle = load_release(cohort)
    print(f"{name}: {len(questions)} puzzles in {cohort.source_repo}/test.csv")
    if traces_root is not None:
        for split in cohort.trace_dirs:
            write_index_from_traces(cohort, split, traces_root, cohorts_dir, by_puzzle)

    out_dir = output_root / name / "dataset" / cohort.dataset
    rows = {split: read_index(cohort, split, cohorts_dir, by_puzzle) for split in cohort.index_files}
    assert set(rows["impossible"]) <= set(rows["test"]), "the DF cohort is not a subset of the test cohort"

    metadata = None
    for split, split_rows in rows.items():
        inputs = np.stack([cohort.encode(questions[r]) for r in tqdm(split_rows, desc=f"Encoding {name}/{split}")])
        labels = np.stack([cohort.encode(answers[r]) for r in split_rows])
        check_labels(cohort, inputs, labels)
        if metadata is None:
            metadata = {
                "num_puzzles": len(inputs),
                "num_groups": len(inputs),
                "num_augmentations": 0,
                "mean_puzzles_per_group": 1.0,
                "grid_size": cohort.grid_size,
                "max_grid_size": cohort.grid_size,
            }
            if cohort.givens:
                givens = np.count_nonzero(inputs != 2, axis=1)
                metadata.update({"min_givens": int(givens.min()), "max_givens": int(givens.max())})
            metadata.update({"seq_len": cohort.grid_size**2, "vocab_size": cohort.vocab_size})
        else:
            metadata = {**metadata, "num_puzzles": len(inputs), "num_groups": len(inputs)}
        write_split(out_dir / split, inputs, labels, metadata)
        print(f"  {split}: {len(inputs)} puzzles -> {out_dir / split}")


@click.command()
@click.option("--backbone", type=click.Choice([*BACKBONES, "all"]), default="all", help="Backbone whose cohorts to build")
@click.option("--output-root", type=click.Path(path_type=Path), default=REPO_ROOT / "data",
              help="Root of the data/ tree; cohorts land at <root>/<backbone>/dataset/<dataset>/{test,impossible}")
@click.option("--cohorts-dir", type=click.Path(path_type=Path), default=REPO_ROOT / "assets" / "cohorts",
              help="Directory of the cohort index files")
@click.option("--from-traces", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None,
              help="Regenerate the index files from the particle traces under DIR/{sudoku_test,sudoku_df,maze_test,maze_df}/trace before building")
def main(backbone: str, output_root: Path, cohorts_dir: Path, from_traces: Path | None):
    for name in BACKBONES if backbone == "all" else [backbone]:
        build_backbone(name, output_root, cohorts_dir, from_traces)


if __name__ == "__main__":
    main()
