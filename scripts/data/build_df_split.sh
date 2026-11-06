#!/usr/bin/env bash
# build_df_split.sh — build the deterministic-failure (DF) split: the test puzzles that
# the frozen TRM fails to solve on its own. Written as <dataset>/impossible/ in the same
# on-disk format as the other splits, exactly as the paper did.
#
#   bash scripts/data/build_df_split.sh sudoku_extreme
#   bash scripts/data/build_df_split.sh maze_hard
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PROFILE="${1:?usage: build_df_split.sh {sudoku_extreme|maze_hard} [extra hydra overrides]}"
shift
case "$PROFILE" in
    sudoku_extreme) CKPT=data/sudoku_extreme/checkpoints/last.ckpt; DATA=data/sudoku_extreme/dataset/sudoku_extreme; EXTRA=(+task=sudoku) ;;
    maze_hard)      CKPT=data/maze_hard/checkpoints/last.ckpt;      DATA=data/maze_hard/dataset/maze-30x30-hard-1k; EXTRA=(+task=maze +grid_size=30) ;;
    *) echo "unknown profile: $PROFILE" >&2; exit 1 ;;
esac
"${PYTHON:-python}" -m trm.nn.evaluate "checkpoint=$CKPT" "data_dir=$DATA" "${EXTRA[@]}" +eval_split=test +num_workers=0 "$@"
echo "DF split written to $DATA/impossible ($("${PYTHON:-python}" -c "import json;print(json.load(open('$DATA/impossible/dataset.json'))['num_puzzles'])" 2>/dev/null || echo '?') puzzles)"
