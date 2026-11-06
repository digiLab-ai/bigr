"""Per-type validation accuracy for the merged PPBench dataset.

The model already logs aggregate ``val/exact_accuracy`` (per-puzzle whole-grid match over
the whole val split). PPBench mixes several puzzle types in one dataset, so this callback
additionally logs the exact-match accuracy **per type**::

    val/exact_accuracy/sudoku
    val/exact_accuracy/lightup
    val/exact_accuracy/nurikabe
    val/exact_accuracy/tapa
    val/exact_accuracy/ppbench_mean   (macro-mean over the types present)

Type is inferred from the input tokens: PPBench uses a **disjoint per-type token vocabulary**
(namespaced ``(pid, symbol)``), so every non-PAD cell of a grid belongs to exactly one type.
The token->type map is read from ``<data_dir>/metadata.json`` (``token_map`` + ``type_index``).

This reads the greedy prediction stashed by the validation step
(``pl_module._last_eval_preds``) and does NOT re-run the model — re-invoking the compiled
inner_forward (CUDA graphs) would corrupt the persistent training carry.

Wire into an experiment config::

    callbacks:
      ppbench_metrics:
        _target_: trm.nn.callbacks.ppbench_metrics.PPBenchTypeAccuracyCallback
        data_dir: ./data/ppbench
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from lightning import Callback

IGNORE_LABEL_ID = -100
PAD_ID = 0


class PPBenchTypeAccuracyCallback(Callback):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = Path(data_dir)
        self._id2type: np.ndarray | None = None      # token id -> type index (-1 = PAD/none)
        self._type_names: list[str] = []
        self._hit: dict[int, int] = {}
        self._tot: dict[int, int] = {}

    def _ensure_maps(self) -> None:
        if self._id2type is not None:
            return
        meta = json.loads((self.data_dir / "metadata.json").read_text())
        type_index = meta["type_index"]                  # {pid: idx}
        self._type_names = [None] * len(type_index)
        for pid, idx in type_index.items():
            self._type_names[idx] = pid
        vocab = meta["vocab_size"]
        id2type = np.full(vocab, -1, dtype=np.int64)
        for key, tid in meta["token_map"].items():       # "pid|symbol" -> id
            pid = key.split("|", 1)[0]
            id2type[int(tid)] = type_index[pid]
        self._id2type = id2type

    def on_validation_epoch_start(self, trainer, pl_module):
        self._ensure_maps()
        self._hit = {}
        self._tot = {}

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        preds_t = getattr(pl_module, "_last_eval_preds", None)
        if preds_t is None:
            return
        preds = preds_t.cpu().numpy()                      # (B, L)
        inputs = batch["input"].detach().cpu().numpy()     # (B, L)
        labels = batch["output"].detach().cpu().numpy()    # (B, L)
        if len(preds) != len(inputs) or len(preds) != len(labels):
            return  # stash/batch shape guard

        id2type = self._id2type
        for i in range(len(preds)):
            # Type = the type of this grid's non-PAD input tokens (disjoint per type).
            toks = inputs[i][inputs[i] != PAD_ID]
            if toks.size == 0:
                continue
            t = id2type[toks]
            t = t[t >= 0]
            if t.size == 0:
                continue
            ti = int(np.bincount(t).argmax())
            mask = labels[i] != IGNORE_LABEL_ID
            exact = bool(np.array_equal(preds[i][mask], labels[i][mask]))
            self._tot[ti] = self._tot.get(ti, 0) + 1
            self._hit[ti] = self._hit.get(ti, 0) + (1 if exact else 0)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._tot:
            return
        per_type_acc = []
        for ti, n in sorted(self._tot.items()):
            acc = self._hit[ti] / n
            per_type_acc.append(acc)
            name = self._type_names[ti] if ti < len(self._type_names) else str(ti)
            pl_module.log(f"val/exact_accuracy/{name}", float(acc), sync_dist=True)
        pl_module.log(
            "val/exact_accuracy/ppbench_mean",
            float(sum(per_type_acc) / len(per_type_acc)),
            prog_bar=True,
            sync_dist=True,
        )
