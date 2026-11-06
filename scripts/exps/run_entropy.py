"""run_entropy.py — Run guided particle inference and save per-token UQ artefacts (Figures 4 and 7).

Outputs (<out>, used as given; with --num-folds k > 1 one fold_XX/ per fold plus
entropy_summary.npz and fold_manifest.json; the driver passes assets/runs/<bench>/entropy*_<split>_k5,
the layout fig_token_entropy.py and fig_token_contraction.py read):
    <out>/config.json          — run parameters
    <out>/entropy_data.npz     — arrays for fig_token_entropy.py:
        inputs          (N, L)     puzzle token inputs
        labels          (N, L)     puzzle labels
        preds           (N, S, L)  per-particle predictions at final step
        weights         (N, S)     particle weights at final step
        qlogits         (N, S)     Q-logits at final step
        flip_rate       (N, L)     fraction of temporal steps where MAP token flips
        path_auc_uncert (N, L)     mean normalised entropy over temporal checkpoints
        tv_path         (N, L)     mean total-variation of token probs over time

Examples
--------
# The cached runs (Figures 4 and 7): assets/runs/sudoku/entropy_{test,impossible}_k5
bash scripts/run_experiments.sh --profile sudoku_extreme --token-entropy

# Directly (S=16, sigma 0.3, beta 0.25, tau 0.3, stride 3, 64 x 160 batches, 5 folds)
python scripts/exps/run_entropy.py --profile sudoku_extreme --split test \\
    --particles 16 --sigma 0.3 --beta 0.25 --ess-threshold 0.3 --temporal-stride 3 \\
    --batch-size 64 --n-batches 160 --num-folds 5 --fold-seed 31 \\
    --out assets/runs/sudoku/entropy_test_k5

python scripts/exps/run_entropy.py --profile sudoku_extreme --split impossible \\
    --particles 16 --sigma 0.3 --beta 0.25 --ess-threshold 0.3 --temporal-stride 3 \\
    --batch-size 64 --n-batches 160 --num-folds 5 --fold-seed 31 \\
    --out assets/runs/sudoku/entropy_impossible_k5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while not (REPO_ROOT / "src").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bigr import (
    BIGRFilter,
    InferenceConfig,
    StochasticFTheta,
    TRMCoreDecoder,
    TRMCoreEmbedder,
    TRMCoreQHead,
    TRMNativeCycleFTheta,
    load_trm_backbone_checkpoint,
    select_device,
)
from fold_utils import FoldBatchInfo, build_example_folds

PROFILE_DEFAULTS = {
    "sudoku_extreme": {
        "checkpoint": REPO_ROOT / "data" / "sudoku_extreme" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "sudoku_extreme" / "dataset" / "sudoku_extreme",
        "out_dir": REPO_ROOT / "assets" / "runs" / "sudoku",
        "out_prefix": "entropy",
    },
    "maze_hard": {
        "checkpoint": REPO_ROOT / "data" / "maze_hard" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "maze_hard" / "dataset" / "maze-30x30-hard-1k",
        "out_dir": REPO_ROOT / "assets" / "runs" / "maze",
        "out_prefix": "entropy_maze_hard",
    },
}


def _default_out(profile: str, split: str) -> Path:
    """assets/runs/sudoku/entropy_<split> or assets/runs/maze/entropy_maze_hard_<split> (the driver appends _k5)."""
    cfg = PROFILE_DEFAULTS[profile]
    return cfg["out_dir"] / f"{cfg['out_prefix']}_{split}"

try:
    from trm.nn.utils.constants import IGNORE_LABEL_ID as _IGNORE
except Exception:
    _IGNORE = -100


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Guided particle inference token-UQ artefact runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--profile",
        default="sudoku_extreme",
        choices=sorted(PROFILE_DEFAULTS),
        help="Dataset/checkpoint profile preset.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override profile checkpoint path.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override profile dataset directory.",
    )
    p.add_argument("--split", default="test", choices=["test", "impossible"])
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory, used as given (default: assets/runs/sudoku/entropy_<split> or "
            "assets/runs/maze/entropy_maze_hard_<split>; the driver appends _k5 for 5-fold runs)."
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        help="Target device (cuda/mps/cpu). Auto-selects CUDA > MPS > CPU if omitted.",
    )
    p.add_argument("--particles", type=int, default=16)
    p.add_argument("--total-steps", type=int, default=48)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--ess-threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-batches", type=int, default=32)
    p.add_argument(
        "--num-folds",
        type=int,
        default=1,
        help="Run k folds over the loaded evaluation subset and summarize them.",
    )
    p.add_argument(
        "--fold-seed",
        type=int,
        default=None,
        help="Seed used to shuffle examples before splitting into folds.",
    )
    p.add_argument(
        "--temporal-stride",
        type=int,
        default=3,
        help="Checkpoint every N steps for temporal metrics (3 = supervision boundaries).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke check: 2 batches, 4 particles, 12 steps.",
    )
    p.add_argument(
        "--disable-step-cpu-metrics",
        action="store_true",
        help="Skip per-step diagnostic materialization/aggregation.",
    )
    p.set_defaults(smoke=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_batches(
    *, data_dir: Path, split: str, batch_size: int, n_batches: int, device: torch.device
) -> list[dict]:
    from trm.nn.data.sudoku_datamodule import SudokuDataModule

    dm = SudokuDataModule(data_dir=str(data_dir), batch_size=batch_size, num_workers=0)
    if split == "impossible":
        dm.setup(stage="impossible")
        loader = dm.impossible_dataloader()
    else:
        dm.setup(stage="test")
        loader = dm.test_dataloader()

    batches = []
    for raw in tqdm(loader, desc="Loading batches", total=n_batches):
        batches.append(
            {
                "inputs": raw["input"],
                "labels": raw["output"],
                "puzzle_identifiers": raw.get(
                    "puzzle_identifiers",
                    torch.zeros(raw["input"].shape[0], dtype=torch.long),
                ),
            }
        )
        if len(batches) >= n_batches:
            break
    return batches


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Temporal metrics
# ---------------------------------------------------------------------------


def _token_probs(
    preds: np.ndarray, weights: np.ndarray, n_classes: int
) -> np.ndarray:
    """Build (N, L, n_classes) weighted predictive distribution."""
    N, S, L = preds.shape
    # One-hot then weighted sum: avoids Python loop over classes
    one_hot = (
        preds[:, :, :, None] == np.arange(n_classes, dtype=preds.dtype)[None, None, None, :]
    ).astype(np.float32)  # (N, S, L, C)
    probs = (weights[:, :, None, None] * one_hot).sum(axis=1)  # (N, L, C)
    probs /= np.clip(probs.sum(axis=-1, keepdims=True), 1e-12, None)
    return probs


def compute_temporal_metrics(
    pred_ckpts: dict[int, np.ndarray],
    w_ckpts: dict[int, np.ndarray],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-token temporal metrics from checkpoint dicts.

    Returns
    -------
    flip_rate       (N, L)  fraction of consecutive step pairs where MAP flips
    path_auc_uncert (N, L)  mean normalised token entropy over all checkpoints
    tv_path         (N, L)  mean total-variation of token probs across steps
    """
    steps = sorted(set(pred_ckpts) & set(w_ckpts))
    if len(steps) < 2:
        sample = pred_ckpts[max(pred_ckpts)]
        N, _S, L = sample.shape
        nan = np.full((N, L), np.nan, dtype=np.float32)
        return nan, nan, nan

    log_C = np.log(float(n_classes))
    map_steps: list[np.ndarray] = []
    ent_steps: list[np.ndarray] = []
    tv_steps: list[np.ndarray] = []
    prev_probs: np.ndarray | None = None

    for step in tqdm(steps, desc="Temporal metrics", leave=False):
        p = pred_ckpts[step].astype(np.int32)
        w = w_ckpts[step].astype(np.float32)
        w /= np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)

        probs = _token_probs(p, w, n_classes)  # (N, L, C)
        map_steps.append(probs.argmax(axis=-1).astype(np.int16))

        safe = np.clip(probs, 1e-12, 1.0)
        ent = -(safe * np.log(safe)).sum(axis=-1) / log_C  # (N, L), normalised
        ent_steps.append(ent.astype(np.float32))

        if prev_probs is not None:
            tv = 0.5 * np.abs(probs - prev_probs).sum(axis=-1)
            tv_steps.append(tv.astype(np.float32))
        prev_probs = probs

    map_stack = np.stack(map_steps, axis=0)  # (T, N, L)
    ent_stack = np.stack(ent_steps, axis=0)  # (T, N, L)

    flip_rate = (map_stack[1:] != map_stack[:-1]).mean(axis=0).astype(np.float32)
    path_auc_uncert = ent_stack.mean(axis=0).astype(np.float32)
    tv_path = (
        np.stack(tv_steps, axis=0).mean(axis=0).astype(np.float32)
        if tv_steps
        else np.full_like(flip_rate, np.nan)
    )
    return flip_rate, path_auc_uncert, tv_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_entropy_single(
    *,
    args: argparse.Namespace,
    trm,
    device: torch.device,
    embedder: TRMCoreEmbedder,
    decoder: TRMCoreDecoder,
    q_head: TRMCoreQHead,
    f_theta: TRMNativeCycleFTheta,
    raw_batches: list[dict],
    out_dir: Path,
    fold_meta: dict | None = None,
) -> dict:
    """Run one entropy evaluation and persist artefacts into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg_dict.update({"out": str(out_dir)})
    if fold_meta is not None:
        cfg_dict.update(fold_meta)
    _write_json(out_dir / "config.json", cfg_dict)

    embedded = []
    for b in tqdm(raw_batches, desc="Embedding batches"):
        x = embedder(b["inputs"].to(device), b["puzzle_identifiers"].to(device))
        embedded.append({"x": x, "labels": b["labels"].to(device)})

    stride = args.temporal_stride
    ckpt_steps = sorted(
        set(range(stride, args.total_steps + 1, stride)) | {args.total_steps}
    )
    print(
        f"Checkpointing at {len(ckpt_steps)} steps (stride={stride}): {ckpt_steps[:6]}…"
    )

    h_cycles = int(getattr(trm.hparams, "H_cycles", 3))
    stoch_f = StochasticFTheta(f_theta, sigma=args.sigma, per_step_inner_noise=True)
    inf_cfg = InferenceConfig(
        particles=args.particles,
        total_steps=args.total_steps,
        sigma=args.sigma,
        beta=args.beta,
        ess_threshold=args.ess_threshold,
        seed=args.seed,
        checkpoint_steps=ckpt_steps,
        save_particle_details=True,
        collect_step_cpu_metrics=not args.disable_step_cpu_metrics,
    )
    filt = BIGRFilter(stoch_f, q_head, decoder, inf_cfg, h_cycles=h_cycles)

    y0_init = trm.z_H_init.detach().clone()
    z0_init = trm.z_L_init.detach().clone()

    print(
        f"Running particle filter: S={args.particles}, N={args.total_steps}, "
        f"σ={args.sigma} …"
    )
    with torch.no_grad():
        trace = filt.run(embedded, y0_init=y0_init, z0_init=z0_init, device=device)

    final_step = max(trace.checkpoint_preds.keys())
    preds = trace.checkpoint_preds[final_step]
    weights_by_step = trace.checkpoint_weights_raw or trace.checkpoint_weights
    qlogits_by_step = trace.checkpoint_q_raw or trace.checkpoint_q_logits
    weights = weights_by_step[final_step]
    qlogits = qlogits_by_step[final_step]

    if preds is None or len(preds) == 0:
        raise RuntimeError("checkpoint_preds is empty — ensure labels are provided.")

    N, S, L = preds.shape
    n_classes = int(max(preds.max(), 1)) + 1

    print("Computing temporal UQ metrics …")
    flip_rate, path_auc_uncert, tv_path = compute_temporal_metrics(
        dict(trace.checkpoint_preds),
        dict(weights_by_step),
        n_classes=n_classes,
    )

    inputs = np.concatenate([b["inputs"].cpu().numpy() for b in raw_batches])[:N]
    labels = np.concatenate([b["labels"].cpu().numpy() for b in raw_batches])[:N]

    out_path = out_dir / "entropy_data.npz"
    np.savez(
        out_path,
        inputs=inputs,
        labels=labels,
        preds=preds,
        weights=weights.astype(np.float32),
        qlogits=qlogits.astype(np.float32),
        flip_rate=flip_rate,
        path_auc_uncert=path_auc_uncert,
        tv_path=tv_path,
    )
    print(f"Saved entropy_data.npz  N={N} S={S} L={L} n_classes={n_classes}")

    final_rec = trace.step_records[-1]
    print(
        f"Final step {final_rec.step}: "
        f"exact_accuracy={final_rec.exact_accuracy:.4f}  "
        f"any_exact_rate={final_rec.any_exact_rate:.4f}"
    )
    print(f"Artefacts saved to: {out_dir}")

    return {
        "flip_rate_mean": flip_rate.mean(axis=0).astype(np.float32),
        "path_auc_uncert_mean": path_auc_uncert.mean(axis=0).astype(np.float32),
        "tv_path_mean": tv_path.mean(axis=0).astype(np.float32),
        "n_examples": int(N),
        "n_classes": int(n_classes),
        "seq_len": int(L),
        "total_steps": int(args.total_steps),
        "final": {
            "step": int(final_rec.step),
            "exact_accuracy": float(final_rec.exact_accuracy),
            "any_exact_rate": float(final_rec.any_exact_rate),
        },
    }


def _summarize_fold_entropy(
    fold_outputs: list[dict], fold_infos: list[FoldBatchInfo], out_dir: Path, args: argparse.Namespace
) -> None:
    """Write fold-level summary arrays and a manifest."""
    flip_stack = np.stack([entry["flip_rate_mean"] for entry in fold_outputs], axis=0)
    path_stack = np.stack(
        [entry["path_auc_uncert_mean"] for entry in fold_outputs], axis=0
    )
    tv_stack = np.stack([entry["tv_path_mean"] for entry in fold_outputs], axis=0)

    summary_path = out_dir / "entropy_summary.npz"
    np.savez(
        summary_path,
        flip_rate_mean=flip_stack.mean(axis=0).astype(np.float32),
        flip_rate_std=flip_stack.std(axis=0).astype(np.float32),
        path_auc_uncert_mean=path_stack.mean(axis=0).astype(np.float32),
        path_auc_uncert_std=path_stack.std(axis=0).astype(np.float32),
        tv_path_mean=tv_stack.mean(axis=0).astype(np.float32),
        tv_path_std=tv_stack.std(axis=0).astype(np.float32),
        fold_sizes=np.array([info.num_examples for info in fold_infos], dtype=np.int32),
        num_folds=np.array(args.num_folds, dtype=np.int32),
        fold_seed=np.array(args.fold_seed, dtype=np.int32),
        total_steps=np.array(args.total_steps, dtype=np.int32),
        temporal_stride=np.array(args.temporal_stride, dtype=np.int32),
        n_classes=np.array(fold_outputs[0]["n_classes"], dtype=np.int32),
        seq_len=np.array(fold_outputs[0]["seq_len"], dtype=np.int32),
    )

    manifest = {
        "split": args.split,
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed,
        "summary_npz": "entropy_summary.npz",
        "folds": [
            {
                "fold_index": info.fold_index,
                "fold_name": f"fold_{info.fold_index + 1:02d}",
                "num_examples": info.num_examples,
                "num_batches": info.num_batches,
                "path": f"fold_{info.fold_index + 1:02d}",
            }
            for info in fold_infos
        ],
    }
    _write_json(out_dir / "fold_manifest.json", manifest)
    _write_json(out_dir / "summary.json", manifest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    profile_cfg = PROFILE_DEFAULTS[args.profile]
    if args.checkpoint is None:
        args.checkpoint = profile_cfg["checkpoint"]
    if args.data_dir is None:
        args.data_dir = profile_cfg["data_dir"]
    if args.out is None:
        args.out = _default_out(args.profile, args.split)

    if args.smoke:
        args.particles = 4
        args.total_steps = 12
        args.batch_size = 4
        args.n_batches = 2
        print("Smoke mode: particles=4, steps=12, batches=2")

    args.fold_seed = args.seed if args.fold_seed is None else args.fold_seed
    if args.num_folds < 1:
        raise ValueError(f"--num-folds must be >= 1, got {args.num_folds}")

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else select_device()

    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    print(f"Loading checkpoint: {args.checkpoint}")
    trm = load_trm_backbone_checkpoint(
        args.checkpoint, map_location=device, use_ema=True
    )
    trm.eval().to(device)

    embedder = TRMCoreEmbedder(trm)
    decoder = TRMCoreDecoder(trm)
    q_head = TRMCoreQHead(trm)
    f_theta = TRMNativeCycleFTheta(trm)

    print(f"Loading {args.n_batches} batches (split={args.split}) …")
    raw_batches = load_batches(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        device=device,
    )
    if args.num_folds == 1:
        _run_entropy_single(
            args=args,
            trm=trm,
            device=device,
            embedder=embedder,
            decoder=decoder,
            q_head=q_head,
            f_theta=f_theta,
            raw_batches=raw_batches,
            out_dir=args.out,
        )
        return

    fold_batches, fold_infos = build_example_folds(
        raw_batches,
        num_folds=args.num_folds,
        seed=args.fold_seed,
        batch_size=args.batch_size,
    )
    print(
        f"Running {args.num_folds}-fold mode with seed={args.fold_seed} "
        f"over {sum(info.num_examples for info in fold_infos)} examples"
    )

    fold_outputs = []
    for fold_idx, (fold_raw_batches, fold_info) in enumerate(
        zip(fold_batches, fold_infos),
        start=1,
    ):
        fold_name = f"fold_{fold_idx:02d}"
        print(
            f"\n[{fold_name}] examples={fold_info.num_examples} batches={fold_info.num_batches}"
        )
        fold_outputs.append(
            _run_entropy_single(
                args=args,
                trm=trm,
                device=device,
                embedder=embedder,
                decoder=decoder,
                q_head=q_head,
                f_theta=f_theta,
                raw_batches=fold_raw_batches,
                out_dir=args.out / fold_name,
                fold_meta={
                    "fold_mode": True,
                    "fold_index": fold_idx,
                    "fold_name": fold_name,
                    "num_folds": args.num_folds,
                    "fold_seed": args.fold_seed,
                    "fold_num_examples": fold_info.num_examples,
                    "fold_num_batches": fold_info.num_batches,
                },
            )
        )

    _summarize_fold_entropy(fold_outputs, fold_infos, args.out, args)
    print(f"Fold summaries saved to: {args.out}")


if __name__ == "__main__":
    main()
