"""run_continuation_labels.py — Guide-value alignment experiment data.

For each checkpoint step n, runs a deterministic (σ=0) continuation from
every saved particle state h_{n+1}^(s) to the terminal horizon T. This
produces binary continuation-solve labels g_s = 1{particle s solves the
puzzle from step n onwards}, enabling direct measurement of the guide-value
alignment: whether Q-scores predict continuation success.

Outputs (<out>, written as given; with --num-folds k > 1 one fold_XX/ per fold plus
continuation_labels_summary.csv and fold_manifest.json; the driver passes
assets/runs/<bench>/continuation_labels*_k5, the layout fig_guide_alignment.py reads)
-------
    config.json
    continuation_labels.npz
        steps       — (K,) checkpoint step numbers
        q_raw       — (K, B_total, S) Q logits, original particle order
        q_score     — (K, B_total, S) guide scores log σ(q_raw), original particle order
        g_labels    — (K, B_total, S) binary continuation-solve labels
        pp_exact    — (K, B_total, S) immediate per-particle exact-solve at checkpoint

A single --sigma controls both belief-update (z) and state-update (y) noise,
so --sigma 0 recovers the fully deterministic case.

Usage
-----
    # The cached runs (Figure 3): assets/runs/sudoku/continuation_labels_k5 and
    # assets/runs/maze/continuation_labels_maze_hard_k5
    bash scripts/run_experiments.sh --profile sudoku_extreme --guide-alignment
    bash scripts/run_experiments.sh --profile maze_hard --guide-alignment

    # Directly, the Sudoku-Extreme protocol (S=16, sigma 0.3, beta 0.25, tau 0.3, 64 x 160, 5 folds)
    python scripts/exps/run_continuation_labels.py --profile sudoku_extreme \\
        --particles 16 --total-steps 48 --sigma 0.3 --beta 0.25 --ess-threshold 0.3 \\
        --batch-size 64 --n-batches 160 --num-folds 5 --fold-seed 31 \\
        --out assets/runs/sudoku/continuation_labels_k5

    # Deterministic reference (sigma 0 collapses the cloud onto one trajectory):
    python scripts/exps/run_continuation_labels.py --sigma 0 \\
        --batch-size 16 --n-batches 32 --out tmp/continuation_labels_det
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while not (REPO_ROOT / "src").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

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
from fold_utils import build_example_folds

try:
    from trm.nn.utils.constants import IGNORE_LABEL_ID as _IGNORE
except Exception:
    _IGNORE = -100

PROFILE_DEFAULTS = {
    "sudoku_extreme": {
        "checkpoint": REPO_ROOT / "data" / "sudoku_extreme" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "sudoku_extreme" / "dataset" / "sudoku_extreme",
        "out": REPO_ROOT / "assets" / "runs" / "sudoku" / "continuation_labels",
    },
    "maze_hard": {
        "checkpoint": REPO_ROOT / "data" / "maze_hard" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "maze_hard" / "dataset" / "maze-30x30-hard-1k",
        "out": REPO_ROOT / "assets" / "runs" / "maze" / "continuation_labels_maze_hard",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Guide-value alignment: continuation labels experiment",
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
    p.add_argument(
        "--device",
        default=None,
        help="Target device (cuda/mps/cpu). Auto-selects CUDA > MPS > CPU if omitted.",
    )
    p.add_argument("--particles",   type=int,   default=16)
    p.add_argument("--total-steps", type=int,   default=48)
    p.add_argument(
        "--sigma", type=float, default=0.3,
        help="Noise std applied to both belief (z) and state (y) updates. "
             "sigma=0 gives a fully deterministic filter run.",
    )
    p.add_argument("--beta",        type=float, default=0.25)
    p.add_argument("--seed",        type=int,   default=31)
    p.add_argument(
        "--ess-threshold",
        type=float,
        default=0.3,
        help="ESS resampling threshold for the main predictive-cloud run.",
    )
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--n-batches",   type=int, default=32)
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
        "--disable-step-cpu-metrics",
        action="store_true",
        help="Skip per-step diagnostic materialization/aggregation.",
    )
    p.add_argument(
        "--out", type=Path,
        default=None,
        help=(
            "Output directory, used as given (default: assets/runs/sudoku/continuation_labels or "
            "assets/runs/maze/continuation_labels_maze_hard; the driver appends _k5 for 5-fold runs)."
        ),
    )
    p.add_argument("--smoke", action="store_true", help="2 batches, quick check")
    p.set_defaults(smoke=False)
    return p.parse_args()


def _load_data(data_dir, batch_size, n_batches):
    from trm.nn.data.sudoku_datamodule import SudokuDataModule

    dm = SudokuDataModule(data_dir=str(data_dir), batch_size=batch_size, num_workers=0)
    dm.setup(stage="test")
    loader = dm.test_dataloader()
    embedded = []
    for raw in tqdm(loader, desc="Loading batches", total=n_batches):
        current_batch = int(raw["input"].shape[0])
        embedded.append({
            "inputs": raw["input"],
            "labels": raw["output"],
            "puzzle_identifiers": raw.get(
                "puzzle_identifiers",
                torch.zeros(current_batch, dtype=torch.long),
            ),
        })
        if len(embedded) >= n_batches:
            break
    return embedded


def _exact_solved(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels != _IGNORE
    return torch.logical_or(~mask, pred == labels).all(-1).float()


def _guide_score_from_raw(q_raw: np.ndarray) -> np.ndarray:
    """Return the implemented guide score q = log σ(q_raw)."""
    return -np.logaddexp(0.0, -q_raw)


def _run_deterministic_continuation(
    stoch_det: StochasticFTheta,
    decoder: TRMCoreDecoder,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    labels: torch.Tensor,
    n_steps: int,
    *,
    progress_desc: str | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Run n_steps deterministic steps, return exact-solve mask (B_cont,)."""
    step_iter = tqdm(
        range(n_steps),
        desc=progress_desc or "Continuation steps",
        leave=False,
        disable=not show_progress,
    )
    for _ in step_iter:
        z = stoch_det.belief_update(x, y, z)
        y = stoch_det.state_update(y, z)
    tokens = decoder(y).argmax(-1)
    return _exact_solved(tokens, labels).cpu().numpy()


def _embed_batches(raw_batches, embedder, device):
    embedded = []
    for batch in tqdm(raw_batches, desc="Embedding batches"):
        x = embedder(batch["inputs"].to(device), batch["puzzle_identifiers"].to(device))
        embedded.append({"x": x, "labels": batch["labels"].to(device)})
    return embedded


def _run_continuation(
    *,
    args: argparse.Namespace,
    embedded: list[dict],
    trm,
    decoder: TRMCoreDecoder,
    q_head: TRMCoreQHead,
    f_theta: TRMNativeCycleFTheta,
    device: torch.device,
    out_dir: Path,
    fold_meta: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    cfg.update({"out": str(out_dir)})
    if fold_meta is not None:
        cfg.update(fold_meta)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    h_cycles = int(getattr(trm.hparams, "H_cycles", 3))
    y0_init = trm.z_H_init.detach().clone()
    z0_init = trm.z_L_init.detach().clone()

    cfg_inf = InferenceConfig(
        particles=args.particles,
        total_steps=args.total_steps,
        sigma=args.sigma,
        ess_threshold=args.ess_threshold,
        beta=args.beta,
        seed=args.seed,
        save_hidden_states=True,
        collect_step_cpu_metrics=not args.disable_step_cpu_metrics,
    )
    stoch = StochasticFTheta(
        f_theta,
        sigma=args.sigma,
        per_step_inner_noise=True,
    )
    filt = BIGRFilter(stoch, q_head, decoder, cfg_inf, h_cycles=h_cycles)

    stoch_det = StochasticFTheta(f_theta, sigma=0.0)
    S = args.particles

    checkpoint_steps: list[int] | None = None
    q_raw_chunks: dict[int, list[np.ndarray]] = {}
    q_score_chunks: dict[int, list[np.ndarray]] = {}
    g_label_chunks: dict[int, list[np.ndarray]] = {}
    pp_exact_chunks: dict[int, list[np.ndarray]] = {}

    print(f"\nRunning main filter + continuation labels (streaming)   σ={args.sigma}  β={args.beta} …")
    for bi, batch in enumerate(tqdm(embedded, desc="Batches")):
        # Match the per-batch seed schedule used by a single multi-batch filter run.
        filt.config.seed = args.seed + bi
        with torch.no_grad():
            trace = filt.run([batch], y0_init=y0_init, z0_init=z0_init, device=device)

        batch_steps = sorted(trace.checkpoint_y.keys())
        if checkpoint_steps is None:
            checkpoint_steps = batch_steps
            print(f"Checkpoint steps: {checkpoint_steps}")
            for n in checkpoint_steps:
                q_raw_chunks[n] = []
                q_score_chunks[n] = []
                g_label_chunks[n] = []
                pp_exact_chunks[n] = []
        elif batch_steps != checkpoint_steps:
            raise RuntimeError(
                "Checkpoint steps changed across batches: "
                f"{checkpoint_steps} vs {batch_steps}"
            )

        x_b = batch["x"]
        lbl_b = batch["labels"]
        B_i = int(x_b.shape[0])

        for n in tqdm(
            checkpoint_steps,
            desc=f"Checkpoint steps (batch {bi + 1}/{len(embedded)})",
            leave=False,
        ):
            y_ckpt = trace.checkpoint_y[n]
            z_ckpt = trace.checkpoint_z[n]
            q_raw = trace.checkpoint_q_raw[n].astype(np.float32, copy=False)
            n_remaining = args.total_steps - n

            if y_ckpt.shape[0] != B_i:
                raise RuntimeError(
                    f"Batch-size mismatch at step {n}: checkpoint has {y_ckpt.shape[0]}, expected {B_i}"
                )

            _B, _, latent_len, D = y_ckpt.shape
            y_b = torch.from_numpy(y_ckpt.astype(np.float32, copy=False)).to(device)
            z_b = torch.from_numpy(z_ckpt.astype(np.float32, copy=False)).to(device)

            label_len = int(lbl_b.shape[-1])
            y_flat = y_b.reshape(B_i * S, latent_len, D)
            z_flat = z_b.reshape(B_i * S, latent_len, D)
            x_flat = (
                x_b.unsqueeze(1)
                .expand(B_i, S, latent_len, D)
                .reshape(B_i * S, latent_len, D)
            )
            lbl_flat = (
                lbl_b.unsqueeze(1)
                .expand(B_i, S, label_len)
                .reshape(B_i * S, label_len)
            )

            with torch.no_grad():
                g_flat = _run_deterministic_continuation(
                    stoch_det,
                    decoder,
                    x_flat,
                    y_flat,
                    z_flat,
                    lbl_flat,
                    n_remaining,
                    progress_desc=f"Continuation steps (batch {bi + 1}, ckpt {n})",
                    show_progress=True,
                )
            g_labels = g_flat.reshape(B_i, S).astype(np.float32, copy=False)

            q_score = _guide_score_from_raw(q_raw).astype(np.float32, copy=False)
            q_raw_chunks[n].append(q_raw)
            q_score_chunks[n].append(q_score)
            g_label_chunks[n].append(g_labels)
            if n in trace.checkpoint_pp_exact:
                pp_exact_chunks[n].append(
                    trace.checkpoint_pp_exact[n].astype(np.float32, copy=False)
                )
            else:
                pp_exact_chunks[n].append(np.zeros((B_i, S), dtype=np.float32))

        del trace

    if checkpoint_steps is None:
        raise RuntimeError("No checkpoint steps were produced by the filter run.")

    all_q_raw = {n: np.concatenate(q_raw_chunks[n], axis=0) for n in checkpoint_steps}
    all_q_score = {n: np.concatenate(q_score_chunks[n], axis=0) for n in checkpoint_steps}
    all_g_labels = {n: np.concatenate(g_label_chunks[n], axis=0) for n in checkpoint_steps}
    all_pp_exact = {n: np.concatenate(pp_exact_chunks[n], axis=0) for n in checkpoint_steps}

    per_step_rows = []
    for n in checkpoint_steps:
        g = all_g_labels[n]
        q = all_q_score[n]
        q_raw = all_q_raw[n]
        p0 = float(g.mean())
        gap = (
            float(q[g == 1].mean() - q[g == 0].mean())
            if (g.any() and (~g.astype(bool)).any())
            else float("nan")
        )
        per_step_rows.append(
            {
                "step": int(n),
                "p_solve": p0,
                "guide_score_gap": gap,
                "pp_exact_rate": float(all_pp_exact[n].mean()),
                "q_score_mean": float(q.mean()),
                "q_logit_mean": float(q_raw.mean()),
            }
        )
        print(f"  step {n:3d}  p(solve)={p0:.3f}  E[q|A]−E[q|A^c]={gap:+.3f}")

    out_path = out_dir / "continuation_labels.npz"
    steps = np.array(checkpoint_steps)
    np.savez(
        out_path,
        steps=steps,
        q_raw=np.stack([all_q_raw[n] for n in checkpoint_steps]),
        q_score=np.stack([all_q_score[n] for n in checkpoint_steps]),
        g_labels=np.stack([all_g_labels[n] for n in checkpoint_steps]),
        pp_exact=np.stack([all_pp_exact[n] for n in checkpoint_steps]),
        total_steps=np.array(args.total_steps),
    )
    print(f"\nSaved {out_path}")

    return {
        "per_step": pd.DataFrame(per_step_rows),
        "steps": steps,
        "final_step": int(checkpoint_steps[-1]) if checkpoint_steps else -1,
        "n_examples": int(sum(batch["labels"].shape[0] for batch in embedded)),
        "n_batches": int(len(embedded)),
    }


def main() -> None:
    args = parse_args()
    profile_cfg = PROFILE_DEFAULTS[args.profile]
    if args.checkpoint is None:
        args.checkpoint = profile_cfg["checkpoint"]
    if args.data_dir is None:
        args.data_dir = profile_cfg["data_dir"]
    if args.out is None:
        args.out = profile_cfg["out"]
    n_batches = 2 if args.smoke else args.n_batches
    args.fold_seed = args.seed if args.fold_seed is None else args.fold_seed
    if args.num_folds < 1:
        raise ValueError(f"--num-folds must be >= 1, got {args.num_folds}")

    device = torch.device(args.device) if args.device else select_device()
    print(f"Device: {device}")

    print("Loading checkpoint …")
    trm = load_trm_backbone_checkpoint(args.checkpoint, map_location=device)
    trm.eval().to(device)

    embedder  = TRMCoreEmbedder(trm)
    decoder   = TRMCoreDecoder(trm)
    q_head    = TRMCoreQHead(trm)
    f_theta   = TRMNativeCycleFTheta(trm)

    print(f"Loading {n_batches} batches …")
    raw_batches = _load_data(args.data_dir, args.batch_size, n_batches)
    n_puzzles = sum(batch["inputs"].shape[0] for batch in raw_batches)
    print(f"Loaded {len(raw_batches)} batches × {args.batch_size} = {n_puzzles} puzzles")

    output_root = args.out
    output_root.mkdir(parents=True, exist_ok=True)

    if args.num_folds == 1:
        embedded = _embed_batches(raw_batches, embedder, device)
        _run_continuation(
            args=args,
            embedded=embedded,
            trm=trm,
            decoder=decoder,
            q_head=q_head,
            f_theta=f_theta,
            device=device,
            out_dir=args.out,
        )
        return

    fold_batches, fold_infos = build_example_folds(
        raw_batches, num_folds=args.num_folds, seed=args.fold_seed, batch_size=args.batch_size
    )
    fold_frames = []
    for fold_idx, (fold_raw, info) in enumerate(zip(fold_batches, fold_infos), start=1):
        fold_name = f"fold_{fold_idx:02d}"
        fold_dir = output_root / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        embedded = _embed_batches(fold_raw, embedder, device)
        fold_result = _run_continuation(
            args=args,
            embedded=embedded,
            trm=trm,
            decoder=decoder,
            q_head=q_head,
            f_theta=f_theta,
            device=device,
            out_dir=fold_dir,
            fold_meta={
                "fold_mode": True,
                "fold_index": fold_idx,
                "fold_name": fold_name,
                "num_folds": args.num_folds,
                "fold_seed": args.fold_seed,
                "fold_num_examples": info.num_examples,
                "fold_num_batches": info.num_batches,
            },
        )
        fold_frames.append(fold_result["per_step"].assign(fold=fold_idx))

    summary = (
        pd.concat(fold_frames, ignore_index=True)
        .groupby("step")[["p_solve", "guide_score_gap", "pp_exact_rate", "q_score_mean", "q_logit_mean"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    flat_columns = []
    for col in summary.columns:
        if isinstance(col, tuple):
            base, stat = col
            flat_columns.append(f"{base}_{stat}" if stat else str(base))
        else:
            flat_columns.append(str(col))
    summary.columns = flat_columns
    summary.to_csv(output_root / "continuation_labels_summary.csv", index=False)
    (output_root / "fold_manifest.json").write_text(
        json.dumps(
            {
                "num_folds": args.num_folds,
                "fold_seed": args.fold_seed,
                "folds": [
                    {
                        "fold_index": info.fold_index,
                        "fold_name": f"fold_{info.fold_index + 1:02d}",
                        "num_examples": info.num_examples,
                        "num_batches": info.num_batches,
                    }
                    for info in fold_infos
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Fold summary saved to: {output_root / 'continuation_labels_summary.csv'}")


if __name__ == "__main__":
    main()
