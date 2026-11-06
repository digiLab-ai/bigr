"""run_sigma_guided.py — σ sweep: guided vs unguided filtering (Figure 2, right panel).

Sweeps noise level σ (applied equally to belief z and state y updates) at
S∈{4,16} × guided∈{on,off,resampled} to show that guide-based weighting
consistently improves MAP accuracy over a uniform-weight (β=0) baseline,
and that adding resampling (τ=0.3) further benefits small-σ regimes.
Sudoku-Extreme only: the guided conditions use the Sudoku-Extreme tempering β=0.25.

At σ=0 both inner and outer updates are deterministic: all particles collapse
to a single trajectory, so guided and unguided converge (useful baseline).

Conditions (6 total per σ value)
---------------------------------
  S=4,  β=0.00  τ=0.00  unguided   — plain particle filter, uniform weights
  S=4,  β=0.25  τ=0.00  guided     — guided particle filter, Q-head weighted, no resampling
  S=4,  β=0.25  τ=0.30  resampled  — guided particle filter, Q-head weighted + resampling
  S=16, β=0.00  τ=0.00  unguided
  S=16, β=0.25  τ=0.00  guided
  S=16, β=0.25  τ=0.30  resampled

Fixed: total_steps=48
Sweep: σ ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.8}

Outputs (<out>, a directory used as given; the driver passes assets/runs/sudoku/sigma_guided_results_k5)
-------
    sigma_guided_results.csv            one row per (σ, S, condition); with --num-folds k > 1 the
                                        fold mean/std, with the per-fold tables in fold_XX/
    fold_manifest.json                  (k-fold only)

Usage
-----
    # The cached run (Figure 2, right panel): assets/runs/sudoku/sigma_guided_results_k5
    bash scripts/run_experiments.sh --profile sudoku_extreme --sigma-guided

    # Directly (64 x 160 batches, 5 folds)
    python scripts/exps/run_sigma_guided.py --profile sudoku_extreme --total-steps 48 \\
        --batch-size 64 --n-batches 160 --num-folds 5 --fold-seed 31 \\
        --out assets/runs/sudoku/sigma_guided_results_k5

    python scripts/exps/run_sigma_guided.py --smoke --out tmp/sigma_guided_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from fold_utils import build_example_folds, summarize_numeric_df

PROFILE_DEFAULTS = {
    "sudoku_extreme": {
        "checkpoint": REPO_ROOT / "data" / "sudoku_extreme" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "sudoku_extreme" / "dataset" / "sudoku_extreme",
        "out": REPO_ROOT / "assets" / "runs" / "sudoku" / "sigma_guided_results",
    },
    "maze_hard": {
        "checkpoint": REPO_ROOT / "data" / "maze_hard" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "maze_hard" / "dataset" / "maze-30x30-hard-1k",
        "out": REPO_ROOT / "assets" / "runs" / "maze" / "sigma_guided_results_maze_hard",
    },
}

SIGMA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
S_VALUES = [4, 16]
BETA_GUIDED = 0.25
BETA_UNGUIDED = 0.00
TAU_RESAMPLE = 0.30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="σ sweep: guided vs unguided filtering (Figure 2, right panel)",
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
    p.add_argument("--total-steps", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=31)
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
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory for sigma_guided_results.csv, used as given (default: "
            "assets/runs/sudoku/sigma_guided_results; the driver appends _k5 for 5-fold runs)."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Fast smoke check: 3 σ values (0, 0.50, 1.00), 2 batches.",
    )
    p.add_argument(
        "--disable-step-cpu-metrics",
        action="store_true",
        help="Skip per-step diagnostic materialization/aggregation.",
    )
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
        embedded.append(
            {
                "inputs": raw["input"],
                "labels": raw["output"],
                "puzzle_identifiers": raw.get(
                    "puzzle_identifiers",
                    torch.zeros(current_batch, dtype=torch.long),
                ),
            }
        )
        if len(embedded) >= n_batches:
            break
    return embedded


def _embed_batches(raw_batches, embedder, device):
    embedded = []
    for batch in tqdm(raw_batches, desc="Embedding batches"):
        x = embedder(batch["inputs"].to(device), batch["puzzle_identifiers"].to(device))
        embedded.append({"x": x, "labels": batch["labels"].to(device)})
    return embedded


def _run_one(
    *,
    embedded,
    f_theta,
    q_head,
    decoder,
    y0_init,
    z0_init,
    device,
    h_cycles,
    beta: float,
    sigma: float,
    tau: float = 0.0,
    particles: int,
    total_steps: int,
    seed: int = 31,
    collect_step_cpu_metrics: bool = True,
) -> dict:
    cfg = InferenceConfig(
        particles=particles,
        total_steps=total_steps,
        sigma=sigma,
        ess_threshold=tau,
        beta=beta,
        seed=seed,
        collect_step_cpu_metrics=collect_step_cpu_metrics,
    )
    stoch = StochasticFTheta(
        f_theta,
        sigma=sigma,
        per_step_inner_noise=True,
    )
    filt = BIGRFilter(stoch, q_head, decoder, cfg, h_cycles=h_cycles)
    with torch.no_grad():
        trace = filt.run(embedded, y0_init=y0_init, z0_init=z0_init, device=device)
    final = trace.step_records[-1]
    return {
        "final_acc": final.exact_accuracy,
        "final_oracle": final.any_exact_rate,
        "final_ess": final.mean_ess,
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
    sigma_values = [0.00, 0.50, 1.00] if args.smoke else SIGMA_VALUES
    n_batches = 2 if args.smoke else args.n_batches
    args.fold_seed = args.seed if args.fold_seed is None else args.fold_seed
    if args.num_folds < 1:
        raise ValueError(f"--num-folds must be >= 1, got {args.num_folds}")

    device = torch.device(args.device) if args.device else select_device()
    print(f"Device: {device}")

    print("Loading checkpoint …")
    trm = load_trm_backbone_checkpoint(args.checkpoint, map_location=device)
    trm.eval().to(device)
    h_cycles = int(getattr(trm.hparams, "H_cycles", 3))

    embedder = TRMCoreEmbedder(trm)
    decoder = TRMCoreDecoder(trm)
    q_head = TRMCoreQHead(trm)
    f_theta = TRMNativeCycleFTheta(trm)
    y0_init = trm.z_H_init.detach().clone()
    z0_init = trm.z_L_init.detach().clone()

    print(f"Loading {n_batches} batches …")
    raw_batches = _load_data(args.data_dir, args.batch_size, n_batches)
    n_puzzles = sum(batch["inputs"].shape[0] for batch in raw_batches)
    print(f"Loaded {len(raw_batches)} batches × {args.batch_size} = {n_puzzles} puzzles\n")

    base = dict(
        f_theta=f_theta,
        q_head=q_head,
        decoder=decoder,
        y0_init=y0_init,
        z0_init=z0_init,
        device=device,
        h_cycles=h_cycles,
        total_steps=args.total_steps,
        seed=args.seed,
        collect_step_cpu_metrics=not args.disable_step_cpu_metrics,
    )

    # 6 conditions: S × {unguided, guided, resampled τ=0.3}
    conditions = [
        (S, beta, tau, label)
        for S in S_VALUES
        for beta, tau, label in [
            (BETA_UNGUIDED, 0.00, "unguided"),
            (BETA_GUIDED, 0.00, "guided"),
            (BETA_GUIDED, TAU_RESAMPLE, "resampled"),
        ]
    ]

    output_root = args.out
    output_root.mkdir(parents=True, exist_ok=True)

    def run_sweep(embedded, label: str):
        rows = []
        for sigma in tqdm(sigma_values, desc=f"σ[{label}]"):
            for S, beta, tau, cond_label in conditions:
                r = _run_one(**base, embedded=embedded, beta=beta, sigma=sigma, tau=tau, particles=S)
                rows.append(
                    {
                        "sigma": sigma,
                        "S": S,
                        "beta": beta,
                        "tau": tau,
                        "guided": cond_label,
                        **r,
                    }
                )
                print(
                    f"  σ={sigma:.2f}  S={S:2d}  τ={tau:.2f}  {cond_label:10s}  "
                    f"MAP={r['final_acc']:.4f}  "
                    f"oracle={r['final_oracle']:.4f}  "
                    f"ESS={r['final_ess']:.3f}"
                )
        return pd.DataFrame(rows)

    if args.num_folds == 1:
        embedded = _embed_batches(raw_batches, embedder, device)
        df = run_sweep(embedded, "single")
        out_path = output_root / "sigma_guided_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to: {out_path}")
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
        fold_df = run_sweep(embedded, fold_name)
        fold_df.to_csv(fold_dir / "sigma_guided_results.csv", index=False)
        fold_frames.append(fold_df)
        print(f"Saved fold {fold_name} with {info.num_examples} examples")

    summary = summarize_numeric_df(
        pd.concat(fold_frames, ignore_index=True),
        ["sigma", "S", "beta", "tau", "guided"],
    )
    summary.to_csv(output_root / "sigma_guided_results.csv", index=False)
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
    print(f"\nFold summary saved to: {output_root / 'sigma_guided_results.csv'}")


if __name__ == "__main__":
    main()
