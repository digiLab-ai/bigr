"""run_ablation_sensitivity.py — sensitivity sweeps behind Figure 6 (fig_ablation_sensitivity).

Two sweeps per benchmark, each run over the same k folds of a held-out subset:

1. Stress sweeps at S=16, N=48 with isotropic kernel noise sigma (the paper's value for
   the profile): tempering ``beta`` over the profile's grid at fixed S, and particle count
   ``S`` over {2, 4, 8, 16} at fixed beta=0.25, each with and without resampling
   (tau_ESS=0.3) — panels (a) and (b).
2. A resampling sweep over ESS thresholds tau_ESS in {0, 0.3, 0.5, 0.7, 1.0}, plus two
   thresholds with rejuvenation noise sigma_r=0.1 — panel (c).

The defaults reproduce the cached runs for each --profile:

    sudoku_extreme  32 batches of 64; beta in {0, 0.1, 0.25, 0.5, 0.75, 1}; sigma 0.3;
                    resampling sweep with sigma 0.3 / beta 0.25, rejuvenation at tau 0.3 and 0.7
    maze_hard       64 batches of 16; beta in {0, 0.05, 0.1, 0.25, 0.5, 1, 2}; sigma 0.5;
                    resampling sweep with sigma 0.3 / beta 0.25 (the Sudoku-Extreme values,
                    which produced the shipped Maze panel), rejuvenation at tau 0.5 and 0.7

Outputs (<out-dir>, used as given; the driver passes assets/runs/<bench>/ablation_sensitivity*_k5)
-------
    fold_XX/resample_stress_results.csv         panels (a, b) inputs for one fold
    fold_XX/stress_config.json
    fold_XX/resample/summary.csv                panel (c) input for one fold
    fold_XX/resample/resample_config.json
    fold_XX/resample/<condition>/{step_metrics.csv,config.json}
    resample_stress_results_summary.csv         fold mean/std (fig_ablation_sensitivity.py reads these)
    resample/summary_summary.csv
    resample/<condition>/step_metrics_summary.csv
    stress_config.json, resample/resample_config.json, fold_manifest.json

With --num-folds 1 the per-fold files are written directly under <out-dir>.

Examples
--------
# The cached runs (Figure 6): assets/runs/sudoku/ablation_sensitivity_k5 and
# assets/runs/maze/ablation_sensitivity_maze_hard_k5
bash scripts/run_experiments.sh --profile sudoku_extreme --ablation-sensitivity
bash scripts/run_experiments.sh --profile maze_hard --ablation-sensitivity

# Directly
python scripts/exps/run_ablation_sensitivity.py --profile maze_hard --num-folds 5 --fold-seed 31 \\
    --out-dir assets/runs/maze/ablation_sensitivity_maze_hard_k5
python scripts/exps/run_ablation_sensitivity.py --smoke --out-dir tmp/ablation_smoke
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
from fold_utils import build_example_folds, summarize_numeric_df

PROFILE_DEFAULTS = {
    "sudoku_extreme": {
        "checkpoint": REPO_ROOT / "data" / "sudoku_extreme" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "sudoku_extreme" / "dataset" / "sudoku_extreme",
        "out_dir": REPO_ROOT / "assets" / "runs" / "sudoku" / "ablation_sensitivity",
        "stress": {
            "batch_size": 64,
            "n_batches": 32,
            "sigma": 0.3,
            "beta_values": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        },
        "resample": {
            "batch_size": 64,
            "n_batches": 32,
            "sigma": 0.3,
            "beta": 0.25,
            "rejuvenation_taus": [0.3, 0.7],
        },
    },
    "maze_hard": {
        "checkpoint": REPO_ROOT / "data" / "maze_hard" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "maze_hard" / "dataset" / "maze-30x30-hard-1k",
        "out_dir": REPO_ROOT / "assets" / "runs" / "maze" / "ablation_sensitivity_maze_hard",
        "stress": {
            "batch_size": 16,
            "n_batches": 64,
            "sigma": 0.5,
            "beta_values": [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        },
        # sigma 0.3 / beta 0.25 are the Sudoku-Extreme values: the shipped Maze tau_ESS panel was produced with them.
        "resample": {
            "batch_size": 16,
            "n_batches": 64,
            "sigma": 0.3,
            "beta": 0.25,
            "rejuvenation_taus": [0.5, 0.7],
        },
    },
}

S_VALUES = [2, 4, 8, 16]
S_SWEEP_BETA = 0.25
STRESS_COMPARE = [(0.0, "no_resample"), (0.3, "tau_0.30")]
TAU_VALUES = [0.3, 0.5, 0.7, 1.0]
SIGMA_REJUVENATE = 0.1


def resample_conditions(rejuvenation_taus: list[float]) -> list[dict]:
    """The tau_ESS sweep: no resampling, each tau in TAU_VALUES, then rejuvenation at the given taus."""
    conditions = [{"tag": "no_resample", "ess_threshold": 0.0, "sigma_rejuvenate": 0.0}]
    conditions += [
        {"tag": f"tau_{tau:.2f}", "ess_threshold": tau, "sigma_rejuvenate": 0.0}
        for tau in TAU_VALUES
    ]
    conditions += [
        {"tag": f"tau_{tau:.2f}_rejuv", "ess_threshold": tau, "sigma_rejuvenate": SIGMA_REJUVENATE}
        for tau in rejuvenation_taus
    ]
    return conditions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Figure 6 sensitivity sweeps (beta, S, tau_ESS) for fig_ablation_sensitivity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        default="sudoku_extreme",
        choices=sorted(PROFILE_DEFAULTS),
        help="Dataset/checkpoint profile preset; also sets the sweep defaults below.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override profile checkpoint path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override profile dataset directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory, used as given (default: assets/runs/sudoku/ablation_sensitivity or "
            "assets/runs/maze/ablation_sensitivity_maze_hard; the driver appends _k5 for 5-fold runs)."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Target device (cuda/mps/cpu). Auto-selects CUDA > MPS > CPU if omitted.",
    )
    parser.add_argument("--seed", type=int, default=31)

    parser.add_argument("--stress-particles", type=int, default=16)
    parser.add_argument("--stress-total-steps", type=int, default=48)
    parser.add_argument("--stress-batch-size", type=int, default=None, help="Default per --profile (64 / 16).")
    parser.add_argument("--stress-n-batches", type=int, default=None, help="Default per --profile (32 / 64).")
    parser.add_argument(
        "--stress-sigma",
        type=float,
        default=None,
        help="Isotropic kernel noise for the beta and S sweeps. Default per --profile (0.3 / 0.5).",
    )
    parser.add_argument(
        "--beta-values",
        type=float,
        nargs="+",
        default=None,
        help="Tempering grid for the beta sweep. Default per --profile.",
    )

    parser.add_argument("--resample-particles", type=int, default=16)
    parser.add_argument("--resample-total-steps", type=int, default=48)
    parser.add_argument("--resample-batch-size", type=int, default=None, help="Default per --profile (64 / 16).")
    parser.add_argument("--resample-n-batches", type=int, default=None, help="Default per --profile (32 / 64).")
    parser.add_argument(
        "--resample-sigma",
        type=float,
        default=None,
        help="Kernel noise for the tau_ESS sweep. Default 0.3 for both profiles (see module docstring).",
    )
    parser.add_argument(
        "--resample-beta",
        type=float,
        default=None,
        help="Tempering for the tau_ESS sweep. Default 0.25 for both profiles (see module docstring).",
    )
    parser.add_argument(
        "--rejuvenation-taus",
        type=float,
        nargs="+",
        default=None,
        help="ESS thresholds run a second time with rejuvenation noise 0.1. Default per --profile.",
    )
    parser.add_argument("--num-folds", type=int, default=1)
    parser.add_argument("--fold-seed", type=int, default=None)
    parser.add_argument(
        "--disable-step-cpu-metrics",
        action="store_true",
        help="Skip per-step diagnostic materialization/aggregation.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke check with reduced steps and batches while preserving the output schema.",
    )
    parser.set_defaults(smoke=False)
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_batches(
    data_dir: Path, batch_size: int, n_batches: int
) -> list[dict[str, torch.Tensor]]:
    from trm.nn.data.sudoku_datamodule import SudokuDataModule

    dm = SudokuDataModule(data_dir=str(data_dir), batch_size=batch_size, num_workers=0)
    dm.setup(stage="test")
    loader = dm.test_dataloader()

    batches: list[dict[str, torch.Tensor]] = []
    for raw in tqdm(loader, desc="Loading batches", total=n_batches):
        current_batch = int(raw["input"].shape[0])
        batches.append(
            {
                "inputs": raw["input"],
                "labels": raw["output"],
                "puzzle_identifiers": raw.get(
                    "puzzle_identifiers",
                    torch.zeros(current_batch, dtype=torch.long),
                ),
            }
        )
        if len(batches) >= n_batches:
            break
    return batches


def _embed_batches(
    batches: list[dict[str, torch.Tensor]],
    *,
    embedder: TRMCoreEmbedder,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    embedded: list[dict[str, torch.Tensor]] = []
    for batch in tqdm(batches, desc="Embedding", leave=False):
        x = embedder(
            batch["inputs"].to(device),
            batch["puzzle_identifiers"].to(device),
        )
        embedded.append({"x": x, "labels": batch["labels"].to(device)})
    return embedded


def _run_trace(
    *,
    embedded: list[dict[str, torch.Tensor]],
    f_theta,
    q_head,
    decoder,
    y0_init: torch.Tensor,
    z0_init: torch.Tensor,
    device: torch.device,
    h_cycles: int,
    particles: int,
    total_steps: int,
    sigma: float,
    beta: float,
    tau: float,
    seed: int,
    sigma_rejuvenate: float = 0.0,
    collect_step_cpu_metrics: bool = True,
):
    cfg = InferenceConfig(
        particles=particles,
        total_steps=total_steps,
        sigma=sigma,
        ess_threshold=tau,
        beta=beta,
        seed=seed,
        sigma_rejuvenate=sigma_rejuvenate,
        collect_step_cpu_metrics=collect_step_cpu_metrics,
    )
    stoch = StochasticFTheta(f_theta, sigma=sigma, per_step_inner_noise=True)
    filt = BIGRFilter(stoch, q_head, decoder, cfg, h_cycles=h_cycles)
    with torch.no_grad():
        trace = filt.run(embedded, y0_init=y0_init, z0_init=z0_init, device=device)
    final = trace.step_records[-1]
    return trace, {
        "final_acc": final.exact_accuracy,
        "final_oracle": final.any_exact_rate,
        "final_gap": final.any_exact_rate - final.exact_accuracy,
        "final_ess": final.mean_ess,
        "final_entropy": final.weight_entropy,
        "final_q_div": final.q_diversity,
        "final_y_div": float(np.mean(final.y_diversity)),
        "final_z_div": float(np.mean(final.z_diversity)),
        "n_events": int(sum(r.resample_fraction > 0.0 for r in trace.step_records)),
        "mean_resample_fraction": float(
            np.mean([r.resample_fraction for r in trace.step_records])
        ),
    }


def _step_metrics(trace) -> pd.DataFrame:
    rows = [
        {
            "step": r.step,
            "exact_accuracy": r.exact_accuracy,
            "any_exact_rate": r.any_exact_rate,
            "oracle_gap": r.any_exact_rate - r.exact_accuracy,
            "mean_ess": r.mean_ess,
            "weight_entropy": r.weight_entropy,
            "q_diversity": r.q_diversity,
            "y_diversity": float(np.mean(r.y_diversity)),
            "z_diversity": float(np.mean(r.z_diversity)),
            "resample_fraction": r.resample_fraction,
        }
        for r in trace.step_records
    ]
    return pd.DataFrame(rows)


def _stress_row(exp: str, *, beta: float, particles: int, sigma: float, tau: float, label: str, metrics: dict) -> dict:
    # The `sigma_trans` column name is shared with the shipped CSVs; the kernel is isotropic, so it holds the one sigma.
    return {
        "exp": exp,
        "beta": beta,
        "S": particles,
        "sigma_trans": sigma,
        "tau": tau,
        "label": label,
        "final_acc": metrics["final_acc"],
        "final_oracle": metrics["final_oracle"],
        "final_gap": metrics["final_gap"],
        "final_ess": metrics["final_ess"],
        "n_events": metrics["n_events"],
    }


def _run_sweeps(
    *,
    stress_embedded: list[dict[str, torch.Tensor]],
    resample_embedded: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
    conditions: list[dict],
    device: torch.device,
    f_theta,
    q_head,
    decoder,
    y0_init: torch.Tensor,
    z0_init: torch.Tensor,
    h_cycles: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resample_dir = output_dir / "resample"
    resample_dir.mkdir(parents=True, exist_ok=True)

    stress_cfg = {
        "particles": args.stress_particles,
        "total_steps": args.stress_total_steps,
        "batch_size": args.stress_batch_size,
        "n_batches": args.stress_n_batches,
        "sigma": args.stress_sigma,
        "seed": args.seed,
        "beta_values": args.beta_values,
        "s_values": S_VALUES,
        "s_sweep_beta": S_SWEEP_BETA,
        "compare_conditions": [label for _, label in STRESS_COMPARE],
        "smoke": args.smoke,
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed,
    }
    _write_json(output_dir / "stress_config.json", stress_cfg)

    stress_common = dict(
        embedded=stress_embedded,
        f_theta=f_theta,
        q_head=q_head,
        decoder=decoder,
        y0_init=y0_init,
        z0_init=z0_init,
        device=device,
        h_cycles=h_cycles,
        total_steps=args.stress_total_steps,
        sigma=args.stress_sigma,
        seed=args.seed,
        collect_step_cpu_metrics=not args.disable_step_cpu_metrics,
    )

    print("\n" + "=" * 72)
    print("Stress sweeps: beta, particle count")
    print("=" * 72)
    stress_rows: list[dict] = []

    for beta in tqdm(args.beta_values, desc="beta sweep"):
        for tau, label in STRESS_COMPARE:
            _, metrics = _run_trace(**stress_common, particles=args.stress_particles, beta=beta, tau=tau)
            stress_rows.append(
                _stress_row(
                    "beta_sweep",
                    beta=beta,
                    particles=args.stress_particles,
                    sigma=args.stress_sigma,
                    tau=tau,
                    label=label,
                    metrics=metrics,
                )
            )
            print(
                f"  beta={beta:>4.2f}  {label:12s}  "
                f"MAP={metrics['final_acc']:.4f}  "
                f"oracle={metrics['final_oracle']:.4f}  "
                f"ESS={metrics['final_ess']:.3f}  "
                f"events={metrics['n_events']}"
            )

    for particles in tqdm(S_VALUES, desc="particle sweep"):
        for tau, label in STRESS_COMPARE:
            _, metrics = _run_trace(**stress_common, particles=particles, beta=S_SWEEP_BETA, tau=tau)
            stress_rows.append(
                _stress_row(
                    "S_sweep",
                    beta=S_SWEEP_BETA,
                    particles=particles,
                    sigma=args.stress_sigma,
                    tau=tau,
                    label=label,
                    metrics=metrics,
                )
            )
            print(
                f"  S={particles:>2d}  {label:12s}  "
                f"MAP={metrics['final_acc']:.4f}  "
                f"oracle={metrics['final_oracle']:.4f}  "
                f"ESS={metrics['final_ess']:.3f}  "
                f"events={metrics['n_events']}"
            )

    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(output_dir / "resample_stress_results.csv", index=False)
    print(f"\nSaved stress results to {output_dir / 'resample_stress_results.csv'}")

    resample_cfg = {
        "particles": args.resample_particles,
        "total_steps": args.resample_total_steps,
        "batch_size": args.resample_batch_size,
        "n_batches": args.resample_n_batches,
        "sigma": args.resample_sigma,
        "beta": args.resample_beta,
        "seed": args.seed,
        "conditions": conditions,
        "smoke": args.smoke,
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed,
    }
    _write_json(resample_dir / "resample_config.json", resample_cfg)

    print("\n" + "=" * 72)
    print("Resampling sweep: ESS-threshold trade-off")
    print("=" * 72)
    summary_rows: list[dict] = []
    step_frames: dict[str, pd.DataFrame] = {}
    for condition in tqdm(conditions, desc="resample sweep"):
        trace, metrics = _run_trace(
            embedded=resample_embedded,
            f_theta=f_theta,
            q_head=q_head,
            decoder=decoder,
            y0_init=y0_init,
            z0_init=z0_init,
            device=device,
            h_cycles=h_cycles,
            particles=args.resample_particles,
            total_steps=args.resample_total_steps,
            sigma=args.resample_sigma,
            beta=args.resample_beta,
            tau=condition["ess_threshold"],
            sigma_rejuvenate=condition["sigma_rejuvenate"],
            seed=args.seed,
            collect_step_cpu_metrics=not args.disable_step_cpu_metrics,
        )

        condition_dir = resample_dir / condition["tag"]
        condition_dir.mkdir(parents=True, exist_ok=True)
        step_df = _step_metrics(trace)
        step_df.to_csv(condition_dir / "step_metrics.csv", index=False)
        _write_json(
            condition_dir / "config.json",
            {
                **condition,
                "particles": args.resample_particles,
                "total_steps": args.resample_total_steps,
                "sigma": args.resample_sigma,
                "beta": args.resample_beta,
                "seed": args.seed,
                "num_folds": args.num_folds,
                "fold_seed": args.fold_seed,
            },
        )

        summary_rows.append(
            {
                **condition,
                "final_accuracy": metrics["final_acc"],
                "final_any_exact": metrics["final_oracle"],
                "final_oracle_gap": metrics["final_gap"],
                "final_ess": metrics["final_ess"],
                "final_entropy": metrics["final_entropy"],
                "final_q_div": metrics["final_q_div"],
                "final_y_div": metrics["final_y_div"],
                "final_z_div": metrics["final_z_div"],
                "n_resample_events": metrics["n_events"],
                "mean_resample_fraction": metrics["mean_resample_fraction"],
            }
        )
        step_frames[condition["tag"]] = step_df
        print(
            f"  {condition['tag']:15s}  tau={condition['ess_threshold']:.2f}  "
            f"sigma_r={condition['sigma_rejuvenate']:.2f}  "
            f"MAP={metrics['final_acc']:.4f}  ESS={metrics['final_ess']:.3f}  "
            f"events={metrics['n_events']}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(resample_dir / "summary.csv", index=False)
    print(f"\nSaved resampling summary to {resample_dir / 'summary.csv'}")
    return stress_df, summary_df, step_frames


def _resolve_profile_defaults(args: argparse.Namespace) -> None:
    """Fill every None argument from the --profile preset (in place)."""
    profile_cfg = PROFILE_DEFAULTS[args.profile]
    if args.checkpoint is None:
        args.checkpoint = profile_cfg["checkpoint"]
    if args.data_dir is None:
        args.data_dir = profile_cfg["data_dir"]
    if args.out_dir is None:
        args.out_dir = profile_cfg["out_dir"]
    stress, resample = profile_cfg["stress"], profile_cfg["resample"]
    if args.stress_batch_size is None:
        args.stress_batch_size = stress["batch_size"]
    if args.stress_n_batches is None:
        args.stress_n_batches = stress["n_batches"]
    if args.stress_sigma is None:
        args.stress_sigma = stress["sigma"]
    if args.beta_values is None:
        args.beta_values = list(stress["beta_values"])
    if args.resample_batch_size is None:
        args.resample_batch_size = resample["batch_size"]
    if args.resample_n_batches is None:
        args.resample_n_batches = resample["n_batches"]
    if args.resample_sigma is None:
        args.resample_sigma = resample["sigma"]
    if args.resample_beta is None:
        args.resample_beta = resample["beta"]
    if args.rejuvenation_taus is None:
        args.rejuvenation_taus = list(resample["rejuvenation_taus"])
    args.fold_seed = args.seed if args.fold_seed is None else args.fold_seed


def main() -> None:
    args = parse_args()
    _resolve_profile_defaults(args)
    conditions = resample_conditions(args.rejuvenation_taus)
    if args.smoke:
        args.stress_total_steps = 8
        args.stress_batch_size = 4
        args.stress_n_batches = 2
        args.resample_total_steps = 12
        args.resample_batch_size = 4
        args.resample_n_batches = 2
        print(
            "Smoke mode: stress=(steps=8, batches=2×4), "
            "resample=(steps=12, batches=2×4)"
        )

    output_root = args.out_dir
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else select_device()
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    trm = load_trm_backbone_checkpoint(args.checkpoint, map_location=device)
    trm.eval().to(device)
    h_cycles = int(getattr(trm.hparams, "H_cycles", 3))

    embedder = TRMCoreEmbedder(trm)
    decoder = TRMCoreDecoder(trm)
    q_head = TRMCoreQHead(trm)
    f_theta = TRMNativeCycleFTheta(trm)
    y0_init = trm.z_H_init.detach().clone()
    z0_init = trm.z_L_init.detach().clone()

    stress_raw = _load_batches(args.data_dir, args.stress_batch_size, args.stress_n_batches)
    resample_raw = _load_batches(
        args.data_dir, args.resample_batch_size, args.resample_n_batches
    )

    sweep_common = dict(
        args=args,
        conditions=conditions,
        device=device,
        f_theta=f_theta,
        q_head=q_head,
        decoder=decoder,
        y0_init=y0_init,
        z0_init=z0_init,
        h_cycles=h_cycles,
    )

    if args.num_folds == 1:
        stress_embedded = _embed_batches(stress_raw, embedder=embedder, device=device)
        resample_embedded = _embed_batches(
            resample_raw, embedder=embedder, device=device
        )
        _run_sweeps(
            stress_embedded=stress_embedded,
            resample_embedded=resample_embedded,
            output_dir=output_root,
            **sweep_common,
        )
        print("\nDone.")
        return

    stress_folds, stress_info = build_example_folds(
        stress_raw,
        num_folds=args.num_folds,
        seed=args.fold_seed,
        batch_size=args.stress_batch_size,
    )
    resample_folds, resample_info = build_example_folds(
        resample_raw,
        num_folds=args.num_folds,
        seed=args.fold_seed,
        batch_size=args.resample_batch_size,
    )

    _write_json(
        output_root / "stress_config.json",
        {
            "num_folds": args.num_folds,
            "fold_seed": args.fold_seed,
            "folds": [info.__dict__ for info in stress_info],
            "smoke": args.smoke,
        },
    )
    _write_json(
        output_root / "resample" / "resample_config.json",
        {
            "num_folds": args.num_folds,
            "fold_seed": args.fold_seed,
            "folds": [info.__dict__ for info in resample_info],
            "smoke": args.smoke,
        },
    )

    fold_stress_frames: list[pd.DataFrame] = []
    fold_summary_frames: list[pd.DataFrame] = []
    fold_step_frames: dict[str, list[pd.DataFrame]] = {
        condition["tag"]: [] for condition in conditions
    }
    for fold_index in range(args.num_folds):
        fold_dir = output_root / f"fold_{fold_index + 1:02d}"
        print(
            f"\nRunning fold {fold_index + 1}/{args.num_folds}: "
            f"stress={stress_info[fold_index].num_examples} examples, "
            f"resample={resample_info[fold_index].num_examples} examples"
        )
        fold_stress_embedded = _embed_batches(
            stress_folds[fold_index], embedder=embedder, device=device
        )
        fold_resample_embedded = _embed_batches(
            resample_folds[fold_index], embedder=embedder, device=device
        )
        stress_df, summary_df, step_frames = _run_sweeps(
            stress_embedded=fold_stress_embedded,
            resample_embedded=fold_resample_embedded,
            output_dir=fold_dir,
            **sweep_common,
        )
        fold_stress_frames.append(stress_df)
        fold_summary_frames.append(summary_df)
        for tag, frame in step_frames.items():
            fold_step_frames[tag].append(frame)

    stress_summary = summarize_numeric_df(
        pd.concat(fold_stress_frames, ignore_index=True),
        group_cols=["exp", "beta", "S", "sigma_trans", "tau", "label"],
    )
    stress_summary.to_csv(output_root / "resample_stress_results_summary.csv", index=False)

    resample_summary = summarize_numeric_df(
        pd.concat(fold_summary_frames, ignore_index=True),
        group_cols=["tag", "ess_threshold", "sigma_rejuvenate"],
    )
    resample_summary.to_csv(output_root / "resample" / "summary_summary.csv", index=False)

    for condition in conditions:
        tag = condition["tag"]
        condition_dir = output_root / "resample" / tag
        condition_dir.mkdir(parents=True, exist_ok=True)
        step_summary = summarize_numeric_df(
            pd.concat(fold_step_frames[tag], ignore_index=True),
            group_cols=["step"],
        )
        step_summary.to_csv(condition_dir / "step_metrics_summary.csv", index=False)

    _write_json(
        output_root / "fold_manifest.json",
        {
            "num_folds": args.num_folds,
            "fold_seed": args.fold_seed,
            "stress_folds": [info.__dict__ for info in stress_info],
            "resample_folds": [info.__dict__ for info in resample_info],
            "outputs": {
                "stress_summary": "resample_stress_results_summary.csv",
                "resample_summary": "resample/summary_summary.csv",
                "step_metrics_summary": {
                    condition["tag"]: f"resample/{condition['tag']}/step_metrics_summary.csv"
                    for condition in conditions
                },
            },
        },
    )
    print(f"\nFold summaries saved to: {output_root}")
    print("\nDone.")


if __name__ == "__main__":
    main()
