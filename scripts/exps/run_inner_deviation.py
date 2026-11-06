"""run_inner_deviation.py — Measure stochastic inner-rollout deviation from the
deterministic lane.

Empirically validates Theorem 1 (local stability of the inner recursion):

    E[||ζ_m − z_m||² · 1_{τ_r > m}] ≤ σ²ν² Σ_{j=0}^{m} ρ_r^{2j}

where τ_r = inf{m : ||ζ_m − z_m|| > r} is the first-exit time from a tube of
radius r.  The masked expectation is the quantity the theorem bounds; plotting
the unconditional mean is invalid.

For each outer step of a deterministic trajectory we run K stochastic inner
rollouts at each σ, compute the indicator-masked squared deviation and the
survival probability Pr(τ_r > m), and estimate the local Lipschitz constant
ρ̂_r via probe perturbations within the tube.

Outputs (<out>, written as given; with --num-folds k > 1 one fold_XX/ per fold plus the
fold-mean summaries below; the driver passes assets/runs/<bench>/inner_deviation*_k5)
--------------------------------------------------------------------------------------
    config.json
    deviation_masked.csv    — E[||e_m||² 1_{τ_r>m}] and Pr(τ_r>m) per
                              (tube_radius, sigma, inner_step)
    survival_at_M.csv       — Pr(τ_r > M) per (tube_radius, sigma)
    rho_local.csv           — local Lipschitz proxy ρ̂_r per (tube_radius,
                              inner_step); used to compute per-step bounds
    *_summary.csv, fold_manifest.json   — k-fold aggregates (fig_inner_deviation.py and
                              fig_maze_case_study.py read these)

Usage
-----
# The cached runs: Figure 2 (assets/runs/sudoku/inner_deviation_k5) and Figure 5
# (assets/runs/maze/inner_deviation_maze_hard_k5)
bash scripts/run_experiments.sh --profile sudoku_extreme --inner-deviation
bash scripts/run_experiments.sh --profile maze_hard --inner-deviation

# Smoke check
python scripts/exps/run_inner_deviation.py --smoke --out tmp/inner_deviation_smoke

# Directly, the Figure 2 protocol (whole batches are loaded, so 2,000 puzzles at batch 64 is 2,048)
python scripts/exps/run_inner_deviation.py --profile sudoku_extreme \\
    --n-puzzles 2000 --outer-steps 48 --n-samples 16 --batch-size 64 \\
    --sigma-values 0.0 0.1 0.2 0.3 0.5 0.8 \\
    --tube-alphas 0.5 1.0 1.25 1.5 2.0 \\
    --n-probes 16 --num-folds 5 --fold-seed 31 \\
    --out assets/runs/sudoku/inner_deviation_k5
"""

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
    TRMCoreEmbedder,
    TRMNativeCycleFTheta,
    load_trm_backbone_checkpoint,
    select_device,
)
from fold_utils import build_example_folds, summarize_numeric_df

PROFILE_DEFAULTS = {
    "sudoku_extreme": {
        "checkpoint": REPO_ROOT / "data" / "sudoku_extreme" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "sudoku_extreme" / "dataset" / "sudoku_extreme",
        "out": REPO_ROOT / "assets" / "runs" / "sudoku" / "inner_deviation",
    },
    "maze_hard": {
        "checkpoint": REPO_ROOT / "data" / "maze_hard" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "maze_hard" / "dataset" / "maze-30x30-hard-1k",
        "out": REPO_ROOT / "assets" / "runs" / "maze" / "inner_deviation_maze_hard",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure stochastic inner-rollout deviation (Theorem 1 validation)",
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
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory, used as given (default: assets/runs/sudoku/inner_deviation or "
            "assets/runs/maze/inner_deviation_maze_hard; the driver appends _k5 for 5-fold runs)."
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        help="Target device (cuda/mps/cpu). Auto-selects CUDA > MPS > CPU if omitted.",
    )
    p.add_argument(
        "--n-puzzles",
        type=int,
        default=256,
        help=(
            "Puzzles to load. Whole batches are loaded, so the count is rounded up to a multiple "
            "of --batch-size (2,000 at batch 64 loads 2,048, the Figure 2 cohort)."
        ),
    )
    p.add_argument("--outer-steps", type=int, default=48)
    p.add_argument(
        "--n-samples", type=int, default=16, help="Stochastic rollouts per outer state."
    )
    p.add_argument(
        "--sigma-values", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.8]
    )
    p.add_argument(
        "--tube-alphas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
        help="Tube radii as multiples of sqrt(L*D) per-element RMS units.",
    )
    p.add_argument(
        "--n-probes",
        type=int,
        default=16,
        help="Number of probe perturbations per lane point for local Lipschitz estimation.",
    )
    p.add_argument(
        "--lipschitz-method",
        choices=["probes", "power-iter"],
        default="probes",
        help=(
            "How to estimate rho_hat_r. 'probes' (default) = max secant ratio over "
            "n_probes random unit-norm perturbations of magnitude r (default; the estimator "
            "used for Figures 2 and 5). "
            "'power-iter' = spectral norm sigma_max(J(z_m)) via JVP+VJP power iteration; "
            "tighter upper bound on the linearised local Lipschitz, but radius-independent "
            "(stored value is identical across tube alphas)."
        ),
    )
    p.add_argument(
        "--power-iter-steps",
        type=int,
        default=20,
        help="Power-iteration steps when --lipschitz-method=power-iter.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--num-folds", type=int, default=1)
    p.add_argument("--fold-seed", type=int, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Fast smoke check: 32 puzzles, 12 steps, 2 rollouts, 2 σ values, 1 tube.",
    )
    p.set_defaults(smoke=False)
    return p.parse_args()


def load_batches(data_dir, batch_size, n_puzzles):
    from trm.nn.data.sudoku_datamodule import SudokuDataModule

    dm = SudokuDataModule(data_dir=str(data_dir), batch_size=batch_size, num_workers=0)
    dm.setup(stage="test")
    loader = dm.test_dataloader()
    batches, n_seen = [], 0
    for raw in loader:
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
        n_seen += raw["input"].shape[0]
        if n_seen >= n_puzzles:
            break
    return batches


def _embed_batches(batches, embedder, device):
    embedded = []
    for batch in tqdm(batches, desc="Embedding", leave=False):
        embedded.append(
            {
                "x": embedder(
                    batch["inputs"].to(device),
                    batch["puzzle_identifiers"].to(device),
                ),
                "labels": batch["labels"].to(device),
            }
        )
    return embedded


def _det_inner_lane(f_theta, x, y, z, M):
    """Run M deterministic inner steps from z. Returns list of M+1 states (m=0..M)."""
    cos_sin = f_theta._cos_sin()
    injection = (x + y).to(f_theta.forward_dtype)
    lane = [z.to(f_theta.forward_dtype)]
    for _ in range(M):
        lane.append(f_theta._belief_update_single_step(injection, lane[-1], cos_sin))
    return lane  # list of M+1 tensors, each (B, L, D)


def _randn(shape, *, device, dtype, gen):
    if getattr(gen, "device", None) is not None and gen.device.type == device.type:
        return torch.randn(shape, generator=gen, device=device, dtype=dtype)
    return torch.randn(shape, generator=gen, dtype=dtype).to(device=device)


def _estimate_rho_power_iter(
    f_theta,
    injection: torch.Tensor,
    z_m: torch.Tensor,
    cos_sin,
    *,
    n_iters: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """Per-batch spectral-norm sigma_max(J(z_m)) via JVP+VJP power iteration.

    J = d f_theta._belief_update_single_step(injection, z, cos_sin) / dz, with
    injection and cos_sin treated as constants. Iterates v <- (J^T J v) / ||...||
    with v unit Frobenius norm per batch element, returning ||J v|| at the final
    step as the operator-norm estimate.

    Returns a tensor of shape (B,) in float32.
    """
    z_in = z_m.detach().to(f_theta.forward_dtype)
    B = z_in.shape[0]

    def func(z):
        return f_theta._belief_update_single_step(injection, z, cos_sin)

    v = _randn(z_in.shape, device=z_in.device, dtype=z_in.dtype, gen=gen)
    v_norm = v.float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-12)
    v = (v.float() / v_norm).to(z_in.dtype)

    sigma_est = torch.zeros(B, device=z_in.device, dtype=torch.float32)

    with torch.enable_grad():
        for _ in range(int(n_iters)):
            _, u = torch.autograd.functional.jvp(
                func, (z_in,), (v,), create_graph=False, strict=False
            )
            sigma_est = u.float().pow(2).sum(dim=(1, 2)).sqrt()

            _, w_tuple = torch.autograd.functional.vjp(
                func, (z_in,), u, create_graph=False, strict=False
            )
            w = w_tuple[0] if isinstance(w_tuple, tuple) else w_tuple
            w_norm = (
                w.float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-12)
            )
            v = (w.float() / w_norm).to(z_in.dtype)

    return sigma_est


def _stoch_inner_rollout_batched(f_theta, x, y, z, sigma, M, gen, n_samples):
    """Run M stochastic inner steps for n_samples rollouts in one batched pass."""
    cos_sin = f_theta._cos_sin()
    injection = (x + y).to(f_theta.forward_dtype)
    B, L_dim, D_dim = z.shape
    K = int(n_samples)

    injection_flat = (
        injection.unsqueeze(0)
        .expand(K, -1, -1, -1)
        .reshape(K * B, L_dim, D_dim)
    )
    z_state = z.to(f_theta.forward_dtype).unsqueeze(0).expand(K, -1, -1, -1).clone()
    z_state = z_state + sigma * _randn(
        z_state.shape, device=z_state.device, dtype=z_state.dtype, gen=gen
    )
    lane = [z_state]
    z_flat = z_state.reshape(K * B, L_dim, D_dim)

    for _ in range(M):
        z_next_flat = f_theta._belief_update_single_step(injection_flat, z_flat, cos_sin)
        z_next = z_next_flat.view(K, B, L_dim, D_dim)
        z_state = z_next + sigma * _randn(
            z_next.shape, device=z_next.device, dtype=z_next.dtype, gen=gen
        )
        lane.append(z_state)
        z_flat = z_state.reshape(K * B, L_dim, D_dim)
    return lane


def _run_analysis(
    *,
    embedded_batches,
    args,
    device,
    trm,
    f_theta,
    y0_init,
    z0_init,
    M,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (output_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    sample_x = embedded_batches[0]["x"]  # (B, L, D)
    _B, L_dim, D_dim = sample_x.shape
    nu_sq = float(L_dim * D_dim)  # E[||σξ||²] = σ² * L * D when ξ ~ N(0,I)
    print(f"Latent shape L={L_dim}, D={D_dim}, ν²=L·D={nu_sq:.0f}")

    tube_radii = {alpha: alpha * nu_sq**0.5 for alpha in args.tube_alphas}
    print(
        f"Tube radii (Frobenius norm): { {a: f'{r:.1f}' for a, r in tube_radii.items()} }"
    )
    print(f"Lipschitz estimator: {args.lipschitz_method}")
    if args.lipschitz_method == "power-iter":
        for p in trm.parameters():
            p.requires_grad_(False)
        print(f"  power-iter steps: {args.power_iter_steps}")

    gen_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(args.seed)

    n_sigma = len(args.sigma_values)
    n_tubes = len(args.tube_alphas)
    tube_items = list(tube_radii.items())

    stats_shape = (n_tubes, n_sigma, M + 1)
    acc_dtype = torch.float32
    masked_sum = torch.zeros(stats_shape, device=device, dtype=acc_dtype)
    masked_sumsq = torch.zeros(stats_shape, device=device, dtype=acc_dtype)
    survival_sum = torch.zeros(stats_shape, device=device, dtype=acc_dtype)
    sample_count = torch.zeros(stats_shape, device=device, dtype=acc_dtype)
    rho_probe_max = torch.zeros((n_tubes, M), device=device, dtype=torch.float32)

    with torch.no_grad():
        for batch in tqdm(embedded_batches, desc="Batches"):
            x = batch["x"]
            B = x.shape[0]

            y = (
                y0_init.view(1, 1, -1)
                .expand(B, L_dim, -1)
                .clone()
                .to(device=device, dtype=x.dtype)
            )
            z = (
                z0_init.view(1, 1, -1)
                .expand(B, L_dim, -1)
                .clone()
                .to(device=device, dtype=x.dtype)
            )

            for _ in range(args.outer_steps):
                det_lane = _det_inner_lane(f_theta, x, y, z, M)
                cos_sin = f_theta._cos_sin()
                injection = (x + y).to(f_theta.forward_dtype)

                if args.lipschitz_method == "power-iter":
                    for m in range(M):
                        z_m = det_lane[m].to(f_theta.forward_dtype)
                        sigma_per_b = _estimate_rho_power_iter(
                            f_theta,
                            injection,
                            z_m,
                            cos_sin,
                            n_iters=args.power_iter_steps,
                            gen=gen,
                        )
                        worst = sigma_per_b.max()
                        for ti in range(n_tubes):
                            rho_probe_max[ti, m] = torch.maximum(
                                rho_probe_max[ti, m], worst
                            )
                else:
                    injection_probe_flat = (
                        injection.unsqueeze(0)
                        .expand(args.n_probes, -1, -1, -1)
                        .reshape(args.n_probes * B, L_dim, D_dim)
                    )

                    for ti, (_alpha, r) in enumerate(tube_items):
                        for m in range(M):
                            z_m = det_lane[m].to(f_theta.forward_dtype)
                            f_det = f_theta._belief_update_single_step(injection, z_m, cos_sin)

                            raw = _randn(
                                (args.n_probes, B, L_dim, D_dim),
                                device=z_m.device,
                                dtype=z_m.dtype,
                                gen=gen,
                            )
                            raw_norm = (
                                raw.pow(2).sum(dim=(2, 3), keepdim=True).sqrt().clamp(min=1e-8)
                            )
                            delta = raw / raw_norm * r
                            z_probe_flat = (z_m.unsqueeze(0) + delta).reshape(
                                args.n_probes * B, L_dim, D_dim
                            )
                            f_probe_flat = f_theta._belief_update_single_step(
                                injection_probe_flat, z_probe_flat, cos_sin
                            )
                            f_probe = f_probe_flat.view(args.n_probes, B, L_dim, D_dim)
                            numer = (
                                (f_probe.float() - f_det.unsqueeze(0).float())
                                .pow(2)
                                .sum(dim=(2, 3))
                                .sqrt()
                            )
                            denom = (
                                delta.float()
                                .pow(2)
                                .sum(dim=(2, 3))
                                .sqrt()
                                .clamp(min=1e-8)
                            )
                            rho_probe_max[ti, m] = torch.maximum(
                                rho_probe_max[ti, m], (numer / denom).max()
                            )

                for si, sigma in enumerate(args.sigma_values):
                    stoch_lane = _stoch_inner_rollout_batched(
                        f_theta, x, y, z, sigma, M, gen, args.n_samples
                    )
                    e_sq = []
                    for m in range(M + 1):
                        diff = stoch_lane[m].float() - det_lane[m].unsqueeze(0).float()
                        e_sq.append(diff.pow(2).sum(dim=(2, 3)))

                    for ti, (_alpha, r) in enumerate(tube_items):
                        r_sq = float(r * r)
                        survived = torch.ones(
                            (args.n_samples, B), dtype=torch.bool, device=device
                        )
                        for m in range(M + 1):
                            survived = survived & (e_sq[m] <= r_sq)
                            survived_f = survived.float()
                            masked_sq = e_sq[m] * survived_f
                            masked_sum[ti, si, m] += masked_sq.sum(dtype=acc_dtype)
                            masked_sumsq[ti, si, m] += (masked_sq * masked_sq).sum(
                                dtype=acc_dtype
                            )
                            survival_sum[ti, si, m] += survived_f.sum(dtype=acc_dtype)
                            sample_count[ti, si, m] += float(masked_sq.numel())

                z_new = det_lane[M]
                y = f_theta.state_update(y.to(f_theta.forward_dtype), z_new)
                z = z_new

    rho_local = rho_probe_max.detach().cpu().numpy().tolist()
    masked_sum_np = masked_sum.detach().cpu().numpy()
    masked_sumsq_np = masked_sumsq.detach().cpu().numpy()
    survival_sum_np = survival_sum.detach().cpu().numpy()
    sample_count_np = sample_count.detach().cpu().numpy()

    def partial_bound(ti, si, m):
        sigma = args.sigma_values[si]
        s = 0.0
        rho = max(rho_local[ti]) if rho_local[ti] else 0.0
        for j in range(m + 1):
            s += rho ** (2 * j)
        return sigma**2 * nu_sq * s

    masked_rows = []
    for ti, alpha in enumerate(args.tube_alphas):
        r = tube_radii[alpha]
        rho_r = max(rho_local[ti]) if rho_local[ti] else 0.0
        for si, sigma in enumerate(args.sigma_values):
            for m in range(M + 1):
                n_total = float(sample_count_np[ti, si, m])
                if n_total > 0:
                    mean_val = float(masked_sum_np[ti, si, m] / n_total)
                    variance = float(masked_sumsq_np[ti, si, m] / n_total - mean_val**2)
                    std_val = float(np.sqrt(max(variance, 0.0)))
                    survival_prob = float(survival_sum_np[ti, si, m] / n_total)
                else:
                    mean_val = 0.0
                    std_val = 0.0
                    survival_prob = 0.0
                masked_rows.append(
                    {
                        "tube_alpha": alpha,
                        "tube_radius": r,
                        "sigma": sigma,
                        "inner_step": m,
                        "mean_masked_dev": mean_val,
                        "std_masked_dev": std_val,
                        "survival_prob": survival_prob,
                        "rho_hat_r": rho_r,
                        "partial_bound": partial_bound(ti, si, m),
                        "n_samples": int(n_total),
                    }
                )
    masked_df = pd.DataFrame(masked_rows)
    masked_df.to_csv(output_dir / "deviation_masked.csv", index=False)
    print(f"Saved deviation_masked.csv ({len(masked_df)} rows)")

    surv_rows = []
    for ti, alpha in enumerate(args.tube_alphas):
        r = tube_radii[alpha]
        for si, sigma in enumerate(args.sigma_values):
            n_total = float(sample_count_np[ti, si, M])
            surv_rows.append(
                {
                    "tube_alpha": alpha,
                    "tube_radius": r,
                    "sigma": sigma,
                    "survival_prob_at_M": (
                        float(survival_sum_np[ti, si, M] / n_total) if n_total > 0 else 0.0
                    ),
                }
            )
    surv_df = pd.DataFrame(surv_rows)
    surv_df.to_csv(output_dir / "survival_at_M.csv", index=False)
    print(f"Saved survival_at_M.csv ({len(surv_df)} rows)")

    rho_rows = []
    for ti, alpha in enumerate(args.tube_alphas):
        for m in range(M):
            rho_rows.append(
                {
                    "tube_alpha": alpha,
                    "tube_radius": tube_radii[alpha],
                    "inner_step": m,
                    "rho_hat_r_max": rho_local[ti][m],
                }
            )
    rho_df = pd.DataFrame(rho_rows)
    rho_df.to_csv(output_dir / "rho_local.csv", index=False)
    print(f"Saved rho_local.csv ({len(rho_df)} rows)")

    print(f"\nSurvival at M={M} (all tubes × σ):")
    for _, row in surv_df.iterrows():
        print(
            f"  α={row['tube_alpha']:.2f}  σ={row['sigma']:.2f}  "
            f"Pr(τ_r>M)={row['survival_prob_at_M']:.3f}"
        )
    print("\nLocal ρ̂_r (max over probes) at step 0:")
    for ti, alpha in enumerate(args.tube_alphas):
        if rho_local[ti]:
            print(f"  α={alpha:.2f}  ρ̂_r={rho_local[ti][0]:.4f}")
    print(f"\nArtefacts saved to: {output_dir}")

    return masked_df, surv_df, rho_df


def main():
    args = parse_args()
    profile_cfg = PROFILE_DEFAULTS[args.profile]
    if args.checkpoint is None:
        args.checkpoint = profile_cfg["checkpoint"]
    if args.data_dir is None:
        args.data_dir = profile_cfg["data_dir"]
    if args.out is None:
        args.out = profile_cfg["out"]
    args.fold_seed = args.seed if args.fold_seed is None else args.fold_seed

    if args.smoke:
        args.n_puzzles = 32
        args.outer_steps = 12
        args.n_samples = 2
        args.sigma_values = [0.1, 0.5]
        args.tube_alphas = [1.0]
        args.n_probes = 4
        print("Smoke mode: 32 puzzles, 12 steps, 2 rollouts, σ∈{0.1,0.5}, 1 tube")

    output_root = args.out
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else select_device()

    print(f"Loading checkpoint: {args.checkpoint}")
    trm = load_trm_backbone_checkpoint(
        args.checkpoint, map_location=device, use_ema=True
    )
    trm.eval().to(device)

    M = int(getattr(trm.hparams, "L_cycles", 6))
    print(f"L_cycles (inner steps M) = {M}")

    embedder = TRMCoreEmbedder(trm)
    f_theta = TRMNativeCycleFTheta(trm)

    y0_init = trm.z_H_init.detach().clone()
    z0_init = trm.z_L_init.detach().clone()

    print(f"Loading batches for {args.n_puzzles} puzzles …")
    raw_batches = load_batches(args.data_dir, args.batch_size, args.n_puzzles)

    if not raw_batches:
        raise RuntimeError("No batches were loaded from the evaluation split")

    if args.num_folds == 1:
        embedded_batches = _embed_batches(raw_batches, embedder, device)
        _run_analysis(
            embedded_batches=embedded_batches,
            args=args,
            device=device,
            trm=trm,
            f_theta=f_theta,
            y0_init=y0_init,
            z0_init=z0_init,
            M=M,
            output_dir=output_root,
        )
        return

    fold_batches, fold_info = build_example_folds(
        raw_batches, num_folds=args.num_folds, seed=args.fold_seed, batch_size=args.batch_size
    )
    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg_dict["folds"] = [info.__dict__ for info in fold_info]
    (output_root / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    fold_masked_frames = []
    fold_survival_frames = []
    fold_rho_frames = []
    for fold_index, batches_for_fold in enumerate(fold_batches, start=1):
        fold_dir = output_root / f"fold_{fold_index:02d}"
        print(
            f"\nRunning fold {fold_index}/{args.num_folds}: "
            f"{fold_info[fold_index - 1].num_examples} examples, "
            f"{fold_info[fold_index - 1].num_batches} batches"
        )
        embedded_batches = _embed_batches(batches_for_fold, embedder, device)
        masked_df, surv_df, rho_df = _run_analysis(
            embedded_batches=embedded_batches,
            args=args,
            device=device,
            trm=trm,
            f_theta=f_theta,
            y0_init=y0_init,
            z0_init=z0_init,
            M=M,
            output_dir=fold_dir,
        )
        fold_masked_frames.append(masked_df)
        fold_survival_frames.append(surv_df)
        fold_rho_frames.append(rho_df)

    masked_summary = summarize_numeric_df(
        pd.concat(fold_masked_frames, ignore_index=True),
        group_cols=["tube_alpha", "tube_radius", "sigma", "inner_step"],
    )
    masked_summary.to_csv(output_root / "deviation_masked_summary.csv", index=False)

    survival_summary = summarize_numeric_df(
        pd.concat(fold_survival_frames, ignore_index=True),
        group_cols=["tube_alpha", "tube_radius", "sigma"],
    )
    survival_summary.to_csv(output_root / "survival_at_M_summary.csv", index=False)

    rho_summary = summarize_numeric_df(
        pd.concat(fold_rho_frames, ignore_index=True),
        group_cols=["tube_alpha", "tube_radius", "inner_step"],
    )
    rho_summary.to_csv(output_root / "rho_local_summary.csv", index=False)

    manifest = {
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed,
        "folds": [info.__dict__ for info in fold_info],
        "outputs": {
            "deviation_masked_summary": "deviation_masked_summary.csv",
            "survival_at_M_summary": "survival_at_M_summary.csv",
            "rho_local_summary": "rho_local_summary.csv",
        },
    }
    (output_root / "fold_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(f"\nFold summaries saved to: {output_root}")


if __name__ == "__main__":
    main()
