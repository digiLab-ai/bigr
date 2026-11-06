"""bench_device — wall-clock and peak-memory benchmark of the guided particle filter (Table 3).

Runs the guided particle filter at the paper's settings for the profile (sigma, beta,
tau_ESS = 0.3) for a fixed number of steps on each requested device and reports run time,
throughput (steps/second), peak CUDA memory and the speedup over CPU when a CPU run is
included. With --json-out it also writes one JSON record per device (device, profile,
particles, steps, batch_size, n_batches, n_puzzles, elapsed_s, steps_per_s, peak_mb,
s_per_puzzle). ``bash scripts/run_experiments.sh --profile sudoku_extreme --timing`` runs
S in {1, 2, 4, 8, 16} with N=48 on 4 batches of 16 and tabulates those records with
scripts/analysis/build_timing_table.py into Table 3; the shipped
assets/runs/analysis/ablations/timing/h100_timing.md is the paper's H100 run.

Examples
--------
# One Table 3 point: S=16, N=48, 4 batches of 16, on CUDA
python scripts/exps/bench_device.py --profile sudoku_extreme --devices cuda --particles 16 --steps 48 \\
    --n-batches 4 --batch-size 16 --json-out tmp/timing/bench_sudoku_cuda_S16.json

# Compare every available device (cpu + cuda/mps) at small settings
python scripts/exps/bench_device.py --particles 8 --steps 12 --n-batches 2 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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

PROFILE_DEFAULTS = {
    # Paper Table 4: sigma / beta per benchmark (tau_ESS = 0.3 for both).
    "sudoku_extreme": {
        "checkpoint": REPO_ROOT / "data" / "sudoku_extreme" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "sudoku_extreme" / "dataset" / "sudoku_extreme",
        "sigma": 0.3,
        "beta": 0.25,
    },
    "maze_hard": {
        "checkpoint": REPO_ROOT / "data" / "maze_hard" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "maze_hard" / "dataset" / "maze-30x30-hard-1k",
        "sigma": 0.5,
        "beta": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Time the guided particle filter on CPU / CUDA / MPS (Table 3)",
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
        "--devices",
        nargs="+",
        default=None,
        help="Devices to benchmark. Defaults to all available (cpu + cuda/mps).",
    )
    p.add_argument("--particles", type=int, default=8)
    p.add_argument("--steps", type=int, default=12, help="Total filter steps per run.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-batches", type=int, default=2)
    p.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Kernel noise scale (isotropic). Default: the paper value for --profile (0.3 / 0.5).",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Guide tempering. Default: the paper value for --profile (0.25 / 1.0).",
    )
    p.add_argument("--ess-threshold", type=float, default=0.3, help="Resample when ESS < threshold * S.")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up runs per device (not counted in timing).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the timing records (one per device) to this JSON file for build_timing_table.py.",
    )
    return p.parse_args()


def _available_devices(requested: list[str] | None) -> list[str]:
    if requested is not None:
        return requested
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def _load_batches(data_dir: Path, batch_size: int, n_batches: int) -> list[dict]:
    from trm.nn.data.sudoku_datamodule import SudokuDataModule

    dm = SudokuDataModule(data_dir=str(data_dir), batch_size=batch_size, num_workers=0)
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


def _run_on_device(
    *,
    checkpoint: Path,
    raw_batches: list[dict],
    device: torch.device,
    particles: int,
    total_steps: int,
    sigma: float,
    beta: float,
    ess_threshold: float,
    seed: int,
) -> tuple[float, float]:
    """Return (wall-clock seconds, peak CUDA MB or nan) for one full filter run on device."""
    trm = load_trm_backbone_checkpoint(checkpoint, map_location=device)
    trm.eval().to(device)

    embedder = TRMCoreEmbedder(trm)
    decoder = TRMCoreDecoder(trm)
    q_head = TRMCoreQHead(trm)
    f_theta = TRMNativeCycleFTheta(trm)
    h_cycles = int(getattr(trm.hparams, "H_cycles", 3))

    embedded = []
    for b in tqdm(raw_batches, desc=f"Embedding ({device.type})", leave=False):
        inputs = b["inputs"].to(device)
        puzzle_ids = b["puzzle_identifiers"].to(device)
        x = embedder(inputs, puzzle_ids)
        embedded.append({"x": x, "labels": b["labels"].to(device)})

    stoch_f = StochasticFTheta(f_theta, sigma=sigma, per_step_inner_noise=True)
    inf_cfg = InferenceConfig(
        particles=particles,
        total_steps=total_steps,
        sigma=sigma,
        beta=beta,
        ess_threshold=ess_threshold,
        seed=seed,
    )
    filt = BIGRFilter(stoch_f, q_head, decoder, inf_cfg, h_cycles=h_cycles)
    y0_init = trm.z_H_init.detach().clone()
    z0_init = trm.z_L_init.detach().clone()

    # Synchronise before timing
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        filt.run(embedded, y0_init=y0_init, z0_init=z0_init, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - t0
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else float("nan")
    )
    return elapsed, peak_mb


def main() -> None:
    args = parse_args()
    profile_cfg = PROFILE_DEFAULTS[args.profile]
    if args.checkpoint is None:
        args.checkpoint = profile_cfg["checkpoint"]
    if args.data_dir is None:
        args.data_dir = profile_cfg["data_dir"]
    if args.sigma is None:
        args.sigma = profile_cfg["sigma"]
    if args.beta is None:
        args.beta = profile_cfg["beta"]
    device_names = _available_devices(args.devices)

    print("\nGuided particle inference device benchmark")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  particles  : {args.particles}")
    print(f"  steps      : {args.steps}")
    print(f"  sigma      : {args.sigma}  beta : {args.beta}  tau_ESS : {args.ess_threshold}")
    print(f"  batch_size : {args.batch_size}  n_batches : {args.n_batches}")
    print(f"  devices    : {device_names}")
    print()

    print("Loading data …")
    raw_batches = _load_batches(args.data_dir, args.batch_size, args.n_batches)
    n_puzzles = sum(int(b["inputs"].shape[0]) for b in raw_batches)

    results: dict[str, tuple[float, float]] = {}

    for dev_name in tqdm(device_names, desc="Devices"):
        device = torch.device(dev_name)
        print(f"Benchmarking {dev_name} …", end="", flush=True)

        kw = dict(
            checkpoint=args.checkpoint,
            raw_batches=raw_batches,
            device=device,
            particles=args.particles,
            total_steps=args.steps,
            sigma=args.sigma,
            beta=args.beta,
            ess_threshold=args.ess_threshold,
            seed=args.seed,
        )

        # Warm-up
        for _ in tqdm(range(args.warmup), desc=f"Warm-up ({dev_name})", leave=False):
            _run_on_device(**kw)

        elapsed, peak_mb = _run_on_device(**kw)
        results[dev_name] = (elapsed, peak_mb)
        steps_per_sec = args.steps / elapsed
        mem_str = f", {peak_mb:.0f} MB" if peak_mb == peak_mb else ""
        print(f"  {elapsed:.2f}s  ({steps_per_sec:.1f} steps/s{mem_str})")

    print()
    print("─" * 62)
    print(f"{'Device':<10} {'Time (s)':>10} {'Steps/s':>10} {'Peak MB':>10} {'Speedup vs CPU':>16}")
    print("─" * 62)
    cpu_time = results.get("cpu", (None, None))[0]
    for dev_name, (elapsed, peak_mb) in results.items():
        steps_per_sec = args.steps / elapsed
        if dev_name == "cpu":
            speedup = "1.00×"
        elif cpu_time:
            speedup = f"{cpu_time / elapsed:.2f}×"
        else:
            speedup = "—"  # no CPU run to compare against
        mem_str = f"{peak_mb:>10.0f}" if peak_mb == peak_mb else f"{'—':>10}"
        print(f"{dev_name:<10} {elapsed:>10.2f} {steps_per_sec:>10.1f} {mem_str} {speedup:>16}")
    print("─" * 62)
    print()

    best = min(results, key=lambda k: results[k][0])
    auto = select_device()
    print(f"Fastest device : {best}")
    print(f"select_device(): {auto}")
    if str(auto) != best:
        print(f"  Note: select_device() chose {auto} (CUDA > MPS > CPU priority).")

    if args.json_out is not None:
        records = [
            {
                "device": dev_name,
                "profile": args.profile,
                "particles": args.particles,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "n_batches": len(raw_batches),
                "n_puzzles": n_puzzles,
                "elapsed_s": elapsed,
                "steps_per_s": args.steps / elapsed,
                "peak_mb": peak_mb if peak_mb == peak_mb else None,
                "s_per_puzzle": elapsed / n_puzzles,
            }
            for dev_name, (elapsed, peak_mb) in results.items()
        ]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(records, indent=2) + "\n")
        print(f"Timing records written to {args.json_out}")


if __name__ == "__main__":
    main()
