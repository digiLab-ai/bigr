"""run_pf — guided particle inference (BIGR) over a frozen TRM checkpoint.

Outputs under --out (one run) or --out/fold_XX (k-fold) or --out/seed_XXXX/fold_XX
(seed-fold grid, the paper's headline protocol):

    config.json              all run parameters
    step_metrics.csv         per-step scalar metrics (guided cloud)
    rank_profiles.csv        per-step, per-rank arrays (flattened)
    trm_depth_profile.csv    deterministic TRM baseline (same puzzles)

k-fold and seed-fold runs add *_summary.csv, fold_manifest.json / seed_manifest.json
and summary.json at each level.

Examples
--------
# Quick smoke run (4 particles, 12 steps, 2 batches)
python scripts/exps/run_pf.py --profile sudoku_extreme --smoke

# Paper Table 1 / Figure 1, Sudoku-Extreme test split (5 seeds x 5 folds)
python scripts/exps/run_pf.py --profile sudoku_extreme --split test \\
    --batch-size 64 --n-batches 160 --seeds 31 32 33 34 35 --num-folds 5 --fold-seed 31 \\
    --out assets/runs/sudoku/pf_seed_fold_test

See scripts/run_experiments.sh for every invocation used in the paper.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo bootstrap — works from any working directory
# ---------------------------------------------------------------------------
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
from fold_utils import FoldBatchInfo, build_example_folds, summarize_numeric_df

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PROFILE_DEFAULTS = {
    # Paper Table 4: sigma / beta / tau_ESS per benchmark (S=16, N=48 common).
    "sudoku_extreme": {
        "checkpoint": REPO_ROOT / "data" / "sudoku_extreme" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "sudoku_extreme" / "dataset" / "sudoku_extreme",
        "out": REPO_ROOT / "assets" / "runs" / "sudoku" / "pf",
        "sigma": 0.3,
        "beta": 0.25,
        "ess_threshold": 0.3,
    },
    "maze_hard": {
        "checkpoint": REPO_ROOT / "data" / "maze_hard" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "maze_hard" / "dataset" / "maze-30x30-hard-1k",
        "out": REPO_ROOT / "assets" / "runs" / "maze" / "pf_maze_hard",
        "sigma": 0.5,
        "beta": 1.0,
        "ess_threshold": 0.3,
    },
    "ppbench": {
        "checkpoint": REPO_ROOT / "data" / "ppbench" / "checkpoints" / "last.ckpt",
        "data_dir": REPO_ROOT / "data" / "ppbench",
        "out": REPO_ROOT / "assets" / "runs" / "analysis" / "ablations" / "ppbench" / "pf_ppbench",
        "sigma": 0.3,
        "beta": 0.25,
        "ess_threshold": 0.5,
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Guided particle inference runner",
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
    p.add_argument("--split", default="test", choices=["test", "impossible", "val"])
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default depends on --profile).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Target device (cuda/mps/cpu). Auto-selects CUDA > MPS > CPU if omitted.",
    )
    # Filter config
    p.add_argument("--particles", type=int, default=16)
    p.add_argument("--total-steps", type=int, default=48)
    p.add_argument(
        "--sigma", type=float, default=None,
        help="Noise scale; default is the paper value for --profile (sudoku 0.3, maze 0.5, ppbench 0.3).",
    )
    p.add_argument(
        "--noise-family",
        default="gaussian",
        choices=["gaussian", "laplace", "student_t"],
        help="Proposal noise family (variance-matched; sigma keeps its meaning).",
    )
    p.add_argument(
        "--beta", type=float, default=None,
        help="Inverse guide temperature; default is the paper value for --profile (sudoku 0.25, maze 1.0, ppbench 0.25). 0 = unguided.",
    )
    p.add_argument(
        "--ess-threshold", type=float, default=None,
        help="Resample when ESS < threshold * S; default is the paper value for --profile (0.3; ppbench 0.5).",
    )
    p.add_argument(
        "--inner-depth",
        type=int,
        default=None,
        help="Override inner recursion depth M (L_cycles) at inference; default: checkpoint value.",
    )
    p.add_argument("--seed", type=int, default=31)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Noise-seed grid: with --num-folds k, runs every (seed, fold) pair and writes "
            "<out>/seed_XXXX/fold_XX plus seed-fold summaries (the pf_seed_fold_* layout read by "
            "fig_pf_main.py and the evidence register). Defaults to a single run with --seed."
        ),
    )
    # Data
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-batches", type=int, default=8)
    # Mode
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke check: 2 batches, 4 particles, 12 steps.",
    )
    p.add_argument(
        "--save-particle-scores",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "If set, save per-step particle arrays to DIR: "
            "particle_qlogits.npy (N,T,S), particle_correct.npy (N,T,S), "
            "ess_ratio.npy (N,T), resample_events.npy (N,T)."
        ),
    )
    p.add_argument(
        "--save-particle-trace",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "If set, save original-order particle weights, Q logits, exact flags, "
            "token predictions, MAP predictions, and resampling ancestors to DIR. "
            "This is the replay-oriented export and implies all-step checkpoints."
        ),
    )
    p.add_argument(
        "--checkpoint-all-steps",
        action="store_true",
        help=(
            "Record checkpoint_q_logits / checkpoint_pp_exact at every outer step "
            "rather than only at H-cycle supervision boundaries (default)."
        ),
    )
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
    p.set_defaults(smoke=False, checkpoint_all_steps=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_batches(
    *, data_dir: Path, split: str, batch_size: int, n_batches: int, device: torch.device
):
    """Load Sudoku batches and embed inputs."""
    from trm.nn.data.sudoku_datamodule import SudokuDataModule

    dm = SudokuDataModule(data_dir=str(data_dir), batch_size=batch_size, num_workers=0)
    if split == "test":
        dm.setup(stage="test")
        loader = dm.test_dataloader()
    elif split == "impossible":
        dm.setup(stage="impossible")
        loader = dm.impossible_dataloader()
    else:
        dm.setup(stage="fit")
        loader = dm.val_dataloader()
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


def _numeric_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate numeric columns with mean/std while preserving grouping columns."""
    return summarize_numeric_df(df, group_cols)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def embed_batches(
    batches, embedder: TRMCoreEmbedder, *, device: torch.device
) -> list[dict]:
    """Embed token-id batches into (B, L, D) tensors."""
    embedded = []
    for b in tqdm(batches, desc="Embedding batches"):
        inputs = b["inputs"].to(device)
        puzzle_ids = b["puzzle_identifiers"].to(device)
        x = embedder(inputs, puzzle_ids)
        embedded.append({"x": x, "labels": b["labels"].to(device)})
    return embedded


def run_trm_baseline(
    trm,
    batches,
    *,
    device: torch.device,
    total_steps: int | None = None,
    return_per_example: bool = False,
):
    """Run checkpoint-faithful deterministic TRM rollout for the baseline curve.

    Reproduces the native TRM recurrence exactly:
        for each supervision step:
            for each H cycle:
                for L_cycles inner steps: z_l = lenet(z_l, z_h + x)
                z_h = lenet(z_h, z_l)              # H-level update
                -> record (global_cycle, exact_solved, q_logit)

    Returns ``(df, final_preds)``: the per-step profile (batch means) and the final greedy
    predictions, row-aligned with the loader. With ``return_per_example=True`` a third
    element is appended: a bool array (N_examples, n_cycles) whose column j is the
    exact-solve flag of each example after global cycle j + 1, in loader row order.
    """
    try:
        from trm.nn.utils.constants import IGNORE_LABEL_ID as _IGNORE
    except Exception:
        _IGNORE = -100

    h_cycles = int(getattr(trm.hparams, "H_cycles", 3))
    l_cycles = int(getattr(trm.hparams, "L_cycles", 6))
    n_supervision = int(
        getattr(
            trm.hparams, "N_supervision_val", getattr(trm.hparams, "N_supervision", 16)
        )
    )
    if total_steps is not None:
        n_supervision = -(-int(total_steps) // h_cycles)
    cos_sin = trm.pos_embedding() if hasattr(trm, "pos_embedding") else None
    prefix = getattr(trm, "puzzle_emb_len", 0)

    rows = []
    final_preds_list = []  # deterministic TRM greedy prediction per example (final step)
    per_example_list = []  # (B, n_cycles) exact-solve flags per batch
    for bi, batch in enumerate(tqdm(batches, desc="TRM baseline")):
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        puzzle_ids = batch["puzzle_identifiers"].to(device)

        x = trm._input_embeddings(inputs, puzzle_ids)
        B, L = x.shape[:2]
        dtype = x.dtype

        z_h = trm.z_H_init.detach().clone()
        z_l = trm.z_L_init.detach().clone()
        if z_h.dim() == 1:
            z_h = (
                z_h.view(1, 1, -1)
                .expand(B, L, -1)
                .to(device=device, dtype=dtype)
                .clone()
            )
            z_l = (
                z_l.view(1, 1, -1)
                .expand(B, L, -1)
                .to(device=device, dtype=dtype)
                .clone()
            )

        global_cycle = 0
        exact_cycles = []
        for _sup in range(n_supervision):
            for _cyc in range(h_cycles):
                for _inner in range(l_cycles):
                    z_l = trm.lenet(z_l, z_h + x, cos_sin=cos_sin)
                z_h = trm.lenet(z_h, z_l, cos_sin=cos_sin)
                global_cycle += 1

                logits = trm.lm_head(z_h)
                if prefix > 0:
                    logits = logits[:, prefix:]
                preds = logits.argmax(-1)
                mask = labels != _IGNORE
                exact = torch.logical_or(~mask, preds == labels).all(-1).float()
                q_logit = trm.q_head(z_h[:, 0]).to(torch.float32)[..., 0]

                rows.append(
                    {
                        "batch_idx": bi,
                        "global_cycle": global_cycle,
                        "exact_solved": float(exact.mean().item()),
                        "q_logit": float(q_logit.mean().item()),
                    }
                )
                if return_per_example:
                    exact_cycles.append(exact.bool().cpu())
        # `preds` now holds the final-step (last supervision cycle) greedy output.
        final_preds_list.append(preds.detach().cpu().numpy())
        if return_per_example:
            per_example_list.append(torch.stack(exact_cycles, dim=1).numpy())

    df = (
        pd.DataFrame(rows)
        .groupby("global_cycle")
        .agg(exact_solved=("exact_solved", "mean"), mean_q_logit=("q_logit", "mean"))
        .reset_index()
        .rename(columns={"global_cycle": "step"})
    )
    final_preds = (
        np.concatenate(final_preds_list, axis=0)
        if final_preds_list
        else np.empty((0, 0), dtype=np.int64)
    )
    if not return_per_example:
        return df, final_preds
    per_example = (
        np.concatenate(per_example_list, axis=0)
        if per_example_list
        else np.empty((0, 0), dtype=bool)
    )
    return df, final_preds, per_example


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _save_particle_scores(trace, out_dir: Path, S: int, T: int, n_batches: int) -> None:
    """Persist per-step, per-particle arrays for guide-margin analysis.

    Saves:
        particle_qlogits.npy   (N_puzzles, T, S)  — Q-logit per particle
        particle_correct.npy   (N_puzzles, T, S)  — bool: particle exactly correct
        ess_ratio.npy          (N_puzzles, T)      — ESS/S at each step
        resample_events.npy    (N_puzzles, T)      — 1 if resampling fired, else 0

    Uses the step_data list stored in the trace (one entry per step, with (B,)
    scalar fields). Per-particle data comes from checkpoint_q_logits and
    checkpoint_pp_exact which are snapshotted at every step when the filter
    stores checkpoint_steps = range(1, T+1).
    """
    # Use the original-particle-order checkpoints (checkpoint_q_raw / checkpoint_weights_raw)
    # when the filter recorded them; otherwise the Q-rank-sorted checkpoints.
    q_source = trace.checkpoint_q_raw or trace.checkpoint_q_logits
    weight_source = trace.checkpoint_weights_raw or trace.checkpoint_weights
    steps_available = sorted(q_source.keys())
    if not steps_available:
        print(
            "  [save-particle-scores] No checkpoint data available; "
            "re-run with checkpoint_steps covering all steps."
        )
        return

    # Determine N (total puzzles across batches * batch_size implicitly)
    # checkpoint arrays are (B, S) per step; we need to stack over batches.
    # The trace stores (B, S) per-step checkpoint arrays accumulated per batch,
    # so here they're (N_total, S) already if the filter concatenated them.
    # Peek at the first step to determine shapes.
    sample = q_source[steps_available[0]]
    N_total = sample.shape[0]

    q_arr = np.full((N_total, T, S), np.nan, dtype=np.float32)
    weight_arr = np.full((N_total, T, S), np.nan, dtype=np.float32)
    correct_arr = np.zeros((N_total, T, S), dtype=bool)
    ess_arr = np.full((N_total, T), np.nan, dtype=np.float32)
    resample_arr = np.zeros((N_total, T), dtype=np.uint8)

    for step in steps_available:
        t_idx = step - 1  # 0-indexed
        if t_idx >= T:
            continue
        q_arr[:, t_idx, :] = q_source[step]  # (N, S)
        if step in weight_source:
            weight_arr[:, t_idx, :] = weight_source[step]  # (N, S)
        if step in trace.checkpoint_pp_exact:
            correct_arr[:, t_idx, :] = trace.checkpoint_pp_exact[step].astype(bool)

    # ESS and resample flags come from step_records
    for rec in trace.step_records:
        t_idx = rec.step - 1
        if t_idx >= T:
            continue
        # mean_ess is already ESS/S averaged over batch; broadcast to N
        ess_arr[:, t_idx] = rec.mean_ess
        resample_arr[:, t_idx] = int(rec.resample_fraction > 0)

    np.save(out_dir / "particle_qlogits.npy", q_arr)
    np.save(out_dir / "particle_weights.npy", weight_arr)
    np.save(out_dir / "particle_correct.npy", correct_arr)
    np.save(out_dir / "ess_ratio.npy", ess_arr)
    np.save(out_dir / "resample_events.npy", resample_arr)
    if trace.checkpoint_ancestors:
        ancestor_arr = np.full((N_total, T, S), -1, dtype=np.int16)
        for step, arr in trace.checkpoint_ancestors.items():
            t_idx = step - 1
            if t_idx < T:
                ancestor_arr[:, t_idx, :] = arr.astype(np.int16, copy=False)
        np.save(out_dir / "particle_ancestors.npy", ancestor_arr)
    print(
        f"  Saved particle score arrays to {out_dir}  shape=(N={N_total}, T={T}, S={S})"
    )


def _save_particle_trace(trace, out_dir: Path, S: int, T: int) -> None:
    """Persist replay-oriented particle arrays in original particle order.

    ``particle_ancestors[:, step - 1, child]`` gives the parent particle index
    at checkpoint ``step`` used to initialise ``child`` at the next step after
    any post-step resampling. Compose ancestors across steps to follow
    descendants from an earlier checkpoint to a later checkpoint.
    """
    if not trace.checkpoint_q_raw or not trace.checkpoint_weights_raw:
        print(
            "  [save-particle-trace] Original-order checkpoints are unavailable; "
            "set InferenceConfig.save_particle_details=True."
        )
        return

    steps_available = sorted(trace.checkpoint_q_raw.keys())
    sample = trace.checkpoint_q_raw[steps_available[0]]
    N_total = sample.shape[0]

    q_arr = np.full((N_total, T, S), np.nan, dtype=np.float32)
    weight_arr = np.full((N_total, T, S), np.nan, dtype=np.float32)
    correct_arr = np.zeros((N_total, T, S), dtype=bool)
    ancestor_arr = np.full((N_total, T, S), -1, dtype=np.int16)

    preds_arr = None
    if trace.checkpoint_preds:
        pred_sample = next(iter(trace.checkpoint_preds.values()))
        pred_len = int(pred_sample.shape[-1])
        preds_arr = np.full((N_total, T, S, pred_len), -1, dtype=np.int16)

    map_preds_arr = None
    if trace.checkpoint_map_preds:
        map_sample = next(iter(trace.checkpoint_map_preds.values()))
        map_len = int(map_sample.shape[-1])
        map_preds_arr = np.full((N_total, T, map_len), -1, dtype=np.int16)

    for step in steps_available:
        t_idx = step - 1
        if t_idx >= T:
            continue
        q_arr[:, t_idx, :] = trace.checkpoint_q_raw[step]
        weight_arr[:, t_idx, :] = trace.checkpoint_weights_raw[step]
        if step in trace.checkpoint_pp_exact:
            correct_arr[:, t_idx, :] = trace.checkpoint_pp_exact[step].astype(bool)
        if preds_arr is not None and step in trace.checkpoint_preds:
            preds_arr[:, t_idx, :, :] = trace.checkpoint_preds[step].astype(
                np.int16, copy=False
            )
        if map_preds_arr is not None and step in trace.checkpoint_map_preds:
            map_preds_arr[:, t_idx, :] = trace.checkpoint_map_preds[step].astype(
                np.int16, copy=False
            )

    for step, arr in trace.checkpoint_ancestors.items():
        t_idx = step - 1
        if t_idx < T:
            ancestor_arr[:, t_idx, :] = arr.astype(np.int16, copy=False)

    np.save(out_dir / "particle_qlogits.npy", q_arr)
    np.save(out_dir / "particle_weights.npy", weight_arr)
    np.save(out_dir / "particle_correct.npy", correct_arr)
    np.save(out_dir / "particle_ancestors.npy", ancestor_arr)
    if preds_arr is not None:
        np.save(out_dir / "particle_preds.npy", preds_arr)
    if map_preds_arr is not None:
        np.save(out_dir / "map_preds.npy", map_preds_arr)

    metadata = {
        "shape": {"num_examples": int(N_total), "total_steps": int(T), "particles": int(S)},
        "particle_order": "original",
        "ancestor_semantics": (
            "particle_ancestors[:, step - 1, child] gives the parent particle "
            "index at checkpoint step used to initialise child at the next "
            "step after any post-step resampling; identity rows mean no "
            "resampling fired for that example."
        ),
    }
    _write_json(out_dir / "particle_trace_manifest.json", metadata)
    print(
        f"  Saved particle trace arrays to {out_dir}  shape=(N={N_total}, T={T}, S={S})"
    )


def _run_pf_single(
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
    save_particle_scores_dir: Path | None,
    save_particle_trace_dir: Path | None,
    fold_meta: dict | None = None,
) -> dict:
    """Run one PF evaluation and persist artefacts into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_particle_scores_dir is not None:
        save_particle_scores_dir.mkdir(parents=True, exist_ok=True)
    if save_particle_trace_dir is not None:
        save_particle_trace_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    cfg_dict.update(
        {
            "out": str(out_dir),
            "save_particle_scores": (
                str(save_particle_scores_dir)
                if save_particle_scores_dir is not None
                else None
            ),
            "save_particle_trace": (
                str(save_particle_trace_dir)
                if save_particle_trace_dir is not None
                else None
            ),
        }
    )
    if fold_meta is not None:
        cfg_dict.update(fold_meta)
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    embedded = embed_batches(raw_batches, embedder, device=device)

    if getattr(args, "inner_depth", None) is not None:
        trm.hparams.L_cycles = int(args.inner_depth)
    h_cycles = int(getattr(trm.hparams, "H_cycles", 3))
    stoch_f = StochasticFTheta(
        f_theta,
        sigma=args.sigma,
        per_step_inner_noise=True,
        noise_family=getattr(args, "noise_family", "gaussian"),
    )
    save_particle_details = (
        save_particle_scores_dir is not None or save_particle_trace_dir is not None
    )
    ckpt_steps = (
        list(range(1, args.total_steps + 1))
        if args.checkpoint_all_steps or save_particle_details
        else None
    )
    inf_cfg = InferenceConfig(
        particles=args.particles,
        total_steps=args.total_steps,
        sigma=args.sigma,
        beta=args.beta,
        ess_threshold=args.ess_threshold,
        seed=args.seed,
        checkpoint_steps=ckpt_steps,
        save_particle_details=save_particle_details,
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

    if save_particle_scores_dir is not None:
        _save_particle_scores(
            trace, save_particle_scores_dir, args.particles, args.total_steps, len(embedded)
        )
    if save_particle_trace_dir is not None:
        _save_particle_trace(
            trace, save_particle_trace_dir, args.particles, args.total_steps
        )
        # The input token sequences, row-aligned with the trace, so that evaluators
        # (eval_ppbench.py) can match predictions to dataset entries by input content
        # rather than by row order, which k-fold shuffling changes.
        inputs_all = np.concatenate(
            [np.asarray(b["inputs"].detach().cpu()) for b in raw_batches]
        )
        np.save(Path(save_particle_trace_dir) / "inputs.npy", inputs_all)

    rows = []
    for r in trace.step_records:
        rows.append(
            {
                "step": r.step,
                "exact_accuracy": r.exact_accuracy,
                "any_exact_rate": r.any_exact_rate,
                "mean_ess": r.mean_ess,
                "weight_entropy": r.weight_entropy,
                "q_diversity": r.q_diversity,
                "y_diversity": float(np.mean(r.y_diversity)),
                "z_diversity": float(np.mean(r.z_diversity)),
                "resample_fraction": r.resample_fraction,
            }
        )
    step_df = pd.DataFrame(rows)
    step_df.to_csv(out_dir / "step_metrics.csv", index=False)
    print(f"Saved step_metrics.csv ({len(step_df)} rows)")

    ra = trace.rank_arrays
    S = inf_cfg.particles
    rank_rows = []
    for step_i, step in enumerate(tqdm(ra["steps"], desc="Building rank profiles")):
        for rank_j in range(S):
            rank_rows.append(
                {
                    "step": int(step),
                    "rank": rank_j + 1,
                    "q_logit": float(ra["q_by_rank"][step_i, rank_j]),
                    "q_logit_std": float(ra["q_std_by_rank"][step_i, rank_j]),
                    "weight": float(ra["w_by_rank"][step_i, rank_j]),
                    "weight_std": float(ra["w_std_by_rank"][step_i, rank_j]),
                    "exact_rate": float(ra["exact_by_rank"][step_i, rank_j]),
                }
            )
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(out_dir / "rank_profiles.csv", index=False)
    print(f"Saved rank_profiles.csv ({len(rank_df)} rows)")

    for step, arr in trace.checkpoint_weights.items():
        np.save(out_dir / f"ckpt_weights_step{step}.npy", arr)
    for step, arr in trace.checkpoint_q_logits.items():
        np.save(out_dir / f"ckpt_qlogits_step{step}.npy", arr)

    trm_path = out_dir / "trm_depth_profile.csv"
    trm_preds_path = (
        Path(save_particle_trace_dir) / "trm_preds.npy"
        if save_particle_trace_dir is not None
        else None
    )
    # The deterministic baseline is computed when its CSV is missing, or when a particle
    # trace was requested and the per-example baseline predictions (trm_preds.npy) are not
    # yet in the trace directory.
    need_trm = (not trm_path.exists()) or (
        trm_preds_path is not None and not trm_preds_path.exists()
    )
    if need_trm:
        print("Running TRM baseline …")
        with torch.no_grad():
            trm_df, trm_preds = run_trm_baseline(
                trm, raw_batches, device=device, total_steps=args.total_steps
            )
        trm_df.to_csv(trm_path, index=False)
        print(f"Saved trm_depth_profile.csv ({len(trm_df)} rows)")
        # The deterministic TRM's final predictions, row-aligned with inputs.npy, so that
        # eval_ppbench.py can score the baseline on the same puzzles.
        if trm_preds_path is not None:
            np.save(trm_preds_path, trm_preds)
    else:
        print(f"TRM baseline already present at {trm_path}")

    final = trace.step_records[-1]
    print(
        f"\nFinal step {final.step}: exact_accuracy={final.exact_accuracy:.4f}, "
        f"any_exact_rate={final.any_exact_rate:.4f}, mean_ess={final.mean_ess:.3f}"
    )
    print(f"Artefacts saved to: {out_dir}")

    return {
        "step_df": step_df,
        "rank_df": rank_df,
        "final": {
            "step": int(final.step),
            "exact_accuracy": float(final.exact_accuracy),
            "any_exact_rate": float(final.any_exact_rate),
            "mean_ess": float(final.mean_ess),
            "weight_entropy": float(final.weight_entropy),
        },
        "num_examples": int(sum(batch["inputs"].shape[0] for batch in raw_batches)),
        "num_batches": int(len(raw_batches)),
    }


def _summarize_fold_outputs(
    fold_outputs: list[dict], out_dir: Path, fold_infos: list[FoldBatchInfo], args: argparse.Namespace
) -> None:
    """Write parent-level summaries for fold mode."""
    step_frames = [entry["step_df"] for entry in fold_outputs]
    rank_frames = [entry["rank_df"] for entry in fold_outputs]

    step_summary = _numeric_summary(pd.concat(step_frames, ignore_index=True), ["step"])
    rank_summary = _numeric_summary(
        pd.concat(rank_frames, ignore_index=True), ["step", "rank"]
    )

    step_summary.to_csv(out_dir / "step_metrics_summary.csv", index=False)
    rank_summary.to_csv(out_dir / "rank_profiles_summary.csv", index=False)

    summary = {
        "split": args.split,
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed if args.fold_seed is not None else args.seed,
        "total_examples": int(sum(info.num_examples for info in fold_infos)),
        "total_batches": int(sum(info.num_batches for info in fold_infos)),
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
        "final_step_summary": {
            "step": int(step_summary["step"].iloc[-1]),
            "exact_accuracy_mean": float(step_summary["exact_accuracy_mean"].iloc[-1]),
            "exact_accuracy_std": float(step_summary["exact_accuracy_std"].iloc[-1]),
            "any_exact_rate_mean": float(step_summary["any_exact_rate_mean"].iloc[-1]),
            "any_exact_rate_std": float(step_summary["any_exact_rate_std"].iloc[-1]),
            "mean_ess_mean": float(step_summary["mean_ess_mean"].iloc[-1]),
            "mean_ess_std": float(step_summary["mean_ess_std"].iloc[-1]),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "fold_manifest.json", summary)

    fold_summary = {
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed if args.fold_seed is not None else args.seed,
        "step_metrics_summary": "step_metrics_summary.csv",
        "rank_profiles_summary": "rank_profiles_summary.csv",
    }
    _write_json(out_dir / "summary_manifest.json", fold_summary)


def _run_fold_loop(
    *,
    args: argparse.Namespace,
    trm,
    device: torch.device,
    embedder: TRMCoreEmbedder,
    decoder: TRMCoreDecoder,
    q_head: TRMCoreQHead,
    f_theta: TRMNativeCycleFTheta,
    fold_batches: list[list[dict]],
    fold_infos: list[FoldBatchInfo],
    out_dir: Path,
    seed_meta: dict | None = None,
    save_particle_scores_root: Path | None = None,
    save_particle_trace_root: Path | None = None,
) -> list[dict]:
    """Run the per-fold loop into ``out_dir`` and return per-fold outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_outputs: list[dict] = []
    for fold_idx, (fold_raw_batches, fold_info) in enumerate(
        zip(fold_batches, fold_infos), start=1
    ):
        fold_name = f"fold_{fold_idx:02d}"
        fold_out = out_dir / fold_name
        fold_ps_dir = (
            save_particle_scores_root / fold_name
            if save_particle_scores_root is not None
            else None
        )
        fold_ptrace_dir = (
            save_particle_trace_root / fold_name
            if save_particle_trace_root is not None
            else None
        )
        print(
            f"\n[{fold_name}] examples={fold_info.num_examples} batches={fold_info.num_batches}"
        )
        fold_meta = {
            "fold_mode": True,
            "fold_index": fold_idx,
            "fold_name": fold_name,
            "num_folds": args.num_folds,
            "fold_seed": args.fold_seed,
            "fold_num_examples": fold_info.num_examples,
            "fold_num_batches": fold_info.num_batches,
        }
        if seed_meta is not None:
            fold_meta.update(seed_meta)
        fold_outputs.append(
            _run_pf_single(
                args=args,
                trm=trm,
                device=device,
                embedder=embedder,
                decoder=decoder,
                q_head=q_head,
                f_theta=f_theta,
                raw_batches=fold_raw_batches,
                out_dir=fold_out,
                save_particle_scores_dir=fold_ps_dir,
                save_particle_trace_dir=fold_ptrace_dir,
                fold_meta=fold_meta,
            )
        )
    return fold_outputs


def _fold_manifest(fold_infos: list[FoldBatchInfo], args: argparse.Namespace) -> dict:
    return {
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
    }


def _summarize_seed_grid(
    *,
    out_dir: Path,
    seeds: list[int],
    seed_dirs: list[Path],
    fold_infos: list[FoldBatchInfo],
    args: argparse.Namespace,
) -> None:
    """Aggregate per-(seed, fold) leaves into a top-level seed-fold summary.

    Writes ``step_metrics_summary.csv``, ``rank_profiles_summary.csv``,
    ``seed_manifest.json``, and ``summary.json`` matching the cached
    ``pf_seed_fold_*`` layout that ``fig_pf_main.py`` reads.
    """
    step_frames: list[pd.DataFrame] = []
    rank_frames: list[pd.DataFrame] = []
    for seed, seed_dir in zip(seeds, seed_dirs):
        for fold_idx in range(1, args.num_folds + 1):
            leaf = seed_dir / f"fold_{fold_idx:02d}"
            step_frames.append(pd.read_csv(leaf / "step_metrics.csv"))
            rank_frames.append(pd.read_csv(leaf / "rank_profiles.csv"))

    step_summary = _numeric_summary(
        pd.concat(step_frames, ignore_index=True), ["step"]
    )
    rank_summary = _numeric_summary(
        pd.concat(rank_frames, ignore_index=True), ["step", "rank"]
    )
    step_summary.to_csv(out_dir / "step_metrics_summary.csv", index=False)
    rank_summary.to_csv(out_dir / "rank_profiles_summary.csv", index=False)

    seed_entries = [
        {
            "seed": int(seed),
            "path": seed_dir.name,
            "num_folds": args.num_folds,
            "num_runs": args.num_folds,
        }
        for seed, seed_dir in zip(seeds, seed_dirs)
    ]
    summary_payload = {
        "split": args.split,
        "num_seeds": len(seeds),
        "seeds": [int(s) for s in seeds],
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed,
        "num_runs": len(seeds) * args.num_folds,
        "seed_entries": seed_entries,
        "outputs": {
            "step_metrics_summary": "step_metrics_summary.csv",
            "rank_profiles_summary": "rank_profiles_summary.csv",
        },
        "final_step_summary": {
            "step": int(step_summary["step"].iloc[-1]),
            "exact_accuracy_mean": float(step_summary["exact_accuracy_mean"].iloc[-1]),
            "exact_accuracy_std": float(step_summary["exact_accuracy_std"].iloc[-1]),
            "any_exact_rate_mean": float(step_summary["any_exact_rate_mean"].iloc[-1]),
            "any_exact_rate_std": float(step_summary["any_exact_rate_std"].iloc[-1]),
            "mean_ess_mean": float(step_summary["mean_ess_mean"].iloc[-1]),
            "mean_ess_std": float(step_summary["mean_ess_std"].iloc[-1]),
        },
    }
    _write_json(out_dir / "seed_manifest.json", summary_payload)
    _write_json(out_dir / "summary.json", summary_payload)



def main() -> None:
    args = parse_args()
    profile_cfg = PROFILE_DEFAULTS[args.profile]

    if args.checkpoint is None:
        args.checkpoint = profile_cfg["checkpoint"]
    if args.data_dir is None:
        args.data_dir = profile_cfg["data_dir"]
    if args.out is None:
        args.out = profile_cfg["out"]
    if args.sigma is None:
        args.sigma = profile_cfg["sigma"]
    if args.beta is None:
        args.beta = profile_cfg["beta"]
    if args.ess_threshold is None:
        args.ess_threshold = profile_cfg["ess_threshold"]

    if args.smoke:
        args.particles = 4
        args.total_steps = 12
        args.batch_size = 4
        args.n_batches = 2
        print(
            "Smoke mode: particles=4, steps=12, batches=2 "
            f"(profile={args.profile})"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else select_device()
    args.fold_seed = args.seed if args.fold_seed is None else args.fold_seed
    if args.num_folds < 1:
        raise ValueError(f"--num-folds must be >= 1, got {args.num_folds}")

    seeds = (
        list(dict.fromkeys(int(s) for s in args.seeds))
        if args.seeds
        else [int(args.seed)]
    )
    seed_grid_mode = len(seeds) > 1
    if seed_grid_mode and args.num_folds < 2:
        raise ValueError(
            "--seeds with multiple values requires --num-folds >= 2 "
            "(the paper's seed-fold runs are 5 seeds x 5 folds)."
        )

    # Save config
    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    print(f"Loading checkpoint: {args.checkpoint}")
    trm = load_trm_backbone_checkpoint(
        args.checkpoint, map_location=device, use_ema=True
    )
    trm.eval().to(device)

    h_cycles = int(getattr(trm.hparams, "H_cycles", 3))
    print(f"H_cycles={h_cycles}")

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

    if not seed_grid_mode and args.num_folds == 1:
        ps_dir = args.save_particle_scores
        if ps_dir is not None:
            ps_dir.mkdir(parents=True, exist_ok=True)
        ptrace_dir = args.save_particle_trace
        if ptrace_dir is not None:
            ptrace_dir.mkdir(parents=True, exist_ok=True)
        args.seed = seeds[0]
        _run_pf_single(
            args=args,
            trm=trm,
            device=device,
            embedder=embedder,
            decoder=decoder,
            q_head=q_head,
            f_theta=f_theta,
            raw_batches=raw_batches,
            out_dir=args.out,
            save_particle_scores_dir=ps_dir,
            save_particle_trace_dir=ptrace_dir,
        )
        return

    fold_batches, fold_infos = build_example_folds(
        raw_batches,
        num_folds=args.num_folds,
        seed=args.fold_seed,
        batch_size=args.batch_size,
    )

    if not seed_grid_mode:
        args.seed = seeds[0]
        print(
            f"Running {args.num_folds}-fold mode with seed={args.seed}, "
            f"fold_seed={args.fold_seed} over "
            f"{sum(info.num_examples for info in fold_infos)} examples"
        )
        ps_root = (
            Path(args.save_particle_scores)
            if args.save_particle_scores is not None
            else None
        )
        ptrace_root = (
            Path(args.save_particle_trace)
            if args.save_particle_trace is not None
            else None
        )
        fold_outputs = _run_fold_loop(
            args=args,
            trm=trm,
            device=device,
            embedder=embedder,
            decoder=decoder,
            q_head=q_head,
            f_theta=f_theta,
            fold_batches=fold_batches,
            fold_infos=fold_infos,
            out_dir=args.out,
            save_particle_scores_root=ps_root,
            save_particle_trace_root=ptrace_root,
        )
        _summarize_fold_outputs(fold_outputs, args.out, fold_infos, args)
        for root in (ps_root, ptrace_root):
            if root is not None:
                _write_json(root / "fold_manifest.json", _fold_manifest(fold_infos, args))
        print(f"Fold summaries saved to: {args.out}")
        return

    # Seed-grid mode: outer loop over noise seeds, fixed fold partition.
    print(
        f"Running seed-grid mode: {len(seeds)} seeds × {args.num_folds} folds = "
        f"{len(seeds) * args.num_folds} runs (fold_seed={args.fold_seed}) over "
        f"{sum(info.num_examples for info in fold_infos)} examples"
    )

    seed_dirs: list[Path] = []
    baseline_source: Path | None = None
    for seed in seeds:
        seed_dir = args.out / f"seed_{seed:04d}"
        seed_dirs.append(seed_dir)
        seed_dir.mkdir(parents=True, exist_ok=True)
        # Reuse the deterministic TRM baseline across seeds (it depends only on
        # the data partition, not the noise seed). Pre-place it so the
        # exists() check in _run_pf_single skips re-running it.
        if baseline_source is not None:
            for fold_idx in range(1, args.num_folds + 1):
                src = baseline_source / f"fold_{fold_idx:02d}" / "trm_depth_profile.csv"
                if not src.exists():
                    continue
                dst_dir = seed_dir / f"fold_{fold_idx:02d}"
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / "trm_depth_profile.csv"
                if not dst.exists():
                    shutil.copy2(src, dst)
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        seed_args.out = seed_dir
        ps_root = (
            Path(args.save_particle_scores) / f"seed_{seed:04d}"
            if args.save_particle_scores is not None
            else None
        )
        ptrace_root = (
            Path(args.save_particle_trace) / f"seed_{seed:04d}"
            if args.save_particle_trace is not None
            else None
        )
        seed_meta = {
            "seed_grid_mode": True,
            "seed_value": int(seed),
            "seeds": [int(s) for s in seeds],
        }
        print(f"\n=== seed {seed} ({seeds.index(seed) + 1}/{len(seeds)}) ===")
        fold_outputs = _run_fold_loop(
            args=seed_args,
            trm=trm,
            device=device,
            embedder=embedder,
            decoder=decoder,
            q_head=q_head,
            f_theta=f_theta,
            fold_batches=fold_batches,
            fold_infos=fold_infos,
            out_dir=seed_dir,
            seed_meta=seed_meta,
            save_particle_scores_root=ps_root,
            save_particle_trace_root=ptrace_root,
        )
        _summarize_fold_outputs(fold_outputs, seed_dir, fold_infos, seed_args)
        for root in (ps_root, ptrace_root):
            if root is not None:
                _write_json(root / "fold_manifest.json", _fold_manifest(fold_infos, args))
        if baseline_source is None:
            baseline_source = seed_dir

    _summarize_seed_grid(
        out_dir=args.out,
        seeds=seeds,
        seed_dirs=seed_dirs,
        fold_infos=fold_infos,
        args=args,
    )
    print(f"Seed-grid summaries saved to: {args.out}")


if __name__ == "__main__":
    main()
