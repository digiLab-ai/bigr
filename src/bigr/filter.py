"""`BIGRFilter`: the guided particle filter over hypothesis states.

One outer step (Algorithm 1 in the paper):

    1. Propose, for each particle s:
           z' = stoch_f.belief_update(x, y_s, z_s, generator=gen)
           y' = stoch_f.state_update(y_s, z', generator=gen)
    2. Tilt by the Q-head: log_w += beta * logsigmoid(Q(y'))
    3. Record per-step diagnostics; the answer is the highest-weight particle.
    4. Systematic resampling when ESS < ess_threshold * S, after which the
       weights of the resampled batch elements are reset to uniform.

`BIGRFilter.run()` iterates this for N steps and aggregates over batches.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from bigr.backbone import TRMCoreDecoder, TRMCoreQHead
from bigr.config import InferenceConfig
from bigr.resampling import (
    broadcast_initial_state,
    effective_sample_size,
    make_generator,
    normalised_weights,
    systematic_resample,
)
from bigr.stochastic import StochasticFTheta
from bigr.types import FilterTrace, StepRecord

try:
    from trm.nn.utils.constants import IGNORE_LABEL_ID as _IGNORE
except Exception:
    _IGNORE = -100


def _exact_solved(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Boolean exact-solve mask, shape (B,).

    A prediction is exact if every non-ignored token matches the label.
    """
    mask = labels != _IGNORE
    return torch.logical_or(~mask, pred == labels).all(-1).float()


def _decode_map_particle(
    decoder: TRMCoreDecoder,
    y_particles: torch.Tensor,
    log_weights: torch.Tensor,
) -> torch.Tensor:
    """Decode the highest-weight particle for each batch element.

    This is a particle-level MAP estimate over hypothesis identity. It differs
    from predictive-mixture decoding, which first averages token
    distributions across particles and then takes a tokenwise argmax.
    """
    B, _, L, D = y_particles.shape
    top_idx = log_weights.argmax(dim=-1)
    gather_idx = top_idx.view(B, 1, 1, 1).expand(B, 1, L, D)
    y_map = y_particles.gather(1, gather_idx).squeeze(1)
    return decoder(y_map).argmax(-1)


# ---------------------------------------------------------------------------
# Single-batch runner
# ---------------------------------------------------------------------------


def _run_one_batch(
    stoch_f: StochasticFTheta,
    q_head: TRMCoreQHead,
    decoder: TRMCoreDecoder,
    *,
    x: torch.Tensor,
    y0: torch.Tensor,
    z0_init: torch.Tensor,
    labels: torch.Tensor | None,
    config: InferenceConfig,
    checkpoint_steps: set[int],
    seed: int,
    device: torch.device,
) -> tuple[
    list[dict],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    """Run the particle filter for one batch.

    Parameters
    ----------
    x:         (B, L, D) embedded input — fixed.
    y0:        (B, L, D) initial hypothesis state broadcast from checkpoint.
    z0_init:   (B, L, D) initial latent state from checkpoint parameter.
    labels:    (B, L) ground-truth tokens or None (disables accuracy metrics).
    config:    InferenceConfig.
    checkpoint_steps: Set of step indices at which to snapshot (B, S) arrays.
    seed:      Per-batch random seed.
    device:    Target device.

    Returns
    -------
    step_data:    List of per-step dicts (scalars + (S,) tensors).
    ckpt_weights: {step: (B, S) normalised weights sorted by Q-rank}.
    ckpt_q:       {step: (B, S) Q logits sorted by Q-rank}.
    ckpt_weights_raw: {step: (B, S) normalised weights in original particle order}
        if save_particle_details.
    ckpt_ancestors: {step: (B, S) parent particle indices for the post-step
        resampling transition} if save_particle_details.
    ckpt_y:       {step: (B, S, L, D) float16 hypothesis states} if save_hidden_states.
    ckpt_z:       {step: (B, S, L, D) float16 latent states} if save_hidden_states.
    ckpt_q_raw:   {step: (B, S) float32 Q logits, original particle order} if
        save_hidden_states or save_particle_details.
    """
    gen = make_generator(seed)
    B, L, D = x.shape
    S = config.particles
    collect_step_metrics = config.collect_step_cpu_metrics

    nan_scalar = torch.full((), float("nan"), device=device, dtype=torch.float32)
    nan_rank = torch.full((S,), float("nan"), device=device, dtype=torch.float32)
    nan_batch = torch.full((B,), float("nan"), device=device, dtype=torch.float32)

    def _flatten_batch_particles(t: torch.Tensor) -> torch.Tensor:
        """Flatten (B, S, ...) -> (B*S, ...) in batch-major order."""
        return t.reshape(B * S, *t.shape[2:])

    # Initialise particle population: y broadcast to (B, S, L, D)
    y = y0.unsqueeze(1).expand(B, S, L, D).clone()

    log_w = torch.zeros(B, S, device=device, dtype=torch.float32)

    step_data: list[dict] = []
    ckpt_weights: dict[int, torch.Tensor] = {}
    ckpt_q: dict[int, torch.Tensor] = {}
    ckpt_weights_raw: dict[int, torch.Tensor] = {}  # (B, S) original particle order
    ckpt_ancestors: dict[int, torch.Tensor] = {}    # (B, S) post-step resampling parents
    ckpt_pp_exact: dict[int, torch.Tensor] = {}  # (B, S) per-particle exact-solve
    ckpt_preds: dict[int, torch.Tensor] = {}  # (B, S, L) per-particle predictions
    ckpt_map_preds: dict[int, torch.Tensor] = {}  # (B, L) MAP-particle predictions
    ckpt_y: dict[int, torch.Tensor] = {}      # (B, S, L, D) float16 — populated when save_hidden_states
    ckpt_z: dict[int, torch.Tensor] = {}      # (B, S, L, D) float16
    ckpt_q_raw: dict[int, torch.Tensor] = {}  # (B, S) float32 unsorted Q logits
    save_particle_details = config.save_particle_details

    # Persistent latent state per particle: (B, S, L, D).
    z = z0_init.unsqueeze(1).expand(B, S, L, D).clone()
    x_particles = x.unsqueeze(1).expand(B, S, L, D).reshape(B * S, L, D)

    for step_idx in tqdm(range(config.total_steps), desc="Filter steps", leave=False):
        y_flat = y.reshape(B * S, L, D)
        z_flat = z.reshape(B * S, L, D)
        z_next_flat = stoch_f.belief_update(
            x_particles, y_flat, z_flat, generator=gen
        )
        y_next_flat = stoch_f.state_update(y_flat, z_next_flat, generator=gen)
        z = z_next_flat.reshape(B, S, L, D)
        y = y_next_flat.reshape(B, S, L, D)

        # Q-head scoring: raw logit then (tempered) log-sigmoid potential
        q_logit_raw = q_head(y.reshape(B * S, L, D)).reshape(B, S)   # (B, S) raw
        q_scores = config.beta * F.logsigmoid(q_logit_raw)     # (B, S) tempered
        log_w = log_w + q_scores

        # Normalised weights and ESS
        w_norm = normalised_weights(log_w)           # (B, S)
        ess_val = effective_sample_size(log_w)        # (B,)

        # Resampling decision computed early so it can be recorded alongside metrics
        do_rs = ess_val < config.ess_threshold * S            # (B,) bool
        step = step_idx + 1
        need_sorted = collect_step_metrics or step in checkpoint_steps
        if need_sorted:
            # Sort particles by Q-logit descending: rank 1 = highest Q score
            q_order = q_logit_raw.argsort(dim=-1, descending=True)  # (B, S)
            q_sorted = q_logit_raw.gather(1, q_order)               # (B, S)
            w_sorted = w_norm.gather(1, q_order)                    # (B, S)
        else:
            q_order = None
            q_sorted = None
            w_sorted = None

        if collect_step_metrics:
            if S > 1:
                resample_fraction = do_rs.float().mean()
                log_w_n = torch.log(w_norm.clamp(min=1e-10))
                raw_entropy = -(w_norm * log_w_n).sum(-1).mean()
                weight_entropy = raw_entropy / float(np.log(S))
            else:
                resample_fraction = torch.tensor(
                    0.0, device=device, dtype=torch.float32
                )
                weight_entropy = torch.tensor(1.0, device=device, dtype=torch.float32)
            # Diversity metrics: mean std across particles, averaged over all other axes.
            q_diversity = q_logit_raw.std(dim=-1, unbiased=False).mean().float()
            y_diversity = (
                y.float()
                .std(dim=1, unbiased=False)
                .mean(dim=(1, 2))
            )  # (B,)
            z_diversity = (
                z.float()
                .std(dim=1, unbiased=False)
                .mean(dim=(1, 2))
            )  # (B,)
        else:
            resample_fraction = nan_scalar
            weight_entropy = nan_scalar
            q_diversity = nan_scalar
            y_diversity = nan_batch
            z_diversity = nan_batch

        exact_pp = None
        preds_pp = None
        exact_sorted = None
        map_tokens = None
        need_predictions = labels is not None and (
            collect_step_metrics or step in checkpoint_steps
        )
        # Per-particle exact-solve rate (requires labels)
        if need_predictions:
            assert q_order is not None
            tok_flat = decoder(y.reshape(B * S, L, D)).argmax(-1)
            pred_len = tok_flat.shape[-1]
            tok_pp = tok_flat.view(B, S, pred_len)
            labels_flat = labels.unsqueeze(1).expand(B, S, pred_len).reshape(B * S, pred_len)
            exact_pp = _exact_solved(tok_flat, labels_flat).view(B, S)
            preds_pp = tok_pp
            exact_sorted = exact_pp.gather(1, q_order)  # (B, S) by Q-rank
            map_tokens = _decode_map_particle(decoder, y, log_w)

        if collect_step_metrics and labels is not None:
            # MAP prediction: decode the highest-weight particle.
            assert map_tokens is not None
            map_exact = _exact_solved(map_tokens, labels).mean()
            valid_mask = labels != _IGNORE
            token_acc = (
                ((map_tokens == labels) & valid_mask).sum().float()
                / valid_mask.sum().clamp(min=1).float()
            )
            assert exact_pp is not None and exact_sorted is not None
            any_exact = exact_pp.max(dim=-1).values.mean()
            exact_by_rank = exact_sorted.detach().float().mean(0)
            assert q_sorted is not None and w_sorted is not None
            q_by_rank = q_sorted.detach().float().mean(0)
            q_std_by_rank = q_sorted.detach().float().std(0)
            w_by_rank = w_sorted.detach().float().mean(0)
            w_std_by_rank = w_sorted.detach().float().std(0)
        elif collect_step_metrics and labels is None:
            map_exact = nan_scalar
            token_acc = nan_scalar
            any_exact = nan_scalar
            exact_by_rank = torch.zeros(S, device=device, dtype=torch.float32)
            assert q_sorted is not None and w_sorted is not None
            q_by_rank = q_sorted.detach().float().mean(0)
            q_std_by_rank = q_sorted.detach().float().std(0)
            w_by_rank = w_sorted.detach().float().mean(0)
            w_std_by_rank = w_sorted.detach().float().std(0)
        else:
            map_exact = nan_scalar
            token_acc = nan_scalar
            any_exact = nan_scalar
            exact_by_rank = nan_rank
            q_by_rank = nan_rank
            q_std_by_rank = nan_rank
            w_by_rank = nan_rank
            w_std_by_rank = nan_rank

        step_data.append(
            {
                "step": step,
                "exact_accuracy": map_exact,
                "token_accuracy": token_acc,
                "any_exact_rate": any_exact,
                "oracle_gap": (
                    any_exact - map_exact
                    if labels is not None
                    else nan_scalar
                ),
                "mean_ess": (ess_val / S).mean(),
                "weight_entropy": weight_entropy,
                "q_diversity": q_diversity,
                "y_diversity": y_diversity.detach(),  # (B,)
                "z_diversity": z_diversity.detach(),  # (B,)
                "resample_fraction": resample_fraction,
                # (S,) arrays — mean over batch dimension
                "q_by_rank": q_by_rank.detach(),
                "q_std_by_rank": q_std_by_rank.detach(),
                "w_by_rank": w_by_rank.detach(),
                "w_std_by_rank": w_std_by_rank.detach(),
                "exact_by_rank": exact_by_rank.detach(),
            }
        )

        if step in checkpoint_steps:
            assert q_sorted is not None and w_sorted is not None
            ckpt_weights[step] = w_sorted.detach().float()   # (B, S)
            ckpt_q[step] = q_sorted.detach().float()         # (B, S)
            if save_particle_details:
                ckpt_weights_raw[step] = w_norm.detach().float()  # (B, S)
                ckpt_q_raw[step] = q_logit_raw.detach().float()   # (B, S)
            if labels is not None:
                assert exact_pp is not None and preds_pp is not None
                ckpt_pp_exact[step] = exact_pp.detach().float()  # (B, S)
                ckpt_preds[step] = preds_pp.detach()              # (B, S, L)
                assert map_tokens is not None
                ckpt_map_preds[step] = map_tokens.detach()        # (B, L)
            if config.save_hidden_states:
                ckpt_y[step] = y.detach().half()             # (B, S, L, D) float16
                ckpt_z[step] = z.detach().half()             # (B, S, L, D) float16
                ckpt_q_raw[step] = q_logit_raw.detach().float()  # (B, S)

        # Systematic resampling when ESS drops below threshold
        step_anc = None
        if save_particle_details:
            identity_anc = torch.arange(S, device=device).view(1, S).expand(B, S)
            step_anc = identity_anc
        if S > 1 and do_rs.any():
            anc = systematic_resample(log_w, generator=gen)        # (B, S)
            if save_particle_details:
                step_anc = torch.where(do_rs.view(B, 1), anc, identity_anc)
            g = anc.unsqueeze(-1).unsqueeze(-1).expand(B, S, L, D)
            y_rs = y.gather(1, g)
            z_rs = z.gather(1, g)
            # Optional post-resample jitter (rejuvenation) to restore diversity
            if config.sigma_rejuvenate > 0.0:
                jitter = torch.empty(B, S, L, D).normal_(
                    std=config.sigma_rejuvenate, generator=gen
                ).to(device=device, dtype=y.dtype)
                y_rs = y_rs + jitter
            do_rs_view = do_rs.view(B, 1, 1, 1)
            y = torch.where(do_rs_view, y_rs, y)
            z = torch.where(do_rs_view, z_rs, z)
            log_w = torch.where(do_rs.view(B, 1), torch.zeros_like(log_w), log_w)
        if save_particle_details:
            assert step_anc is not None
            ckpt_ancestors[step] = step_anc.detach().long()

    return (
        step_data,
        ckpt_weights,
        ckpt_q,
        ckpt_weights_raw,
        ckpt_ancestors,
        ckpt_pp_exact,
        ckpt_preds,
        ckpt_map_preds,
        ckpt_y,
        ckpt_z,
        ckpt_q_raw,
    )


# ---------------------------------------------------------------------------
# Guided particle filter runtime
# ---------------------------------------------------------------------------


class BIGRFilter:
    """Particle filter over hypothesis states.

    Wraps ``_run_one_batch`` over multiple batches and aggregates results
    into a ``FilterTrace``.

    Parameters
    ----------
    stoch_f:
        StochasticFTheta instance (can be sigma=0 for deterministic).
    q_head:
        TRMCoreQHead instance.
    decoder:
        TRMCoreDecoder instance.
    config:
        InferenceConfig — particles, steps, sigmas, seed, etc.
    h_cycles:
        TRM H_cycles parameter (used to compute supervision boundaries
        when config.checkpoint_steps is None). Defaults to 3.
    """

    def __init__(
        self,
        stoch_f: StochasticFTheta,
        q_head: TRMCoreQHead,
        decoder: TRMCoreDecoder,
        config: InferenceConfig,
        h_cycles: int = 3,
    ) -> None:
        self.stoch_f = stoch_f
        self.q_head = q_head
        self.decoder = decoder
        self.config = config
        self.h_cycles = h_cycles

    def run(
        self,
        batches: Sequence[dict],
        *,
        y0_init: torch.Tensor,
        z0_init: torch.Tensor,
        device: torch.device,
    ) -> FilterTrace:
        """Run the particle filter over all batches and return a FilterTrace.

        Parameters
        ----------
        batches:
            Sequence of dicts with keys:
              'x'      — (B, L, D) embedded inputs (on device)
              'labels' — (B, L) ground-truth tokens or None
            Alternatively, provide raw batches with 'inputs' (token ids),
            'puzzle_identifiers', and 'labels' — the filter will embed them
            if an 'embedder' key is absent.
        y0_init:
            (L, D) or (B, L, D) learnt initial hypothesis state from checkpoint.
        z0_init:
            (L, D) or (B, L, D) learnt initial latent state from checkpoint.
        device:
            Target device.

        Returns
        -------
        FilterTrace with aggregated step records and checkpoint arrays.
        """
        ckpt_steps = set(self.config.resolve_checkpoint_steps(self.h_cycles))

        all_step_data: list[list[dict]] = []
        ckpt_w_acc: dict[int, list[torch.Tensor]] = {s: [] for s in ckpt_steps}
        ckpt_q_acc: dict[int, list[torch.Tensor]] = {s: [] for s in ckpt_steps}
        ckpt_w_raw_acc: dict[int, list[torch.Tensor]] = {s: [] for s in ckpt_steps}
        ancestor_steps = (
            set(range(1, self.config.total_steps + 1))
            if self.config.save_particle_details
            else set()
        )
        ckpt_ancestor_acc: dict[int, list[torch.Tensor]] = {
            s: [] for s in ancestor_steps
        }
        ckpt_pp_exact_acc: dict[int, list[torch.Tensor]] = {s: [] for s in ckpt_steps}
        ckpt_preds_acc: dict[int, list[torch.Tensor]] = {s: [] for s in ckpt_steps}
        ckpt_map_preds_acc: dict[int, list[torch.Tensor]] = {s: [] for s in ckpt_steps}
        ckpt_y_acc: dict[int, list[torch.Tensor]] = {s: [] for s in ckpt_steps}
        ckpt_z_acc: dict[int, list[torch.Tensor]] = {s: [] for s in ckpt_steps}
        ckpt_q_raw_acc: dict[int, list[torch.Tensor]] = {s: [] for s in ckpt_steps}

        for bi, batch in enumerate(tqdm(batches, desc="Batches")):
            x = batch["x"]
            labels = batch.get("labels")
            B, L, D = x.shape

            y0 = broadcast_initial_state(
                y0_init, batch_size=B, seq_len=L, device=device, dtype=x.dtype
            )
            z0 = broadcast_initial_state(
                z0_init, batch_size=B, seq_len=L, device=device, dtype=x.dtype
            )

            sd, cw, cq, cwr, ca, cppe, cpp, cmp, cy, cz, cqr = _run_one_batch(
                self.stoch_f,
                self.q_head,
                self.decoder,
                x=x,
                y0=y0,
                z0_init=z0,
                labels=labels,
                config=self.config,
                checkpoint_steps=ckpt_steps,
                seed=self.config.seed + bi,
                device=device,
            )
            all_step_data.append(sd)
            for s in ckpt_steps:
                if s in cw:
                    ckpt_w_acc[s].append(cw[s])
                    ckpt_q_acc[s].append(cq[s])
                if s in cwr:
                    ckpt_w_raw_acc[s].append(cwr[s])
                if s in cppe:
                    ckpt_pp_exact_acc[s].append(cppe[s])
                    ckpt_preds_acc[s].append(cpp[s])
                    ckpt_map_preds_acc[s].append(cmp[s])
                if s in cy:
                    ckpt_y_acc[s].append(cy[s])
                    ckpt_z_acc[s].append(cz[s])
                if s in cqr:
                    ckpt_q_raw_acc[s].append(cqr[s])
            for s in ancestor_steps:
                if s in ca:
                    ckpt_ancestor_acc[s].append(ca[s])

        return self._aggregate(
            all_step_data, ckpt_w_acc, ckpt_q_acc, ckpt_steps,
            ckpt_pp_exact_acc, ckpt_preds_acc, ckpt_map_preds_acc,
            ckpt_y_acc, ckpt_z_acc, ckpt_q_raw_acc,
            ckpt_w_raw_acc, ckpt_ancestor_acc,
        )

    @staticmethod
    def _aggregate(
        all_step_data: list[list[dict]],
        ckpt_w_acc: dict[int, list[torch.Tensor]],
        ckpt_q_acc: dict[int, list[torch.Tensor]],
        ckpt_steps: set[int],
        ckpt_pp_exact_acc: dict[int, list[torch.Tensor]] | None = None,
        ckpt_preds_acc: dict[int, list[torch.Tensor]] | None = None,
        ckpt_map_preds_acc: dict[int, list[torch.Tensor]] | None = None,
        ckpt_y_acc: dict[int, list[torch.Tensor]] | None = None,
        ckpt_z_acc: dict[int, list[torch.Tensor]] | None = None,
        ckpt_q_raw_acc: dict[int, list[torch.Tensor]] | None = None,
        ckpt_w_raw_acc: dict[int, list[torch.Tensor]] | None = None,
        ckpt_ancestor_acc: dict[int, list[torch.Tensor]] | None = None,
    ) -> FilterTrace:
        n_steps = len(all_step_data[0])

        def _as_tensor(value) -> torch.Tensor:
            if torch.is_tensor(value):
                return value
            return torch.as_tensor(value)

        def _scalar_mean(values: list) -> float:
            stacked = torch.stack([_as_tensor(v).float().reshape(()) for v in values], dim=0)
            finite = stacked[torch.isfinite(stacked)]
            if finite.numel() == 0:
                return float("nan")
            return float(finite.mean().detach().cpu())

        def _mean_rank_array(step_i: int, key: str) -> np.ndarray:
            stacked = torch.stack(
                [_as_tensor(b[step_i][key]).float() for b in all_step_data], dim=0
            )
            return stacked.mean(dim=0).detach().cpu().numpy()

        def _concat_to_numpy(values: list) -> np.ndarray:
            if not values:
                return np.array([])
            first = values[0]
            if torch.is_tensor(first):
                stacked = torch.cat(
                    [_as_tensor(v).detach() for v in values],
                    dim=0,
                )
                return stacked.cpu().numpy()
            return np.concatenate(values, axis=0)

        # Scalar metrics: mean across batches
        step_records: list[StepRecord] = []
        for step_i in range(n_steps):
            row = {}
            for key in (
                "exact_accuracy", "token_accuracy", "any_exact_rate", "oracle_gap",
                "mean_ess", "weight_entropy", "q_diversity", "resample_fraction",
            ):
                vals = [b[step_i][key] for b in all_step_data]
                row[key] = _scalar_mean(vals)
            row["y_diversity"] = (
                torch.cat(
                    [_as_tensor(b[step_i]["y_diversity"]).float() for b in all_step_data],
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            row["z_diversity"] = (
                torch.cat(
                    [_as_tensor(b[step_i]["z_diversity"]).float() for b in all_step_data],
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            step_records.append(
                StepRecord(
                    step=all_step_data[0][step_i]["step"],
                    exact_accuracy=row["exact_accuracy"],
                    token_accuracy=row["token_accuracy"],
                    any_exact_rate=row["any_exact_rate"],
                    mean_ess=row["mean_ess"],
                    weight_entropy=row["weight_entropy"],
                    q_diversity=row["q_diversity"],
                    y_diversity=row["y_diversity"],
                    z_diversity=row["z_diversity"],
                    resample_fraction=row["resample_fraction"],
                    q_by_rank=_mean_rank_array(step_i, "q_by_rank"),
                    q_std_by_rank=_mean_rank_array(step_i, "q_std_by_rank"),
                    w_by_rank=_mean_rank_array(step_i, "w_by_rank"),
                    w_std_by_rank=_mean_rank_array(step_i, "w_std_by_rank"),
                    exact_by_rank=_mean_rank_array(step_i, "exact_by_rank"),
                )
            )

        # (N, S) stacked rank arrays
        rank_arrays: dict[str, np.ndarray] = {
            "steps": np.array([r.step for r in step_records]),
        }
        for key in ("q_by_rank", "q_std_by_rank", "w_by_rank", "w_std_by_rank", "exact_by_rank"):
            rank_arrays[key] = np.stack([getattr(r, key) for r in step_records])

        checkpoint_weights = {
            s: _concat_to_numpy(ckpt_w_acc[s])
            for s in ckpt_steps
            if ckpt_w_acc[s]
        }
        checkpoint_q_logits = {
            s: _concat_to_numpy(ckpt_q_acc[s])
            for s in ckpt_steps
            if ckpt_q_acc[s]
        }
        checkpoint_pp_exact = {}
        checkpoint_preds = {}
        checkpoint_map_preds = {}
        checkpoint_weights_raw = {}
        checkpoint_ancestors = {}
        if ckpt_w_raw_acc:
            checkpoint_weights_raw = {
                s: _concat_to_numpy(ckpt_w_raw_acc[s])
                for s in ckpt_steps
                if ckpt_w_raw_acc.get(s)
            }
        if ckpt_ancestor_acc:
            checkpoint_ancestors = {
                s: _concat_to_numpy(ckpt_ancestor_acc[s])
                for s in sorted(ckpt_ancestor_acc)
                if ckpt_ancestor_acc.get(s)
            }
        if ckpt_pp_exact_acc:
            checkpoint_pp_exact = {
                s: _concat_to_numpy(ckpt_pp_exact_acc[s])
                for s in ckpt_steps
                if ckpt_pp_exact_acc.get(s)
            }
        if ckpt_preds_acc:
            checkpoint_preds = {
                s: _concat_to_numpy(ckpt_preds_acc[s])
                for s in ckpt_steps
                if ckpt_preds_acc.get(s)
            }
        if ckpt_map_preds_acc:
            checkpoint_map_preds = {
                s: _concat_to_numpy(ckpt_map_preds_acc[s])
                for s in ckpt_steps
                if ckpt_map_preds_acc.get(s)
            }

        checkpoint_y = {}
        checkpoint_z = {}
        checkpoint_q_raw = {}
        if ckpt_y_acc:
            checkpoint_y = {
                s: _concat_to_numpy(ckpt_y_acc[s])
                for s in ckpt_steps
                if ckpt_y_acc.get(s)
            }
        if ckpt_z_acc:
            checkpoint_z = {
                s: _concat_to_numpy(ckpt_z_acc[s])
                for s in ckpt_steps
                if ckpt_z_acc.get(s)
            }
        if ckpt_q_raw_acc:
            checkpoint_q_raw = {
                s: _concat_to_numpy(ckpt_q_raw_acc[s])
                for s in ckpt_steps
                if ckpt_q_raw_acc.get(s)
            }

        return FilterTrace(
            step_records=step_records,
            rank_arrays=rank_arrays,
            checkpoint_weights=checkpoint_weights,
            checkpoint_q_logits=checkpoint_q_logits,
            checkpoint_weights_raw=checkpoint_weights_raw,
            checkpoint_ancestors=checkpoint_ancestors,
            checkpoint_pp_exact=checkpoint_pp_exact,
            checkpoint_preds=checkpoint_preds,
            checkpoint_map_preds=checkpoint_map_preds,
            checkpoint_y=checkpoint_y,
            checkpoint_z=checkpoint_z,
            checkpoint_q_raw=checkpoint_q_raw,
        )
