"""Data containers for the guided particle filter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ParticleState:
    """Mutable outer filter state for S particles over a batch of B examples.

    Attributes
    ----------
    y:
        Hypothesis states, shape (B, S, L, D).
    log_w:
        Unnormalised log-weights, shape (B, S).
        Initialised to zeros; accumulated via Q-head log-potentials.
    """

    y: torch.Tensor        # (B, S, L, D)
    log_w: torch.Tensor    # (B, S)


@dataclass
class StepRecord:
    """Per-step diagnostic record produced by `BIGRFilter`.

    Scalar metrics are Python floats averaged over the batch.
    Rank-array metrics have shape (S,), averaged over the batch.
    State-diversity metrics have shape (B_total,) after aggregation across batches.
    Accuracy fields are NaN when the filter runs without labels.
    """

    step: int
    exact_accuracy: float        # MAP-particle exact-solve rate
    token_accuracy: float        # MAP-particle token-level accuracy
    any_exact_rate: float        # oracle rate (best particle)
    mean_ess: float              # mean normalised ESS = mean(ESS/S)
    q_by_rank: np.ndarray        # (S,) mean Q logit per Q-rank (best first)
    q_std_by_rank: np.ndarray    # (S,) std of Q logit per rank
    w_by_rank: np.ndarray        # (S,) mean normalised weight per rank
    w_std_by_rank: np.ndarray    # (S,) std of weight per rank
    exact_by_rank: np.ndarray    # (S,) mean exact-solve rate per rank
    weight_entropy: float = 0.0     # normalised weight entropy H(w)/log(S) in [0,1]
    q_diversity: float = 0.0        # mean std of Q-logits across particles
    y_diversity: np.ndarray | None = None  # (B_total,) mean std of y across particles
    z_diversity: np.ndarray | None = None  # (B_total,) mean std of z across particles
    resample_fraction: float = 0.0  # fraction of batch elements that resampled at this step


@dataclass
class FilterTrace:
    """Full trace returned by `BIGRFilter.run()`.

    Attributes
    ----------
    step_records:
        Per-step diagnostics (one entry per outer step).
    rank_arrays:
        Stacked rank arrays with keys:
          'steps'          — (N,) int
          'q_by_rank'      — (N, S)
          'q_std_by_rank'  — (N, S)
          'w_by_rank'      — (N, S)
          'w_std_by_rank'  — (N, S)
          'exact_by_rank'  — (N, S)
    checkpoint_weights:
        {step: np.ndarray of shape (B_total, S)} normalised weights at
        checkpoint steps, sorted by Q-rank.
    checkpoint_q_logits:
        {step: np.ndarray of shape (B_total, S)} Q logits at checkpoint
        steps, sorted by Q-rank.
    checkpoint_weights_raw:
        {step: np.ndarray of shape (B_total, S)} normalised weights at
        checkpoint steps, in original particle order.
    checkpoint_ancestors:
        {step: np.ndarray of shape (B_total, S)} parent particle index used
        by the resampling transition after that step. Identity rows mean no
        resampling fired for that example.
    """

    step_records: list[StepRecord] = field(default_factory=list)
    rank_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    checkpoint_weights: dict[int, np.ndarray] = field(default_factory=dict)
    checkpoint_q_logits: dict[int, np.ndarray] = field(default_factory=dict)
    checkpoint_weights_raw: dict[int, np.ndarray] = field(default_factory=dict)
    """Normalised weights in original particle order at checkpoint steps. Shape (B_total, S)."""
    checkpoint_ancestors: dict[int, np.ndarray] = field(default_factory=dict)
    """Per-step resampling ancestors in original particle order. Shape (B_total, S)."""
    checkpoint_pp_exact: dict[int, np.ndarray] = field(default_factory=dict)
    """Per-particle exact-solve flags at checkpoint steps. Shape (B_total, S)."""
    checkpoint_preds: dict[int, np.ndarray] = field(default_factory=dict)
    """Per-particle token predictions at checkpoint steps. Shape (B_total, S, L)."""
    checkpoint_map_preds: dict[int, np.ndarray] = field(default_factory=dict)
    """MAP-particle token predictions at checkpoint steps. Shape (B_total, L)."""
    checkpoint_y: dict[int, np.ndarray] = field(default_factory=dict)
    """Per-particle hypothesis states at checkpoint steps (float16). Shape (B_total, S, L, D).
    Only populated when InferenceConfig.save_hidden_states=True."""
    checkpoint_z: dict[int, np.ndarray] = field(default_factory=dict)
    """Per-particle latent states at checkpoint steps (float16). Shape (B_total, S, L, D).
    Only populated when InferenceConfig.save_hidden_states=True."""
    checkpoint_q_raw: dict[int, np.ndarray] = field(default_factory=dict)
    """Per-particle Q logits in original particle order at checkpoint steps. Shape (B_total, S).
    Populated when InferenceConfig.save_hidden_states=True or
    InferenceConfig.save_particle_details=True."""
