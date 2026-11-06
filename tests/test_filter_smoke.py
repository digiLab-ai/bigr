"""Shape and bound contracts of `BIGRFilter` on the tiny in-memory backbone."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from bigr.config import InferenceConfig
from bigr.filter import BIGRFilter
from bigr.stochastic import StochasticFTheta


# ---------------------------------------------------------------------------
# Shape smoke tests (tiny backbone, no real checkpoint)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_filter_output_shapes(tiny_trm, tiny_native_f, tiny_decoder, tiny_q_head, tiny_embedder):
    """`BIGRFilter` returns a FilterTrace with the correct array shapes."""
    device = torch.device("cpu")
    B, L = 2, 4
    S, N = 3, 5

    inputs = torch.randint(0, 5, (B, L))
    with torch.no_grad():
        x = tiny_embedder(inputs, torch.zeros(B, dtype=torch.long))

    y0_param = tiny_trm.z_H_init.detach().clone()
    z0_param = tiny_trm.z_L_init.detach().clone()

    stoch_f = StochasticFTheta(tiny_native_f, sigma_inner=0.3, sigma_trans=0.1)
    cfg = InferenceConfig(
        particles=S,
        total_steps=N,
        sigma_inner=0.3,
        sigma_trans=0.1,
        ess_threshold=0.5,
        seed=0,
    )
    filt = BIGRFilter(stoch_f, tiny_q_head, tiny_decoder, cfg, h_cycles=1)

    labels = torch.randint(0, 5, (B, L))
    batches = [{"x": x, "labels": labels}]

    with torch.no_grad():
        trace = filt.run(batches, y0_init=y0_param, z0_init=z0_param, device=device)

    # Step records
    assert len(trace.step_records) == N
    for rec in trace.step_records:
        assert rec.q_by_rank.shape == (S,)
        assert rec.w_by_rank.shape == (S,)
        assert rec.exact_by_rank.shape == (S,)

    # Rank arrays
    assert trace.rank_arrays["steps"].shape == (N,)
    assert trace.rank_arrays["q_by_rank"].shape == (N, S)
    assert trace.rank_arrays["w_by_rank"].shape == (N, S)


@pytest.mark.unit
def test_filter_mean_ess_bounded(tiny_trm, tiny_native_f, tiny_decoder, tiny_q_head, tiny_embedder):
    """mean_ess must be in [0, 1] at every step."""
    device = torch.device("cpu")
    B, S, N = 2, 4, 4
    inputs = torch.randint(0, 5, (B, 4))
    with torch.no_grad():
        x = tiny_embedder(inputs, torch.zeros(B, dtype=torch.long))

    stoch_f = StochasticFTheta(tiny_native_f, sigma_inner=0.5, sigma_trans=0.1)
    cfg = InferenceConfig(particles=S, total_steps=N, sigma_inner=0.5, sigma_trans=0.1,
                          ess_threshold=0.5, seed=1)
    filt = BIGRFilter(stoch_f, tiny_q_head, tiny_decoder, cfg, h_cycles=1)

    with torch.no_grad():
        trace = filt.run(
            [{"x": x, "labels": None}],
            y0_init=tiny_trm.z_H_init.detach().clone(),
            z0_init=tiny_trm.z_L_init.detach().clone(),
            device=device,
        )

    for rec in trace.step_records:
        assert 0.0 <= rec.mean_ess <= 1.0 + 1e-5, f"mean_ess out of range: {rec.mean_ess}"


@pytest.mark.unit
def test_filter_no_labels_produces_nan_accuracy(tiny_trm, tiny_native_f, tiny_decoder, tiny_q_head, tiny_embedder):
    """Without labels, exact_accuracy should be nan (not raise)."""
    device = torch.device("cpu")
    B = 2
    inputs = torch.randint(0, 5, (B, 4))
    with torch.no_grad():
        x = tiny_embedder(inputs, torch.zeros(B, dtype=torch.long))

    stoch_f = StochasticFTheta(tiny_native_f, sigma_inner=0.0, sigma_trans=0.0)
    cfg = InferenceConfig(particles=2, total_steps=2, sigma_inner=0.0, sigma_trans=0.0,
                          ess_threshold=0.5, seed=0)
    filt = BIGRFilter(stoch_f, tiny_q_head, tiny_decoder, cfg, h_cycles=1)

    with torch.no_grad():
        trace = filt.run(
            [{"x": x, "labels": None}],
            y0_init=tiny_trm.z_H_init.detach().clone(),
            z0_init=tiny_trm.z_L_init.detach().clone(),
            device=device,
        )

    for rec in trace.step_records:
        assert np.isnan(rec.exact_accuracy)


@pytest.mark.unit
def test_filter_particle_details_trace_shapes(tiny_trm, tiny_native_f, tiny_decoder, tiny_q_head, tiny_embedder):
    """Particle-detail tracing keeps original-order weights, Q logits, and ancestry."""
    device = torch.device("cpu")
    B, L = 2, 4
    S, N = 3, 3
    inputs = torch.randint(0, 5, (B, L))
    with torch.no_grad():
        x = tiny_embedder(inputs, torch.zeros(B, dtype=torch.long))

    stoch_f = StochasticFTheta(tiny_native_f, sigma_inner=0.0, sigma_trans=0.0)
    cfg = InferenceConfig(
        particles=S,
        total_steps=N,
        sigma_inner=0.0,
        sigma_trans=0.0,
        ess_threshold=0.0,
        seed=0,
        checkpoint_steps=list(range(1, N + 1)),
        save_particle_details=True,
    )
    filt = BIGRFilter(stoch_f, tiny_q_head, tiny_decoder, cfg, h_cycles=1)

    labels = torch.randint(0, 5, (B, L))
    with torch.no_grad():
        trace = filt.run(
            [{"x": x, "labels": labels}],
            y0_init=tiny_trm.z_H_init.detach().clone(),
            z0_init=tiny_trm.z_L_init.detach().clone(),
            device=device,
        )

    identity = np.tile(np.arange(S), (B, 1))
    for step in range(1, N + 1):
        assert trace.checkpoint_weights_raw[step].shape == (B, S)
        assert trace.checkpoint_q_raw[step].shape == (B, S)
        assert trace.checkpoint_ancestors[step].shape == (B, S)
        np.testing.assert_array_equal(trace.checkpoint_ancestors[step], identity)
