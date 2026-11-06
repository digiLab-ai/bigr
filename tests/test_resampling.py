"""Unit tests for resampling utilities.

All tests are fast and isolated — no checkpoint or GPU required.
"""

import pytest
import torch

from bigr.resampling import (
    broadcast_initial_state,
    effective_sample_size,
    make_generator,
    normalised_weights,
    systematic_resample,
)


@pytest.mark.unit
def test_normalised_weights_sum_to_one():
    log_w = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]])
    w = normalised_weights(log_w)
    assert w.shape == log_w.shape
    torch.testing.assert_close(w.sum(-1), torch.ones(2))
    assert (w >= 0).all()


@pytest.mark.unit
def test_normalised_weights_uniform_gives_one_over_s():
    log_w = torch.zeros(3, 8)
    w = normalised_weights(log_w)
    torch.testing.assert_close(w, torch.full((3, 8), 1.0 / 8))


@pytest.mark.unit
def test_ess_uniform_equals_s():
    S = 8
    log_w = torch.zeros(4, S)
    ess = effective_sample_size(log_w)
    torch.testing.assert_close(ess, torch.full((4,), float(S)))


@pytest.mark.unit
def test_ess_degenerate_equals_one():
    """All weight on one particle -> ESS = 1."""
    log_w = torch.full((2, 6), -1e9)
    log_w[:, 0] = 0.0
    ess = effective_sample_size(log_w)
    torch.testing.assert_close(ess, torch.ones(2), atol=1e-3, rtol=0)


@pytest.mark.unit
def test_systematic_resample_output_shape():
    B, S = 3, 8
    log_w = torch.randn(B, S)
    gen = make_generator(42)
    anc = systematic_resample(log_w, generator=gen)
    assert anc.shape == (B, S)
    assert (anc >= 0).all()
    assert (anc < S).all()


@pytest.mark.unit
def test_systematic_resample_concentrates_on_high_weight():
    """All mass on particle 0 -> all ancestors point to particle 0."""
    B, S = 1, 8
    log_w = torch.full((B, S), -100.0)
    log_w[0, 0] = 0.0
    gen = make_generator(7)
    anc = systematic_resample(log_w, generator=gen)
    assert (anc == 0).all(), "All ancestors should point to particle 0"


@pytest.mark.unit
def test_make_generator_reproducible():
    gen1 = make_generator(123)
    gen2 = make_generator(123)
    v1 = torch.randn(5, generator=gen1)
    v2 = torch.randn(5, generator=gen2)
    torch.testing.assert_close(v1, v2)


@pytest.mark.unit
def test_broadcast_initial_state_1d():
    base = torch.randn(16)
    out = broadcast_initial_state(
        base, batch_size=4, seq_len=9, device=torch.device("cpu"), dtype=torch.float32
    )
    assert out.shape == (4, 9, 16)


@pytest.mark.unit
def test_broadcast_initial_state_2d():
    base = torch.randn(9, 16)
    out = broadcast_initial_state(
        base, batch_size=4, seq_len=9, device=torch.device("cpu"), dtype=torch.float32
    )
    assert out.shape == (4, 9, 16)
