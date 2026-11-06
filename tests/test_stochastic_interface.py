"""Unit tests for StochasticFTheta.

Verifies:
  - sigma=0 path produces output identical to the deterministic adapter
  - sigma>0 path differs from deterministic output
  - Generator reproducibility
  - Outputs stay on the correct device (no unintended CPU round-trips)
"""

import pytest
import torch

from bigr.resampling import make_generator
from bigr.stochastic import StochasticFTheta


@pytest.fixture
def tiny_x(tiny_trm):
    """Minimal embedded input (B=2, L=4, D=32)."""
    return torch.randn(2, 4, 32)


@pytest.fixture
def tiny_yz(tiny_trm):
    y = torch.randn(2, 4, 32)
    z = torch.randn(2, 4, 32)
    return y, z


@pytest.mark.unit
def test_sigma_zero_belief_matches_deterministic(tiny_f_theta, tiny_x, tiny_yz):
    """sigma_inner=0 -> StochasticFTheta.belief_update == f_theta.belief_update."""
    y, z = tiny_yz
    sf = StochasticFTheta(tiny_f_theta, sigma_inner=0.0, sigma_trans=0.0)
    gen = make_generator(0)
    with torch.no_grad():
        det = tiny_f_theta.belief_update(tiny_x, y, z)
        sto = sf.belief_update(tiny_x, y, z, generator=gen)
    torch.testing.assert_close(det, sto)


@pytest.mark.unit
def test_sigma_zero_state_matches_deterministic(tiny_f_theta, tiny_yz):
    """sigma_trans=0 -> StochasticFTheta.state_update == f_theta.state_update."""
    y, z = tiny_yz
    sf = StochasticFTheta(tiny_f_theta, sigma_inner=0.0, sigma_trans=0.0)
    gen = make_generator(0)
    with torch.no_grad():
        det = tiny_f_theta.state_update(y, z)
        sto = sf.state_update(y, z, generator=gen)
    torch.testing.assert_close(det, sto)


@pytest.mark.unit
def test_sigma_nonzero_belief_differs(tiny_f_theta, tiny_x, tiny_yz):
    """sigma_inner>0 -> stochastic output differs from deterministic."""
    y, z = tiny_yz
    sf = StochasticFTheta(tiny_f_theta, sigma_inner=1.0, sigma_trans=0.0)
    gen = make_generator(1)
    with torch.no_grad():
        det = tiny_f_theta.belief_update(tiny_x, y, z)
        sto = sf.belief_update(tiny_x, y, z, generator=gen)
    assert not torch.allclose(det, sto), "Noisy output should differ from deterministic"


@pytest.mark.unit
def test_generator_reproducibility(tiny_f_theta, tiny_x, tiny_yz):
    """Same seed -> same stochastic output."""
    y, z = tiny_yz
    sf = StochasticFTheta(tiny_f_theta, sigma_inner=0.5, sigma_trans=0.1)
    with torch.no_grad():
        out1 = sf.belief_update(tiny_x, y, z, generator=make_generator(99))
        out2 = sf.belief_update(tiny_x, y, z, generator=make_generator(99))
    torch.testing.assert_close(out1, out2)


@pytest.mark.unit
def test_output_device_matches_input(tiny_f_theta, tiny_x, tiny_yz):
    """Output device == input device (no silent CPU round-trips)."""
    y, z = tiny_yz
    sf = StochasticFTheta(tiny_f_theta, sigma_inner=0.3, sigma_trans=0.1)
    gen = make_generator(0)
    with torch.no_grad():
        out_z = sf.belief_update(tiny_x, y, z, generator=gen)
        out_y = sf.state_update(y, z, generator=gen)
    assert out_z.device == tiny_x.device
    assert out_y.device == y.device


@pytest.mark.unit
def test_output_shape_unchanged(tiny_f_theta, tiny_x, tiny_yz):
    y, z = tiny_yz
    sf = StochasticFTheta(tiny_f_theta, sigma_inner=0.3, sigma_trans=0.1)
    gen = make_generator(0)
    with torch.no_grad():
        out_z = sf.belief_update(tiny_x, y, z, generator=gen)
        out_y = sf.state_update(y, z, generator=gen)
    assert out_z.shape == z.shape
    assert out_y.shape == y.shape
