"""The deterministic limit: with S = 1 and sigma = 0 the filter is the TRM.

Algorithm 1 reduces to the deterministic TRM rollout when there is one particle and
no noise. The test runs ``TRMModule.inner_forward`` with the model's own carry for N
outer steps and checks that ``BIGRFilter`` reproduces the hypothesis state, the Q-logit
and the decoded tokens at every step. With sigma = 0 every particle follows the same
path, so the same holds for S > 1.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bigr.config import InferenceConfig
from bigr.filter import BIGRFilter
from bigr.stochastic import StochasticFTheta

N_STEPS = 4
B, L = 2, 4


def _native_rollout(trm, inputs: torch.Tensor, puzzle_ids: torch.Tensor):
    """N outer steps of the TRM's own recurrence (H_cycles = 1 in the tiny backbone)."""
    batch = {"input": inputs, "puzzle_identifiers": puzzle_ids}
    carry = trm.reset_carry(torch.ones(B, dtype=torch.bool), trm.empty_carry(B, inputs.device))
    ys, qs, toks = [], [], []
    for _ in range(N_STEPS):
        carry, logits, q = trm.inner_forward(carry, batch)
        ys.append(carry.z_H.clone())
        qs.append(q.clone())
        toks.append(logits.argmax(-1))
    return ys, qs, toks


@pytest.mark.unit
@pytest.mark.parametrize("particles", [1, 3])
def test_zero_noise_filter_matches_native_trm(
    tiny_trm, tiny_native_f, tiny_decoder, tiny_q_head, tiny_embedder, particles
):
    assert int(tiny_trm.hparams.H_cycles) == 1, "one inner_forward call must be one outer step"
    inputs = torch.randint(0, 5, (B, L))
    puzzle_ids = torch.zeros(B, dtype=torch.long)
    labels = torch.randint(0, 5, (B, L))

    with torch.no_grad():
        ys, qs, toks = _native_rollout(tiny_trm, inputs, puzzle_ids)
        x = tiny_embedder(inputs, puzzle_ids)
        cfg = InferenceConfig(
            particles=particles,
            total_steps=N_STEPS,
            sigma=0.0,
            ess_threshold=0.3,
            seed=0,
            checkpoint_steps=list(range(1, N_STEPS + 1)),
            save_hidden_states=True,
        )
        filt = BIGRFilter(StochasticFTheta(tiny_native_f, sigma=0.0), tiny_q_head, tiny_decoder, cfg, h_cycles=1)
        trace = filt.run(
            [{"x": x, "labels": labels}],
            y0_init=tiny_trm.z_H_init.detach().clone(),
            z0_init=tiny_trm.z_L_init.detach().clone(),
            device=torch.device("cpu"),
        )

    for step in range(1, N_STEPS + 1):
        y_native, q_native, tok_native = ys[step - 1], qs[step - 1], toks[step - 1]
        y_filter = torch.from_numpy(trace.checkpoint_y[step]).float()        # (B, S, L, D), stored as float16
        q_filter = torch.from_numpy(trace.checkpoint_q_raw[step])            # (B, S)
        for s in range(particles):
            torch.testing.assert_close(y_filter[:, s], y_native, rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(q_filter[:, s], q_native, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(trace.checkpoint_map_preds[step], tok_native.numpy())

    # Identical particles carry identical weights: ESS/S = 1 and no resampling.
    for rec in trace.step_records:
        assert abs(rec.mean_ess - 1.0) < 1e-5
        assert rec.resample_fraction == 0.0
