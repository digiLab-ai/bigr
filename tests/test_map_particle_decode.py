from __future__ import annotations

import torch

from bigr.filter import _decode_map_particle


class _IdentityDecoder:
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def particle_logits(
        self,
        y_particles: torch.Tensor,
        log_weights: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _, _ = y_particles.shape
        log_w = log_weights - log_weights.logsumexp(dim=-1, keepdim=True)
        log_probs = torch.log_softmax(y_particles.float(), dim=-1)
        return torch.logsumexp(log_probs + log_w.view(B, S, 1, 1), dim=1)


def test_decode_map_particle_uses_highest_weight_particle() -> None:
    """MAP decode should select the top-weight particle, not mixture argmax."""
    decoder = _IdentityDecoder()
    y_particles = torch.tensor(
        [[[[5.0, 0.0]], [[0.0, 1.0]]]],
        dtype=torch.float32,
    )  # (B=1, S=2, L=1, V=2)
    log_weights = torch.log(torch.tensor([[0.4, 0.6]], dtype=torch.float32))

    map_tokens = _decode_map_particle(decoder, y_particles, log_weights)
    mixture_tokens = decoder.particle_logits(y_particles, log_weights).argmax(-1)

    assert map_tokens.item() == 1
    assert mixture_tokens.item() == 0
