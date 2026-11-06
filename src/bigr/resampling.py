"""Particle resampling utilities (systematic resampling, log-weight ESS)."""

from __future__ import annotations

import torch


def normalised_weights(log_w: torch.Tensor) -> torch.Tensor:
    """Convert unnormalised log-weights to normalised weights via softmax.

    Parameters
    ----------
    log_w:
        Shape (B, S).

    Returns
    -------
    Normalised weights, shape (B, S), summing to 1 along dim=-1.
    """
    return torch.softmax(log_w.float(), dim=-1)


def effective_sample_size(log_w: torch.Tensor) -> torch.Tensor:
    """Compute ESS = 1 / sum_s(w_s^2) for each batch element.

    Parameters
    ----------
    log_w:
        Shape (B, S).

    Returns
    -------
    ESS values, shape (B,). Range: [1, S].
    """
    w = normalised_weights(log_w)
    return 1.0 / w.pow(2).sum(-1)


def systematic_resample(
    log_w: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Systematic resampling — returns ancestor indices (B, S).

    Uses a single uniform offset shared across all particles in each
    batch element, giving lower variance than independent multinomial
    resampling.

    Parameters
    ----------
    log_w:
        Unnormalised log-weights, shape (B, S).
    generator:
        Seeded CPU torch.Generator for reproducibility.

    Returns
    -------
    Ancestor index tensor, shape (B, S), values in [0, S-1].
    """
    B, S = log_w.shape
    w = normalised_weights(log_w)
    cdf = w.cumsum(-1)
    # Single uniform offset per batch element, generated on CPU then moved
    u0 = torch.rand(B, 1, generator=generator).to(log_w.device) / S
    grid = u0 + torch.arange(S, device=log_w.device).view(1, S) / S
    return torch.searchsorted(cdf, grid).clamp_max(S - 1)


def broadcast_initial_state(
    base: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand a learnt initial state parameter to (B, L, D).

    Handles 1-D (D,), 2-D (L, D), and 3-D (B, L, D) inputs.

    Parameters
    ----------
    base:
        Learnt initial tensor from the checkpoint.
    batch_size:
        Target batch size B.
    seq_len:
        Target sequence length L.
    device, dtype:
        Target device and dtype.

    Returns
    -------
    Tensor of shape (B, L, D).
    """
    if base.dim() == 1:
        return (
            base.view(1, 1, -1)
            .expand(batch_size, seq_len, -1)
            .to(device=device, dtype=dtype)
        )
    if base.dim() == 2:
        return (
            base.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .to(device=device, dtype=dtype)
        )
    return base.expand(batch_size, -1, -1).to(device=device, dtype=dtype)


def make_generator(seed: int) -> torch.Generator:
    """Create a seeded CPU torch.Generator.

    Noise is always generated on CPU then moved to the target device.
    This ensures reproducibility across cpu/cuda/mps without relying on
    device-specific generator APIs.

    Parameters
    ----------
    seed:
        Integer seed.

    Returns
    -------
    Seeded torch.Generator (on CPU).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
