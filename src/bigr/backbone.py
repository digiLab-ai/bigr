"""TRM backbone adapters: the single source of truth for F_theta.

The rest of ``bigr`` reaches the backbone only through these adapters.

Five public classes:
    TRMCoreEmbedder   — embed (inputs, puzzle_ids) -> x: (B, L, D)
    TRMCoreDecoder    — lm_head(y) -> logits; particle_logits(y, log_w) -> blended
    TRMCoreQHead      — q_head(y[:,0]) -> scalar logit: (B,)
    TRMCoreFTheta     — single lenet call; .belief_update(x,y,z), .state_update(y,z)
    TRMNativeCycleFTheta — checkpoint-faithful L_cycles belief + 1 H state update

One public function:
    load_trm_backbone_checkpoint(path, map_location) -> TRMModule
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _import_trm_module():
    try:
        from trm.nn.models.trm import TRMModule
    except Exception as exc:
        raise RuntimeError(
            "Unable to import TRMModule from trm. Install the TRM runtime "
            "dependencies (lightning, rich)."
        ) from exc
    return TRMModule


def load_trm_backbone_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    *,
    use_ema: bool | None = None,
):
    """Load a TRM backbone checkpoint into TRMModule.

    Security note: only call with checkpoints from trusted local sources.
    Uses weights_only=False because Lightning saves non-tensor objects in
    hyper_parameters.
    """
    TRMModule = _import_trm_module()
    module = TRMModule.load_from_checkpoint(
        checkpoint_path=Path(checkpoint_path),
        map_location=map_location,
        weights_only=False,
    )

    if use_ema is False:
        return module

    try:
        raw = torch.load(
            Path(checkpoint_path), map_location=map_location, weights_only=False
        )
    except Exception as exc:
        if use_ema is True:
            warnings.warn(
                f"use_ema=True but failed to read checkpoint callback state from "
                f"{checkpoint_path}: {exc}. Falling back to raw weights.",
                stacklevel=2,
            )
        return module

    shadow = _extract_ema_shadow(raw)
    if shadow is None:
        if use_ema is True:
            warnings.warn(
                f"use_ema=True but no EMA state found in {checkpoint_path}; "
                "falling back to raw weights.",
                stacklevel=2,
            )
        return module

    applied = 0
    for name, param in module.named_parameters():
        ema_param = shadow.get(name)
        if ema_param is None or ema_param.shape != param.data.shape:
            continue
        param.data.copy_(ema_param.to(device=param.device, dtype=param.dtype))
        applied += 1

    if applied == 0 and use_ema is True:
        warnings.warn(
            f"use_ema=True but no EMA params matched model weights for {checkpoint_path}; "
            "falling back to raw weights.",
            stacklevel=2,
        )
    return module


def _extract_ema_shadow(obj) -> dict | None:
    """Recursively find EMA shadow weights in a Lightning checkpoint.

    Searches for a dict with key ``"shadow"`` whose values are all tensors.
    This matches the EMA callback state written by training.
    """
    if isinstance(obj, dict):
        shadow = obj.get("shadow")
        if isinstance(shadow, dict) and shadow:
            if all(torch.is_tensor(v) for v in shadow.values()):
                return shadow
        for value in obj.values():
            found = _extract_ema_shadow(value)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _extract_ema_shadow(value)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# TRMCoreEmbedder
# ---------------------------------------------------------------------------


class TRMCoreEmbedder(nn.Module):
    """Embed task inputs via the TRM's _input_embeddings layer.

    Parameters
    ----------
    core:
        A loaded TRMModule instance.
    """

    def __init__(self, core) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        inputs: torch.Tensor,
        puzzle_identifiers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed inputs.

        Parameters
        ----------
        inputs:
            Token ids, shape (B, L).
        puzzle_identifiers:
            Puzzle type ids, shape (B,). Required when puzzle_emb_dim > 0.

        Returns
        -------
        Tensor of shape (B, L, D).
        """
        if puzzle_identifiers is None:
            if getattr(self.core.hparams, "puzzle_emb_dim", 0) > 0:
                raise ValueError(
                    "puzzle_identifiers is required when puzzle_emb_dim > 0."
                )
            puzzle_identifiers = torch.zeros(
                inputs.shape[0],
                device=inputs.device,
                dtype=torch.long,
            )
        return self.core._input_embeddings(inputs, puzzle_identifiers)


# ---------------------------------------------------------------------------
# TRMCoreDecoder
# ---------------------------------------------------------------------------


class TRMCoreDecoder(nn.Module):
    """Decode hypothesis states to token logits.

    Parameters
    ----------
    core:
        A loaded TRMModule instance.
    """

    def __init__(self, core) -> None:
        super().__init__()
        self.core = core

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Decode hypothesis state to logits.

        Parameters
        ----------
        y:
            Hypothesis state, shape (B, L, D) or (B, L+prefix, D).

        Returns
        -------
        Logits of shape (B, L, V).
        """
        logits = self.core.lm_head(y)
        prefix = getattr(self.core, "puzzle_emb_len", 0)
        if prefix > 0:
            return logits[:, prefix:]
        return logits

    def particle_logits(
        self,
        y_particles: torch.Tensor,
        log_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Blend per-particle logits under particle weights.

        Parameters
        ----------
        y_particles:
            Shape (B, S, L, D).
        log_weights:
            Shape (B, S) — unnormalised log-weights.

        Returns
        -------
        Log-probability tensor of shape (B, L, V).
        """
        B, S, L, D = y_particles.shape
        flat = y_particles.view(B * S, L, D)
        logits_flat = self.forward(flat)
        seq_len, V = logits_flat.shape[1], logits_flat.shape[2]
        logits = logits_flat.view(B, S, seq_len, V)
        log_w = log_weights - log_weights.logsumexp(dim=-1, keepdim=True)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return torch.logsumexp(log_probs + log_w.unsqueeze(-1).unsqueeze(-1), dim=1)


# ---------------------------------------------------------------------------
# TRMCoreQHead
# ---------------------------------------------------------------------------


class TRMCoreQHead(nn.Module):
    """Halting Q-head used as a learned particle score.

    Parameters
    ----------
    core:
        A loaded TRMModule instance.
    """

    def __init__(self, core) -> None:
        super().__init__()
        self.core = core

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Compute raw Q-head logit.

        Matches TRMModule.inner_forward semantics:
          q_logits = self.q_head(z_H[:, 0]).to(float32)[..., 0]

        Parameters
        ----------
        y:
            Hypothesis state, shape (B, L, D).

        Returns
        -------
        Scalar logit per batch element, shape (B,).
        """
        return self.core.q_head(y[:, 0]).to(torch.float32)[..., 0]


# ---------------------------------------------------------------------------
# TRMCoreFTheta
# ---------------------------------------------------------------------------


class TRMCoreFTheta(nn.Module):
    """Single-pass F_theta adapter (one lenet call per update).

    Base class of ``TRMNativeCycleFTheta``, the adapter the paper uses; this
    variant is kept for unit tests. Additive convention:
        belief_update(x, y, z) = lenet(x + y + z)
        state_update(y, z)     = lenet(y + z)

    Parameters
    ----------
    core:
        A loaded TRMModule instance.
    """

    def __init__(self, core) -> None:
        super().__init__()
        self.core = core
        self.forward_dtype = getattr(core, "forward_dtype", None) or torch.float32

    @property
    def layers(self):
        return self.core.lenet.layers

    def _cos_sin(self):
        if hasattr(self.core, "pos_embedding") and self.core.pos_embedding is not None:
            return self.core.pos_embedding()
        return None

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Single lenet pass.

        Parameters
        ----------
        u:
            Combined input tensor, shape (B, L, D).

        Returns
        -------
        Output tensor, shape (B, L, D).
        """
        h = u.to(self.forward_dtype)
        zeros = torch.zeros_like(h)
        cos_sin = self._cos_sin()
        if cos_sin is not None:
            cos, sin = cos_sin
            seq_len = h.shape[1]
            if cos.shape[0] != seq_len:
                cos, sin = cos[:seq_len], sin[:seq_len]
            cos_sin = (cos, sin)
        return self.core.lenet(
            hidden_states=zeros,
            input_injection=h,
            cos_sin=cos_sin,
        )

    def belief_update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """z' = F_theta(x + y + z)."""
        return self.forward(x + y + z)

    def state_update(
        self,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """y' = F_theta(y + z)."""
        return self.forward(y + z)


# ---------------------------------------------------------------------------
# TRMNativeCycleFTheta
# ---------------------------------------------------------------------------


class TRMNativeCycleFTheta(TRMCoreFTheta):
    """Checkpoint-faithful cycle adapter.

    Reproduces the native TRM recurrence exactly:
      - belief_update: L_cycles lenet passes (low-level latent update)
      - state_update:  1 lenet pass          (high-level hypothesis update)

    This is the adapter used throughout the paper: with S = 1 and sigma = 0 the
    filter reproduces the deterministic TRM rollout exactly.
    """

    @property
    def n_inner_steps(self) -> int:
        """Number of fast inner lenet passes in belief_update (= L_cycles)."""
        return int(getattr(self.core.hparams, "L_cycles", 6))

    def _belief_update_single_step(
        self,
        injection: torch.Tensor,
        z: torch.Tensor,
        cos_sin,
    ) -> torch.Tensor:
        """Single inner lenet pass: z' = lenet(z, injection).

        Used by StochasticFTheta when per_step_inner_noise=True so that
        noise can be injected between each of the L_cycles passes.
        """
        return self.core.lenet(
            hidden_states=z,
            input_injection=injection,
            cos_sin=cos_sin,
        )

    def belief_update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """z' = L_cycles applications of lenet with input_injection=(x+y)."""
        cos_sin = self._cos_sin()
        z_next = z.to(self.forward_dtype)
        injection = (x + y).to(self.forward_dtype)
        for _ in range(self.n_inner_steps):
            z_next = self._belief_update_single_step(injection, z_next, cos_sin)
        return z_next

    def state_update(
        self,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """y' = one lenet pass with hidden=y, input_injection=z."""
        return self.core.lenet(
            hidden_states=y.to(self.forward_dtype),
            input_injection=z.to(self.forward_dtype),
            cos_sin=self._cos_sin(),
        )
