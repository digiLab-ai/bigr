"""Configuration of the guided particle filter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Configuration for the guided particle filter.

    Parameters
    ----------
    particles:
        Number of particles S.
    total_steps:
        Total number of outer steps N.
    sigma:
        Shared standard deviation applied to both belief and state updates.
        This shorthand is equivalent to setting both ``sigma_inner`` and
        ``sigma_trans`` to the same value.
    sigma_inner:
        Standard deviation for the belief-update noise.
    sigma_trans:
        Standard deviation for the state-update noise.
    ess_threshold:
        Resample when ESS < ess_threshold * S. Typical value 0.3.
    seed:
        Random seed for the particle generator.
    checkpoint_steps:
        Steps at which to snapshot full (B, S) weight and Q-logit arrays.
        If None, defaults to supervision cycle boundaries + final step.
    """

    particles: int = 16
    total_steps: int = 48
    sigma: float | None = 0.3
    sigma_inner: float | None = None
    sigma_trans: float | None = None
    ess_threshold: float = 0.3
    seed: int = 31
    checkpoint_steps: list[int] | None = None
    beta: float = 0.25
    """Inverse temperature of the Q-head tilt, log_w += beta * logsigmoid(Q(y')).
    beta = 0 leaves the weights uniform (unguided)."""
    sigma_rejuvenate: float = 0.0
    """Standard deviation of Gaussian jitter added to y after resampling.
    0.0 (the paper's setting) disables it."""
    save_hidden_states: bool = False
    """Save per-particle y and z tensors at checkpoint steps (float16).
    Required for guide-value alignment experiments. Increases memory use."""
    save_particle_details: bool = False
    """Save original-order particle arrays and resampling ancestry.
    Required for particle-level replay experiments. Increases memory use."""
    collect_step_cpu_metrics: bool = True
    """If True, materialize per-step diagnostics on CPU (numpy/scalars).
    Disable to reduce GPU<->CPU synchronization overhead during long runs."""

    def __post_init__(self) -> None:
        self._normalise_noise()

    def _normalise_noise(self) -> None:
        """Resolve the shared ``sigma`` shorthand into ``sigma_inner`` and ``sigma_trans``."""
        if self.sigma is not None:
            shared_sigma = float(self.sigma)
            if self.sigma_inner is None:
                self.sigma_inner = shared_sigma
            if self.sigma_trans is None:
                self.sigma_trans = shared_sigma
        else:
            if self.sigma_inner is None and self.sigma_trans is None:
                self.sigma_inner = 0.3
                self.sigma_trans = 0.3
            else:
                if self.sigma_inner is None:
                    self.sigma_inner = 0.0
                if self.sigma_trans is None:
                    self.sigma_trans = 0.0

        self.sigma_inner = float(self.sigma_inner)
        self.sigma_trans = float(self.sigma_trans)
        self.sigma = (
            self.sigma_inner if self.sigma_inner == self.sigma_trans else None
        )

    def resolve_checkpoint_steps(self, h_cycles: int) -> list[int]:
        """Return supervision boundary steps if checkpoint_steps is None."""
        if self.checkpoint_steps is not None:
            return list(self.checkpoint_steps)
        n_sup = self.total_steps // h_cycles
        return [
            k * h_cycles
            for k in range(1, n_sup + 1)
            if k * h_cycles <= self.total_steps
        ]
