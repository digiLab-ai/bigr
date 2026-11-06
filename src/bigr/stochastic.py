"""`StochasticFTheta`: the stochastic proposal kernel of Algorithm 1.

Wraps a deterministic backbone adapter with isotropic mean-zero noise of scale
``sigma`` (Gaussian by default; Laplace and Student-t are variance-matched):

    zeta_{m+1} = F_theta(x + y + zeta_m) + sigma * xi_m,  m = 0, ..., M-1   (inner recursion)
    y'         = F_theta(y + zeta_M)     + sigma * xi_M                     (read-out step)

With ``per_step_inner_noise=True`` (the default and the paper's kernel) the noise
enters after each of the M inner passes of ``TRMNativeCycleFTheta``; with
``False`` it is drawn once after the inner loop. ``sigma_inner`` and
``sigma_trans`` give the two updates different scales. ``sigma = 0`` recovers the
deterministic adapter exactly.
"""

from __future__ import annotations

import torch

from bigr.backbone import TRMCoreFTheta, TRMNativeCycleFTheta


NOISE_FAMILIES = ("gaussian", "laplace", "student_t")


def _unit_noise(
    shape: torch.Size,
    family: str,
    *,
    generator: torch.Generator | None,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw unit-variance noise of the given family.

    All families are scaled to E||xi||^2 = dim (unit per-coordinate variance),
    so ``sigma`` retains the same second-moment meaning across families:
    laplace uses scale b = 1/sqrt(2); student_t uses df = 3 divided by sqrt(3).
    """
    kwargs = {"generator": generator, "device": device, "dtype": dtype}
    if family == "gaussian":
        return torch.randn(shape, **kwargs)
    if family == "laplace":
        # Inverse-CDF transform in float32, then cast: in bf16/fp16 |u| can round to
        # 0.5, where log1p(-2|u|) is -inf.
        b = 1.0 / (2.0**0.5)
        u = torch.rand(shape, generator=generator, device=device, dtype=torch.float32) - 0.5
        x = -b * torch.sign(u) * torch.log1p((-2.0 * u.abs()).clamp(min=-1.0 + 1e-6))
        return x.to(dtype)
    if family == "student_t":
        df = 3
        z = torch.randn(shape, **kwargs)
        chi2 = sum(torch.randn(shape, **kwargs) ** 2 for _ in range(df))
        return z / torch.sqrt(chi2 / df) / (df / (df - 2)) ** 0.5
    raise ValueError(f"Unknown noise family: {family!r}")


def _add_noise(
    out: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None,
    family: str = "gaussian",
) -> torch.Tensor:
    """Add isotropic mean-zero noise of the given family if sigma > 0.

    For CUDA tensors, noise is drawn directly on-device to avoid CPU->GPU
    transfer overhead in the inner filter loop.

    Parameters
    ----------
    out:
        Deterministic output tensor — the noise centre.
    sigma:
        Noise scale (per-coordinate standard deviation for every family).
        If 0.0, returns ``out`` unchanged.
    generator:
        Seeded CPU torch.Generator. Pass None for a non-reproducible draw.
    family:
        One of NOISE_FAMILIES; all variance-matched (see _unit_noise).

    Returns
    -------
    Tensor with the same shape and device as ``out``.
    """
    if sigma == 0.0:
        return out

    gen_device = getattr(generator, "device", None) if generator is not None else None
    gen_device_type = (
        torch.device(gen_device).type if gen_device is not None else None
    )

    if out.device.type in {"cuda", "mps"}:
        # Prefer on-device sampling for accelerator paths. If the provided
        # generator is on the same device, use it; otherwise use device RNG.
        if generator is not None and gen_device_type == out.device.type:
            noise = _unit_noise(
                out.shape, family, generator=generator, device=out.device, dtype=out.dtype
            )
        else:
            noise = _unit_noise(
                out.shape, family, generator=None, device=out.device, dtype=out.dtype
            )
    else:
        noise = _unit_noise(
            out.shape, family, generator=generator, device=None, dtype=out.dtype
        ).to(out.device)
    return out + sigma * noise


class StochasticFTheta:
    """Wraps a deterministic F_theta adapter with isotropic noise.

    Parameters
    ----------
    f_theta:
        A deterministic backbone adapter exposing ``belief_update(x, y, z)``
        and ``state_update(y, z)`` (``TRMNativeCycleFTheta`` or ``TRMCoreFTheta``).
    sigma:
        Shared standard deviation for the belief (z) and state (y) updates.
        0.0 is deterministic.
    sigma_inner:
        Separate scale for the belief update; overrides ``sigma``.
    sigma_trans:
        Separate scale for the state update; overrides ``sigma``.
    per_step_inner_noise:
        If True and ``f_theta`` is a ``TRMNativeCycleFTheta``, noise is injected
        after each of the M inner passes, as in Algorithm 1; if False, once
        after the inner loop. No effect for the single-pass ``TRMCoreFTheta``
        or when ``sigma == 0``.
    noise_family:
        One of ``NOISE_FAMILIES``; every family has unit variance per coordinate.

    Examples
    --------
    sf = StochasticFTheta(f, sigma=0.0)   # deterministic TRM
    sf = StochasticFTheta(f, sigma=0.3)   # the paper's Sudoku-Extreme setting
    """

    def __init__(
        self,
        f_theta: TRMCoreFTheta,
        sigma: float | None = None,
        sigma_inner: float | None = None,
        sigma_trans: float | None = None,
        per_step_inner_noise: bool = True,
        noise_family: str = "gaussian",
    ) -> None:
        if noise_family not in NOISE_FAMILIES:
            raise ValueError(f"noise_family must be one of {NOISE_FAMILIES}")
        self.noise_family = noise_family
        self.f_theta = f_theta
        if sigma is not None:
            shared_sigma = float(sigma)
            if sigma_inner is None:
                sigma_inner = shared_sigma
            if sigma_trans is None:
                sigma_trans = shared_sigma
        else:
            if sigma_inner is None and sigma_trans is None:
                sigma_inner = 0.0
                sigma_trans = 0.0
            else:
                if sigma_inner is None:
                    sigma_inner = 0.0
                if sigma_trans is None:
                    sigma_trans = 0.0

        self.sigma_inner = float(sigma_inner)
        self.sigma_trans = float(sigma_trans)
        self.sigma = (
            self.sigma_inner
            if self.sigma_inner == self.sigma_trans
            else None
        )
        self.per_step_inner_noise = per_step_inner_noise

    def belief_update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """z' ~ F_theta(x + y + z) + sigma * eps.

        When per_step_inner_noise=True and the backbone is TRMNativeCycleFTheta,
        noise is injected after each individual inner lenet pass:
            for k in range(L_cycles):
                z = lenet(z, x+y) + sigma * eps_k
        """
        if (
            self.per_step_inner_noise
            and isinstance(self.f_theta, TRMNativeCycleFTheta)
            and self.sigma_inner > 0.0
        ):
            cos_sin = self.f_theta._cos_sin()
            injection = (x + y).to(self.f_theta.forward_dtype)
            z_next = z.to(self.f_theta.forward_dtype)
            for _ in range(self.f_theta.n_inner_steps):
                z_next = self.f_theta._belief_update_single_step(
                    injection, z_next, cos_sin
                )
                z_next = _add_noise(z_next, self.sigma_inner, generator, self.noise_family)
            return z_next
        else:
            z_det = self.f_theta.belief_update(x, y, z)
            return _add_noise(z_det, self.sigma_inner, generator, self.noise_family)

    def state_update(
        self,
        y: torch.Tensor,
        z: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """y' ~ F_theta(y + z) + sigma * eps."""
        y_det = self.f_theta.state_update(y, z)
        return _add_noise(y_det, self.sigma_trans, generator, self.noise_family)
