from __future__ import annotations

import torch
from torch import Tensor, nn

from Sampling.schedules import append_dims, beta_t, lambda_t, scalar_covariances, sigma_t


@torch.no_grad()
def sample_reverse_kinetic(
    r_net: nn.Module,
    score_net: nn.Module,
    *,
    batch_size: int,
    image_size: int,
    channels: int = 3,
    steps: int = 100,
    tau: float = 0.05,
    eta: float = 0.0,
    lambda_const: float = 2.0,
    rho: float = 2.0,
    num_quad: int = 128,
    prompt_id: Tensor | None = None,
    text_embed: Tensor | None = None,
    text_tokens: Tensor | None = None,
    text_mask: Tensor | None = None,
    score_scale_override: float | None = None,
    clamp_output: bool = True,
    device: torch.device | None = None,
) -> Tensor:
    """Generate samples with a reverse kinetic ODE/SDE integrator.

    eta=0 uses the deterministic probability-flow ODE. eta=1 uses the full
    reverse SDE noise level. Values between 0 and 1 interpolate the randomness.
    """

    device = device or next(r_net.parameters()).device
    shape = (batch_size, channels, image_size, image_size)
    x = torch.randn(shape, device=device)

    t_one = torch.ones(batch_size, device=device)
    tau_vec = torch.full((batch_size,), tau, device=device)
    _, _, q_var = scalar_covariances(
        t_one,
        tau_vec,
        lambda_const=lambda_const,
        rho=rho,
        num_quad=num_quad,
    )
    v = append_dims(q_var.clamp_min(1e-12).sqrt(), 4) * torch.randn(shape, device=device)

    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    eta = float(eta)
    for idx in range(steps):
        t_now = times[idx].expand(batch_size)
        t_next = times[idx + 1].expand(batch_size)
        dt = t_next - t_now
        tau_vec = torch.full((batch_size,), tau, device=device)

        r = r_net(
            x,
            v,
            t_now,
            tau_vec,
            prompt_id=prompt_id,
            text_embed=text_embed,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )
        score_v = score_net(
            x,
            v,
            t_now,
            tau_vec,
            prompt_id=prompt_id,
            text_embed=text_embed,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )

        lam = append_dims(lambda_t(t_now, lambda_const), 4)
        beta = append_dims(beta_t(t_now, lambda_const), 4)
        sigma = append_dims(sigma_t(t_now, tau_vec, lambda_const=lambda_const, rho=rho), 4)
        sigma_sq = sigma.square()

        score_scale = 0.5 * (1.0 + eta * eta) if score_scale_override is None else float(score_scale_override)
        drift_x = v
        drift_v = beta * r - lam * v - score_scale * sigma_sq * score_v

        x = x + append_dims(dt, 4) * drift_x
        v = v + append_dims(dt, 4) * drift_v
        if eta > 0.0:
            v = v + eta * sigma * (-append_dims(dt, 4)).clamp_min(0.0).sqrt() * torch.randn_like(v)

    return x.clamp(-1.0, 1.0) if clamp_output else x
