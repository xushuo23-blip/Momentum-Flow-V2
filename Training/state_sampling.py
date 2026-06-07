from __future__ import annotations

import torch
from torch import Tensor

from Training.schedules import c_prime_t, c_t, scalar_covariances


def append_dims(x: Tensor, target_ndim: int) -> Tensor:
    return x.reshape(*x.shape, *((1,) * (target_ndim - x.ndim)))


def sample_tau(batch: int, device: torch.device, tau_min: float, tau_max: float) -> Tensor:
    log_min = torch.log(torch.tensor(tau_min, device=device))
    log_max = torch.log(torch.tensor(tau_max, device=device))
    return torch.exp(torch.rand(batch, device=device) * (log_max - log_min) + log_min)


def sample_conditional_state(
    x0: Tensor,
    *,
    tau_min: float = 0.02,
    tau_max: float = 0.2,
    t_eps: float = 1e-3,
    lambda_const: float = 2.0,
    rho: float = 2.0,
    num_quad: int = 128,
) -> dict[str, Tensor]:
    """Sample y_t=(x_t,v_t) from the conditional kinetic Gaussian flow."""

    batch = x0.shape[0]
    device = x0.device
    t = torch.rand(batch, device=device) * (1.0 - 2.0 * t_eps) + t_eps
    tau = sample_tau(batch, device, tau_min, tau_max)
    eps = torch.randn_like(x0)
    r = eps - x0

    c = append_dims(c_t(t), x0.ndim)
    cp = append_dims(c_prime_t(t), x0.ndim)
    mu_x = (1.0 - c) * x0 + c * eps
    mu_v = cp * r

    s_var, c_cov, q_var = scalar_covariances(
        t,
        tau,
        lambda_const=lambda_const,
        rho=rho,
        num_quad=num_quad,
    )

    s_b = append_dims(s_var, x0.ndim)
    c_b = append_dims(c_cov, x0.ndim)
    q_b = append_dims(q_var, x0.ndim)

    z_v = torch.randn_like(x0)
    z_x = torch.randn_like(x0)
    eta = q_b.clamp_min(1e-12).sqrt() * z_v
    xi_std = (s_b - c_b.square() / q_b.clamp_min(1e-12)).clamp_min(1e-12).sqrt()
    xi = c_b / q_b.clamp_min(1e-12).sqrt() * z_v + xi_std * z_x

    x_t = mu_x + xi
    v_t = mu_v + eta

    return {
        "x_t": x_t,
        "v_t": v_t,
        "t": t,
        "tau": tau,
        "r": r,
        "mu_x": mu_x,
        "mu_v": mu_v,
        "S": s_var,
        "C": c_cov,
        "Q": q_var,
    }
