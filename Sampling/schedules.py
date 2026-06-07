from __future__ import annotations

import torch
from torch import Tensor


def append_dims(x: Tensor, target_ndim: int) -> Tensor:
    return x.reshape(*x.shape, *((1,) * (target_ndim - x.ndim)))


def c_prime_t(t: Tensor) -> Tensor:
    return 2.0 - 2.0 * t


def c_double_prime_t(t: Tensor) -> Tensor:
    return torch.full_like(t, -2.0)


def lambda_t(t: Tensor, lambda_const: float = 2.0) -> Tensor:
    return torch.full_like(t, lambda_const)


def beta_t(t: Tensor, lambda_const: float = 2.0) -> Tensor:
    return c_double_prime_t(t) + lambda_t(t, lambda_const) * c_prime_t(t)


def g_t(t: Tensor, rho: float = 2.0) -> Tensor:
    return t.clamp_min(0.0).pow(rho)


def g_prime_t(t: Tensor, rho: float = 2.0) -> Tensor:
    safe_t = t.clamp_min(1e-8)
    return rho * safe_t.pow(rho - 1.0)


def sigma_t(t: Tensor, tau: Tensor, lambda_const: float = 2.0, rho: float = 2.0) -> Tensor:
    budget_derivative = g_prime_t(t, rho) + 2.0 * lambda_t(t, lambda_const) * g_t(t, rho)
    return tau.sqrt() * budget_derivative.clamp_min(1e-12).sqrt()


def phi(t: Tensor, s: Tensor, lambda_const: float = 2.0) -> Tensor:
    return torch.exp(-lambda_const * (t - s))


def k_kernel(t: Tensor, s: Tensor, lambda_const: float = 2.0) -> Tensor:
    if abs(lambda_const) < 1e-8:
        return t - s
    return (1.0 - torch.exp(-lambda_const * (t - s))) / lambda_const


def scalar_covariances(
    t: Tensor,
    tau: Tensor,
    *,
    lambda_const: float = 2.0,
    rho: float = 2.0,
    num_quad: int = 128,
) -> tuple[Tensor, Tensor, Tensor]:
    batch = t.shape[0]
    u = torch.linspace(0.0, 1.0, num_quad, device=t.device, dtype=t.dtype)
    s = t[:, None] * u[None, :]
    t_grid = t[:, None].expand(batch, num_quad)
    tau_grid = tau[:, None].expand(batch, num_quad)

    sigma_sq = sigma_t(s, tau_grid, lambda_const=lambda_const, rho=rho).square()
    phi_vals = phi(t_grid, s, lambda_const=lambda_const)
    k_vals = k_kernel(t_grid, s, lambda_const=lambda_const)

    integrand_q = phi_vals.square() * sigma_sq
    integrand_s = k_vals.square() * sigma_sq
    integrand_c = k_vals * phi_vals * sigma_sq

    q = torch.trapezoid(integrand_q, u, dim=-1) * t
    s_var = torch.trapezoid(integrand_s, u, dim=-1) * t
    c_cov = torch.trapezoid(integrand_c, u, dim=-1) * t
    return s_var, c_cov, q
