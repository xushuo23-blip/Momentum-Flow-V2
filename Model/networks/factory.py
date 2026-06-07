from __future__ import annotations

import inspect

from torch import nn

from .mmdit import MomentumMMDiT


def build_network(cfg: dict) -> nn.Module:
    """Build a MomentumMMDiT backbone for latent-space Momentum Flow."""
    cfg = dict(cfg)
    cfg.pop("name", None)  # name is informational only
    return MomentumMMDiT(**_filter_kwargs(MomentumMMDiT, cfg))


def _filter_kwargs(cls: type[nn.Module], cfg: dict) -> dict:
    allowed = set(inspect.signature(cls.__init__).parameters)
    allowed.discard("self")
    return {key: value for key, value in cfg.items() if key in allowed}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
