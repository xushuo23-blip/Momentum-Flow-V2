from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class LatentShape:
    channels: int
    image_size: int
    vae_scale_factor: int


class FrozenLatentVAE(nn.Module):
    """Frozen SD-style VAE wrapper used by training."""

    def __init__(
        self,
        model_name: str,
        *,
        subfolder: str | None = "vae",
        sample: bool = False,
        dtype: str = "float32",
    ):
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise ImportError("latent.enabled=True requires `pip install diffusers accelerate safetensors`") from exc

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        kwargs = {"torch_dtype": torch_dtype}
        if subfolder:
            kwargs["subfolder"] = subfolder
        try:
            self.vae = AutoencoderKL.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load VAE from {model_name!r}. If this is an SD3 gated model, "
                "accept the Hugging Face license and run `huggingface-cli login`, or use a local model path."
            ) from exc
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad_(False)

        config = self.vae.config
        self.sample = sample
        self.latent_channels = int(getattr(config, "latent_channels", 4))
        block_out_channels = getattr(config, "block_out_channels", (1, 1, 1, 1))
        self.vae_scale_factor = 2 ** (len(block_out_channels) - 1)
        self.scaling_factor = float(getattr(config, "scaling_factor", 1.0))
        self.shift_factor = float(getattr(config, "shift_factor", 0.0))

    def infer_shape(self, image_size: int) -> LatentShape:
        if image_size % self.vae_scale_factor != 0:
            raise ValueError(f"dataset.image_size={image_size} must be divisible by VAE scale {self.vae_scale_factor}")
        return LatentShape(
            channels=self.latent_channels,
            image_size=image_size // self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        dtype = next(self.vae.parameters()).dtype
        images = images.to(dtype=dtype)
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() if self.sample else posterior.mode()
        return (latents - self.shift_factor) * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=dtype) / self.scaling_factor + self.shift_factor
        images = self.vae.decode(latents).sample
        return images.float().clamp(-1.0, 1.0)


def build_latent_vae(cfg: dict, device: torch.device) -> FrozenLatentVAE | None:
    if not cfg or not cfg.get("enabled", False):
        return None
    vae = FrozenLatentVAE(
        cfg["vae_model"],
        subfolder=cfg.get("subfolder", "vae"),
        sample=bool(cfg.get("sample", False)),
        dtype=cfg.get("dtype", "float32"),
    )
    return vae.to(device)
