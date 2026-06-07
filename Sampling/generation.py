from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from Model.networks import build_network
from Sampling.latent_vae import FrozenLatentVAE, build_latent_vae
from Sampling.reverse import sample_reverse_kinetic
from Sampling.text_encoder import FrozenTextEncoder


@dataclass
class GeneratorBundle:
    r_net: nn.Module
    score_net: nn.Module
    text_encoder: FrozenTextEncoder | None
    latent_vae: FrozenLatentVAE | None
    config: dict
    device: torch.device


def expand_prompts(prompts: list[str], batch_size: int) -> list[str]:
    if len(prompts) == 1:
        return prompts * batch_size
    if len(prompts) != batch_size:
        raise ValueError("--prompt must be supplied once or exactly batch-size times")
    return prompts


def load_generator(
    checkpoint_path: str,
    *,
    config: dict | None = None,
    device: torch.device | None = None,
) -> GeneratorBundle:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = config or ckpt.get("config")
    if cfg is None:
        raise ValueError("checkpoint does not contain config; pass the training config explicitly")
    model_cfg = dict(cfg["model"])
    latent_vae = build_latent_vae(cfg.get("latent", {}), device)
    if latent_vae is not None:
        latent_shape = latent_vae.infer_shape(int(cfg["dataset"]["image_size"]))
        model_cfg["image_size"] = latent_shape.image_size
        model_cfg["out_channels"] = latent_shape.channels
        model_cfg["in_channels"] = latent_shape.channels * 2

    text_encoder = None
    text_cfg = cfg.get("text_encoder", {})
    if text_cfg.get("enabled", False):
        text_encoder = FrozenTextEncoder(
            model_name=text_cfg.get("model_name", "openai/clip-vit-base-patch32"),
            max_length=text_cfg.get("max_length"),
        ).to(device)
        text_encoder.eval()
        if model_cfg.get("text_embed_dim", 0) <= 0:
            model_cfg["text_embed_dim"] = text_encoder.out_dim
        if model_cfg.get("text_token_dim", 0) <= 0:
            model_cfg["text_token_dim"] = text_encoder.out_dim

    r_net = build_network(model_cfg).to(device)
    score_net = build_network(model_cfg).to(device)
    r_net.load_state_dict(ckpt["r_net"])
    score_net.load_state_dict(ckpt["score_net"])
    r_net.eval()
    score_net.eval()

    return GeneratorBundle(
        r_net=r_net,
        score_net=score_net,
        text_encoder=text_encoder,
        latent_vae=latent_vae,
        config=cfg,
        device=device,
    )


@torch.no_grad()
def encode_generation_prompts(bundle: GeneratorBundle, prompts: list[str], batch_size: int) -> dict[str, Tensor] | None:
    if bundle.text_encoder is None:
        return None
    prompts = expand_prompts(prompts, batch_size)
    return bundle.text_encoder.encode(prompts, bundle.device)


@torch.no_grad()
def generate_samples(
    bundle: GeneratorBundle,
    *,
    prompts: list[str],
    batch_size: int = 16,
    steps: int = 100,
    tau: float | None = None,
    eta: float = 0.0,
    prompt_id: int | None = None,
    score_scale_override: float | None = None,
) -> Tensor:
    model_cfg = dict(bundle.config["model"])
    kinetic_cfg = bundle.config["kinetic"]

    prompt_id_tensor = None
    if model_cfg.get("num_prompts", 0) > 0:
        if prompt_id is None:
            raise ValueError("prompt_id is required for prompt-id conditioned checkpoints")
        prompt_id_tensor = torch.full((batch_size,), prompt_id, dtype=torch.long, device=bundle.device)

    text_embed = None
    text_tokens = None
    text_mask = None
    if bundle.text_encoder is not None:
        if not prompts:
            raise ValueError("prompts are required for text-conditioned checkpoints")
        text_encoded = encode_generation_prompts(bundle, prompts, batch_size)
        text_embed = text_encoded["pooled"]
        if bundle.config.get("text_encoder", {}).get("return_tokens", True):
            text_tokens = text_encoded["tokens"]
            text_mask = text_encoded["mask"]

    tau = tau if tau is not None else float(kinetic_cfg.get("tau_max", kinetic_cfg.get("tau_min", 0.05)))
    samples = sample_reverse_kinetic(
        bundle.r_net,
        bundle.score_net,
        batch_size=batch_size,
        image_size=model_cfg["image_size"],
        channels=model_cfg.get("out_channels", 3),
        steps=steps,
        tau=tau,
        eta=eta,
        lambda_const=kinetic_cfg.get("lambda_const", 2.0),
        rho=kinetic_cfg.get("rho", 2.0),
        num_quad=kinetic_cfg.get("num_quad", 128),
        prompt_id=prompt_id_tensor,
        text_embed=text_embed,
        text_tokens=text_tokens,
        text_mask=text_mask,
        score_scale_override=score_scale_override,
        clamp_output=bundle.latent_vae is None,
        device=bundle.device,
    )
    if bundle.latent_vae is not None:
        samples = bundle.latent_vae.decode(samples)
    return samples


@torch.no_grad()
def generate_samples_from_checkpoint(
    checkpoint_path: str,
    *,
    config: dict | None = None,
    prompts: list[str],
    batch_size: int = 16,
    steps: int = 100,
    tau: float | None = None,
    eta: float = 0.0,
    prompt_id: int | None = None,
    score_scale_override: float | None = None,
    device: torch.device | None = None,
) -> Tensor:
    bundle = load_generator(checkpoint_path, config=config, device=device)
    return generate_samples(
        bundle,
        prompts=prompts,
        batch_size=batch_size,
        steps=steps,
        tau=tau,
        eta=eta,
        prompt_id=prompt_id,
        score_scale_override=score_scale_override,
    )
