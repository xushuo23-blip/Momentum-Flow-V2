from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Data.train_dataset import build_train_dataset
from Model.networks import build_network, count_parameters
from Training.latent_vae import FrozenLatentVAE, build_latent_vae
from Training.loss import compute_training_losses
from Training.preview_sampling import sample_preview_reverse_kinetic
from Training.text_encoder import FrozenTextEncoder
from configs import load_config


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split(".")[-1])
    except ValueError:
        return -1


def save_model_checkpoint(
    output_dir: Path,
    *,
    step: int,
    r_net: nn.Module,
    score_net: nn.Module,
    cfg: dict,
    keep_last: int,
) -> Path:
    path = output_dir / f"checkpoint.{step}.pt"
    torch.save(
        {
            "step": step,
            "config": cfg,
            "r_net": r_net.state_dict(),
            "score_net": score_net.state_dict(),
        },
        path,
    )
    checkpoints = sorted(output_dir.glob("checkpoint.*.pt"), key=checkpoint_step)
    for old_path in checkpoints[: max(0, len(checkpoints) - keep_last)]:
        old_path.unlink()
    return path


def image_grid(samples: torch.Tensor, max_images: int = 4) -> torch.Tensor:
    samples = (samples.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5
    samples = samples[:max_images]
    n, c, h, w = samples.shape
    cols = min(n, max_images)
    rows = math.ceil(n / cols)
    grid = torch.zeros(c, rows * h, cols * w)
    for idx, image in enumerate(samples):
        row = idx // cols
        col = idx % cols
        grid[:, row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    return grid.permute(1, 2, 0).numpy()


@torch.no_grad()
def log_preview_images(
    wandb,
    *,
    step: int,
    cfg: dict,
    r_net: nn.Module,
    score_net: nn.Module,
    latent_vae: FrozenLatentVAE | None,
    prompt_id: torch.Tensor | None,
    text_embed: torch.Tensor | None,
    text_tokens: torch.Tensor | None,
    text_mask: torch.Tensor | None,
    prompts: list[str],
    device: torch.device,
) -> None:
    if wandb is None or not cfg.get("preview", {}).get("enabled", False):
        return

    preview_cfg = cfg["preview"]
    kinetic_cfg = cfg["kinetic"]
    model_cfg = cfg["model"]
    batch_size = int(preview_cfg.get("batch_size", 4))
    if text_embed is not None:
        batch_size = min(batch_size, text_embed.shape[0])
        text_embed = text_embed[:batch_size]
    if text_tokens is not None:
        batch_size = min(batch_size, text_tokens.shape[0])
        text_tokens = text_tokens[:batch_size]
    if text_mask is not None:
        text_mask = text_mask[:batch_size]
    if prompt_id is not None:
        batch_size = min(batch_size, prompt_id.shape[0])
        prompt_id = prompt_id[:batch_size]
    if batch_size <= 0:
        return

    was_training = (r_net.training, score_net.training)
    r_net.eval()
    score_net.eval()
    samples = sample_preview_reverse_kinetic(
        r_net,
        score_net,
        batch_size=batch_size,
        image_size=model_cfg["image_size"],
        channels=model_cfg.get("out_channels", 3),
        steps=int(preview_cfg.get("steps", 30)),
        tau=float(preview_cfg.get("tau", kinetic_cfg.get("tau_max", kinetic_cfg.get("tau_min", 0.05)))),
        eta=float(preview_cfg.get("eta", 0.0)),
        lambda_const=float(kinetic_cfg.get("lambda_const", 2.0)),
        rho=float(kinetic_cfg.get("rho", 2.0)),
        num_quad=int(kinetic_cfg.get("num_quad", 128)),
        prompt_id=prompt_id,
        text_embed=text_embed,
        text_tokens=text_tokens,
        text_mask=text_mask,
        score_scale_override=preview_cfg.get("score_scale_override"),
        clamp_output=latent_vae is None,
        device=device,
    )
    if latent_vae is not None:
        samples = latent_vae.decode(samples)
    if was_training[0]:
        r_net.train()
    if was_training[1]:
        score_net.train()

    caption = prompts[0] if prompts else f"step {step}"
    wandb.log({"preview/images": wandb.Image(image_grid(samples, batch_size), caption=caption)}, step=step)


def _build_optimizer(cfg_opt: dict, params: list) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params,
        lr=cfg_opt["lr"],
        betas=tuple(cfg_opt["betas"]),
        weight_decay=cfg_opt["weight_decay"],
        eps=cfg_opt["eps"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_training.py")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_train_dataset(cfg["dataset"])
    latent_vae = build_latent_vae(cfg.get("latent", {}), device)
    if latent_vae is not None:
        latent_shape = latent_vae.infer_shape(int(cfg["dataset"]["image_size"]))
        cfg["model"]["image_size"] = latent_shape.image_size
        cfg["model"]["out_channels"] = latent_shape.channels
        cfg["model"]["in_channels"] = latent_shape.channels * 2
        cfg.setdefault("latent", {})["latent_channels"] = latent_shape.channels
        cfg["latent"]["latent_image_size"] = latent_shape.image_size
        cfg["latent"]["vae_scale_factor"] = latent_shape.vae_scale_factor

    text_encoder = None
    text_cfg = cfg.get("text_encoder", {})
    use_text_encoder = bool(text_cfg.get("enabled", False))
    use_prompt_id = bool(cfg.get("conditioning", {}).get("use_prompt_id", not use_text_encoder))

    if hasattr(dataset, "num_prompts") and use_prompt_id:
        cfg["model"]["num_prompts"] = dataset.num_prompts
    if use_text_encoder:
        if not hasattr(dataset, "id_to_prompt"):
            raise ValueError("text_encoder.enabled requires a dataset with prompt text")
        text_encoder = FrozenTextEncoder(
            model_name=text_cfg.get("model_name", "openai/clip-vit-base-patch32"),
            max_length=text_cfg.get("max_length"),
        ).to(device)
        cfg["model"]["text_embed_dim"] = text_encoder.out_dim
        cfg["model"]["text_token_dim"] = text_encoder.out_dim
        cfg["text_encoder"]["out_dim"] = text_encoder.out_dim

    use_wandb = bool(cfg.get("wandb", {}).get("enabled", False)) and not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise ImportError("wandb.enabled=True requires `pip install wandb` or run with --no-wandb") from exc

        wandb = wandb_module
        wandb_kwargs = {
            "project": cfg["wandb"]["project"],
            "name": cfg["wandb"]["run_name"],
            "config": cfg,
            "mode": cfg["wandb"].get("mode", "online"),
        }
        if cfg["wandb"].get("entity"):
            wandb_kwargs["entity"] = cfg["wandb"]["entity"]
        if cfg["wandb"].get("dir"):
            wandb_kwargs["dir"] = cfg["wandb"]["dir"]
        if cfg["wandb"].get("tags"):
            wandb_kwargs["tags"] = list(cfg["wandb"]["tags"])
        if cfg["wandb"].get("notes"):
            wandb_kwargs["notes"] = cfg["wandb"]["notes"]
        try:
            wandb.init(**wandb_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "wandb initialization failed. If mode='online', run `wandb login` first "
                "or set WANDB_API_KEY outside the code. You can also set wandb.mode='offline' "
                "or run Training/train.py with --no-wandb."
            ) from exc

    dl = DataLoader(
        dataset,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=True,
        drop_last=bool(cfg["dataset"].get("drop_last", True)),
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
    )
    if len(dl) == 0:
        raise ValueError(
            "dataloader has zero batches; reduce dataset.batch_size, set dataset.drop_last=False, "
            "or provide more training images"
        )

    r_net = build_network(cfg["model"]).to(device)
    score_net = build_network(cfg["model"]).to(device)
    print(f"r_net params: {count_parameters(r_net) / 1e6:.2f}M")
    print(f"score_net params: {count_parameters(score_net) / 1e6:.2f}M")
    text_params = count_parameters(text_encoder) if text_encoder is not None else 0
    print(f"text_encoder trainable params: {text_params / 1e6:.2f}M")
    print(f"total trainable params: {(count_parameters(r_net) + count_parameters(score_net) + text_params) / 1e6:.2f}M")

    optimizer_r = _build_optimizer(cfg["optimizer_r"], r_net.parameters())
    optimizer_s = _build_optimizer(cfg["optimizer_s"], score_net.parameters())

    output_dir = Path(cfg["train"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grad_clip = cfg["train"]["grad_clip_norm"]
    step = 0
    while step < cfg["train"]["steps"]:
        for batch in dl:
            prompt_id = None
            text_embed = None
            text_tokens = None
            text_mask = None
            prompt_text = []
            if isinstance(batch, dict):
                x0 = batch["image"]
                prompt_text = list(batch.get("prompt_text", []))
                if use_prompt_id:
                    prompt_id = batch["prompt_id"].to(device, non_blocking=True)
                if text_encoder is not None:
                    text_encoded = text_encoder.encode(prompt_text, device)
                    text_embed = text_encoded["pooled"]
                    if cfg.get("text_encoder", {}).get("return_tokens", True):
                        text_tokens = text_encoded["tokens"]
                        text_mask = text_encoded["mask"]
            else:
                x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
            x0 = x0.to(device, non_blocking=True)
            if latent_vae is not None:
                x0 = latent_vae.encode(x0).float()

            # --- compute both losses on the same sampled state ---
            loss_r, loss_s, logs = compute_training_losses(
                r_net,
                score_net,
                x0,
                prompt_id=prompt_id,
                text_embed=text_embed,
                text_tokens=text_tokens,
                text_mask=text_mask,
                **cfg["kinetic"],
            )

            # --- r_net update ---
            optimizer_r.zero_grad(set_to_none=True)
            loss_r.backward(retain_graph=True)
            grad_norm_r = torch.nn.utils.clip_grad_norm_(r_net.parameters(), grad_clip)
            optimizer_r.step()

            # --- score_net update ---
            optimizer_s.zero_grad(set_to_none=True)
            loss_s.backward()
            grad_norm_s = torch.nn.utils.clip_grad_norm_(score_net.parameters(), grad_clip)
            optimizer_s.step()

            logs["grad_norm_r"] = grad_norm_r.detach()
            logs["grad_norm_s"] = grad_norm_s.detach()

            step += 1
            if step % cfg["train"]["log_every"] == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in logs.items())
                print(f"[{step}] {msg}")
                if wandb is not None:
                    wandb.log({k: v.item() for k, v in logs.items()}, step=step)

            if step % cfg.get("preview", {}).get("every", 10**12) == 0:
                log_preview_images(
                    wandb,
                    step=step,
                    cfg=cfg,
                    r_net=r_net,
                    score_net=score_net,
                    latent_vae=latent_vae,
                    prompt_id=prompt_id,
                    text_embed=text_embed,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                    prompts=prompt_text,
                    device=device,
                )

            if step % cfg["train"]["save_every"] == 0:
                ckpt_path = save_model_checkpoint(
                    output_dir,
                    step=step,
                    r_net=r_net,
                    score_net=score_net,
                    cfg=cfg,
                    keep_last=int(cfg["train"].get("keep_last_checkpoints", 5)),
                )
                print(f"saved {ckpt_path}")

            if step >= cfg["train"]["steps"]:
                break

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
