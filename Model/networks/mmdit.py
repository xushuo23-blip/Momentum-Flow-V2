from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class TimestepTauEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim * 2, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: Tensor, tau: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        emb_t = t[:, None] * freqs[None, :]
        emb_tau = tau.log()[:, None] * freqs[None, :]
        emb = torch.cat([emb_t.sin(), emb_t.cos(), emb_tau.sin(), emb_tau.cos()], dim=-1)
        return self.mlp(emb)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = multiple_of * math.ceil(hidden_dim / multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MMDiTBlock(nn.Module):
    """Two-stream MMDiT block with modality-specific QKV/MLP and joint attention."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, multiple_of: int = 256, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout

        self.img_norm1 = RMSNorm(dim)
        self.txt_norm1 = RMSNorm(dim)
        self.img_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.txt_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.img_out = nn.Linear(dim, dim, bias=False)
        self.txt_out = nn.Linear(dim, dim, bias=False)

        hidden = int(dim * mlp_ratio)
        self.img_norm2 = RMSNorm(dim)
        self.txt_norm2 = RMSNorm(dim)
        self.img_mlp = SwiGLU(dim, hidden, multiple_of=multiple_of)
        self.txt_mlp = SwiGLU(dim, hidden, multiple_of=multiple_of)

        self.img_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.txt_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def _qkv(self, layer: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, n, _ = x.shape
        qkv = layer(x).reshape(b, n, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def forward(self, img: Tensor, txt: Tensor, cond: Tensor, txt_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        img_shift_a, img_scale_a, img_gate_a, img_shift_m, img_scale_m, img_gate_m = self.img_ada(cond).chunk(6, -1)
        txt_shift_a, txt_scale_a, txt_gate_a, txt_shift_m, txt_scale_m, txt_gate_m = self.txt_ada(cond).chunk(6, -1)

        img_a = modulate(self.img_norm1(img), img_shift_a, img_scale_a)
        txt_a = modulate(self.txt_norm1(txt), txt_shift_a, txt_scale_a)

        q_img, k_img, v_img = self._qkv(self.img_qkv, img_a)
        q_txt, k_txt, v_txt = self._qkv(self.txt_qkv, txt_a)
        q = torch.cat([q_txt, q_img], dim=2)
        k = torch.cat([k_txt, k_img], dim=2)
        v = torch.cat([v_txt, v_img], dim=2)

        attn_mask = None
        if txt_mask is not None:
            b, n_txt = txt_mask.shape
            n_img = img.shape[1]
            keep = torch.cat([txt_mask.bool(), torch.ones(b, n_img, device=img.device, dtype=torch.bool)], dim=1)
            attn_mask = keep[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(img.shape[0], txt.shape[1] + img.shape[1], self.dim)
        txt_out, img_out = out.split([txt.shape[1], img.shape[1]], dim=1)

        txt = txt + txt_gate_a[:, None, :] * self.txt_out(txt_out)
        img = img + img_gate_a[:, None, :] * self.img_out(img_out)

        txt_m = modulate(self.txt_norm2(txt), txt_shift_m, txt_scale_m)
        img_m = modulate(self.img_norm2(img), img_shift_m, img_scale_m)
        txt = txt + txt_gate_m[:, None, :] * self.txt_mlp(txt_m)
        img = img + img_gate_m[:, None, :] * self.img_mlp(img_m)
        return img, txt


class MomentumMMDiT(nn.Module):
    """SD3-inspired MMDiT-lite backbone for latent-space Momentum Flow."""

    def __init__(
        self,
        *,
        image_size: int = 16,
        in_channels: int = 32,
        out_channels: int = 16,
        patch_size: int = 2,
        dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        multiple_of: int = 256,
        dropout: float = 0.0,
        text_embed_dim: int = 0,
        text_token_dim: int = 0,
        num_prompts: int = 0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_prompts = num_prompts
        patch_dim = in_channels * patch_size * patch_size
        out_patch_dim = out_channels * patch_size * patch_size
        num_patches = (image_size // patch_size) ** 2

        self.img_in = nn.Linear(patch_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.time_embed = TimestepTauEmbedding(dim)
        self.prompt_embed = nn.Embedding(num_prompts, dim) if num_prompts > 0 else None
        self.pooled_proj = nn.Linear(text_embed_dim, dim) if text_embed_dim > 0 else None
        token_dim = text_token_dim or text_embed_dim
        self.text_proj = nn.Linear(token_dim, dim) if token_dim > 0 else None
        self.null_text = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = nn.ModuleList(
            [MMDiTBlock(dim, heads, mlp_ratio=mlp_ratio, multiple_of=multiple_of, dropout=dropout) for _ in range(depth)]
        )
        self.final_norm = RMSNorm(dim)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.out = nn.Linear(dim, out_patch_dim)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def patchify(self, x: Tensor) -> Tensor:
        p = self.patch_size
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, (h // p) * (w // p), c * p * p)

    def unpatchify(self, x: Tensor) -> Tensor:
        p = self.patch_size
        b, n, _ = x.shape
        h = w = int(math.sqrt(n))
        c = self.out_channels
        x = x.reshape(b, h, w, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, c, h * p, w * p)

    def forward(
        self,
        x_t: Tensor,
        v_t: Tensor,
        t: Tensor,
        tau: Tensor,
        prompt_id: Tensor | None = None,
        text_embed: Tensor | None = None,
        text_tokens: Tensor | None = None,
        text_mask: Tensor | None = None,
    ) -> Tensor:
        img = torch.cat([x_t, v_t], dim=1)
        img = self.img_in(self.patchify(img)) + self.pos

        cond = self.time_embed(t, tau)
        if self.prompt_embed is not None:
            if prompt_id is None:
                raise ValueError("prompt_id is required when num_prompts > 0")
            cond = cond + self.prompt_embed(prompt_id)
        if self.pooled_proj is not None and text_embed is not None:
            cond = cond + self.pooled_proj(text_embed)

        if text_tokens is not None:
            if self.text_proj is None:
                raise ValueError("text_token_dim or text_embed_dim must be set when text_tokens are provided")
            txt = self.text_proj(text_tokens)
        elif text_embed is not None:
            if self.text_proj is None:
                raise ValueError("text_embed_dim must be set when text_embed is provided")
            txt = self.text_proj(text_embed[:, None, :])
            text_mask = torch.ones(text_embed.shape[0], 1, device=text_embed.device, dtype=torch.bool)
        else:
            txt = self.null_text.expand(img.shape[0], -1, -1)
            text_mask = torch.ones(img.shape[0], 1, device=img.device, dtype=torch.bool)

        for block in self.blocks:
            img, txt = block(img, txt, cond, txt_mask=text_mask)

        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        img = modulate(self.final_norm(img), shift, scale)
        return self.unpatchify(self.out(img))
