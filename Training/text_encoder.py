from __future__ import annotations

import torch
from torch import Tensor, nn


class FrozenTextEncoder(nn.Module):
    """Frozen pretrained text encoder for prompt conditioning."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        max_length: int | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as exc:
            raise ImportError("text_encoder requires `pip install transformers`") from exc
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.encoder = CLIPTextModel.from_pretrained(model_name)
        self.out_dim = int(self.encoder.config.hidden_size)
        if self.max_length is None:
            self.max_length = int(self.tokenizer.model_max_length)

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def encode(self, prompts: list[str], device: torch.device) -> dict[str, Tensor]:
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            outputs = self.encoder(**tokens)

        return {
            "pooled": outputs.pooler_output,
            "tokens": outputs.last_hidden_state,
            "mask": tokens["attention_mask"],
        }

    def forward(self, prompts: list[str], device: torch.device) -> Tensor:
        return self.encode(prompts, device)["pooled"]
