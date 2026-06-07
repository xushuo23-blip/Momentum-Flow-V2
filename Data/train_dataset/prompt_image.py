from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class PromptFolderDataset(Dataset):
    """One prompt folder may contain many images."""

    def __init__(self, root: str | Path, transform=None, exts: tuple[str, ...] = ("jpg", "jpeg", "png", "webp")):
        self.root = Path(root)
        self.transform = transform
        prompts_path = self.root / "prompts.json"
        if not prompts_path.exists():
            raise FileNotFoundError(f"missing prompts file: {prompts_path}")

        with open(prompts_path, "r", encoding="utf-8") as f:
            prompt_map = json.load(f)

        self.prompt_items = sorted(prompt_map.items())
        self.prompt_to_id = {folder: idx for idx, (folder, _) in enumerate(self.prompt_items)}
        self.id_to_prompt = {idx: text for idx, (_, text) in enumerate(self.prompt_items)}

        self.samples: list[tuple[Path, int]] = []
        for folder, _ in self.prompt_items:
            prompt_dir = self.root / folder
            if not prompt_dir.is_dir():
                raise FileNotFoundError(f"missing prompt image folder: {prompt_dir}")
            for ext in exts:
                self.samples.extend((path, self.prompt_to_id[folder]) for path in sorted(prompt_dir.glob(f"*.{ext}")))

        if not self.samples:
            raise RuntimeError(f"no images found under {self.root}")

    @property
    def num_prompts(self) -> int:
        return len(self.prompt_items)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path, prompt_id = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "prompt_id": prompt_id, "prompt_text": self.id_to_prompt[prompt_id]}


class PromptImageDataset(Dataset):
    """One prompt / one image manifest dataset."""

    def __init__(self, root: str | Path, manifest: str = "samples.json", transform=None):
        self.root = Path(root)
        self.transform = transform
        manifest_path = self.root / manifest
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing prompt-image manifest: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"{manifest_path} must contain a list of image-prompt entries")

        prompts = sorted({str(row["prompt"]) for row in rows})
        self.prompt_to_id = {prompt: idx for idx, prompt in enumerate(prompts)}
        self.id_to_prompt = {idx: prompt for prompt, idx in self.prompt_to_id.items()}
        self.samples: list[tuple[Path, int]] = []
        for row in rows:
            image_path = self.root / row["image"]
            prompt = str(row["prompt"])
            self.samples.append((image_path, self.prompt_to_id[prompt]))

        if not self.samples:
            raise RuntimeError(f"no prompt-image entries found in {manifest_path}")

    @property
    def num_prompts(self) -> int:
        return len(self.prompt_to_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path, prompt_id = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "prompt_id": prompt_id, "prompt_text": self.id_to_prompt[prompt_id]}


class LimitedDataset(Dataset):
    def __init__(self, dataset: Dataset, max_samples: int):
        self.dataset = dataset
        self.max_samples = min(int(max_samples), len(dataset))
        self.indices = list(range(self.max_samples))

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)


def build_image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )


def build_train_dataset(cfg: dict):
    transform = build_image_transform(cfg["image_size"])
    name = cfg["name"].lower()
    if name == "imagefolder":
        dataset = ImageFolder(root=cfg["folder"], transform=transform)
    elif name == "prompt_folder":
        dataset = PromptFolderDataset(root=cfg["folder"], transform=transform)
    elif name == "prompt_image":
        dataset = PromptImageDataset(root=cfg["folder"], manifest=cfg.get("manifest", "samples.json"), transform=transform)
    else:
        raise ValueError(f"unknown dataset {cfg['name']}")

    max_samples = cfg.get("max_samples")
    if max_samples is not None:
        dataset = LimitedDataset(dataset, max_samples)
    return dataset
