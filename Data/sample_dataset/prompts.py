from __future__ import annotations

import json
from pathlib import Path


def load_sampling_prompts(path: str | Path) -> list[str]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"missing sampling prompt file: {prompt_path}")

    if prompt_path.suffix == ".json":
        with open(prompt_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = rows.get("prompts", [])
        prompts = [str(row["prompt"] if isinstance(row, dict) else row) for row in rows]
    else:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        raise RuntimeError(f"no prompts found in {prompt_path}")
    return prompts
