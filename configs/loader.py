from __future__ import annotations

import importlib.util
from pathlib import Path


def _to_plain_dict(cfg):
    return cfg.to_dict() if hasattr(cfg, "to_dict") else cfg


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if config_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("experiment_config", config_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load config: {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "get_config"):
            raise ValueError(f"{config_path} must define get_config()")
        return _to_plain_dict(module.get_config())
    raise ValueError(f"unsupported config type: {config_path.suffix}; use a Python config with get_config()")
