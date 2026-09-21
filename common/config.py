"""Config loading, seeding and device selection.

Every run is driven by a YAML file under <task>/<member>/configs/ so that no
hyperparameter or path is hard-coded in a notebook (workplan section 5).
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config, apply any `extends:` parent, and resolve every
    *_dir / *_path value against the repo root."""
    path = Path(path)
    cfg = _resolve_paths(_load_with_extends(path))
    cfg["_config_path"] = str(path.resolve().relative_to(REPO_ROOT))
    return cfg


def _load_with_extends(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular `extends` chain at {path}")
    _seen.add(path)
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    parent_ref = cfg.pop("extends", None)
    if parent_ref is None:
        return cfg
    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        parent_path = REPO_ROOT / parent_path
    return _deep_merge(_load_with_extends(parent_path, _seen), cfg)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Child wins. Nested dicts merge key-by-key; lists are replaced wholesale."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_paths(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: (str(REPO_ROOT / v) if _is_path_key(k, v) else _resolve_paths(v))
                for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_paths(v) for v in node]
    return node


def _is_path_key(key: str, value: Any) -> bool:
    return (isinstance(value, str)
            and (key.endswith("_dir") or key.endswith("_path"))
            and not os.path.isabs(value))


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch. Determinism is best-effort on MPS."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> "torch.device":
    """Resolve 'auto' to cuda > mps > cpu. Task 1 runs on the local M5 (mps);
    tasks 2 and 3 stay portable to the GPU Lab (cuda)."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_hardware(device: "torch.device") -> dict[str, str]:
    """Hardware disclosure row required per model (workplan 2.2 / section 4)."""
    import platform

    import torch

    info = {
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
    return info


def config_arg_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True, help="path to the run's YAML config")
    ap.add_argument("--run-id", default=None, help="override the auto-generated run id")
    return ap


@dataclass(frozen=True)
class RunPaths:
    """Where one run writes. Created eagerly so a crash still leaves the log."""
    run_id: str
    log_dir: Path
    ckpt_dir: Path
    out_dir: Path

    @staticmethod
    def build(cfg: dict[str, Any], run_id: str) -> "RunPaths":
        paths = cfg["paths"]
        rp = RunPaths(
            run_id=run_id,
            log_dir=Path(paths["log_dir"]),
            ckpt_dir=Path(paths["checkpoint_dir"]),
            out_dir=Path(paths["output_dir"]),
        )
        for d in (rp.log_dir, rp.ckpt_dir, rp.out_dir):
            d.mkdir(parents=True, exist_ok=True)
        return rp
