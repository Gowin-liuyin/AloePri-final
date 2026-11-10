"""读取普通或分片 safetensors；在加载前检查索引和实际文件是否一致。"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def _checkpoint_files(directory: str | Path) -> tuple[Path, list[Path], dict[str, str] | None]:
    root = Path(directory).resolve()
    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    if index_path.exists() and single_path.exists():
        raise ValueError("Both single-file and sharded checkpoints exist; use a clean export directory.")
    if not index_path.exists():
        if not single_path.is_file():
            raise FileNotFoundError(f"No safetensors checkpoint or shard index in {root}")
        return root, [single_path], None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint index must contain a nonempty weight_map.")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()):
        raise ValueError("Checkpoint tensor names and shard filenames must be strings.")
    files: list[Path] = []
    for filename in sorted(set(weight_map.values())):
        if not isinstance(filename, str):
            raise ValueError("Shard filenames must be strings.")
        path = (root / filename).resolve()
        if path.parent != root or path.suffix != ".safetensors":
            raise ValueError(f"Invalid shard filename: {filename}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint shard: {path}")
        files.append(path)
    return root, files, weight_map


def safetensors_shapes(directory: str | Path) -> dict[str, tuple[int, ...]]:
    """只读文件头检查全部名称和形状，不把大模型全部装进内存。"""
    _, files, weight_map = _checkpoint_files(directory)
    shapes: dict[str, tuple[int, ...]] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in shapes:
                    raise ValueError(f"Duplicate tensor across shards: {key}")
                if weight_map is not None and weight_map.get(key) != path.name:
                    raise ValueError(f"Tensor/shard index mismatch: {key}")
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    if weight_map is not None and set(shapes) != set(weight_map):
        raise ValueError("The shard index and actual tensors do not contain the same keys.")
    return shapes


def read_safetensors_state(directory: str | Path) -> dict[str, torch.Tensor]:
    """读取权重字典；大模型会占用相应 CPU 内存，形状验收请用上面的函数。"""
    safetensors_shapes(directory)
    _, files, _ = _checkpoint_files(directory)
    state: dict[str, torch.Tensor] = {}
    for path in files:
        state.update(load_file(str(path), device="cpu"))
    return state
