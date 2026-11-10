"""Extract the usable HF checkpoint from Models.zip, omitting incomplete original weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.defaults import DEFAULT_LLAMA_MODEL_DIR

MODEL_NAME = "Llama-3.2-1B-Instruct"
REQUIRED = {"config.json", "generation_config.json", "model.safetensors",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
OPTIONAL = {"README.md", "LICENSE.txt", "USE_POLICY.md", ".gitattributes"}


def validate_checkpoint(model_dir: Path) -> dict:
    missing = sorted(name for name in REQUIRED - {"model.safetensors"} if not (model_dir / name).is_file())
    if missing:
        raise ValueError(f"Incomplete checkpoint: {missing}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    expected = dict(model_type="llama", hidden_size=2048, intermediate_size=8192,
                    num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8,
                    head_dim=64, vocab_size=128256, tie_word_embeddings=True)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Expected Llama 3.2 1B {key}={value!r}, got {config.get(key)!r}")
    if config.get("rope_scaling", {}).get("rope_type") != "llama3":
        raise ValueError("Expected Llama 3 scaled RoPE")
    single = model_dir / "model.safetensors"
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        if single.exists():
            raise ValueError("Both single-file and sharded checkpoints exist")
        weight_map = json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
        files = []
        for name in sorted(set(weight_map.values())):
            file = (model_dir / name).resolve()
            if file.parent != model_dir.resolve() or file.suffix != ".safetensors":
                raise ValueError("Invalid shard filename")
            files.append(file)
    else:
        weight_map = None
        files = [single]
    shapes = {}
    for weight_file in files:
        header_size, tensors = read_header(weight_file)
        data_size = weight_file.stat().st_size - 8 - header_size
        entries = {name: value for name, value in tensors.items() if name != "__metadata__"}
        if max(value["data_offsets"][1] for value in entries.values()) != data_size:
            raise ValueError(f"Safetensors payload length mismatch: {weight_file}")
        for name, value in entries.items():
            if name in shapes or (weight_map is not None and weight_map.get(name) != weight_file.name):
                raise ValueError(f"Duplicate tensor or incorrect shard index: {name}")
            shapes[name] = value["shape"]
    if weight_map is not None and set(shapes) != set(weight_map):
        raise ValueError("Shard index does not match tensor names")
    if shapes.get("model.embed_tokens.weight") != [128256, 2048]:
        raise ValueError("Embedding shape does not match the 1B checkpoint")
    for index in range(16):
        prefix = f"model.layers.{index}"
        for name, shape in {"self_attn.q_proj.weight": [2048, 2048],
                            "self_attn.k_proj.weight": [512, 2048],
                            "self_attn.v_proj.weight": [512, 2048],
                            "self_attn.o_proj.weight": [2048, 2048],
                            "mlp.gate_proj.weight": [8192, 2048],
                            "mlp.up_proj.weight": [8192, 2048],
                            "mlp.down_proj.weight": [2048, 8192]}.items():
            if shapes.get(f"{prefix}.{name}") != shape:
                raise ValueError(f"Missing or incorrect tensor: {prefix}.{name}")
    return {"model_dir": str(model_dir.resolve()), **expected,
            "tensor_count": len(shapes), "weight_bytes": sum(file.stat().st_size for file in files),
            "shard_count": len(files)}


def read_header(path: Path):
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < size < 16 * 1024 * 1024:
            raise ValueError("Invalid safetensors header")
        return size, json.loads(handle.read(size))


def shard_checkpoint(model_dir: Path, max_bytes=512_000_000):
    """Repack bytes without loading Torch; retain the source single-file checkpoint."""
    original = model_dir / "model.safetensors"
    if not original.is_file() or original.stat().st_size <= max_bytes:
        return
    old_size, header = read_header(original)
    groups, current, current_size = [], [], 0
    for name, value in header.items():
        if name == "__metadata__":
            continue
        length = value["data_offsets"][1] - value["data_offsets"][0]
        if current and current_size + length > max_bytes:
            groups.append(current)
            current, current_size = [], 0
        current.append((name, value))
        current_size += length
    if current:
        groups.append(current)
    staging = model_dir / "sharded-preparation"
    staging.mkdir(exist_ok=False)
    weight_map, total_size = {}, 0
    with original.open("rb") as source:
        for index, group in enumerate(groups, 1):
            filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            new_header = {"__metadata__": header.get("__metadata__", {"format": "pt"})}
            offset = 0
            for name, value in group:
                length = value["data_offsets"][1] - value["data_offsets"][0]
                new_header[name] = {**value, "data_offsets": [offset, offset + length]}
                offset += length
                weight_map[name] = filename
            total_size += offset
            encoded = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
            encoded += b" " * (-len(encoded) % 8)
            print(f"Writing shard {index}/{len(groups)} ({offset:,} bytes)", flush=True)
            with (staging / filename).open("xb") as target:
                target.write(struct.pack("<Q", len(encoded)))
                target.write(encoded)
                for name, value in group:
                    start, end = value["data_offsets"]
                    source.seek(8 + old_size + start)
                    remaining = end - start
                    while remaining:
                        chunk = source.read(min(4 * 1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Truncated source weights")
                        target.write(chunk)
                        remaining -= len(chunk)
    (staging / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2), encoding="utf-8")
    preserved = model_dir / "source_single_file"
    preserved.mkdir(exist_ok=True)
    original.rename(preserved / original.name)
    for file in staging.iterdir():
        file.rename(model_dir / file.name)
    staging.rmdir()


def prepare_checkpoint(archive: Path, destination: Path) -> dict:
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        validate_checkpoint(destination)
        shard_checkpoint(destination)
        return validate_checkpoint(destination)
    with zipfile.ZipFile(archive) as source:
        configs = [info for info in source.infolist()
                   if info.filename.endswith(f"{MODEL_NAME}/config.json")]
        if len(configs) != 1:
            raise ValueError("Archive must contain exactly one Llama-3.2-1B-Instruct checkpoint")
        prefix = configs[0].filename.removesuffix("config.json")
        # Copy allowlisted direct children; archive paths never determine destination paths.
        selected = {info.filename[len(prefix):]: info for info in source.infolist()
                    if info.filename.startswith(prefix)
                    and info.filename[len(prefix):] in REQUIRED | OPTIONAL}
        if REQUIRED - selected.keys():
            raise ValueError(f"Missing archive files: {sorted(REQUIRED - selected.keys())}")
        destination.mkdir(parents=True, exist_ok=True)
        for name, info in selected.items():
            print(f"Extracting {name} ({info.file_size:,} bytes)", flush=True)
            with source.open(info) as reader, (destination / name).open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
    validate_checkpoint(destination)
    shard_checkpoint(destination)
    return validate_checkpoint(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("D:/Models.zip"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / DEFAULT_LLAMA_MODEL_DIR)
    args = parser.parse_args()
    print(json.dumps(prepare_checkpoint(args.archive, args.model_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
