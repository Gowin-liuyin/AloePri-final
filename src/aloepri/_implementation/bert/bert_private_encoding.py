from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from src.aloepri._implementation.bert.bert_layernorm_bridge import KeyMatBertLayerNormBridge
from src.aloepri._implementation.export.checkpoint_io import load_stage_i_hf_bundle
from src.aloepri._implementation.analysis.keymat_family import build_diag_friendly_keymat_transform
from src.aloepri._implementation.export.square_transform import SquareMonomialTransform
from src.aloepri.adapters.bert import BertArchitectureAdapter
from src.hidden_keys import build_hidden_transform
from src.keymat import apply_inverse_keymat_transform, build_keymat_transform
from src.model_loader import resolve_torch_dtype
from src.transforms import apply_inverse_hidden_transform, map_input_ids


def _read_catalog_profile(bundle_dir: Path, profile: str) -> dict[str, Any]:
    catalog = json.loads((bundle_dir / "catalog.json").read_text(encoding="utf-8"))
    profile_map = {item["name"]: item for item in catalog["profiles"]}
    if profile not in profile_map:
        raise ValueError(f"Unknown release profile: {profile}")
    return profile_map[profile]


def _load_bert_bundle_with_fallback(
    *,
    server_dir: Path,
    client_secret_path: Path,
    dtype: str,
) -> dict[str, Any]:
    load_error: Exception | None = None
    try:
        bundle = load_stage_i_hf_bundle(
            server_dir,
            client_secret_path=client_secret_path,
            device="cpu",
            dtype=dtype,
        )
        try:
            BertArchitectureAdapter.from_model(bundle["model"])
            return bundle
        except Exception as exc:
            load_error = exc
    except Exception as exc:
        load_error = exc

    tokenizer = AutoTokenizer.from_pretrained(server_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        server_dir,
        trust_remote_code=True,
        torch_dtype=resolve_torch_dtype(dtype),
    )
    model.eval()

    secret: dict[str, Any] | None = None
    perm_vocab = None
    inv_perm_vocab = None
    if client_secret_path.exists():
        secret = torch.load(client_secret_path, map_location="cpu", weights_only=True)
        perm_vocab = torch.as_tensor(secret["perm_vocab"], dtype=torch.long)
        inv_perm_vocab = torch.as_tensor(secret["inv_perm_vocab"], dtype=torch.long)

    metadata_path = server_dir.parent / "stage_i_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if load_error is not None:
        metadata = {**metadata, "_bert_loader_fallback": repr(load_error)}

    return {
        "tokenizer": tokenizer,
        "model": model,
        "perm_vocab": perm_vocab,
        "inv_perm_vocab": inv_perm_vocab,
        "secret": secret,
        "metadata": metadata,
    }


class _SquareKeyMatProxy:
    def __init__(self, square_transform: SquareMonomialTransform) -> None:
        self.hidden_size = int(square_transform.dim)
        self.expansion_size = 0
        self.lam = 0.0
        self.key = square_transform.key(dtype=torch.float32)
        self.inverse = square_transform.inverse(dtype=torch.float32)


def _load_square_transform(metadata: dict[str, Any]) -> SquareMonomialTransform:
    square_transform = metadata.get("square_transform")
    if not isinstance(square_transform, dict):
        raise ValueError("Missing square_transform metadata required for square-profile BERT load.")
    return SquareMonomialTransform(
        perm=torch.as_tensor(square_transform["perm"], dtype=torch.long),
        inv_perm=torch.as_tensor(square_transform["inv_perm"], dtype=torch.long),
        signs=torch.as_tensor(square_transform["signs"], dtype=torch.float32),
        global_scale=float(square_transform.get("global_scale", 1.0)),
    )


def _wrap_square_bert_layernorms_inplace(model, metadata: dict[str, Any]) -> None:
    transform_family = str(metadata.get("transform_family", "")).lower()
    if "square" not in transform_family and "square_transform" not in metadata:
        return

    BertArchitectureAdapter.from_model(model)
    square_transform = _load_square_transform(metadata)
    proxy = _SquareKeyMatProxy(square_transform)

    def wrap(norm_layer):
        if isinstance(norm_layer, KeyMatBertLayerNormBridge):
            return norm_layer
        return KeyMatBertLayerNormBridge(norm_layer, proxy)

    model.embeddings.LayerNorm = wrap(model.embeddings.LayerNorm)
    for layer in model.encoder.layer:
        layer.attention.output.LayerNorm = wrap(layer.attention.output.LayerNorm)
        layer.output.LayerNorm = wrap(layer.output.LayerNorm)


def load_bert_release_profile(bundle_dir: str | Path, profile: str, *, dtype: str = "float32") -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    selected = _read_catalog_profile(bundle_dir, profile)
    server_dir = bundle_dir / selected["server_dir"]
    client_secret_path = bundle_dir / selected["client_secret"]
    bundle = _load_bert_bundle_with_fallback(
        server_dir=server_dir,
        client_secret_path=client_secret_path,
        dtype=dtype,
    )
    secret_metadata = bundle.get("secret", {}).get("metadata", {}) if isinstance(bundle.get("secret"), dict) else {}
    selected_metadata = selected.get("metadata", {}) if isinstance(selected, dict) else {}
    metadata = {**selected_metadata, **secret_metadata, **bundle.get("metadata", {})}
    _wrap_square_bert_layernorms_inplace(bundle["model"], metadata)
    return {
        "tokenizer": bundle["tokenizer"],
        "model": bundle["model"],
        "perm_vocab": bundle["perm_vocab"],
        "inv_perm_vocab": bundle["inv_perm_vocab"],
        "metadata": metadata,
    }


def _validate_return_type(return_type: str) -> None:
    if return_type not in {"last_hidden_state", "pooler_output"}:
        raise ValueError("return_type must be 'last_hidden_state' or 'pooler_output'")


def _resolve_square_hidden_transform(metadata: dict[str, Any]):
    square_transform = _load_square_transform(metadata)
    scale = square_transform.signs * float(square_transform.global_scale)
    return build_hidden_transform(perm=square_transform.perm, scale=scale)


def _resolve_keymat_inverse_matrix(
    metadata: dict[str, Any],
    *,
    hidden_size: int,
) -> torch.Tensor | None:
    keymat_cfg = metadata.get("keymat")
    if isinstance(keymat_cfg, dict) and "inverse" in keymat_cfg:
        return torch.as_tensor(keymat_cfg["inverse"], dtype=torch.float32)
    if "keymat_inverse" in metadata:
        return torch.as_tensor(metadata["keymat_inverse"], dtype=torch.float32)

    family = str(
        (keymat_cfg or {}).get("family")
        or metadata.get("keymat_family")
        or "algorithm1"
    ).lower()
    expansion_size_raw = (
        (keymat_cfg or {}).get("expansion_size")
        or (keymat_cfg or {}).get("h")
        or metadata.get("expansion_size")
        or metadata.get("h")
    )
    if expansion_size_raw is None:
        raise ValueError("Missing KeyMat expansion_size/h metadata required for hidden-state restore.")
    expansion_size = int(expansion_size_raw)
    lam = float(
        (keymat_cfg or {}).get("lambda")
        or (keymat_cfg or {}).get("lam")
        or metadata.get("lambda")
        or metadata.get("lam")
        or 0.0
    )
    base_hidden_size_raw = (
        (keymat_cfg or {}).get("hidden_size")
        or metadata.get("hidden_size")
    )
    if base_hidden_size_raw is not None:
        base_hidden_size = int(base_hidden_size_raw)
    else:
        derived_hidden_size = int(hidden_size) - 2 * expansion_size
        if derived_hidden_size <= 0:
            raise ValueError(
                "Unable to derive base hidden size for KeyMat restore from observed hidden width and expansion size."
            )
        base_hidden_size = derived_hidden_size
    init_seed = (keymat_cfg or {}).get("init_seed")
    key_seed = (keymat_cfg or {}).get("key_seed")
    inv_seed = (keymat_cfg or {}).get("inv_seed")
    seed = (keymat_cfg or {}).get("seed")
    root_seed = metadata.get("seed")

    if family == "diag_friendly":
        transform = build_diag_friendly_keymat_transform(
            hidden_size=base_hidden_size,
            expansion_size=expansion_size,
            seed=int(seed if seed is not None else root_seed if root_seed is not None else 0),
            dtype=torch.float32,
        )
    else:
        if init_seed is None:
            base_seed = int(seed if seed is not None else root_seed if root_seed is not None else 0)
            init_seed = base_seed + 12000
        if key_seed is None:
            key_seed = int(init_seed) + 100
        if inv_seed is None:
            inv_seed = int(init_seed) + 200
        transform = build_keymat_transform(
            d=base_hidden_size,
            h=expansion_size,
            lam=lam,
            init_seed=int(init_seed),
            key_seed=int(key_seed),
            inv_seed=int(inv_seed),
            dtype=torch.float32,
        )
    return transform.inverse


def _restore_hidden_tensor(hidden: torch.Tensor, metadata: dict[str, Any], *, hidden_size: int) -> torch.Tensor:
    transform_family = str(metadata.get("transform_family", "")).lower()
    if "square" in transform_family or "square_transform" in metadata:
        hidden_transform = _resolve_square_hidden_transform(metadata)
        return apply_inverse_hidden_transform(hidden, hidden_transform)

    has_keymat_metadata = any(
        key in metadata for key in ("keymat", "lambda", "lam", "h", "expansion_size", "keymat_inverse")
    )
    if "keymat" in transform_family or has_keymat_metadata:
        inverse = _resolve_keymat_inverse_matrix(metadata, hidden_size=hidden_size)
        if inverse is None:
            raise ValueError("Missing KeyMat inverse metadata required for hidden-state restore.")

        class _KeyMatLike:
            def __init__(self, inverse_matrix: torch.Tensor):
                self.inverse = inverse_matrix

        return apply_inverse_keymat_transform(hidden, _KeyMatLike(inverse))

    raise ValueError("Unable to determine hidden transform family from metadata.")


def _restore_pooler_output_tensor(pooler_output: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
    if str(metadata.get("pooler_output_space", "obfuscated")).lower() == "plain":
        return pooler_output
    return _restore_hidden_tensor(pooler_output, metadata, hidden_size=int(pooler_output.shape[-1]))


def _encode_private_outputs(
    *,
    bundle_dir: str | Path,
    profile: str,
    text: str,
    dtype: str,
    return_type: str,
) -> dict[str, Any]:
    _validate_return_type(return_type)
    bundle = load_bert_release_profile(bundle_dir, profile, dtype=dtype)
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    perm_vocab = bundle["perm_vocab"]

    if perm_vocab is None:
        raise ValueError("Release bundle is missing perm_vocab.")

    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    obfuscated_batch = dict(encoded)
    obfuscated_batch["input_ids"] = map_input_ids(encoded["input_ids"], perm_vocab)

    outputs = model(**obfuscated_batch, return_dict=True)
    result: dict[str, Any] = {
        "tokenizer": tokenizer,
        "perm_vocab": perm_vocab,
        "metadata": bundle["metadata"],
        "input_ids": encoded["input_ids"],
        "obfuscated_input_ids": obfuscated_batch["input_ids"],
        "attention_mask": encoded.get("attention_mask"),
    }
    if "token_type_ids" in encoded:
        result["token_type_ids"] = encoded["token_type_ids"]
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        result["last_hidden_state"] = outputs.last_hidden_state.detach().cpu().to(torch.float32)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        result["pooler_output"] = outputs.pooler_output.detach().cpu().to(torch.float32)

    if return_type == "pooler_output":
        result.pop("last_hidden_state", None)
    if return_type == "last_hidden_state":
        result.pop("pooler_output", None)

    if return_type == "pooler_output" and "pooler_output" not in result:
        raise ValueError("Requested pooler_output, but the model did not return one.")
    if "last_hidden_state" not in result and "pooler_output" not in result:
        raise ValueError("BERT model did not return last_hidden_state or pooler_output.")
    return result


@torch.inference_mode()
def encode_private(
    *,
    bundle_dir: str | Path,
    profile: str,
    text: str,
    dtype: str = "float32",
    return_type: str = "last_hidden_state",
) -> dict[str, Any]:
    return _encode_private_outputs(
        bundle_dir=bundle_dir,
        profile=profile,
        text=text,
        dtype=dtype,
        return_type=return_type,
    )


@torch.inference_mode()
def encode_private_with_restore(
    *,
    bundle_dir: str | Path,
    profile: str,
    text: str,
    dtype: str = "float32",
    return_type: str = "last_hidden_state",
) -> dict[str, Any]:
    result = _encode_private_outputs(
        bundle_dir=bundle_dir,
        profile=profile,
        text=text,
        dtype=dtype,
        return_type=return_type,
    )
    metadata = result["metadata"]
    restored = dict(result)

    if "last_hidden_state" in result:
        restored["last_hidden_state"] = _restore_hidden_tensor(
            result["last_hidden_state"],
            metadata,
            hidden_size=int(result["last_hidden_state"].shape[-1]),
        )
    if "pooler_output" in result:
        restored["pooler_output"] = _restore_pooler_output_tensor(result["pooler_output"], metadata)
    return restored
