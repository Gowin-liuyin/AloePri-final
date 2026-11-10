from __future__ import annotations

import torch
from torch import nn
from dataclasses import replace
from typing import Optional, Iterable, Dict, Any

from src.aloepri.adapters import get_architecture_adapter
from src.aloepri.config import AloePriConfig
from src.aloepri.keys import build_aloepri_keys, AloePriKeys
from src.aloepri.layers.norm import wrap_norm
from src.aloepri.layers.ffn import wrap_mlp
from src.aloepri.layers.attention import wrap_attention
from src.aloepri.layers.embeddings import wrap_embedding, wrap_head
from src.key_manager import ordinary_token_ids
from src.aloepri._implementation.runtime.token_space_runtime import TraceRecorder, prepare_stage_a_model
from src.aloepri._implementation.runtime.keymat_bridge_runtime import _clear_copied_hooks
from src.aloepri._implementation.runtime.keymat_fusion_runtime import KeyMatDecoderLayerHandoff
from src.keymat_norm import estimate_kappa_for_keymat
from src.obfuscate_ffn import build_ffn_transform, generate_ffn_permutation, generate_ffn_scaling

class AloePriEngine:
    """
    Main engine to apply AloePri obfuscation to a model.
    """
    def __init__(self, config: AloePriConfig, tokenizer: Any):
        self.tokenizer = tokenizer
        self.keys = build_aloepri_keys(config, tokenizer.vocab_size, ordinary_token_ids(tokenizer))
        self.config = self.keys.config

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        tokenizer: Any,
        **config_overrides,
    ) -> "AloePriEngine":
        get_architecture_adapter(model)
        config = AloePriConfig.from_model(model, **config_overrides)
        return cls(config=config, tokenizer=tokenizer)

    def obfuscate_model(self, model: nn.Module, recorder: Optional[TraceRecorder] = None) -> nn.Module:
        """
        Build the Qwen research runtime. Exact normalization remains a custom reference.
        """
        adapter = get_architecture_adapter(model)
        if self.config.architecture_family != "qwen_decoder" or "qwen" not in adapter.model_type:
            raise ValueError("AloePriEngine.obfuscate_model currently supports Qwen decoders only; use the dedicated BERT/Llama export APIs.")
        if len(model.model.layers) != self.config.num_hidden_layers:
            raise ValueError("The configured decoder layer count does not match the model.")
        model_vocab_size = int(model.get_input_embeddings().weight.shape[0])
        if self.keys.token_perm.numel() != model_vocab_size:
            new_config = replace(self.config, model_vocab_size=model_vocab_size)
            new_keys = build_aloepri_keys(new_config, self.tokenizer.vocab_size, ordinary_token_ids(self.tokenizer))
            self.keys = replace(new_keys, seed_source=self.keys.seed_source)
            self.config = self.keys.config
        # 1. Prepare Stage A model (token permutation)
        stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(
            model, self.tokenizer, self.config.seed,
            perm_vocab=self.keys.token_perm, inv_perm_vocab=self.keys.inv_token_perm,
        )
        _clear_copied_hooks(stage_a_model)

        target_dtype = stage_a_model.dtype
        device = stage_a_model.device

        # 2. Obfuscate Embedding
        movable_ids = ordinary_token_ids(self.tokenizer)
        stage_a_model.model.embed_tokens = wrap_embedding(
            embed_module=stage_a_model.model.embed_tokens,
            keymat_transform=self.keys.keymat_transform,
            alpha_e=self.config.alpha_e,
            seed=self.config.seed,
            movable_ids=movable_ids,
            recorder=recorder,
            record_name="embed_tokens_out"
        ).to(device=device, dtype=target_dtype)

        # 4. Obfuscate Layers
        total_layers = stage_a_model.config.num_hidden_layers
        adapted_layers = sorted(self.config.adapted_layers)

        # Handoff layer logic
        handoff_layer = len(adapted_layers) if len(adapted_layers) < total_layers else None

        for layer_idx in adapted_layers:
            layer = stage_a_model.model.layers[layer_idx]

            # Get input/post-attn weights for fusion
            input_norm_weight = layer.input_layernorm.weight.detach().to(torch.float32)
            post_norm_weight = layer.post_attention_layernorm.weight.detach().to(torch.float32)

            # Estimate kappas
            input_kappa = estimate_kappa_for_keymat(
                self.keys.keymat_transform,
                hidden_size=self.config.hidden_size,
                num_samples=1024,
                seed=self.config.seed + 2000 + layer_idx
            )
            post_attn_kappa = estimate_kappa_for_keymat(
                self.keys.keymat_transform,
                hidden_size=self.config.hidden_size,
                num_samples=1024,
                seed=self.config.seed + 3000 + layer_idx
            )
            self.keys.kappas[layer_idx] = {"input_norm": input_kappa, "post_attn_norm": post_attn_kappa}

            # Apply obfuscation
            layer.input_layernorm = wrap_norm(
                layer.input_layernorm,
                keymat_transform=self.keys.keymat_transform,
                kappa=input_kappa,
                mode=self.config.normalization_mode,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_input_norm_out"
            ).to(device=device, dtype=target_dtype)

            layer.self_attn = wrap_attention(
                attention_module=layer.self_attn,
                keymat_transform=self.keys.keymat_transform,
                input_norm_weight=input_norm_weight,
                attention_profile=self.config.attention_profile,
                seed=self.config.seed + 1000 + layer_idx,
                qk_scale_range=self.config.qk_scale_range,
                beta=self.config.beta,
                gamma=self.config.gamma,
                rope_base=self.config.rope_theta,
                layer_idx=layer_idx,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_attn_out"
            ).to(device=device, dtype=target_dtype)

            layer.post_attention_layernorm = wrap_norm(
                layer.post_attention_layernorm,
                keymat_transform=self.keys.keymat_transform,
                kappa=post_attn_kappa,
                mode=self.config.normalization_mode,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_post_attn_norm_out"
            ).to(device=device, dtype=target_dtype)

            # FFN Transform - build it if needed
            ffn_intermediate_size = layer.mlp.gate_proj.out_features
            ffn_perm = generate_ffn_permutation(ffn_intermediate_size, seed=self.config.seed + 5000 + layer_idx)
            ffn_scale = generate_ffn_scaling(ffn_intermediate_size, scale_range=self.config.ffn_scale_range, seed=self.config.seed + 6000 + layer_idx)
            ffn_transform = build_ffn_transform(ffn_perm, ffn_scale)
            self.keys.ffn_transforms[layer_idx] = ffn_transform

            layer.mlp = wrap_mlp(
                mlp_module=layer.mlp,
                keymat_transform=self.keys.keymat_transform,
                ffn_transform=ffn_transform,
                input_norm_weight=post_norm_weight,
                seed=self.config.seed + 7000 + layer_idx,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_mlp_out"
            ).to(device=device, dtype=target_dtype)

        # 5. Handle Handoff
        if handoff_layer is not None:
            original_handoff = stage_a_model.model.layers[handoff_layer]
            stage_a_model.model.layers[handoff_layer] = KeyMatDecoderLayerHandoff(
                layer_module=original_handoff,
                keymat_transform=self.keys.keymat_transform,
                layer_idx=handoff_layer,
                recorder=recorder,
            )

        # 6. Obfuscate Head and Final Norm
        if handoff_layer is None:
            # Full model obfuscated - fuse final norm into head
            final_norm = stage_a_model.model.norm
            final_norm_weight = final_norm.weight.detach().to(torch.float32)

            final_kappa = estimate_kappa_for_keymat(
                self.keys.keymat_transform,
                hidden_size=self.config.hidden_size,
                num_samples=1024,
                seed=self.config.seed + 12345
            )
            self.keys = replace(self.keys, final_norm_kappa=final_kappa)

            stage_a_model.model.norm = wrap_norm(
                final_norm,
                keymat_transform=self.keys.keymat_transform,
                kappa=final_kappa,
                mode=self.config.normalization_mode,
                recorder=recorder,
                record_name="final_norm_out"
            ).to(device=device, dtype=target_dtype)

            stage_a_model.lm_head = wrap_head(
                head_module=stage_a_model.lm_head,
                keymat_transform=self.keys.keymat_transform,
                alpha_h=self.config.alpha_h,
                seed=self.config.seed + 202,
                movable_ids=movable_ids,
                final_norm_weight=final_norm_weight,
                expects_obfuscated_input=True,
                recorder=recorder,
                record_name="head_out"
            ).to(device=device, dtype=target_dtype)
        else:
            # Handoff - head is not fused with final norm because final norm input is already restored
            stage_a_model.lm_head = wrap_head(
                head_module=stage_a_model.lm_head,
                keymat_transform=self.keys.keymat_transform,
                alpha_h=self.config.alpha_h,
                seed=self.config.seed + 202,
                movable_ids=movable_ids,
                expects_obfuscated_input=False,
                recorder=recorder,
                record_name="head_out"
            ).to(device=device, dtype=target_dtype)

        # Attach permutation for easy access
        stage_a_model.perm_vocab = perm_vocab
        stage_a_model.inv_perm_vocab = inv_perm_vocab
        stage_a_model.aloepri_runtime_metadata = {
            "normalization_mode": self.config.normalization_mode,
            "requires_custom_runtime": True,
            "adapted_layers": adapted_layers,
            "handoff_layer": handoff_layer,
            "shared_residual_key": True,
            "seed_source": self.keys.seed_source,
            "scalar_epsilon_rule": "epsilon_obfuscated = (kappa * sqrt(d / D))**2 * epsilon_original",
        }

        return stage_a_model
