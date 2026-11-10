"""将Algorithm-1扩维矩阵写入Qwen2/Llama/Qwen2-MoE/Mixtral标准权重。

归一化仍是明确标注的标量近似，MoE路由归一化还可能改变专家选择。

与旧调试bridge不同，此入口从原模型直接构建，不把恢复矩阵或密钥配置交给服务器。
头数和每头维度保持原值；残差宽度改为D，不能用D/头数重算head_dim。
"""
from __future__ import annotations
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import secrets
import torch
from transformers import AutoModelForCausalLM
from src.keymat import build_keymat_transform, sample_compatible_inverse
from src.key_manager import generate_vocab_permutation, invert_permutation, ordinary_token_ids
from src.keymat_embed_head import add_embed_noise, add_head_noise
from src.hidden_keys import build_hidden_transform, generate_hidden_scaling,generate_hidden_permutation
from src.obfuscate_ffn import fuse_ffn_intermediate_weights
from src.aloepri._implementation.qwen.static_attention_obfuscation import StaticizedQwen2Attention
from src.aloepri._implementation.runtime.keymat_ffn_fusion import KeyMatFusedQwen2MLP
from src.aloepri._implementation.export.checkpoint_io import export_stage_i_vllm_checkpoint, summarize_token_partitions
from src.aloepri._implementation.analysis.standard_weight_proof import build_stage_j_standard_weight_proof
from src.secure_random import secure_export,random_metadata,random_stream,random_namespace


def _fuse_expert_mlp(module,key,gamma,seed):
    """把打包专家逐一融合后再打乱专家顺序；路由器使用相同专家顺序。"""
    joined=module.experts.gate_up_proj.detach().float().cpu()
    downs=module.experts.down_proj.detach().float().cpu()
    count,twice_width,_=joined.shape
    if twice_width%2 or downs.shape!=(count,key.hidden_size,twice_width//2):
        raise ValueError("Unsupported packed MoE expert dimensions")
    width=twice_width//2
    order=generate_hidden_permutation(count,seed)
    new_joined,new_down=[],[]
    for old in order.tolist():
        offset=seed+100+old*11
        qg=sample_compatible_inverse(key,offset);qu=sample_compatible_inverse(key,offset+1)
        gate=joined[old,:width]@(qg*gamma).T
        up=joined[old,width:]@(qu*gamma).T
        down=key.key.T@downs[old]
        ff=build_hidden_transform(generate_hidden_permutation(width,offset+2),generate_hidden_scaling(width,(.95,1.05),offset+3))
        gate,up,down,_,_=fuse_ffn_intermediate_weights(gate,up,down,ff)
        new_joined.append(torch.cat([gate,up],0));new_down.append(down)
    router=module.gate.weight.detach().float().cpu()
    lengths=router.norm(dim=1,keepdim=True)
    if (lengths<=0).any():raise ValueError("Cannot normalize zero router expert vectors")
    # 归一化每个专家向量是论文的额外改写，可能改变原模型路由，不宣称严格等价。
    qr=sample_compatible_inverse(key,seed+1)
    result={"experts.gate_up_proj":torch.stack(new_joined),"experts.down_proj":torch.stack(new_down),
            "gate.weight":((router/lengths)@(qr*gamma).T)[order]}
    shared=getattr(module,"shared_expert",None)
    if shared is not None:
        ff=build_hidden_transform(generate_hidden_permutation(shared.gate_proj.out_features,seed+2),
              generate_hidden_scaling(shared.gate_proj.out_features,(.95,1.05),seed+3))
        fused=KeyMatFusedQwen2MLP(shared,key,gamma,ff,seed=seed+4)
        for kind in ("gate","up","down"):
            result[f"shared_expert.{kind}_proj.weight"]=getattr(fused,f"{kind}_weight")
        q=sample_compatible_inverse(key,seed+5)
        result["shared_expert_gate.weight"]=module.shared_expert_gate.weight.detach().float().cpu()@(q*gamma).T
    return result


def build_expanded_decoder_model(baseline, tokenizer, *, expansion_size=4, lam=.1, seed=20261020,
                              kappa=None, attention_profile="rqk_hqk_block_taukv_taugroup",
                              alpha_e=0., alpha_h=0.):
    """全层使用同一残差P，各线性投影使用相容的Q；只在离线阶段构造这些矩阵。

    全部norm共用一个kappa，以便使用HF的全局rms_norm_eps。未提供kappa时用
    ||P||_F/sqrt(d)，即各方向均匀分布时的均方伸缩估计，不冒称真实语料校准。
    """
    if baseline.config.model_type not in ("qwen2", "llama","qwen2_moe","mixtral"):
        raise ValueError("Expanded export supports Qwen2/Llama/Qwen2-MoE/Mixtral")
    if type(expansion_size) is not int or expansion_size <= 0 or expansion_size % 2:
        raise ValueError("expansion_size must be a positive even integer")
    if any(not math.isfinite(x) or x < 0 for x in (lam,alpha_e,alpha_h)):
        raise ValueError("lambda and noise scales must be finite and nonnegative")
    if kappa is not None and (not math.isfinite(kappa) or kappa <= 0):
        raise ValueError("kappa must be finite and positive")
    d=int(baseline.config.hidden_size)
    transform=build_keymat_transform(d,expansion_size,lam,init_seed=seed)
    p=transform.key.float()
    if not torch.isfinite(p).all() or not torch.isfinite(transform.inverse).all() or not torch.allclose(p@transform.inverse,torch.eye(d),atol=1e-4,rtol=1e-4):
        raise ValueError("Sampled residual matrices are not numerically compatible in float32")
    kappa=float(p.double().square().sum().sqrt()/math.sqrt(d)) if kappa is None else float(kappa)
    c=kappa*math.sqrt(d/transform.expanded_size)
    scalar_eps=c*c*float(baseline.config.rms_norm_eps)
    if not math.isfinite(scalar_eps) or scalar_eps < torch.finfo(torch.float32).tiny or not math.isfinite(c) or not torch.finfo(torch.float32).tiny <= c <= torch.finfo(torch.float32).max:
        raise ValueError("Scalar normalization parameters are outside finite float32 range")
    config=deepcopy(baseline.config)
    config.hidden_size=transform.expanded_size
    config.head_dim=int(baseline.model.layers[0].self_attn.head_dim)
    config.rms_norm_eps=scalar_eps
    config.tie_word_embeddings=False
    config._attn_implementation="eager"
    # 普通HF模型的参数形状在新config下创建，服务器不需要此构建模块。
    # 先只创建形状，再将融合后的参数直接装入；避免多分配一份随机大模型权重。
    with torch.device("meta"):
        model=AutoModelForCausalLM.from_config(config,trust_remote_code=False).float().eval()
    movable=ordinary_token_ids(tokenizer)
    perm=generate_vocab_permutation(config.vocab_size,seed+1,movable)
    inv=invert_permutation(perm)
    state={}
    e=baseline.get_input_embeddings().weight.detach().float().cpu()
    h=baseline.get_output_embeddings().weight.detach().float().cpu()
    state["model.embed_tokens.weight"]=(add_embed_noise(e,alpha_e,seed+2,movable)@p)[inv].contiguous()
    final_gamma=baseline.model.norm.weight.detach().float().cpu()
    head_inverse=sample_compatible_inverse(transform,seed+3)
    state["lm_head.weight"]=((add_head_noise(h,alpha_h,seed+4,movable)*final_gamma)@head_inverse.T)[inv].contiguous()
    state["model.norm.weight"]=torch.full((config.hidden_size,),c)
    for index,layer in enumerate(baseline.model.layers):
        prefix=f"model.layers.{index}"
        state[f"{prefix}.input_layernorm.weight"]=torch.full((config.hidden_size,),c)
        state[f"{prefix}.post_attention_layernorm.weight"]=torch.full((config.hidden_size,),c)
        with random_namespace(f"decoder.layer.{index}.attention"):
            attention=StaticizedQwen2Attention(layer.self_attn,transform,layer.input_layernorm.weight,
                        recorder=None,layer_idx=index,attention_profile=attention_profile,seed=seed+10000+index)
        for kind in ("q","k","v","o"):
            state[f"{prefix}.self_attn.{kind}_proj.weight"]=getattr(attention,f"{kind}_weight").contiguous()
            bias=getattr(attention,f"{kind}_bias")
            if bias is not None:
                state[f"{prefix}.self_attn.{kind}_proj.bias"]=bias.contiguous()
        if hasattr(layer.mlp,"experts"):
            with random_namespace(f"decoder.layer.{index}.moe"):
                moe=_fuse_expert_mlp(layer.mlp,transform,layer.post_attention_layernorm.weight.detach().float().cpu(),seed+50000+index*1000)
            state.update({f"{prefix}.mlp.{name}":value.contiguous() for name,value in moe.items()})
            continue
        ff_width=layer.mlp.gate_proj.out_features
        generator=torch.Generator().manual_seed(seed+20000+index)
        stream=random_stream("expanded.ffn_permutation",seed+20000+index)
        ff=build_hidden_transform(stream.permutation(ff_width) if stream is not None else torch.randperm(ff_width,generator=generator),
                    generate_hidden_scaling(ff_width,scale_range=(.95,1.05),seed=seed+30000+index))
        with random_namespace(f"decoder.layer.{index}.ffn"):
            mlp=KeyMatFusedQwen2MLP(layer.mlp,transform,layer.post_attention_layernorm.weight,ff,seed=seed+40000+index)
        for kind in ("gate","up","down"):
            state[f"{prefix}.mlp.{kind}_proj.weight"]=getattr(mlp,f"{kind}_weight").contiguous()
            bias=getattr(mlp,f"{kind}_bias")
            if bias is not None:
                state[f"{prefix}.mlp.{kind}_proj.bias"]=bias.contiguous()
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Non-finite fused weight; export was not written")
    model.load_state_dict(state,strict=True,assign=True)
    for module in model.modules():
        if hasattr(module,"compute_default_rope_parameters") and hasattr(module,"inv_freq") and module.inv_freq.device.type=="meta":
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            rope_type = module.config.rope_parameters.get("rope_type", "default")
            initialize = (module.compute_default_rope_parameters if rope_type == "default"
                          else ROPE_INIT_FUNCTIONS[rope_type])
            frequency,scaling=initialize(module.config,device=torch.device("cpu"))
            module.inv_freq=frequency;module.original_inv_freq=frequency.clone();module.attention_scaling=scaling
    remaining=[name for name,value in model.named_buffers() if value.device.type=="meta"]
    if remaining:raise ValueError(f"Unmaterialized model buffers require explicit support: {remaining}")
    return model,perm,inv,transform,dict(kappa=kappa,coefficient=c,scalar_epsilon=config.rms_norm_eps,
           original_hidden_size=d,expanded_hidden_size=config.hidden_size,attention_head_dim=config.head_dim)


@secure_export
def export_expanded_decoder_checkpoint(export_dir, *, model_dir, expansion_size=4, lam=.1, seed=None,
                                     kappa=None, attention_profile="rqk_hqk_block_taukv_taugroup",alpha_e=0.,alpha_h=0.,expected_model_type=None):
    """导出独立server/client目录；这是扩维标量近似版本，需要另行测真实任务质量。"""
    from src.model_loader import load_model_and_tokenizer, set_global_seed
    source,target=Path(model_dir).resolve(),Path(export_dir).resolve()
    if source==target or source in target.parents or target in source.parents:
        raise ValueError("Source and target directories must not overlap")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Export directory must be empty")
    actual_seed=secrets.randbits(63) if seed is None else int(seed)
    set_global_seed(actual_seed)
    tokenizer,baseline=load_model_and_tokenizer(str(source),device="cpu",dtype="float32")
    model_type=baseline.config.model_type
    if expected_model_type is not None and model_type!=expected_model_type:
        raise ValueError(f"Expected {expected_model_type}, got {model_type}")
    model,perm,inv,transform,details=build_expanded_decoder_model(baseline,tokenizer,expansion_size=expansion_size,
        lam=lam,seed=actual_seed,kappa=kappa,attention_profile=attention_profile,alpha_e=alpha_e,alpha_h=alpha_h)
    metadata=dict(stage="J",track=f"expanded_{model_type}_scalar_approximation",model_line=model_type,
        implementation_variant="algorithm1_residual_with_scalar_norm",paper_algorithm1_reproduced=False,
        requires_custom_runtime=False,seed=actual_seed,seed_source="system_random" if seed is None else "explicit_reproducible",
        dtype="float32",alpha_e=alpha_e,alpha_h=alpha_h,num_layers=model.config.num_hidden_layers,
        residual_matrix_construction="Algorithm 1: D=d+2h",normalization_mode="scalar_rms_approximation",
        kappa_source="isotropic_frobenius_estimate" if kappa is None else "caller_supplied_shared_scalar",
        router_normalization="unit_norm_per_expert_before_fusion" if model_type in ("qwen2_moe","mixtral") else "not_applicable",
        attention_profile=attention_profile,**details,**random_metadata(),
        security_scope="Experimental standard-HF expanded export; scalar norm and possible RoPE block approximation; no full paper, downstream-quality or privacy replication claim.",
        **summarize_token_partitions(tokenizer=tokenizer,model_vocab_size=model.config.vocab_size,perm_vocab=perm))
    paths=export_stage_i_vllm_checkpoint(target,tokenizer=tokenizer,stage_a_model=model,perm_vocab=perm,inv_perm_vocab=inv,metadata=metadata)
    # 便于客户端后续误差分析；绝不写入server/。
    secret=torch.load(paths["client_secret_path"],map_location="cpu",weights_only=True)
    secret["residual_key"]=transform.key
    secret["residual_inverse"]=transform.inverse
    torch.save(secret,paths["client_secret_path"])
    # Windows映射大型safetensors文件也会计入提交内存；保存后先释放两套模型再验收。
    del model,baseline
    gc.collect()
    proof=build_stage_j_standard_weight_proof(paths["server_dir"])
    if not proof["is_standard_weight_export"]:
        raise ValueError(f"Expanded checkpoint failed structural validation: {proof}")
    manifest_path=target/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(track=metadata["track"],paper_algorithm1_reproduced=False,standard_weight_proof=proof,
                    normalization_mode="scalar_rms_approximation",attention_profile=attention_profile,**details)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return {**paths,"manifest_path":manifest_path}


def build_expanded_qwen_model(baseline,tokenizer,**kwargs):
    """保留原Qwen2入口；通用构建不靠修改model_type来冒充另一种模型。"""
    if baseline.config.model_type!="qwen2":
        raise ValueError("Expected Qwen2 baseline")
    return build_expanded_decoder_model(baseline,tokenizer,**kwargs)


def export_expanded_qwen_checkpoint(export_dir,*,model_dir,**kwargs):
    return export_expanded_decoder_checkpoint(export_dir,model_dir=model_dir,expected_model_type="qwen2",**kwargs)
