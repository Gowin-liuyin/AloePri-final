"""只读取公开模型和服务器权重的 Gate-IA、VMA 与 Attn-IA 实验。

VMA的排序/L1/投票规则参考桌面 attack_lab/vma.py；不读取客户端置换或测试明文。
Attn-IA采用文档中明示的二维列空间解释，不能称为原文公式的唯一实现。
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from src.aloepri._implementation.export.tensor_io import read_safetensors_state


def checked_ids(values, size: int, name: str) -> torch.Tensor:
    """检查调用者独立提供的行号，避免重复、越界或小数被悄悄转成整数。"""
    ids = torch.as_tensor(values)
    if ids.ndim != 1 or not ids.numel() or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a nonempty integer vector")
    ids = ids.long().cpu()
    if ids.min() < 0 or ids.max() >= size or ids.unique().numel() != ids.numel():
        raise ValueError(f"{name} contains duplicate or out-of-range IDs")
    return ids


def nearest_l1(observed: torch.Tensor, plain: torch.Tensor, candidate_ids, *, topk=5,
               query_chunk=32, candidate_chunk=256) -> tuple[torch.Tensor, torch.Tensor]:
    """分批计算完整行的L1距离；相同距离按明文编号排序，结果不受分批大小影响。"""
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
           for x in (topk, query_chunk, candidate_chunk)):
        raise ValueError("topk and chunk sizes must be positive integers")
    x, y = torch.as_tensor(observed).double().cpu(), torch.as_tensor(plain).double().cpu()
    if x.ndim != 2 or y.ndim != 2 or not x.shape[0] or not y.shape[0] or not x.shape[1] or x.shape[1] != y.shape[1]:
        raise ValueError("Matching requires nonempty matrices with equal feature width")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Features contain NaN or infinity")
    ids = torch.as_tensor(candidate_ids, dtype=torch.long)
    if ids.ndim != 1 or ids.numel() != y.shape[0] or ids.unique().numel() != ids.numel():
        raise ValueError("candidate_ids must identify each candidate row once")
    order = ids.argsort(stable=True)
    ids, y = ids[order], y[order]
    k = min(topk, len(ids))
    all_ids, all_dist = [], []
    for start in range(0, len(x), query_chunk):
        q = x[start:start + query_chunk]
        best_d = torch.empty((len(q), 0), dtype=torch.float64)
        best_i = torch.empty((len(q), 0), dtype=torch.long)
        for j in range(0, len(y), candidate_chunk):
            distances = torch.cdist(q, y[j:j + candidate_chunk], p=1)
            if not torch.isfinite(distances).all():
                raise ValueError("Distance calculation overflowed")
            merged_d = torch.cat([best_d, distances], 1)
            merged_i = torch.cat([best_i, ids[j:j + candidate_chunk].expand(len(q), -1)], 1)
            # 两次稳定排序明确给出距离并列时的处理方法。
            by_id = merged_i.argsort(dim=1, stable=True)
            merged_d, merged_i = merged_d.gather(1, by_id), merged_i.gather(1, by_id)
            by_d = merged_d.argsort(dim=1, stable=True)[:, :k]
            best_d, best_i = merged_d.gather(1, by_d), merged_i.gather(1, by_d)
        all_ids.append(best_i)
        all_dist.append(best_d)
    return torch.cat(all_ids), torch.cat(all_dist)


def vote_rankings(rankings: Sequence[torch.Tensor], *, topk=5) -> torch.Tensor:
    """每组权重的第一候选投一票；并列时比较各组名次之和，再比较编号。"""
    if not rankings or topk <= 0:
        raise ValueError("At least one ranking and positive topk are required")
    shape = rankings[0].shape
    if len(shape) != 2 or any(r.shape != shape for r in rankings):
        raise ValueError("Rankings must have equal two-dimensional shapes")
    output = []
    for row in range(shape[0]):
        lists = [r[row].tolist() for r in rankings]
        votes = Counter(items[0] for items in lists)
        candidates = set(item for items in lists for item in items)
        rank_sum = {c: sum(items.index(c) if c in items else len(items) + 1 for items in lists) for c in candidates}
        output.append(sorted(candidates, key=lambda c: (-votes[c], rank_sum[c], c))[:min(topk, shape[1])])
    return torch.tensor(output, dtype=torch.long)


def _tensor(state: Mapping, key: str) -> torch.Tensor:
    if key not in state:
        raise ValueError(f"Required public weight missing: {key}")
    value = torch.as_tensor(state[key]).double().cpu()
    if not torch.isfinite(value).all():
        raise ValueError(f"Non-finite weight: {key}")
    return value


def _head(state: Mapping, config: Mapping) -> torch.Tensor:
    if "lm_head.weight" in state:
        head = _tensor(state, "lm_head.weight")
    elif config.get("tie_word_embeddings") is True:
        head = _tensor(state, "model.embed_tokens.weight")
    else:
        raise ValueError("lm_head.weight is missing and embeddings are not declared tied")
    return head


def _projection(state, layer, kind, *, fuse_norms=False):
    """HF存储矩阵为[输出,输入]；这里保持该格式，在相乘时再转置。"""
    block = "self_attn" if kind in ("q", "k") else "mlp"
    name = "gate" if kind == "router" else f"{kind}_proj"
    key = f"model.layers.{layer}.{block}.{name}.weight"
    if kind=="router" and key not in state:
        key=f"model.layers.{layer}.block_sparse_moe.gate.weight"
    value = _tensor(state, key)
    if fuse_norms and kind not in ("down", "router"):
        norm = "input_layernorm" if block == "self_attn" else "post_attention_layernorm"
        value = value * _tensor(state, f"model.layers.{layer}.{norm}.weight")[None, :]
    return value


def _ffn_projection(state,config,layer,kind,*,fuse_norms=False):
    """兼容普通FFN、HF打包专家及旧版逐专家权重。

    使用全部路由专家并合并中间通道，再统一排序/取均值；不假定两边专家编号相同。
    Qwen2-MoE的shared_expert不是路由专家，明确不把它混进此组合。
    """
    if config.get("model_type") not in ("qwen2_moe","mixtral"):
        return _projection(state,layer,kind,fuse_norms=fuse_norms)
    count=int(config.get("num_experts") or config.get("num_local_experts") or 0)
    if count<=0:
        raise ValueError("MoE config must declare a positive expert count")
    base=f"model.layers.{layer}.mlp"
    if not any(k.startswith(base+".experts.") for k in state):
        base=f"model.layers.{layer}.block_sparse_moe"
    packed=f"{base}.experts.{'down_proj' if kind=='down' else 'gate_up_proj'}"
    values=[]
    if packed in state:
        weight=_tensor(state,packed)
        if weight.ndim!=3 or weight.shape[0]!=count:
            raise ValueError("Packed expert tensor shape does not match config")
        if kind in ("gate","up"):
            if weight.shape[1]%2:
                raise ValueError("Packed gate/up width must be even")
            half=weight.shape[1]//2
            weight=weight[:,:half] if kind=="gate" else weight[:,half:]
        values=list(weight.unbind(0))
    else:
        old_names={"gate":"w1","up":"w3","down":"w2"}
        for expert in range(count):
            key=f"{base}.experts.{expert}.{kind}_proj.weight"
            if key not in state:
                key=f"{base}.experts.{expert}.{old_names[kind]}.weight"
            values.append(_tensor(state,key))
    result=torch.cat(values,dim=1 if kind=="down" else 0)
    if fuse_norms and kind!="down":
        result=result*_tensor(state,f"model.layers.{layer}.post_attention_layernorm.weight")[None,:]
    return result


def _bounded_product(a, b, max_elements):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible matrix dimensions: {tuple(a.shape)} @ {tuple(b.shape)}")
    if a.shape[0] * b.shape[1] > max_elements:
        raise ValueError("Full Table-9 matrix exceeds max_feature_elements; select other combinations or explicitly increase the limit. No quantile approximation was used.")
    return a @ b


def table9_features(state, config, *, layers=(0,), combinations=("gate",),
                    fuse_norms=False, normalize_router=True, max_feature_elements=16_000_000):
    """生成表9的六类组合；返回的每一行始终对应词表中的一个token。

    embed_head、qk产生词表平方大小的表，默认预算会明确拒绝过大计算。
    up/down含非均匀缩放时，排序未必保持不变，调用者不能把失败解释为程序缺陷。
    """
    if max_feature_elements <= 0 or not layers or not combinations:
        raise ValueError("Positive memory limit, layers and combinations are required")
    if len(set(layers)) != len(layers) or any(type(i) is not int or i < 0 for i in layers):
        raise ValueError("Layer indices must be distinct nonnegative integers")
    if len(set(combinations)) != len(combinations):
        raise ValueError("Combinations must not repeat (each source receives one vote)")
    embed = _tensor(state, "model.embed_tokens.weight")
    output = {}
    for kind in combinations:
        if kind == "embed_head":
            head = _head(state, config)
            if fuse_norms:
                head = head * _tensor(state, "model.norm.weight")[None, :]
            output[kind] = _bounded_product(embed, head.T, max_feature_elements)
            continue
        for layer in layers:
            if kind in ("gate", "up", "router"):
                w = (_projection(state,layer,kind,fuse_norms=fuse_norms) if kind=="router" else
                     _ffn_projection(state,config,layer,kind,fuse_norms=fuse_norms))
                if kind == "router" and normalize_router:
                    # 只在原侧归一化每个专家向量；服务器权重已经包含此步骤，不能再次归一化。
                    w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)
                output[f"{kind}:{layer}"] = _bounded_product(embed, w.T, max_feature_elements)
            elif kind == "down_head":
                head = _head(state, config)
                if fuse_norms:
                    head = head * _tensor(state, "model.norm.weight")[None, :]
                output[f"{kind}:{layer}"] = _bounded_product(head, _ffn_projection(state,config,layer,"down"), max_feature_elements)
            elif kind == "qk":
                q = _bounded_product(embed, _projection(state, layer, "q", fuse_norms=fuse_norms).T, max_feature_elements)
                k = _bounded_product(embed, _projection(state, layer, "k", fuse_norms=fuse_norms).T, max_feature_elements)
                heads, kv = int(config["num_attention_heads"]), int(config.get("num_key_value_heads", config["num_attention_heads"]))
                width = int(config.get("head_dim") or q.shape[1] // heads)
                if heads <= 0 or kv <= 0 or heads % kv or q.shape[1] != heads * width or k.shape[1] != kv * width:
                    raise ValueError("Invalid attention head dimensions")
                if len(embed) * len(embed) * heads > max_feature_elements:
                    raise ValueError("Combined QK heads exceed max_feature_elements")
                head_tables = []
                for h in range(heads):
                    kh = h // (heads // kv)
                    head_tables.append(_bounded_product(q[:, h*width:(h+1)*width], k[:, kh*width:(kh+1)*width].T, max_feature_elements))
                # 将各头的表拼成同一行后排序，才能同时消除词表列置换和头置换。
                if sum(t.numel() for t in head_tables) > max_feature_elements:
                    raise ValueError("Combined QK heads exceed max_feature_elements")
                output[f"qk:{layer}"] = torch.cat(head_tables, dim=1)
            else:
                raise ValueError(f"Unknown Table-9 combination: {kind}")
    return output


def attention_leverage_features(state, config, *, layers=(0,), fuse_norms=False,
                                max_feature_elements=16_000_000):
    """Attn-IA的一种解释：每个RoPE二维块Q的对角元素 diag(Q @ pinv(Q))。

    用SVD求列空间投影，避免直接求逆。Qwen/Llama的RoPE配对是前后半部，非相邻列。
    保留所有词表行求投影，再选候选；提前截断词表会改变这个量。
    """
    features, ranks = [], []
    e = _tensor(state, "model.embed_tokens.weight")
    for layer in layers:
        q = _bounded_product(e, _projection(state, layer, "q", fuse_norms=fuse_norms).T, max_feature_elements)
        heads = int(config["num_attention_heads"])
        width = int(config.get("head_dim") or q.shape[1] // heads)
        if heads <= 0 or width <= 0 or width % 2 or q.shape[1] != heads * width:
            raise ValueError("Attn-IA requires valid even RoPE head dimensions")
        for h in range(heads):
            for j in range(width // 2):
                block = q[:, [h*width+j, h*width+j+width//2]]
                u, s, _ = torch.linalg.svd(block, full_matrices=False)
                threshold = torch.finfo(torch.float64).eps * max(block.shape) * s.max()
                rank = int((s > threshold).sum())
                ranks.append(rank)
                features.append(u[:, :rank].square().sum(dim=1))
    if not features:
        raise ValueError("No attention blocks selected")
    return torch.stack(features, dim=1).sort(dim=1).values, ranks


def recover_weight_mapping(plain_state, server_state, plain_config, server_config, *,
                           attack="vma", candidate_ids=None, observed_ids=None, layers=(0,),
                           combinations=("gate",), topk=5, fuse_original_norms=False,
                           max_feature_elements=16_000_000):
    """预测入口：参数中没有密钥或真实对应表；评分请在预测保存后另行完成。"""
    n = _tensor(plain_state, "model.embed_tokens.weight").shape[0]
    m = _tensor(server_state, "model.embed_tokens.weight").shape[0]
    if n != m:
        raise ValueError("These attacks require a vocabulary permutation with unchanged vocabulary size")
    candidates = checked_ids(list(range(n)) if candidate_ids is None else candidate_ids, n, "candidate_ids")
    observed = checked_ids(list(range(m)) if observed_ids is None else observed_ids, m, "observed_ids")
    notes, ranks = [], None
    if plain_config.get("model_type") in ("qwen2_moe","mixtral"):
        notes.append("MoE: all routed expert channels are combined before matching; expert numbering need not agree; shared experts are not included in these features.")
    if attack in ("vma", "gate-ia"):
        combos = ("gate",) if attack == "gate-ia" else tuple(combinations)
        a = table9_features(plain_state, plain_config, layers=layers, combinations=combos, fuse_norms=fuse_original_norms, max_feature_elements=max_feature_elements)
        b = table9_features(server_state, server_config, layers=layers, combinations=combos, normalize_router=False, max_feature_elements=max_feature_elements)
        if attack == "gate-ia":
            x = torch.stack([v.mean(1) for v in a.values()], 1)
            y = torch.stack([v.mean(1) for v in b.values()], 1)
            scale = x.std(0, unbiased=False).clamp_min(1e-12)
            a, b = {"gate_means": x/scale}, {"gate_means": y/scale}
            notes.append("Gate means assume no per-column scaling and no added embedding noise.")
        else:
            a = {key: value.sort(1).values for key, value in a.items()}
            b = {key: value.sort(1).values for key, value in b.items()}
            notes.append("Sort complete rows, match L1, one top-1 vote per selected combination/layer; QK heads merged before sorting. Nonuniform scaling and noise may destroy the invariant.")
    elif attack == "attn-ia":
        x, ra = attention_leverage_features(plain_state, plain_config, layers=layers, fuse_norms=fuse_original_norms, max_feature_elements=max_feature_elements)
        y, rb = attention_leverage_features(server_state, server_config, layers=layers, max_feature_elements=max_feature_elements)
        a, b, ranks = {"leverage": x}, {"leverage": y}, {"plain": ra, "server": rb}
        notes.append("Implementation interpretation: sorted diagonal of the column-space projection for each RoPE pair; not a unique reconstruction of the ambiguous paper formula. SVD rank uses relative machine-precision cutoff.")
    else:
        raise ValueError("attack must be vma, gate-ia or attn-ia")
    rankings = []
    for key in a:
        ids, _ = nearest_l1(b[key][observed], a[key][candidates], candidates, topk=topk)
        rankings.append(ids)
    final = vote_rankings(rankings, topk=topk)
    return {"attack": attack, "observed_ids": observed.tolist(), "predicted_plain_ids": final[:, 0].tolist(),
            "topk_plain_ids": final.tolist(), "sources": list(a), "assumptions": notes,
            "block_ranks": ranks, "prediction_uses_client_secret": False,
            "fuse_original_norms": fuse_original_norms, "candidate_count": len(candidates),
            "paper_attack_fully_reproduced": False}


def recover_from_directories(baseline_model_dir, server_dir, **kwargs):
    """读取标准HF单文件或分片权重；自定义buffer产物需要先明确转换，不能猜测格式。"""
    configs = [json.loads((Path(p)/"config.json").read_text(encoding="utf-8")) for p in (baseline_model_dir, server_dir)]
    if any(c.get("model_type") not in ("qwen2", "llama", "qwen2_moe", "mixtral") for c in configs):
        raise ValueError("Supported HF layouts: Qwen2, Llama, Qwen2-MoE and Mixtral")
    states = [read_safetensors_state(p) for p in (baseline_model_dir, server_dir)]
    return recover_weight_mapping(*states, *configs, **kwargs)
