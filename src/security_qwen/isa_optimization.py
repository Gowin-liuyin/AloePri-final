"""ISA迭代优化实验：冻结公开模型，只调整猜测的输入向量，使内部状态接近观测。

此模块不读取密钥。states模式要求观测与公开模型指定层处于同一坐标空间。
cosine_gram模式比较各位置间的余弦相似度，允许未知正交变换和等比例扩维；
它不能消除一般的非等比例变换，也不能保证唯一恢复输入。
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from src.security_qwen.weight_attacks import checked_ids


def observation_features(states,mode):
    """cosine_gram去掉未知正交/等比例表示的影响；一般各向异性矩阵不满足此前提。"""
    if mode=="states":return states
    if mode!="cosine_gram":raise ValueError("observation_mode must be states or cosine_gram")
    if states.ndim!=3 or states.shape[1]<2:raise ValueError("Gram matching requires at least two observed positions")
    unit=F.normalize(states,dim=-1,eps=1e-8)
    return unit@unit.transpose(-1,-2)


def optimize_input_embeddings(observation_fn, target_states, initial_embeddings, *,
                              steps=200, learning_rate=.05, prior_weight=0.,observation_mode="states"):
    """按观测误差迭代更新输入；返回误差最低的猜测和逐步损失。

    observation_fn必须允许对输入求导，不能带no_grad。模型参数由外层冻结。
    prior_weight可约束猜测不要离初值太远；默认不加此约束。
    """
    if type(steps) is not int or steps <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError("Invalid optimization parameters")
    initial = torch.as_tensor(initial_embeddings).detach().clone()
    target = torch.as_tensor(target_states, device=initial.device, dtype=initial.dtype).detach()
    target_features=observation_features(target,observation_mode)
    if initial.ndim != 3 or not initial.numel() or not torch.isfinite(initial).all() or not torch.isfinite(target).all():
        raise ValueError("Nonempty finite [batch, sequence, embedding] inputs are required")
    x = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=learning_rate)
    history, best, best_x = [], float("inf"), None
    for step in range(steps+1):
        output = observation_fn(x)
        output_features=observation_features(output,observation_mode)
        if output_features.shape != target_features.shape:
            raise ValueError(f"Observed and predicted states must share coordinates and shape: {tuple(target.shape)} versus {tuple(output.shape)}")
        mse = (output_features-target_features).square().mean()
        loss = mse + prior_weight*(x-initial).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("ISA optimization loss became non-finite")
        history.append(dict(step=step, state_mse=float(mse.detach()), objective=float(loss.detach())))
        if float(loss.detach()) < best:
            best, best_x = float(loss.detach()), x.detach().clone()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if x.grad is None or not torch.isfinite(x.grad).all():
                raise ValueError("ISA input gradient missing or non-finite")
            optimizer.step()
    return {"embeddings": best_x, "history": history, "best_objective": best}


def run_isa_optimization(model, observed_states, *, candidate_ids=None, layer=-1, initial_embeddings=None,
                         seed=20261020, steps=200, learning_rate=.05, prior_weight=0., topk=5,observation_mode="states",restarts=1):
    """冻结模型求输入，再与公开词表embedding比较余弦距离，得到token候选。"""
    from src.security_qwen.ima import _rank_candidate_plain_ids
    if type(topk) is not int or topk <= 0:
        raise ValueError("topk must be a positive integer")
    if type(restarts) is not int or restarts<=0:raise ValueError("restarts must be a positive integer")
    table = model.get_input_embeddings().weight.detach()
    target = torch.as_tensor(observed_states, device=table.device, dtype=table.dtype)
    if target.ndim != 3 or not target.numel():
        raise ValueError("observed_states must be nonempty [batch, sequence, hidden]")
    candidates = checked_ids(list(range(len(table))) if candidate_ids is None else candidate_ids, len(table), "candidate_ids").to(table.device)
    if initial_embeddings is None:
        generator = torch.Generator(device=table.device).manual_seed(seed)
        ids = candidates[torch.randint(len(candidates), target.shape[:2], device=table.device, generator=generator)]
        initial_embeddings = table[ids].clone()
    else:
        initial_embeddings = torch.as_tensor(initial_embeddings, device=table.device, dtype=table.dtype)
    if initial_embeddings.shape != (*target.shape[:2], table.shape[1]):
        raise ValueError("Initial embedding dimensions do not match batch, sequence and public embedding width")
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        def observe(x):
            output = model(inputs_embeds=x, output_hidden_states=True, use_cache=False, return_dict=True)
            if not -len(output.hidden_states) <= layer < len(output.hidden_states):
                raise ValueError("Requested hidden-state layer is out of range")
            return output.hidden_states[layer]
        best_result,restart_summaries=None,[]
        for restart in range(restarts):
            start=initial_embeddings
            if restart:
                generator=torch.Generator(device=table.device).manual_seed(seed+restart)
                choices=candidates[torch.randint(len(candidates),target.shape[:2],device=table.device,generator=generator)]
                start=table[choices].clone()
            attempt=optimize_input_embeddings(observe,target,start,steps=steps,learning_rate=learning_rate,
                       prior_weight=prior_weight,observation_mode=observation_mode)
            restart_summaries.append(dict(restart=restart,best_objective=attempt["best_objective"]))
            if best_result is None or attempt["best_objective"]<best_result["best_objective"]:best_result=attempt
        result=best_result
        with torch.no_grad():
            ids = _rank_candidate_plain_ids(predicted_embeddings=result["embeddings"].reshape(-1, table.shape[1]),
                       candidate_plain_ids=candidates, baseline_embed=table, topk=topk)
        return {"attack": "isa_optimization", "predicted_sequences": ids[:, 0].reshape(target.shape[:2]).cpu().tolist(),
                "topk_plain_ids": ids.reshape(*target.shape[:2], -1).cpu().tolist(),
                "history": result["history"], "best_objective": result["best_objective"], "layer": layer,
                "observation_mode":observation_mode,"restarts":restart_summaries,
                "prediction_uses_client_secret": False, "paper_attack_fully_reproduced": False,
                "observation_requirement": ("Same selected hidden-state coordinates required" if observation_mode=="states" else
                    "Cosine Gram only: invariant to unknown orthogonal/scaled isometric transforms; not invariant to general Algorithm-1 transforms; matching need not uniquely identify tokens")}
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, training in modes:
            module.training = training
