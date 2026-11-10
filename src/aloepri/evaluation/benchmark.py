"""本地任务对照：同一tokenizer/数据分别评原模型、同宽导出和扩维导出。

支持选择题条件似然、文本困惑度和生成精确匹配。数据由调用者提供，
不自动把自定义格式结果称为MMLU/HumanEval等官方评测结果。
"""
from __future__ import annotations
import gc
import json
import math
from pathlib import Path
import time
import torch
from src.key_manager import validate_permutation
from src.aloepri.private_inference import generate_mapped_ids


def load_records(path):
    records=[json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not records:raise ValueError("Evaluation dataset is empty")
    identifiers=[]
    for row in records:
        if not isinstance(row,dict) or not isinstance(row.get("id"),str) or not row["id"]:
            raise ValueError("Each record requires a nonempty string id")
        identifiers.append(row["id"])
        task=row.get("task")
        if task=="choice":
            choices=row.get("choices")
            if not isinstance(row.get("prompt"),str) or not isinstance(choices,dict) or len(choices)<2 or row.get("answer") not in choices:
                raise ValueError("Choice record requires prompt, >=2 choices and a valid answer")
            if any(not isinstance(k,str) or not isinstance(v,str) or not v for k,v in choices.items()):
                raise ValueError("Choice labels/text must be nonempty strings")
        elif task=="generation":
            if not isinstance(row.get("prompt"),str) or not isinstance(row.get("expected"),str):raise ValueError("Generation record requires prompt and expected text")
        elif task=="perplexity":
            if not isinstance(row.get("text"),str) or not row["text"]:raise ValueError("Perplexity record requires text")
        else:raise ValueError("task must be choice, generation or perplexity")
    if len(set(identifiers))!=len(identifiers):raise ValueError("Evaluation ids must be unique")
    return records


@torch.inference_mode()
def evaluate_model(model,tokenizer,records,*,perm_vocab=None,inv_perm_vocab=None,max_new_tokens=32,length_normalize=False):
    """这是模型持有者的评分程序，可使用客户端置换；不是攻击者预测入口。"""
    vocab=model.get_input_embeddings().num_embeddings
    perm=torch.arange(vocab) if perm_vocab is None else torch.as_tensor(perm_vocab,device="cpu")
    inv=torch.arange(vocab) if inv_perm_vocab is None else torch.as_tensor(inv_perm_vocab,device="cpu")
    if perm.dtype!=torch.long or inv.dtype!=torch.long or not validate_permutation(perm) or len(perm)!=vocab or inv.shape!=perm.shape or not torch.equal(inv[perm],torch.arange(vocab)):
        raise ValueError("Evaluation permutation is invalid")
    if type(max_new_tokens) is not int or max_new_tokens<=0:raise ValueError("max_new_tokens must be positive")
    device=model.get_input_embeddings().weight.device;context=int(model.config.max_position_embeddings)
    if context<2:raise ValueError("Model context must contain at least two positions")
    def encode(text):
        ids=tokenizer.encode(text,add_special_tokens=False)
        if not ids or min(ids)<0 or max(ids)>=vocab:raise ValueError("Text produced empty/out-of-range token IDs")
        return ids
    def likelihood(ids,start=1):
        if len(ids)>context or not 1<=start<len(ids):raise ValueError("Invalid scoring context/continuation")
        mapped=perm[torch.tensor(ids,dtype=torch.long)].to(device)[None,:]
        logits=model(mapped,use_cache=False).logits[0,:-1].float()
        if not torch.isfinite(logits).all():raise ValueError("Non-finite evaluation logits")
        selected=logits.log_softmax(-1).gather(1,mapped[0,1:,None]).squeeze(1)[start-1:]
        return float(selected.sum()),len(selected)
    output=[];loss_sum=0.;token_count=0;started=time.perf_counter()
    was_training=model.training;model.eval()
    try:
        for row in records:
            result=dict(id=row["id"],task=row["task"])
            if row["task"]=="choice":
                prefix=encode(row["prompt"]);scores={}
                for label,text in row["choices"].items():
                    continuation=encode(text)
                    score,count=likelihood(prefix+continuation,len(prefix))
                    scores[label]=score/count if length_normalize else score
                predicted=max(sorted(scores),key=scores.get)
                result.update(scores=scores,predicted=predicted,correct=predicted==row["answer"])
            elif row["task"]=="generation":
                prefix=encode(row["prompt"])
                if len(prefix)+max_new_tokens>context:raise ValueError("Generation exceeds model context")
                ids=torch.tensor([prefix],device=device)
                generated=generate_mapped_ids(model,ids,attention_mask=torch.ones_like(ids),perm_vocab=perm,inv_perm_vocab=inv,max_new_tokens=max_new_tokens)
                text=tokenizer.decode(generated[0].cpu(),skip_special_tokens=True)
                result.update(generated_text=text,generated_ids=generated[0].cpu().tolist(),correct=text.strip()==row["expected"].strip())
            else:
                ids=encode(row["text"])
                if len(ids)<2:raise ValueError("Perplexity text requires at least two tokens")
                score,count=0.,0
                # 相邻片段共享一个边界token，所有目标token恰好评分一次。
                for start in range(0,len(ids)-1,context-1):
                    part,n=likelihood(ids[start:start+context]);score+=part;count+=n
                loss_sum-=score;token_count+=count
                mean=-score/count
                result.update(scored_tokens=count,mean_nll=mean,perplexity=math.exp(mean) if mean<700 else None)
            output.append(result)
    finally:model.train(was_training)
    summary={}
    for task in ("choice","generation"):
        selected=[r for r in output if r["task"]==task]
        if selected:summary[task]=dict(count=len(selected),accuracy=sum(r["correct"] for r in selected)/len(selected))
    if token_count:
        mean=loss_sum/token_count
        summary["perplexity"]=dict(scored_tokens=token_count,mean_nll=mean,perplexity=math.exp(mean) if mean<700 else None)
    return dict(summary=summary,records=output,seconds=time.perf_counter()-started,
                conventions=dict(choice="prompt token IDs plus separately encoded continuation; sum log-probability" if not length_normalize else "mean continuation log-probability",
                  generation="greedy; exact text match after strip; no implicit chat template",perplexity="context-sized chunks sharing one boundary token"),
                scope="Caller-supplied local dataset; not an automatic claim of an official benchmark or paper replication")


def benchmark_directories(baseline_model_dir,dataset,checkpoints,*,device="cpu",dtype="float32",max_new_tokens=32,length_normalize=False):
    """逐个加载模型，避免将原模型和多个变体同时放进显存。checkpoints为(name,root)列表。"""
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from src.model_loader import resolve_torch_dtype
    names=[name for name,_ in checkpoints]
    if "baseline" in names or len(names)!=len(set(names)):raise ValueError("Checkpoint aliases must be unique and not baseline")
    rows=load_records(dataset)
    tokenizer=AutoTokenizer.from_pretrained(baseline_model_dir,local_files_only=True,trust_remote_code=False)
    result={}
    for name,folder in [("baseline",Path(baseline_model_dir)),*[(n,Path(p)) for n,p in checkpoints]]:
        perm=inv=None
        server=folder
        if name!="baseline":
            server=folder/"server"
            secret=torch.load(folder/"client/client_secret.pt",map_location="cpu",weights_only=True)
            perm,inv=secret["perm_vocab"],secret["inv_perm_vocab"]
        model=AutoModelForCausalLM.from_pretrained(server,local_files_only=True,trust_remote_code=False,dtype=resolve_torch_dtype(dtype)).to(device).eval()
        try:result[name]=evaluate_model(model,tokenizer,rows,perm_vocab=perm,inv_perm_vocab=inv,max_new_tokens=max_new_tokens,length_normalize=length_normalize)
        finally:
            del model;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    return dict(models=result,dataset=str(Path(dataset).resolve()),sample_count=len(rows),dtype=dtype,device=device)
