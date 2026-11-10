"""独立评分：按人工标注的token区间计算完整PII实体恢复率。

此处明确采用同位置、整个实体所有token均恢复才成功的规则。
原论文只给出概括定义，不能据此声称已与作者PUPA评测脚本完全一致。
"""
from __future__ import annotations


def prepare_entity_truth(tokenizer,records):
    """将原文字符区间转换为token区间；仅供评分，不能把输出交给攻击预测。"""
    if not getattr(tokenizer,"is_fast",False):raise ValueError("Character annotations require a fast tokenizer with offsets")
    if not isinstance(records,list) or not records:raise ValueError("Annotated corpus is empty")
    output=[];seen=set()
    for row in records:
        text=row.get("text");identifier=row.get("id")
        if not isinstance(identifier,str) or identifier in seen or not isinstance(text,str) or not text:
            raise ValueError("Unique ids and nonempty annotated text are required")
        seen.add(identifier)
        encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        spans=[]
        for entity in row.get("entities",[]):
            start,end=entity.get("start"),entity.get("end")
            if type(start) is not int or type(end) is not int or not 0<=start<end<=len(text):raise ValueError("Invalid character span")
            positions=[i for i,(a,b) in enumerate(encoded["offset_mapping"]) if a<b and a<end and b>start]
            if not positions:raise ValueError("Entity annotation did not overlap any token")
            spans.append(dict(start=min(positions),end=max(positions)+1,label=str(entity.get("label","PII")),text=text[start:end]))
        output.append(dict(id=identifier,true_ids=encoded["input_ids"],entities=spans))
    return output


def evaluate_entities(predicted_sequences,truth):
    """未猜出(-1)计失败；分母包含全部原文token和全部标注实体，不筛选容易的部分。"""
    if not truth or len(predicted_sequences)!=len(truth):raise ValueError("Predictions and truth must align")
    correct_tokens=total_tokens=correct_entities=total_entities=0
    categories={};samples=[];identifiers=set()
    for prediction,row in zip(predicted_sequences,truth):
        original=row.get("true_ids");identifier=row.get("id")
        if not isinstance(identifier,str) or identifier in identifiers or not isinstance(original,list) or not original or len(prediction)!=len(original):
            raise ValueError("Unique sample ids and aligned nonempty token sequences are required")
        identifiers.add(identifier)
        if any(type(t) is not int or t<0 for t in original) or any(type(t) is not int or t< -1 for t in prediction):raise ValueError("Invalid token IDs")
        hits=[a==b for a,b in zip(prediction,original)]
        correct_tokens+=sum(hits);total_tokens+=len(hits)
        seen=set();successes=0
        for entity in row.get("entities",[]):
            start,end,label=entity.get("start"),entity.get("end"),str(entity.get("label","PII"))
            if type(start) is not int or type(end) is not int or not 0<=start<end<=len(original):raise ValueError("Invalid entity token span")
            key=(start,end,label)
            if key in seen:raise ValueError("Duplicate entity annotation")
            seen.add(key);hit=all(hits[start:end]);successes+=hit
            category=categories.setdefault(label,dict(correct=0,total=0));category["total"]+=1;category["correct"]+=hit
        total_entities+=len(seen);correct_entities+=successes
        samples.append(dict(id=identifier,tokens=len(hits),recovered_tokens=sum(hits),entities=len(seen),recovered_entities=successes))
    for row in categories.values():row["recovery_rate"]=row["correct"]/row["total"]
    return dict(TTRSR=correct_tokens/total_tokens,PIIRSR=correct_entities/total_entities if total_entities else None,
                total_tokens=total_tokens,total_entities=total_entities,recovered_entities=correct_entities,
                per_category=categories,samples=samples,matching_rule="exact aligned annotated token span; all entity tokens must match",
                paper_dataset_or_official_script_reproduced=False)
