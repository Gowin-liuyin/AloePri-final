"""独立TFMA入口，参考桌面attack_lab/tfma.py：公开语料与观测编号分别计数。"""
from collections import Counter
from src.security_qwen.sda_transformer import _validate_sequences


def recover_frequency_mapping(reference_sequences, observed_sequences, *, topk=5):
    """按出现次数排名配对；并列次数不能确定唯一对应，结果中会明确标出。"""
    _validate_sequences(reference_sequences,allow_empty=True)
    _validate_sequences(observed_sequences,allow_empty=True)
    if type(topk) is not int or topk <= 0:
        raise ValueError("topk must be a positive integer")
    reference=Counter(t for seq in reference_sequences for t in seq)
    observed=Counter(t for seq in observed_sequences for t in seq)
    if not reference or not observed:
        raise ValueError("Reference and observations must both contain tokens")
    plain_order=sorted(reference,key=lambda token:(-reference[token],token))
    observed_order=sorted(observed,key=lambda token:(-observed[token],token))
    rp,ro=sum(reference.values()),sum(observed.values())
    rankings=[]
    for rank,token in enumerate(observed_order):
        first=plain_order[rank] if rank<len(plain_order) else -1
        rest=sorted(plain_order,key=lambda p:(abs(reference[p]/rp-observed[token]/ro),p))
        rankings.append(([first]+[p for p in rest if p!=first])[:topk])
    return dict(attack="tfma_frequency_rank",observed_ids=observed_order,
                predicted_plain_ids=[row[0] for row in rankings],topk_plain_ids=rankings,
                reference_total_tokens=rp,observed_total_tokens=ro,
                reference_has_frequency_ties=len(set(reference.values()))<len(reference),
                observed_has_frequency_ties=len(set(observed.values()))<len(observed),
                prediction_uses_client_secret=False,paper_attack_fully_reproduced=False,
                scope="Frequency rank top-1; remaining candidates use frequency difference; ties broken by numeric ID are not unique evidence; -1 means insufficient reference candidates")
