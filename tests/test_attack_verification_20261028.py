"""IMA训练标准化的独立检查：训练统计、可逆还原和词表匹配的原始尺度。"""
import pytest
import torch
from src.security_qwen.ima import (fit_ima_standardization,normalize_ima_tensor,
    restore_ima_predictions,_rank_candidate_plain_ids)


def test_ima_training_statistics_and_restore_do_not_use_evaluation_data():
    x=torch.tensor([[[1.,10.],[3.,20.]],[[5.,30.],[7.,40.]]])
    y=torch.tensor([[[10.,1.],[10.,2.]],[[10.,3.],[10.,4.]]])
    stats=fit_ima_standardization(x,y)
    torch.testing.assert_close(stats['x_mean'],torch.tensor([4.,25.]))
    torch.testing.assert_close(stats['y_mean'],torch.tensor([10.,2.5]))
    assert stats['y_std'][0]==pytest.approx(1e-6)
    snapshot={key:value.clone() for key,value in stats.items()}
    normalized=normalize_ima_tensor(x,stats)
    torch.testing.assert_close(normalized.mean((0,1)),torch.zeros(2),atol=1e-6,rtol=0)
    torch.testing.assert_close(normalized.std((0,1),unbiased=False),torch.ones(2))
    unusual=y+1000
    torch.testing.assert_close(restore_ima_predictions(normalize_ima_tensor(unusual,stats,target=True),stats),unusual)
    for key in stats:torch.testing.assert_close(stats[key],snapshot[key],atol=0,rtol=0)


def test_ima_must_restore_output_scale_before_public_vocabulary_matching():
    public=torch.tensor([[10.,1.],[10.,2.],[0.,1.]])
    stats=fit_ima_standardization(torch.ones(1,2,3),public[:2][None])
    normalized=normalize_ima_tensor(public[1:2],stats,target=True)
    candidates=torch.arange(3)
    raw=_rank_candidate_plain_ids(predicted_embeddings=normalized,candidate_plain_ids=candidates,baseline_embed=public,topk=1)
    restored=_rank_candidate_plain_ids(predicted_embeddings=restore_ima_predictions(normalized,stats),candidate_plain_ids=candidates,baseline_embed=public,topk=1)
    assert raw.item()==2 and restored.item()==1


def test_ima_invalid_training_statistics_are_rejected():
    with pytest.raises(ValueError,match='nonempty'):fit_ima_standardization(torch.empty(0,2,3),torch.empty(0,2,4))
    with pytest.raises(ValueError,match='finite'):fit_ima_standardization(torch.full((1,2,3),float('nan')),torch.ones(1,2,4))
    stats=fit_ima_standardization(torch.ones(1,2,3),torch.ones(1,2,4))
    with pytest.raises(ValueError,match='width'):normalize_ima_tensor(torch.ones(1,2,5),stats)


def test_ima_cli_exposes_legacy_training_option(monkeypatch,tmp_path):
    import sys
    from scripts.evaluation.privacy import run_ima
    captured=[]
    monkeypatch.setattr(run_ima,'run_ima_paper_like',lambda **kw:captured.append(kw) or {})
    for disable in (False,True):
        args=['ima','--target','fixture','--mode','paper_like','--allow-simulated-known-pairs','--output-path',str(tmp_path/f'{disable}.json')]
        if disable:args.append('--disable-standardization')
        monkeypatch.setattr(sys,'argv',args);run_ima.main()
        assert captured[-1]['standardize'] is (not disable)
