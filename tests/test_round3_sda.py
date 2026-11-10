"""受约束SDA必须跨句一致；测试特意提供偏向同一个词的分数。"""
import pytest
import torch
from src.security_qwen.sda_transformer import _recover_consistent


class AlwaysSame(torch.nn.Module):
    def forward(self,x):
        logits=torch.zeros(*x.shape,10)
        logits[:,:,-3:]=torch.tensor([3.,2.,1.])
        return logits


def test_sda_global_mapping_does_not_change_for_repeated_symbols():
    sequences=[[80,31,80],[31,97,80],[]]
    result=_recover_consistent(AlwaysSame(),sequences,dict(max_length=20),3,"first_occurrence","cpu","corpus")
    assert result["predicted_sequences"]==[[0,1,0],[1,2,0],[]]
    assert len(result["observed_mapping"])==3
    assert result["prediction_uses_client_secret"] is False


def test_sda_mapping_scope_and_impossible_alphabet():
    seq=[[1,2],[2,1]]
    result=_recover_consistent(AlwaysSame(),seq,dict(max_length=20),3,"frequency_rank","cpu","sequence")
    assert result["predicted_sequences"]==[[0,1],[0,1]]
    with pytest.raises(ValueError,match="injective"):
        _recover_consistent(AlwaysSame(),[[1,2,3,4]],dict(max_length=20),3,"first_occurrence","cpu","corpus")
