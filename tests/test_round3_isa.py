"""坐标未知时只声明能证明的不变量，不能把Gram匹配泛化到任意矩阵。"""
import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from src.security_qwen.isa_optimization import observation_features,optimize_input_embeddings,run_isa_optimization


def test_gram_features_survive_unknown_rotation_and_isometric_expansion():
    torch.manual_seed(811)
    x=torch.randn(2,4,6,dtype=torch.float64)
    q,_=torch.linalg.qr(torch.randn(9,6,dtype=torch.float64))
    expanded=3*x@q.T
    torch.testing.assert_close(observation_features(x,"cosine_gram"),observation_features(expanded,"cosine_gram"))
    anisotropic=x*torch.tensor([.1,.5,1.,2.,3.,6.])
    assert not torch.allclose(observation_features(x,"cosine_gram"),observation_features(anisotropic,"cosine_gram"))
    with pytest.raises(ValueError,match="two"):observation_features(x[:,:1],"cosine_gram")


def test_gram_optimization_runs_with_different_observed_width():
    torch.manual_seed(812)
    truth=torch.randn(1,4,3)
    q,_=torch.linalg.qr(torch.randn(7,3))
    target=truth@q.T
    result=optimize_input_embeddings(lambda x:x,target,torch.randn(1,4,3),steps=120,learning_rate=.05,observation_mode="cosine_gram")
    assert result["best_objective"]<result["history"][0]["objective"]*.02


def test_actual_model_gram_and_multiple_initializations():
    torch.manual_seed(813)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,head_dim=4)).eval()
    with torch.no_grad():state=model(torch.tensor([[3,5,7]]),output_hidden_states=True,use_cache=False).hidden_states[-1]
    q,_=torch.linalg.qr(torch.randn(11,8))
    output=run_isa_optimization(model,state@q.T,steps=5,restarts=2,observation_mode="cosine_gram")
    assert len(output["restarts"])==2 and output["observation_mode"]=="cosine_gram"
    assert "not invariant to general" in output["observation_requirement"]
