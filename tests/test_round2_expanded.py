"""扩维导出的数学对照与标准HF加载：只检验小模型，不代替真实任务评测。"""
from copy import deepcopy
import json
import pytest
import torch
from torch import nn
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2Config,Qwen2ForCausalLM,Qwen2Tokenizer
from src.aloepri._implementation.export.expanded_qwen_checkpoint import build_expanded_qwen_model,export_expanded_qwen_checkpoint
from src.aloepri.private_inference import generate_mapped_ids


def setup_model():
    torch.manual_seed(101)
    vocab={"[UNK]":0,"[BOS]":1,"[EOS]":2,**{token:i+3 for i,token in enumerate(sorted(ByteLevel.alphabet()))}}
    tokenizer=Qwen2Tokenizer(vocab=vocab,merges=[],unk_token="[UNK]",bos_token="[BOS]",eos_token="[EOS]",pad_token="[EOS]")
    config=Qwen2Config(vocab_size=len(tokenizer),hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=4,max_position_embeddings=128,bos_token_id=1,eos_token_id=2,pad_token_id=2)
    return tokenizer,Qwen2ForCausalLM(config).eval()


@pytest.mark.parametrize("profile",["simplified","rqk_hqk"])
def test_expanded_model_matches_independent_directional_norm_reference(profile):
    tokenizer,baseline=setup_model()
    expanded,perm,inv,transform,details=build_expanded_qwen_model(baseline,tokenizer,expansion_size=2,seed=174,attention_profile=profile)
    assert expanded.config.hidden_size==20
    assert expanded.model.layers[0].self_attn.q_proj.out_features==16
    assert expanded.config.head_dim==4
    reference=deepcopy(baseline)
    class DirectionalNorm(nn.Module):
        # 原维度参考：用xP的RMS做除数，然后仍在原坐标中继续计算。
        def __init__(self,norm):
            super().__init__()
            self.weight=nn.Parameter(norm.weight.detach().clone())
        def forward(self,x):
            transformed=x.float()@transform.key
            divisor=(transformed.square().mean(-1,keepdim=True)+details["scalar_epsilon"]).sqrt()
            return (x.float()*details["coefficient"]/divisor*self.weight).to(x.dtype)
    for layer in reference.model.layers:
        layer.input_layernorm=DirectionalNorm(layer.input_layernorm)
        layer.post_attention_layernorm=DirectionalNorm(layer.post_attention_layernorm)
    reference.model.norm=DirectionalNorm(reference.model.norm)
    ids=torch.tensor([[4,7,9,5]])
    with torch.no_grad():
        actual=expanded(perm[ids]).logits[...,perm]
        expected=reference(ids).logits
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        output=generate_mapped_ids(expanded,ids,attention_mask=torch.ones_like(ids),perm_vocab=perm,inv_perm_vocab=inv,max_new_tokens=6)
        expected_tokens=reference.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=6,do_sample=False)[:,4:]
    assert torch.equal(output,expected_tokens)


def test_expanded_export_reloads_without_client_keys_and_records_approximation(tmp_path):
    tokenizer,baseline=setup_model()
    original,output=tmp_path/"original",tmp_path/"expanded"
    baseline.save_pretrained(original,max_shard_size="2KB")
    tokenizer.save_pretrained(original)
    paths=export_expanded_qwen_checkpoint(output,model_dir=original,expansion_size=2,seed=178)
    manifest=json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["paper_algorithm1_reproduced"] is False
    assert manifest["normalization_mode"]=="scalar_rms_approximation"
    assert manifest["standard_weight_proof"]["is_standard_weight_export"]
    assert manifest["original_hidden_size"]==16 and manifest["expanded_hidden_size"]==20
    secret=torch.load(paths["client_secret_path"],weights_only=True)
    assert secret["residual_key"].shape==(16,20)
    for file in paths["server_dir"].rglob("*"):
        assert "secret" not in file.name and "obfuscation_config" not in file.name
        if file.suffix==".json":
            text=file.read_text(encoding="utf-8")
            assert '"residual_key"' not in text and '"seed"' not in text
    loaded=Qwen2ForCausalLM.from_pretrained(paths["server_dir"],local_files_only=True).eval()
    built,*_=build_expanded_qwen_model(baseline,tokenizer,expansion_size=2,seed=178)
    ids=secret["perm_vocab"][torch.tensor([[4,7,9]])]
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits,built(ids).logits,atol=1e-6,rtol=1e-5)
        tokens=loaded.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=5)
    assert tokens.shape[1]>3
    with pytest.raises(ValueError,match="empty"):
        export_expanded_qwen_checkpoint(output,model_dir=original)


def test_expanded_invalid_parameters_fail_before_export(tmp_path):
    tokenizer,baseline=setup_model()
    for kwargs in (dict(expansion_size=3),dict(lam=float("nan")),dict(kappa=0),dict(alpha_e=-1)):
        with pytest.raises(ValueError):
            build_expanded_qwen_model(baseline,tokenizer,**kwargs)
