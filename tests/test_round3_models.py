"""实际Llama和MoE结构检查，不把模型名替换当作架构支持。"""
from copy import deepcopy
import json
import pytest
import torch
from torch import nn
from transformers import AutoConfig,AutoModelForCausalLM,LlamaConfig,LlamaForCausalLM
from src.aloepri._implementation.export.expanded_llama_checkpoint import build_expanded_llama_model,export_expanded_llama_checkpoint
from src.aloepri._implementation.export.expanded_qwen_checkpoint import build_expanded_decoder_model,export_expanded_decoder_checkpoint
from src.aloepri.private_inference import generate_mapped_ids
from src.security_qwen.weight_attacks import recover_from_directories,recover_weight_mapping


def llama_setup(bias=False):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    tok=Tokenizer(WordLevel({"[UNK]":0,"[BOS]":1,"[EOS]":2,**{f"t{i}":i for i in range(3,32)}},unk_token="[UNK]"))
    tok.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token="[UNK]",bos_token="[BOS]",eos_token="[EOS]",pad_token="[EOS]")
    torch.manual_seed(221)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=4,max_position_embeddings=64,attention_bias=bias,mlp_bias=bias,
        eos_token_id=2,pad_token_id=2,bos_token_id=1)).eval()
    if bias:
        with torch.no_grad():
            for name,param in model.named_parameters():
                if name.endswith("bias"):param.normal_(std=.01)
    return tokenizer,model


@pytest.mark.parametrize("bias",[False,True])
@pytest.mark.parametrize("profile",["simplified","rqk_hqk"])
def test_llama_expansion_matches_independent_reference_with_bias(bias,profile):
    tokenizer,model=llama_setup(bias)
    expanded,perm,inv,key,details=build_expanded_llama_model(model,tokenizer,expansion_size=2,seed=441,attention_profile=profile)
    reference=deepcopy(model)
    class Norm(nn.Module):
        def __init__(self,norm):
            super().__init__();self.weight=nn.Parameter(norm.weight.detach().clone())
        def forward(self,x):
            return x*details["coefficient"]*torch.rsqrt((x@key.key).square().mean(-1,keepdim=True)+details["scalar_epsilon"])*self.weight
    for layer in reference.model.layers:
        layer.input_layernorm=Norm(layer.input_layernorm)
        layer.post_attention_layernorm=Norm(layer.post_attention_layernorm)
    reference.model.norm=Norm(reference.model.norm)
    ids=torch.tensor([[3,7,9,4]])
    with torch.no_grad():
        torch.testing.assert_close(expanded(perm[ids]).logits[...,perm],reference(ids).logits,atol=3e-6,rtol=3e-5)
        expected=reference.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=4,do_sample=False)[:,4:]
        actual=generate_mapped_ids(expanded,ids,attention_mask=torch.ones_like(ids),perm_vocab=perm,inv_perm_vocab=inv,max_new_tokens=4)
    assert torch.equal(actual,expected)
    assert expanded.config.model_type=="llama" and expanded.config.hidden_size==20 and expanded.config.head_dim==4


def test_llama_expanded_save_reload_and_release(tmp_path):
    from src.aloepri.release.bundle import export_release_bundle,ReleaseProfile
    tokenizer,model=llama_setup(True)
    original=tmp_path/"original"
    model.save_pretrained(original,max_shard_size="2KB");tokenizer.save_pretrained(original)
    paths=export_expanded_llama_checkpoint(tmp_path/"expanded",model_dir=original,expansion_size=2,seed=221)
    loaded=AutoModelForCausalLM.from_pretrained(paths["server_dir"],local_files_only=True).eval()
    expected,*_=build_expanded_llama_model(model,tokenizer,expansion_size=2,seed=221)
    ids=torch.tensor([[3,7,9]])
    with torch.no_grad():torch.testing.assert_close(loaded(ids).logits,expected(ids).logits)
    export_release_bundle(tmp_path/"release",profiles=[ReleaseProfile("default",str(tmp_path/"expanded"),"Llama检查","数值对照")],reference_profile="default",materialize=True)
    manifest=json.loads(paths["manifest_path"].read_text())
    assert manifest["paper_algorithm1_reproduced"] is False


@pytest.mark.parametrize("family",["qwen2_moe","mixtral"])
@pytest.mark.parametrize("layout",["packed","individual"])
def test_actual_moe_layouts_all_combinations_and_expert_permutation(tmp_path,family,layout):
    from safetensors.torch import save_file
    torch.manual_seed(222)
    config=AutoConfig.for_model(family,vocab_size=32,hidden_size=16,intermediate_size=16,moe_intermediate_size=8,
        shared_expert_intermediate_size=16,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=4,
        num_experts=3,num_local_experts=3,num_experts_per_tok=2,max_position_embeddings=64)
    model=AutoModelForCausalLM.from_config(config).eval()
    plain={k:v.double().clone() for k,v in model.state_dict().items()}
    server={k:v.clone() for k,v in plain.items()}
    inv=torch.randperm(32);order=torch.tensor([2,0,1]);base="model.layers.0.mlp"
    for k in ("model.embed_tokens.weight","lm_head.weight"):server[k]=plain[k][inv]
    router=plain[base+".gate.weight"]
    server[base+".gate.weight"]=(router/router.norm(dim=1,keepdim=True))[order]
    for kind in ("gate_up_proj","down_proj"):
        server[f"{base}.experts.{kind}"]=plain[f"{base}.experts.{kind}"][order]
    if layout=="individual":
        for state in (plain,server):
            joined=state.pop(base+".experts.gate_up_proj");down=state.pop(base+".experts.down_proj");width=joined.shape[1]//2
            target=base if family=="qwen2_moe" else "model.layers.0.block_sparse_moe"
            if target!=base:state[target+".gate.weight"]=state.pop(base+".gate.weight")
            for expert in range(3):
                for kind,value in (("gate",joined[expert,:width]),("up",joined[expert,width:]),("down",down[expert])):
                    name=kind+"_proj" if family=="qwen2_moe" else {"gate":"w1","up":"w3","down":"w2"}[kind]
                    state[f"{target}.experts.{expert}.{name}.weight"]=value.contiguous()
    for name,state in (("plain",plain),("server",server)):
        folder=tmp_path/name;folder.mkdir();config.save_pretrained(folder)
        save_file({k:v.contiguous() for k,v in state.items()},str(folder/"model.safetensors"))
    for kind in ("gate","up","down_head","router","qk","embed_head"):
        result=recover_from_directories(tmp_path/"plain",tmp_path/"server",combinations=(kind,),topk=2)
        assert result["predicted_plain_ids"]==inv.tolist(),kind
    result=recover_weight_mapping(plain,server,config.to_dict(),config.to_dict(),attack="gate-ia",topk=2)
    assert result["predicted_plain_ids"]==inv.tolist()


@pytest.mark.parametrize("family",["qwen2_moe","mixtral"])
def test_moe_expansion_forward_and_native_reload_against_independent_reference(tmp_path,family):
    tokenizer,_=llama_setup()
    torch.manual_seed(261)
    config=AutoConfig.for_model(family,vocab_size=32,hidden_size=16,intermediate_size=16,moe_intermediate_size=8,
        shared_expert_intermediate_size=16,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=4,
        num_experts=3,num_local_experts=3,num_experts_per_tok=2,max_position_embeddings=64,eos_token_id=2,pad_token_id=2)
    baseline=AutoModelForCausalLM.from_config(config).eval()
    expanded,perm,inv,key,details=build_expanded_decoder_model(baseline,tokenizer,expansion_size=2,seed=261,attention_profile="rqk_hqk")
    reference=deepcopy(baseline)
    class Norm(nn.Module):
        def __init__(self,norm):super().__init__();self.weight=nn.Parameter(norm.weight.detach().clone())
        def forward(self,x):return x*details["coefficient"]*torch.rsqrt((x@key.key).square().mean(-1,keepdim=True)+details["scalar_epsilon"])*self.weight
    for layer in reference.model.layers:
        layer.input_layernorm=Norm(layer.input_layernorm);layer.post_attention_layernorm=Norm(layer.post_attention_layernorm)
        with torch.no_grad():layer.mlp.gate.weight.div_(layer.mlp.gate.weight.norm(dim=1,keepdim=True))
    reference.model.norm=Norm(reference.model.norm)
    ids=torch.tensor([[3,7,9]])
    with torch.no_grad():
        torch.testing.assert_close(expanded(perm[ids]).logits[...,perm],reference(ids).logits,atol=3e-6,rtol=3e-5)
        actual=generate_mapped_ids(expanded,ids,attention_mask=torch.ones_like(ids),perm_vocab=perm,inv_perm_vocab=inv,max_new_tokens=4)
        expected=reference.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=4,do_sample=False)[:,3:]
    assert torch.equal(actual,expected)
    original=tmp_path/"original";baseline.save_pretrained(original,max_shard_size="2KB");tokenizer.save_pretrained(original)
    paths=export_expanded_decoder_checkpoint(tmp_path/"expanded",model_dir=original,seed=261,expansion_size=2,attention_profile="rqk_hqk")
    loaded=AutoModelForCausalLM.from_pretrained(paths["server_dir"],local_files_only=True).eval()
    with torch.no_grad():torch.testing.assert_close(loaded(perm[ids]).logits,expanded(perm[ids]).logits)
    assert loaded.config.model_type==family and loaded.config.hidden_size==20


@pytest.mark.parametrize("family",["qwen2_moe","mixtral"])
def test_moe_serialized_proof_rejects_missing_wrong_and_duplicate_experts(family):
    from src.aloepri._implementation.analysis.standard_weight_proof import _moe_serialized_shapes
    base="model.layers.0.mlp"
    packed={base+".experts.gate_up_proj":(2,8,6),base+".experts.down_proj":(2,6,4),base+".gate.weight":(2,6)}
    target=base if family=="qwen2_moe" else "model.layers.0.block_sparse_moe"
    canonical={target+".gate.weight":(2,6)}
    for expert in range(2):
        for kind,shape in (("gate",(4,6)),("up",(4,6)),("down",(6,4))):
            name=kind+"_proj" if family=="qwen2_moe" else {"gate":"w1","up":"w3","down":"w2"}[kind]
            canonical[f"{target}.experts.{expert}.{name}.weight"]=shape
    assert _moe_serialized_shapes(packed,canonical,family)==canonical
    bad=dict(canonical);bad.pop(next(k for k in bad if ".experts." in k))
    expected=_moe_serialized_shapes(packed,bad,family)
    assert set(expected)-set(bad)
    bad=dict(canonical);key=next(k for k in bad if ".experts." in k);bad[key]=(9,9)
    expected=_moe_serialized_shapes(packed,bad,family)
    assert expected[key]!=bad[key]
    bad={**canonical,**packed}
    assert set(bad)-set(_moe_serialized_shapes(packed,bad,family))


def test_square_indexing_matches_independent_dense_multiplication_and_keeps_original():
    from src.aloepri._implementation.export.square_transform import build_square_monomial_transform
    from src.aloepri._implementation.export.standard_shape_obfuscation import (
        adapt_input_linear_weight_for_square,adapt_output_linear_weight_for_square,
        adapt_output_bias_for_square,obfuscate_head_with_square_transform_stage_a,build_stage_j_square_model)
    key=build_square_monomial_transform(16,54,global_scale=1.7)
    x=torch.randn(7,16);y=torch.randn(16,9);b=torch.randn(16)
    torch.testing.assert_close(adapt_input_linear_weight_for_square(x,key),x@key.key())
    torch.testing.assert_close(adapt_output_linear_weight_for_square(y,key),key.key().T@y)
    torch.testing.assert_close(adapt_output_bias_for_square(b,key),b@key.key())
    torch.testing.assert_close(obfuscate_head_with_square_transform_stage_a(x,key,alpha_h=0),x@key.inverse().T)
    tokenizer,baseline=llama_setup();original={k:v.clone() for k,v in baseline.state_dict().items()}
    result,*_=build_stage_j_square_model(baseline,tokenizer,adapted_layers=[0,1],seed=54)
    for name,value in baseline.state_dict().items():torch.testing.assert_close(value,original[name],rtol=0,atol=0)
    assert all(value.device.type!="meta" for value in result.parameters())
