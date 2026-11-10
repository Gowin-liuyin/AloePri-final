"""新增攻击的独立数值检查：保持不变量时能恢复，条件失效时不能假称严格一致。"""
from copy import deepcopy
import json
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from src.security_qwen.weight_attacks import (recover_weight_mapping, recover_from_directories,
    table9_features, attention_leverage_features, nearest_l1, vote_rankings)
from src.security_qwen.sda_transformer import (encode_symbols, train_sda_transformer,
    recover_sda_transformer, SDATransformer)
from src.security_qwen.isa_optimization import optimize_input_embeddings, run_isa_optimization
from src.security_qwen.ima import build_paper_like_inverter_config, _PaperLikeIMAInverter
from src.security_qwen.frequency_mapping import recover_frequency_mapping


def weight_fixture():
    g = torch.Generator().manual_seed(832)
    rand = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64)
    n, d, ff, expanded = 17, 8, 12, 11
    e, h, p = rand(n,d), rand(n,d), rand(d,expanded)
    q = torch.linalg.pinv(p)
    inv = torch.randperm(n, generator=g)
    state, server = {"model.embed_tokens.weight":e, "lm_head.weight":h}, {"model.embed_tokens.weight":e[inv]@p, "lm_head.weight":h[inv]@q.T}
    for layer in range(2):
        z = torch.randperm(ff, generator=g)
        for kind in ("gate", "up"):
            key = f"model.layers.{layer}.mlp.{kind}_proj.weight"
            w = rand(ff,d)
            state[key], server[key] = w, w[z]@q.T
        key = f"model.layers.{layer}.mlp.down_proj.weight"
        w = rand(d,ff)
        state[key], server[key] = w, p.T@w[:,z]
        # 两个头交换；每个头内部二维RoPE块采用可逆变换。
        for kind in ("q", "k"):
            key = f"model.layers.{layer}.self_attn.{kind}_proj.weight"
            w = rand(8,d)
            head_order = torch.tensor([4,5,6,7,0,1,2,3])
            state[key], server[key] = w, w[head_order]@q.T
    config = dict(model_type="qwen2", hidden_size=d, num_attention_heads=2, num_key_value_heads=2, head_dim=4, tie_word_embeddings=False)
    return state, server, config, dict(config, hidden_size=expanded), inv


@pytest.mark.parametrize("attack,combinations", [
    ("gate-ia", ("gate",)), ("attn-ia", ("gate",)),
    *[("vma", (kind,)) for kind in ("gate", "up", "down_head", "embed_head", "qk")],
    ("vma", ("gate", "up", "down_head", "embed_head", "qk"))])
def test_weight_attacks_recover_rectangular_transform_and_head_permutation(attack, combinations):
    plain, server, a, b, truth = weight_fixture()
    result = recover_weight_mapping(plain, server, a, b, attack=attack, layers=(0,1), combinations=combinations, topk=3)
    assert result["predicted_plain_ids"] == truth.tolist()
    assert result["prediction_uses_client_secret"] is False


def test_attn_features_survive_nonorthogonal_rope_pair_transforms():
    plain, server, a, b, inv = weight_fixture()
    for layer in range(2):
        key = f"model.layers.{layer}.self_attn.q_proj.weight"
        w = server[key].clone()
        for base in (0,4):
            for j in (0,1):
                idx = [base+j,base+j+2]
                w[idx] = torch.tensor([[1.5,.2],[-.4,.8]], dtype=w.dtype)@w[idx]
        server[key] = w
    x,_ = attention_leverage_features(plain,a,layers=(0,1))
    y,_ = attention_leverage_features(server,b,layers=(0,1))
    torch.testing.assert_close(y,x[inv],atol=1e-12,rtol=1e-12)


def test_vma_nonuniform_scaling_changes_sorted_rows():
    plain, server, a, b, inv = weight_fixture()
    key = "model.layers.0.mlp.up_proj.weight"
    server[key] *= torch.linspace(.2,3,12,dtype=torch.float64)[:,None]
    x = table9_features(plain,a,combinations=("up",))["up:0"].sort(1).values
    y = table9_features(server,b,combinations=("up",))["up:0"].sort(1).values
    assert not torch.allclose(y,x[inv])
    result = recover_weight_mapping(plain,server,a,b,combinations=("up",))
    assert "Nonuniform scaling" in result["assumptions"][0]


def test_vma_candidates_are_not_augmented_with_truth_and_ties_are_stable():
    p,s,a,b,inv = weight_fixture()
    result = recover_weight_mapping(p,s,a,b,candidate_ids=[9,2,5],observed_ids=[0,4],topk=10)
    assert all(set(row)=={9,2,5} for row in result["topk_plain_ids"])
    for chunk in (1,2,8):
        ids,_ = nearest_l1(torch.zeros(2,4),torch.zeros(3,4),[9,2,5],topk=3,candidate_chunk=chunk)
        assert ids.tolist()==[[2,5,9],[2,5,9]]
    assert vote_rankings([torch.tensor([[2,5]]),torch.tensor([[5,2]])],topk=2).tolist()==[[2,5]]


def test_features_refuse_missing_router_and_unbounded_full_vocabulary_matrix():
    p,s,a,b,_=weight_fixture()
    with pytest.raises(ValueError,match="missing"):
        table9_features(p,a,combinations=("router",))
    with pytest.raises(ValueError,match="exceeds"):
        table9_features(p,a,combinations=("embed_head",),max_feature_elements=10)
    with pytest.raises(ValueError,match="repeat"):
        table9_features(p,a,combinations=("gate","gate"))
    p["model.embed_tokens.weight"][0,0]=float("nan")
    with pytest.raises(ValueError,match="Non-finite"):
        recover_weight_mapping(p,s,a,b)


def test_router_normalizes_original_expert_vectors_once():
    p,s,a,b,inv=weight_fixture()
    # 使用E的满秩关系构造同一P；恢复矩阵仅用来搭建模拟服务器，预测API不接收它。
    residual_p=torch.linalg.lstsq(p["model.embed_tokens.weight"][inv],s["model.embed_tokens.weight"]).solution
    inverse=torch.linalg.pinv(residual_p)
    torch.manual_seed(199)
    for layer in range(2):
        router=torch.randn(5,8,dtype=torch.float64)
        order=torch.tensor([2,4,1,0,3])
        p[f"model.layers.{layer}.mlp.gate.weight"]=router
        s[f"model.layers.{layer}.mlp.gate.weight"]=(router/router.norm(dim=1,keepdim=True))[order]@inverse.T
    result=recover_weight_mapping(p,s,a,b,combinations=("router",),layers=(0,1))
    assert result["predicted_plain_ids"]==inv.tolist()


def test_hf_shards_and_norm_fusion_option(tmp_path):
    from safetensors.torch import save_file
    p,s,a,b,inv=weight_fixture()
    # 把原始gamma乘到原侧投影，服务器保留已经吸收gamma后的版本。
    norm=torch.linspace(.5,1.5,8,dtype=torch.float64)
    for layer in range(2):
        p[f"model.layers.{layer}.post_attention_layernorm.weight"]=norm.clone()
        p[f"model.layers.{layer}.mlp.gate_proj.weight"]/=norm
    for directory,state,config in ((tmp_path/"original",p,a),(tmp_path/"server",s,b)):
        directory.mkdir()
        keys=list(state)
        midpoint=len(keys)//2
        chunks=[dict((k,state[k].contiguous()) for k in keys[:midpoint]),dict((k,state[k].contiguous()) for k in keys[midpoint:])]
        wm={}
        for i,chunk in enumerate(chunks):
            filename=f"part-{i}.safetensors"
            save_file(chunk,str(directory/filename))
            wm.update(dict.fromkeys(chunk,filename))
        (directory/"model.safetensors.index.json").write_text(json.dumps(dict(weight_map=wm)))
        (directory/"config.json").write_text(json.dumps(config))
    result=recover_from_directories(tmp_path/"original",tmp_path/"server",attack="gate-ia",layers=(0,1),fuse_original_norms=True)
    assert result["predicted_plain_ids"]==inv.tolist()


@pytest.mark.parametrize("mode",["first_occurrence","frequency_rank"])
def test_sda_encoding_is_invariant_even_when_frequencies_tie(mode):
    assert encode_symbols([8,3,3,8,9],mode)==encode_symbols([71,19,19,71,5],mode)
    assert encode_symbols([8,3,3,8,3,9],"first_occurrence")==[0,1,1,0,1,2]
    assert encode_symbols([8,3,3,8,3,9],"frequency_rank")==[1,0,0,1,0,2]


@pytest.mark.parametrize("device",["cpu",pytest.param("cuda",marks=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable"))])
def test_sda_trains_and_prediction_is_independent_of_symbol_names(tmp_path,device):
    path=tmp_path/f"sda-{device}.pt"
    result=train_sda_transformer([[0,1,0],[1,2],[2,0,1]],[[0,2,0],[2,1]],vocab_size=3,
           model_path=path,epochs=2,batch_size=2,hidden=16,device=device)
    assert result["history"][-1]["train_loss"]<result["history"][0]["train_loss"]
    a=recover_sda_transformer(path,[[8,3,8],[3,9],[]],device=device)
    b=recover_sda_transformer(path,[[72,10,72],[10,42],[]],device=device)
    assert a["predicted_sequences"]==b["predicted_sequences"]
    assert list(map(len,a["predicted_sequences"]))==[3,2,0]
    with pytest.raises(FileExistsError):
        train_sda_transformer([[0]],[[1]],vocab_size=3,model_path=path,epochs=1,hidden=16)
    with pytest.raises(ValueError,match="exceeds"):
        recover_sda_transformer(path,[[1]*20])


def test_sda_validation_and_causal_mask(tmp_path):
    for kw in (dict(epochs=0),dict(hidden=15)):
        with pytest.raises(ValueError):
            train_sda_transformer([[0]],[[1]],vocab_size=3,model_path=tmp_path/"bad.pt",**kw)
    with pytest.raises(ValueError,match="identical"):
        train_sda_transformer([[0]],[[0]],vocab_size=3,model_path=tmp_path/"same.pt")
    with pytest.raises(ValueError):
        train_sda_transformer([[]],[[0]],vocab_size=3,model_path=tmp_path/"empty.pt")
    model=SDATransformer(10,16,4,2,16).eval()
    with torch.no_grad():
        a=model(torch.tensor([[1,2,3,4]]))
        b=model(torch.tensor([[1,2,8,9]]))
    torch.testing.assert_close(a[:,:2],b[:,:2])


@pytest.mark.parametrize("width",[1,15,16,19,33])
def test_ima_adapter_runs_nonmultiple_dimensions_and_learns(tmp_path,width):
    Qwen2Config(hidden_size=16,num_attention_heads=2,num_key_value_heads=2,intermediate_size=32).save_pretrained(tmp_path)
    config=build_paper_like_inverter_config(observed_hidden_size=width,vocab_size=8,baseline_model_dir=str(tmp_path))
    model=_PaperLikeIMAInverter(backbone_config=config,target_embedding_dim=7)
    x=torch.randn(2,3,width)
    y=torch.randn(2,3,7)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    losses=[]
    for _ in range(5):
        optimizer.zero_grad()
        output=model(x)
        assert output.shape==y.shape
        loss=(output-y).square().mean()
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1]<losses[0]
    assert config.hidden_size%16==0 and config.head_dim%2==0


def test_isa_iterates_on_input_and_reduces_known_linear_state_error():
    torch.manual_seed(44)
    target=torch.randn(1,3,4)
    result=optimize_input_embeddings(lambda x:x@torch.diag(torch.tensor([1.,2.,3.,4.])),target,
        torch.zeros(1,3,4),steps=150,learning_rate=.1)
    assert min(row["state_mse"] for row in result["history"])<1e-5
    assert result["history"][0]["state_mse"]>result["best_objective"]*100


def test_independent_tfma_ranks_reports_ties_and_unknowns():
    result=recover_frequency_mapping([[0,0,0,1,1,2]],[[8,8,8,9,9,7]])
    assert result["observed_ids"]==[8,9,7]
    assert result["predicted_plain_ids"]==[0,1,2]
    assert not result["observed_has_frequency_ties"]
    result=recover_frequency_mapping([[0,0]],[[7,9]],topk=2)
    assert result["observed_has_frequency_ties"]
    assert result["predicted_plain_ids"]==[0,-1]
    with pytest.raises(ValueError):
        recover_frequency_mapping([[]],[[0]])


@pytest.mark.parametrize("device",["cpu",pytest.param("cuda",marks=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable"))])
def test_isa_on_actual_qwen_hidden_states_preserves_model_parameters(device):
    torch.manual_seed(155)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=16,hidden_size=8,intermediate_size=16,
          num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=4)).to(device).eval()
    ids=torch.tensor([[3,7,4]],device=device)
    with torch.no_grad():
        states=model(ids,output_hidden_states=True,use_cache=False).hidden_states[-1]
    snapshot={k:v.clone() for k,v in model.state_dict().items()}
    flags=[p.requires_grad for p in model.parameters()]
    result=run_isa_optimization(model,states,initial_embeddings=torch.zeros(1,3,8,device=device),steps=90,learning_rate=.005)
    assert result["best_objective"]<result["history"][0]["objective"]*.2
    assert [p.requires_grad for p in model.parameters()]==flags
    for k,v in model.state_dict().items():
        assert torch.equal(v,snapshot[k])
    assert not model.training
    with pytest.raises(ValueError,match="coordinates"):
        run_isa_optimization(model,torch.zeros(1,3,11,device=device),steps=1)
