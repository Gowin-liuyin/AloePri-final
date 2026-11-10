"""真正分进程请求token ID，并检验缓存生成、EOS和错误请求。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from src.aloepri.token_id_transport import remote_generate_ids, generate_server_ids, validate_request, tokenizer_roundtrip
from src.aloepri._implementation.analysis.rms_calibration import report_scalar_rms_error


def test_real_separate_process_server_only_weights_matches_baseline(tmp_path):
    torch.manual_seed(172)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
       num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=4,max_position_embeddings=128,
       bos_token_id=1,eos_token_id=2,pad_token_id=2,tie_word_embeddings=False)).eval()
    perm=torch.arange(32)
    perm[3:]=torch.randperm(29)+3
    inv=torch.argsort(perm)
    # 独立构造只有词表置换的模型，参考输出由原模型产生。
    server=Qwen2ForCausalLM(model.config).eval()
    server.load_state_dict(model.state_dict())
    with torch.no_grad():
        server.model.embed_tokens.weight.copy_(model.model.embed_tokens.weight[inv])
        server.lm_head.weight.copy_(model.lm_head.weight[inv])
    directory=tmp_path/"server_only"
    server.save_pretrained(directory,max_shard_size="2KB")
    assert not list(directory.rglob("*secret*"))
    ready=tmp_path/"ready.json"
    script=Path(__file__).resolve().parents[1]/"scripts/release/serve_token_ids.py"
    command=[sys.executable,str(script),"--server-dir",str(directory),"--port","0","--ready-file",str(ready)]
    with (tmp_path/"server.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,cwd=tmp_path,stdout=log,stderr=subprocess.STDOUT,
               creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            deadline=time.monotonic()+50
            while not ready.exists() and time.monotonic()<deadline:
                if process.poll() is not None:
                    pytest.fail((tmp_path/"server.log").read_text(encoding="utf-8"))
                time.sleep(.1)
            assert ready.exists(), "Server failed to start"
            url=json.loads(ready.read_text())["url"]
            ids=torch.tensor([[4,7,9]])
            with torch.no_grad():
                expected=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=6,do_sample=False,use_cache=True)[0,3:].tolist()
            actual=remote_generate_ids(url,ids[0],perm_vocab=perm,inv_perm_vocab=inv,max_new_tokens=6,eos_token_ids=[2],pad_token_id=2)
            assert actual==expected
            assert remote_generate_ids(url,[4],perm_vocab=perm,inv_perm_vocab=inv,max_new_tokens=0)==[]
            with pytest.raises(HTTPError) as err:
                urlopen(Request(url,data=json.dumps({"input_ids":[99]}).encode(),method="POST"))
            assert err.value.code==400
            # 请求出错后服务器仍可处理有效请求。
            assert remote_generate_ids(url,ids[0],perm_vocab=perm,inv_perm_vocab=inv,max_new_tokens=6)==expected
        finally:
            process.terminate()
            process.wait(timeout=15)


def test_server_generate_uses_cache_and_stops_at_eos():
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=16,
       num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=4,max_position_embeddings=128,
       eos_token_id=None,pad_token_id=0)).eval()
    lengths=[]
    handle=model.register_forward_pre_hook(lambda _module,args,kwargs:lengths.append(kwargs["input_ids"].shape[1]),with_kwargs=True)
    try:
        result=generate_server_ids(model,dict(input_ids=[4,7,9],max_new_tokens=4))
        assert lengths==[3,1,1,1]
        assert len(result["generated_ids"])==4
        # 将第一步最优token作为EOS，检验终止逻辑，而不是碰运气等随机模型输出EOS。
        eos=result["generated_ids"][0]
        stopped=generate_server_ids(model,dict(input_ids=[4,7,9],max_new_tokens=4,eos_token_id=[eos],pad_token_id=0))
        assert stopped["generated_ids"]==[eos]
    finally:
        handle.remove()


def test_server_defaults_also_allow_tiny_vocabularies():
    _,kwargs=validate_request(dict(input_ids=[1],max_new_tokens=1,do_sample=True),vocab_size=8,max_context=16)
    assert kwargs["top_k"]==8


@pytest.mark.parametrize("bad",[
    dict(input_ids=[]),dict(input_ids=[True]),dict(input_ids=[-1]),dict(input_ids=[0],max_new_tokens=-1),
    dict(input_ids=[0],max_new_tokens=1000),dict(input_ids=[0],temperature=float("nan")),
    dict(input_ids=[0],top_p=1.5),dict(input_ids=[0],eos_token_id=[32]),
    dict(input_ids=[0],client_secret="forbidden"),dict(input_ids=[0],do_sample="true"),
])
def test_transport_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        validate_request(bad,vocab_size=32,max_context=128)


def test_rms_calibration_exact_isometry_and_directional_failure():
    torch.manual_seed(9)
    train,test=torch.randn(30,3),torch.eye(3)
    p=torch.cat([2*torch.eye(3),torch.zeros(3,2)],1)
    exact=report_scalar_rms_error(p,train,test)
    assert exact["relative_error_max"]<1e-12
    p[0,0]=6
    error=report_scalar_rms_error(p,train,test)
    assert error["relative_error_max"]>.2
    assert error["condition_number"]==pytest.approx(3.)


def test_tokenizer_roundtrip_can_expose_lossy_text_api():
    class LossyTokenizer:
        def decode(self,ids,**kwargs):
            return "same text"
        def encode(self,text,**kwargs):
            return [1,2]
    assert tokenizer_roundtrip(LossyTokenizer(),[7,9])["equal"] is False
