"""真正的TLS/认证/并发请求测试，不能用配置字段存在替代网络验证。"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from queue import Full
import ssl
import threading
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen
import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,LlamaConfig,LlamaForCausalLM
from src.aloepri.token_id_transport import make_server,remote_generate_ids
from src.aloepri.transport_server import BatchScheduler,generate_server_batch,validate_bind
from src.aloepri.transport_credentials import create_local_credentials


def model_for(family="qwen2"):
    torch.manual_seed(711)
    cls,config=(Qwen2ForCausalLM,Qwen2Config) if family=="qwen2" else (LlamaForCausalLM,LlamaConfig)
    return cls(config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,head_dim=4,max_position_embeddings=64,
        eos_token_id=None,pad_token_id=0)).eval()


@contextmanager
def running(model,**kwargs):
    server=make_server(model,port=0,**kwargs)
    thread=threading.Thread(target=server.serve_forever,kwargs=dict(poll_interval=.01),daemon=True);thread.start()
    scheme="https" if kwargs.get("tls_cert") else "http"
    try:yield server,f"{scheme}://127.0.0.1:{server.server_port}/generate"
    finally:server.shutdown();thread.join(timeout=5);server.server_close()


def request(url,ids,**kwargs):
    return remote_generate_ids(url,ids,perm_vocab=torch.arange(32),inv_perm_vocab=torch.arange(32),max_new_tokens=4,**kwargs)


def test_real_https_authentication_and_certificate_verification(tmp_path):
    files=create_local_credentials(tmp_path/"tls")
    token=open(files["auth_token.txt"]).read().strip()
    with running(model_for(),auth_token=token,tls_cert=files["server.crt"],tls_key=files["server.key"]) as (server,url):
        result=request(url,[3,7],auth_token=token,ca_file=files["server.crt"])
        assert len(result)==4
        for bad in (None,"A"*43):
            with pytest.raises(HTTPError) as error:request(url,[3,7],auth_token=bad,ca_file=files["server.crt"])
            assert error.value.code==401
        with pytest.raises(URLError):request(url,[3,7],auth_token=token)
        wrong=create_local_credentials(tmp_path/"wrong")
        with pytest.raises(URLError):request(url,[3,7],auth_token=token,ca_file=wrong["server.crt"])
        req=Request(url.replace("/generate","/health"),headers={"Authorization":"Bearer "+token})
        with urlopen(req,context=ssl.create_default_context(cafile=files["server.crt"])) as response:
            health=json.loads(response.read())
        assert health["status"]=="ready" and health["scheduler"]["completed"]==1
    with pytest.raises(ValueError,match="empty"):create_local_credentials(tmp_path/"tls")


def test_client_refuses_redirect_without_forwarding_authentication():
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_args):pass
        def do_POST(self):
            received.append((self.path,self.headers.get("Authorization")))
            self.send_response(307);self.send_header("Location",f"http://127.0.0.1:{self.server.server_port}/other")
            self.send_header("Content-Length","0");self.end_headers()
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    worker=threading.Thread(target=server.serve_forever,kwargs=dict(poll_interval=.01),daemon=True);worker.start()
    try:
        with pytest.raises(HTTPError) as error:
            request(f"http://127.0.0.1:{server.server_port}/generate",[3],auth_token="t"*43)
        assert error.value.code==307
        assert received==[("/generate","Bearer "+"t"*43)]
    finally:server.shutdown();worker.join(timeout=5);server.server_close()


def test_http_queue_overload_returns_503_and_recovers(monkeypatch):
    import src.aloepri.transport_server as transport
    entered,release=threading.Event(),threading.Event()
    original=transport.generate_server_batch
    def blocked(model,payloads):
        entered.set();assert release.wait(timeout=5)
        return original(model,payloads)
    monkeypatch.setattr(transport,"generate_server_batch",blocked)
    with running(model_for(),max_queue_size=1,batch_wait_ms=0) as (server,url):
        with ThreadPoolExecutor(max_workers=1) as pool:
            first=pool.submit(request,url,[3,7]);assert entered.wait(timeout=5)
            try:
                with pytest.raises(HTTPError) as error:request(url,[3,7])
                assert error.value.code==503 and error.value.headers["Retry-After"]=="1"
            finally:release.set()
            assert len(first.result(timeout=5))==4
        assert len(request(url,[3,7]))==4


@pytest.mark.parametrize("family",["qwen2","llama"])
def test_parallel_requests_are_batched_and_match_individual_greedy_generation(family):
    model=model_for(family)
    inputs=[[3,7],[4,8,9],[5,6,3,7],[13]]
    expected=[]
    with torch.no_grad():
        for ids in inputs:
            tokens=torch.tensor([ids]);expected.append(model.generate(tokens,attention_mask=torch.ones_like(tokens),max_new_tokens=4,do_sample=False)[0,len(ids):].tolist())
    barrier=threading.Barrier(4)
    with running(model,batch_wait_ms=100,max_batch_size=4) as (server,url):
        def call(ids):barrier.wait(timeout=5);return request(url,ids)
        with ThreadPoolExecutor(max_workers=4) as pool:actual=list(pool.map(call,inputs))
        assert actual==expected
        stats=server.scheduler.snapshot()
        assert stats["completed"]==4 and stats["max_batch_seen"]>=2


def test_batch_eos_trimming_matches_individual_stop():
    model=model_for()
    with torch.no_grad():eos=int(model(torch.tensor([[3,7]])).logits[0,-1].argmax())
    payloads=[dict(input_ids=ids,max_new_tokens=4,eos_token_id=[eos],pad_token_id=0) for ids in ([3,7],[9,4,7])]
    actual=generate_server_batch(model,payloads)
    assert actual[0]["generated_ids"]==[eos]
    for payload,result in zip(payloads,actual):
        ids=torch.tensor([payload["input_ids"]])
        with torch.no_grad():expected=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=4,eos_token_id=eos,pad_token_id=0)[0,ids.shape[1]:].tolist()
        assert result["generated_ids"]==expected


def test_queue_limit_and_recovery(monkeypatch):
    import src.aloepri.transport_server as transport
    entered,release=threading.Event(),threading.Event()
    original=transport.generate_server_batch
    def blocked(model,payloads):
        entered.set();assert release.wait(5);return original(model,payloads)
    monkeypatch.setattr(transport,"generate_server_batch",blocked)
    scheduler=BatchScheduler(model_for(),max_batch_size=1,max_queue_size=1)
    try:
        first=scheduler.submit(dict(input_ids=[3],max_new_tokens=1))
        assert entered.wait(5)
        with pytest.raises(Full):scheduler.submit(dict(input_ids=[3],max_new_tokens=1))
        release.set();assert len(first.result(timeout=5)["generated_ids"])==1
        assert len(scheduler.submit(dict(input_ids=[7],max_new_tokens=2)).result(timeout=5)["generated_ids"])==2
    finally:release.set();scheduler.close()


def test_mixed_generation_options_and_invalid_remote_binding():
    scheduler=BatchScheduler(model_for(),max_batch_size=2,batch_wait_ms=20)
    try:
        a=scheduler.submit(dict(input_ids=[3],max_new_tokens=1))
        b=scheduler.submit(dict(input_ids=[7],max_new_tokens=3))
        assert len(a.result(timeout=5)["generated_ids"])==1
        assert len(b.result(timeout=5)["generated_ids"])==3
    finally:scheduler.close()
    with pytest.raises(ValueError,match="Non-loopback"):validate_bind("0.0.0.0",None,None,None)
    with pytest.raises(ValueError,match="together"):validate_bind("127.0.0.1",None,"cert",None)
    with pytest.raises(ValueError,match="token"):validate_bind("127.0.0.1","too-short",None,None)


def test_client_will_not_forward_token_over_remote_plain_http():
    with pytest.raises(ValueError,match="HTTPS"):
        request("http://example.com/generate",[3],auth_token="A"*43)
