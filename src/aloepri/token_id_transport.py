"""本机HTTP token-ID实验接口：服务端只加载server权重，客户端保管置换。

默认只监听127.0.0.1；可配置认证、TLS及有界队列动态批处理。
正文传编号，不经过服务端的decode/encode，避免分词往返改变编号。
"""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
from urllib.request import Request,build_opener,HTTPSHandler,HTTPRedirectHandler,ProxyHandler
from urllib.parse import urlsplit
from urllib.error import HTTPError
import ssl
import torch


MAX_BODY_BYTES = 2*1024*1024


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def validate_request(payload, *, vocab_size, max_context):
    """约束请求大小和编号范围；一次只处理一条序列，协议不接收模型路径或密钥。"""
    allowed = {"input_ids", "max_new_tokens", "do_sample", "temperature", "top_k", "top_p", "eos_token_id", "pad_token_id"}
    if not isinstance(payload, dict) or set(payload)-allowed:
        raise ValueError("Unknown request fields")
    ids = payload.get("input_ids")
    if not isinstance(ids, list) or not ids or len(ids) > min(max_context, 16384):
        raise ValueError("input_ids must be a nonempty bounded list")
    for token in ids:
        _integer(token, "input ID", 0, vocab_size-1)
    count = _integer(payload.get("max_new_tokens", 32), "max_new_tokens", 0, min(max_context-len(ids), 4096))
    sample = payload.get("do_sample", False)
    if type(sample) is not bool:
        raise ValueError("do_sample must be boolean")
    kwargs = dict(max_new_tokens=count, do_sample=sample, use_cache=True)
    # 即使greedy模式未使用这些参数，也验证请求内容，避免NaN混入日志或结果。
    temperature, top_p = payload.get("temperature", .65), payload.get("top_p", .95)
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (temperature, top_p)) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0,1]")
    top_k = _integer(payload.get("top_k", min(20,vocab_size)), "top_k", 0, vocab_size)
    if sample:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
    if "eos_token_id" in payload:
        eos = payload["eos_token_id"]
        if not isinstance(eos, list) or not eos or len(eos)>32:
            raise ValueError("eos_token_id must be a bounded list")
        kwargs["eos_token_id"] = [_integer(i, "EOS", 0, vocab_size-1) for i in eos]
    if "pad_token_id" in payload:
        kwargs["pad_token_id"] = _integer(payload["pad_token_id"], "PAD", 0, vocab_size-1)
    return ids, kwargs


@torch.inference_mode()
def generate_server_ids(model, payload):
    """服务器在混淆编号中生成，复用KV cache；只返回新增编号，不返回logits。"""
    ids, kwargs = validate_request(payload, vocab_size=model.get_input_embeddings().num_embeddings,
                    max_context=int(model.config.max_position_embeddings))
    if kwargs["max_new_tokens"] == 0:
        return {"generated_ids": []}
    device = model.get_input_embeddings().weight.device
    def finite_logits(_module, _args, output):
        if hasattr(output, "logits") and not torch.isfinite(output.logits).all():
            raise ValueError("Non-finite raw logits")
    hook = model.register_forward_hook(finite_logits)
    try:
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        result = model.generate(tokens, attention_mask=torch.ones_like(tokens), **kwargs)
        return {"generated_ids": result[0, len(ids):].cpu().tolist()}
    finally:
        hook.remove()


def make_server(model,**kwargs):
    """保留旧入口，实际服务使用认证/TLS/批处理模块。"""
    from src.aloepri.transport_server import make_server as implementation
    return implementation(model,**kwargs)


def serve_model(server_dir, *, host="127.0.0.1", port=8765, dtype="float32", device="cpu", ready_file=None,
                auth_token_file=None,tls_cert=None,tls_key=None,max_batch_size=4,max_queue_size=64,batch_wait_ms=5,max_connections=32):
    """仅从server目录加载标准HF模型；没有client_secret_path参数。"""
    from transformers import AutoModelForCausalLM
    from src.model_loader import resolve_torch_dtype
    from src.aloepri._implementation.analysis.standard_weight_proof import build_stage_j_standard_weight_proof
    from src.aloepri.transport_server import validate_bind
    auth_token=Path(auth_token_file).read_text(encoding="utf-8").strip() if auth_token_file else None
    validate_bind(host,auth_token,tls_cert,tls_key)
    directory=Path(server_dir)
    if (directory/"obfuscation_config.json").exists() or list(directory.rglob("*client_secret*")):
        raise ValueError("Use an isolated server directory without debug configuration or client secrets")
    if not build_stage_j_standard_weight_proof(directory)["is_standard_weight_export"]:
        raise ValueError("Server checkpoint is not a complete standard model")
    model = AutoModelForCausalLM.from_pretrained(server_dir, local_files_only=True,
                trust_remote_code=False, dtype=resolve_torch_dtype(dtype)).to(device).eval()
    with make_server(model,host=host,port=port,auth_token=auth_token,tls_cert=tls_cert,tls_key=tls_key,
                     max_batch_size=max_batch_size,max_queue_size=max_queue_size,batch_wait_ms=batch_wait_ms,max_connections=max_connections) as server:
        if ready_file is not None:
            path = Path(ready_file)
            with path.open("x", encoding="utf-8") as handle:
                json.dump({"url": f"{'https' if tls_cert else 'http'}://{host}:{server.server_port}/generate"}, handle)
        server.serve_forever(poll_interval=.1)


def remote_generate_ids(url, input_ids, *, perm_vocab, inv_perm_vocab, max_new_tokens=32,
                        eos_token_ids=None, pad_token_id=None, timeout=120,auth_token=None,ca_file=None, **sampling):
    """客户端执行置换、HTTP请求和逆置换；HTTP正文从来不包含置换表。"""
    from src.key_manager import validate_permutation
    from src.aloepri.transport_server import validate_token
    validate_token(auth_token)
    address=urlsplit(url)
    if address.scheme not in ("http","https") or not address.hostname or address.username or address.password or address.fragment:
        raise ValueError("Use an HTTP(S) endpoint without embedded credentials or fragments")
    if auth_token and address.scheme!="https" and address.hostname not in ("127.0.0.1","localhost","::1"):
        raise ValueError("Authenticated remote requests require HTTPS")
    if ca_file and address.scheme!="https":
        raise ValueError("CA certificate requires HTTPS")
    perm = torch.as_tensor(perm_vocab, device="cpu")
    inv = torch.as_tensor(inv_perm_vocab, device="cpu")
    if perm.dtype != torch.long or inv.dtype != torch.long or not validate_permutation(perm) or inv.shape != perm.shape or not torch.equal(inv[perm], torch.arange(len(perm))):
        raise ValueError("Invalid permutation or inverse")
    ids = torch.as_tensor(input_ids)
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or not ids.numel() or ids.min()<0 or ids.max()>=len(perm):
        raise ValueError("input_ids must be a nonempty valid token-ID vector")
    payload = dict(input_ids=perm[ids.long().cpu()].tolist(), max_new_tokens=max_new_tokens, **sampling)
    if eos_token_ids is not None:
        payload["eos_token_id"] = [int(perm[_integer(i, "EOS", 0, len(perm)-1)]) for i in eos_token_ids]
    if pad_token_id is not None:
        payload["pad_token_id"] = int(perm[_integer(pad_token_id, "PAD", 0, len(perm)-1)])
    validate_request(payload, vocab_size=len(perm), max_context=32768)
    raw = json.dumps(payload, allow_nan=False).encode("utf-8")
    headers={"Content-Type":"application/json"}
    if auth_token:headers["Authorization"]="Bearer "+auth_token
    request = Request(url, data=raw, headers=headers, method="POST")
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl):
            raise HTTPError(req.full_url,code,"Generation endpoint redirects are not accepted",headers,fp)
    context=ssl.create_default_context(cafile=str(ca_file) if ca_file else None)
    opener=build_opener(ProxyHandler({}),NoRedirect(),HTTPSHandler(context=context))
    with opener.open(request, timeout=timeout) as response:
        body = response.read(MAX_BODY_BYTES+1)
    if len(body)>MAX_BODY_BYTES:
        raise ValueError("Server response too large")
    result = json.loads(body)
    generated = result.get("generated_ids")
    if not isinstance(generated, list) or len(generated)>max_new_tokens:
        raise ValueError("Invalid server response")
    for token in generated:
        _integer(token, "output ID", 0, len(perm)-1)
    return inv[torch.tensor(generated, dtype=torch.long)].tolist()


def tokenizer_roundtrip(tokenizer, ids):
    """检查把编号交给文字接口会不会改变序列；该检查通过也不证明其他序列都通过。"""
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    recovered = tokenizer.encode(text, add_special_tokens=False)
    return {"input_ids": list(ids), "reencoded_ids": recovered, "equal": list(ids)==recovered}
