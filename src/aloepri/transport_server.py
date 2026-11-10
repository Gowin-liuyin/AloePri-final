"""带认证、TLS、限流和动态批处理的编号服务；一个模型由单独工作线程调用。

网络线程可以并发接收请求，模型线程把参数相同的请求合批，避免多个generate互相干扰。
这提供可运行的服务功能，不代表已经测出论文集群吞吐量。
"""
from __future__ import annotations
from collections import deque
from concurrent.futures import Future,TimeoutError
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from queue import Queue,Empty,Full
import hmac
import ipaddress
import json
import re
import ssl
import threading
import time
import torch
from src.aloepri.token_id_transport import validate_request,MAX_BODY_BYTES


def validate_token(token):
    if token is not None and (not isinstance(token,str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,512}",token)):
        raise ValueError("Authentication token must contain 32-512 URL-safe characters")
    return token


def validate_bind(host,auth_token,tls_cert,tls_key):
    validate_token(auth_token)
    if bool(tls_cert)!=bool(tls_key):
        raise ValueError("TLS certificate and private key must be provided together")
    try:local=ipaddress.ip_address(host).is_loopback
    except ValueError:local=host.lower()=="localhost"
    if not local and (auth_token is None or not tls_cert):
        raise ValueError("Non-loopback binding requires both authentication and TLS")


@torch.inference_mode()
def generate_server_batch(model,payloads):
    """左侧补位并合批生成；每条结果在自己的EOS处截断，去掉批处理额外补位。"""
    parsed=[validate_request(p,vocab_size=model.get_input_embeddings().num_embeddings,
                max_context=int(model.config.max_position_embeddings)) for p in payloads]
    if not parsed:
        return []
    kwargs=parsed[0][1]
    if any(k!=kwargs for _,k in parsed):
        raise ValueError("A batch must have identical generation options")
    if not kwargs["max_new_tokens"]:
        return [{"generated_ids":[]} for _ in parsed]
    kwargs=dict(kwargs)
    eos=kwargs.get("eos_token_id",model.generation_config.eos_token_id)
    eos_ids=set([] if eos is None else ([eos] if isinstance(eos,int) else eos))
    pad=kwargs.get("pad_token_id",model.generation_config.pad_token_id)
    if pad is None:pad=next(iter(eos_ids),0)
    kwargs["pad_token_id"]=pad
    width=max(len(ids) for ids,_ in parsed)
    device=model.get_input_embeddings().weight.device
    tokens=torch.full((len(parsed),width),pad,dtype=torch.long,device=device)
    mask=torch.zeros_like(tokens)
    for i,(ids,_) in enumerate(parsed):
        tokens[i,-len(ids):]=torch.tensor(ids,device=device)
        mask[i,-len(ids):]=1
    def check(_module,_args,output):
        if hasattr(output,"logits") and not torch.isfinite(output.logits).all():
            raise ValueError("Non-finite raw generation logits")
    handle=model.register_forward_hook(check)
    try:
        generated=model.generate(tokens,attention_mask=mask,**kwargs)[:,width:].cpu().tolist()
    finally:
        handle.remove()
    output=[]
    for row in generated:
        stop=next((i+1 for i,t in enumerate(row) if t in eos_ids),len(row))
        output.append({"generated_ids":row[:stop]})
    return output


class BatchScheduler:
    """有界队列和单模型工作线程；队列容量包含尚在等待匹配的请求。"""
    def __init__(self,model,*,max_batch_size=4,max_queue_size=64,batch_wait_ms=5):
        if any(type(x) is not int or x<=0 for x in (max_batch_size,max_queue_size)) or not 0<=batch_wait_ms<=1000:
            raise ValueError("Invalid batch/queue configuration")
        self.model=model;self.max_batch_size=max_batch_size;self.wait=batch_wait_ms/1000
        self.queue=Queue(max_queue_size);self.capacity=threading.BoundedSemaphore(max_queue_size)
        self.stop=threading.Event();self.lock=threading.Lock()
        self.stats=dict(submitted=0,completed=0,failed=0,batches=0,max_batch_seen=0)
        self.worker=threading.Thread(target=self._run,name="aloepri-generation",daemon=True);self.worker.start()

    def submit(self,payload):
        _,kwargs=validate_request(payload,vocab_size=self.model.get_input_embeddings().num_embeddings,
                       max_context=int(self.model.config.max_position_embeddings))
        if self.stop.is_set() or not self.capacity.acquire(blocking=False):
            raise Full("Generation queue is full")
        future=Future()
        try:self.queue.put_nowait((payload,json.dumps(kwargs,sort_keys=True),future))
        except Full:self.capacity.release();raise
        with self.lock:self.stats["submitted"]+=1
        return future

    def _finish(self,item,*,result=None,error=None):
        future=item[2]
        with self.lock:self.stats["failed" if error is not None else "completed"]+=1
        self.capacity.release()
        if not future.cancelled():
            if error is None:future.set_result(result)
            else:future.set_exception(error)

    def _run(self):
        backlog=deque()
        while not self.stop.is_set():
            try:first=backlog.popleft() if backlog else self.queue.get(timeout=.1)
            except Empty:continue
            if first[2].cancelled():self._finish(first,error=RuntimeError("Request cancelled"));continue
            batch=[first];deadline=time.monotonic()+self.wait
            # 优先使用之前等待相同参数的请求，再收取短时间内新到达的请求。
            remaining=deque()
            while backlog:
                item=backlog.popleft()
                if item[1]==first[1] and len(batch)<self.max_batch_size:batch.append(item)
                else:remaining.append(item)
            backlog=remaining
            while len(batch)<self.max_batch_size:
                try:item=self.queue.get(timeout=max(0,deadline-time.monotonic()))
                except Empty:break
                if item[1]==first[1]:batch.append(item)
                else:backlog.append(item)
                if time.monotonic()>=deadline:break
            live=[item for item in batch if item[2].set_running_or_notify_cancel()]
            for item in batch:
                if item not in live:self._finish(item,error=RuntimeError("Request cancelled"))
            if not live:continue
            with self.lock:
                self.stats["batches"]+=1;self.stats["max_batch_seen"]=max(self.stats["max_batch_seen"],len(live))
            try:
                outputs=generate_server_batch(self.model,[item[0] for item in live])
                for item,result in zip(live,outputs):self._finish(item,result=result)
            except Exception:
                for item in live:self._finish(item,error=RuntimeError("Model generation failed"))
        while True:
            try:backlog.append(self.queue.get_nowait())
            except Empty:break
        for item in backlog:self._finish(item,error=RuntimeError("Server stopped"))

    def snapshot(self):
        with self.lock:return dict(self.stats)

    def close(self):
        self.stop.set();self.worker.join(timeout=5)


def make_server(model,*,host="127.0.0.1",port=8765,auth_token=None,tls_cert=None,tls_key=None,
                max_batch_size=4,max_queue_size=64,batch_wait_ms=5,max_connections=32,request_timeout=120):
    validate_bind(host,auth_token,tls_cert,tls_key)
    if type(max_connections) is not int or max_connections<=0 or request_timeout<=0:
        raise ValueError("Connection limit and request timeout must be positive")
    context=None
    if tls_cert:
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version=ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(tls_cert),str(tls_key))
    scheduler=BatchScheduler(model,max_batch_size=max_batch_size,max_queue_size=max_queue_size,batch_wait_ms=batch_wait_ms)
    class Handler(BaseHTTPRequestHandler):
        server_version="AloePriTokenID/1"
        sys_version=""
        def setup(self):
            self.request.settimeout(15)
            super().setup()
        def log_message(self,*_args):pass
        def _send(self,status,data):
            raw=json.dumps(data,allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(raw)));self.send_header("Connection","close")
            if status==503:self.send_header("Retry-After","1")
            self.end_headers();self.close_connection=True
            try:self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):pass
        def _authenticated(self):
            if auth_token is None:return True
            header=self.headers.get("Authorization","")
            expected="Bearer "+auth_token
            if len(header)>520 or not hmac.compare_digest(header.encode(),expected.encode()):
                self._send(401,{"error":"Authentication required"});return False
            return True
        def do_GET(self):
            if not self._authenticated():return
            if self.path!="/health":self._send(404,{"error":"Unknown endpoint"});return
            self._send(200,dict(status="ready",protocol="token_ids_v1",scheduler=scheduler.snapshot()))
        def do_POST(self):
            if not self._authenticated():return
            if self.path!="/generate":self._send(404,{"error":"Use /generate"});return
            future=None
            try:
                if self.headers.get("Transfer-Encoding"):raise ValueError("Chunked bodies unsupported")
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<=MAX_BODY_BYTES:raise ValueError("Body size")
                raw=self.rfile.read(length)
                if len(raw)!=length:raise ValueError("Incomplete body")
                payload=json.loads(raw)
                future=scheduler.submit(payload)
                result=future.result(timeout=request_timeout)
            except Full:self._send(503,{"error":"Generation queue is full"});return
            except TimeoutError:
                if future is not None:future.cancel()
                self._send(504 if future is not None else 408,{"error":"Request timed out"});return
            except (ValueError,TypeError,KeyError,UnicodeError,OverflowError):
                self._send(400,{"error":"Invalid generation request"});return
            except Exception:self._send(500,{"error":"Generation failed"});return
            self._send(200,result)
    class Server(ThreadingHTTPServer):
        daemon_threads=True
        def __init__(self,*args,**kwargs):
            self.slots=threading.BoundedSemaphore(max_connections)
            super().__init__(*args,**kwargs)
            self.scheduler=scheduler
        def process_request(self,request,address):
            if not self.slots.acquire(blocking=False):
                try:request.sendall(b"HTTP/1.0 503 Busy\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                except OSError:pass
                self.shutdown_request(request);return
            try:super().process_request(request,address)
            except Exception:self.slots.release();raise
        def process_request_thread(self,*args):
            try:super().process_request_thread(*args)
            finally:self.slots.release()
        def server_close(self):
            super().server_close();scheduler.close()
    try:
        server=Server((host,port),Handler)
        if context:server.socket=context.wrap_socket(server.socket,server_side=True,do_handshake_on_connect=False)
        return server
    except Exception:scheduler.close();raise
