"""导出时的密码学随机源：系统256位密钥、HMAC域分离、ChaCha20字节流。

只替换变换矩阵、置换和噪声的随机来源；矩阵算法的隐私性质仍需独立论证。
显式seed的旧实验保留Torch采样，不能当成部署密钥。默认导出使用系统密钥。
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import hmac
import json
import math
from pathlib import Path
import secrets
import numpy as np
import torch

_MASTER=ContextVar("aloepri_random_master",default=None)
_NAMESPACE=ContextVar("aloepri_random_namespace",default=())


class CryptoRandom:
    """一个独立用途的字节流；相同私钥/用途/种子可复验，不依赖Torch PRNG。"""
    def __init__(self,key):
        from cryptography.hazmat.primitives.ciphers import Cipher,algorithms
        if not isinstance(key,bytes) or len(key)!=32:raise ValueError("Random key must contain 32 bytes")
        self.encryptor=Cipher(algorithms.ChaCha20(key,b"\0"*16),mode=None).encryptor()
        self.buffer=b"";self.offset=0

    def bytes(self,count):
        if count<0:raise ValueError("Negative random byte count")
        if count==0:return b""
        if count>=65536:
            prefix=self.buffer[self.offset:]
            self.buffer=b"";self.offset=0
            return prefix+self.encryptor.update(b"\0"*(count-len(prefix)))
        if len(self.buffer)-self.offset<count:
            self.buffer=self.buffer[self.offset:]+self.encryptor.update(b"\0"*65536);self.offset=0
        result=self.buffer[self.offset:self.offset+count];self.offset+=count
        return result

    def uniform(self,shape,low=0.,high=1.,dtype=torch.float64):
        if not math.isfinite(low) or not math.isfinite(high) or high<low:raise ValueError("Invalid uniform range")
        count=math.prod(shape)
        integers=np.frombuffer(self.bytes(count*8),dtype="<u8")>>11
        values=(integers.astype(np.float64)+.5)/(2**53)
        return torch.from_numpy((low+(high-low)*values).copy()).reshape(shape).to(dtype)

    def normal(self,shape,dtype=torch.float64):
        count=math.prod(shape)
        result=torch.empty(count,dtype=dtype)
        # 大词表的噪声也分块转换，避免同时保留数份float64中间数组。
        for start in range(0,count,262144):
            size=min(262144,count-start);pairs=(size+1)//2
            u=self.uniform((pairs,2)).numpy()
            radius=np.sqrt(-2*np.log(u[:,0]));angle=2*np.pi*u[:,1]
            values=np.stack((radius*np.cos(angle),radius*np.sin(angle)),axis=1).reshape(-1)[:size].copy()
            result[start:start+size]=torch.from_numpy(values).to(dtype)
        return result.reshape(shape)

    def randbelow(self,n):
        if type(n) is not int or not 0<n<=2**64:raise ValueError("Bound must be an integer in 1..2^64")
        limit=2**64-(2**64%n)
        while True:
            value=int.from_bytes(self.bytes(8),"little")
            if value<limit:return value%n

    def permutation(self,n):
        if type(n) is not int or n<0:raise ValueError("Permutation size must be a nonnegative integer")
        # Fisher-Yates；randbelow使用拒绝采样，不能用带偏差的简单取模。
        values=list(range(n))
        for i in range(n-1,0,-1):
            j=self.randbelow(i+1);values[i],values[j]=values[j],values[i]
        return torch.tensor(values,dtype=torch.long)

    def categorical(self,probabilities):
        probs=torch.as_tensor(probabilities,dtype=torch.float64)
        if probs.ndim!=1 or not probs.numel() or not torch.isfinite(probs).all() or (probs<0).any() or probs.sum()<=0:
            raise ValueError("Invalid categorical probabilities")
        u=float(self.uniform((1,)))
        return min(int(torch.searchsorted((probs/probs.sum()).cumsum(0),torch.tensor(u,dtype=torch.float64))),len(probs)-1)


@contextmanager
def secure_random_context(master_key):
    if master_key is not None and (not isinstance(master_key,bytes) or len(master_key)!=32):
        raise ValueError("Random master key must contain 32 bytes")
    token=_MASTER.set(master_key)
    try:yield
    finally:_MASTER.reset(token)


def random_stream(domain,seed):
    master=_MASTER.get()
    if master is None:return None
    message=json.dumps(["AloePri-random-v1",list(_NAMESPACE.get()),domain,int(seed)],ensure_ascii=True,separators=(",",":")).encode()
    return CryptoRandom(hmac.digest(master,message,"sha256"))


@contextmanager
def random_namespace(name):
    """层与模块各自分域，避免偏移seed偶然碰撞时复用同一条随机流。"""
    token=_NAMESPACE.set((*_NAMESPACE.get(),str(name)))
    try:yield
    finally:_NAMESPACE.reset(token)


def random_metadata():
    return dict(random_source="ChaCha20_HMAC_SHA256" if _MASTER.get() is not None else "torch_seeded_experiment",
                randomness_mode="system_entropy" if _MASTER.get() is not None else "explicit_seed_experiment")


def secure_export(function):
    """默认导出启用系统随机源，私钥仅写客户端文件；显式seed保持旧实验可复验。"""
    @wraps(function)
    def wrapped(*args,**kwargs):
        master=_MASTER.get()
        if master is None and kwargs.get("seed") is None:master=secrets.token_bytes(32)
        with secure_random_context(master):
            paths=function(*args,**kwargs)
            secret_path=Path(paths["client_secret_path"])
            secret=torch.load(secret_path,map_location="cpu",weights_only=True)
            secret["randomness"]=random_metadata()
            if master is not None:secret["random_master_key"]=master
            torch.save(secret,secret_path)
            return paths
    return wrapped
