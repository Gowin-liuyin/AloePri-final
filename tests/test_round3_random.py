"""随机源使用已知字节向量、私钥域分离和默认导出泄漏检查，不以统计测试证明安全。"""
import json
import pytest
import torch
from src.secure_random import CryptoRandom,secure_random_context,random_stream
from src.secure_random import random_namespace
from src.aloepri._implementation.export.expanded_llama_checkpoint import build_expanded_llama_model,export_expanded_llama_checkpoint
from src.aloepri._implementation.export.standard_square_checkpoint import export_standard_square_checkpoint


def test_chacha20_zero_key_known_block_and_chunking():
    expected=bytes.fromhex("76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586")
    assert CryptoRandom(b"\0"*32).bytes(64)==expected
    one=CryptoRandom(b"x"*32);two=CryptoRandom(b"x"*32)
    assert one.bytes(13)+one.bytes(70000)+one.bytes(201)==two.bytes(70214)


def test_random_domain_replay_and_context_cleanup():
    assert random_stream("test",1) is None
    with secure_random_context(b"a"*32):
        x=random_stream("gate",4).bytes(64)
        assert random_stream("gate",4).bytes(64)==x
        assert random_stream("up",4).bytes(64)!=x
        assert random_stream("gate",5).bytes(64)!=x
    assert random_stream("gate",4) is None
    with pytest.raises(RuntimeError):
        with secure_random_context(b"b"*32):raise RuntimeError()
    assert random_stream("gate",4) is None


def test_finite_normal_permutation_and_weighted_choices():
    rng=CryptoRandom(b"g"*32)
    values=rng.normal((100000,))
    assert torch.isfinite(values).all() and abs(float(values.mean()))<.02 and abs(float(values.std())-1)<.02
    assert rng.permutation(97).sort().values.tolist()==list(range(97))
    assert all(rng.categorical([0.,1.,0.])==1 for _ in range(20))
    with pytest.raises(ValueError):rng.randbelow(2**64+1)


def test_chunked_normal_matches_single_box_muller_sequence():
    import numpy as np
    count=262147
    actual=CryptoRandom(b"q"*32).normal((count,))
    unit=CryptoRandom(b"q"*32).uniform(((count+1)//2,2)).numpy()
    radius=np.sqrt(-2*np.log(unit[:,0]));angle=2*np.pi*unit[:,1]
    expected=torch.from_numpy(np.stack((radius*np.cos(angle),radius*np.sin(angle)),1).reshape(-1)[:count].copy())
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_layer_namespaces_do_not_reuse_random_streams_and_restore_after_errors():
    with secure_random_context(b"n"*32):
        base=random_stream("inverse",99).bytes(128)
        with random_namespace("layer.0"):
            first=random_stream("inverse",99).bytes(128)
            with pytest.raises(RuntimeError):
                with random_namespace("branch"):raise RuntimeError("test cleanup")
            assert random_stream("inverse",99).bytes(128)==first
        with random_namespace("layer.1"):
            second=random_stream("inverse",99).bytes(128)
        assert len({base,first,second})==3
        assert random_stream("inverse",99).bytes(128)==base


def test_secure_transform_sampling_does_not_use_torch_prng(monkeypatch):
    from tests.test_round3_models import llama_setup
    tokenizer,baseline=llama_setup()
    def forbidden(*_args,**_kwargs):raise AssertionError("Transform sampling reached Torch PRNG")
    monkeypatch.setattr(torch,"randn",forbidden);monkeypatch.setattr(torch,"randperm",forbidden)
    with secure_random_context(b"s"*32):
        model,perm,inv,transform,_=build_expanded_llama_model(baseline,tokenizer,expansion_size=2,seed=244,alpha_e=.1,alpha_h=.1)
    assert torch.isfinite(model.model.embed_tokens.weight).all()
    torch.testing.assert_close(transform.key@transform.inverse,torch.eye(16),atol=1e-5,rtol=1e-5)
    assert torch.equal(inv[perm],torch.arange(32))


@pytest.mark.parametrize("kind",["square","expanded"])
def test_default_export_uses_system_crypto_key_and_keeps_it_on_client(tmp_path,kind):
    from tests.test_round3_models import llama_setup
    tokenizer,baseline=llama_setup()
    original=tmp_path/"original";baseline.save_pretrained(original);tokenizer.save_pretrained(original)
    if kind=="square":
        paths=export_standard_square_checkpoint(tmp_path/"out",model_dir=original,expected_model_type="llama")
    else:
        paths=export_expanded_llama_checkpoint(tmp_path/"out",model_dir=original,expansion_size=2)
    secret=torch.load(paths["client_secret_path"],weights_only=True)
    master=secret["random_master_key"]
    assert isinstance(master,bytes) and len(master)==32
    assert secret["randomness"]["random_source"]=="ChaCha20_HMAC_SHA256"
    public=json.loads((tmp_path/"out/stage_i_metadata.json").read_text())
    assert public["random_source"]=="ChaCha20_HMAC_SHA256"
    for file in paths["server_dir"].iterdir():
        if file.is_file():assert master not in file.read_bytes()
    assert "random_master_key" not in public
