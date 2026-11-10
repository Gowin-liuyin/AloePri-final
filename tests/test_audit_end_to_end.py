"""独立端到端检查：真正保存、分片读取、发布、重新加载和多步生成。

这里的随机小模型只验证程序衔接，没有任何真实语言任务或隐私成功率声明。
"""
import json
import shutil

import pytest
import torch
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

from src.aloepri.checkpoints.qwen import export_qwen_paper_checkpoint
from src.aloepri.release.bundle import ReleaseProfile, export_release_bundle
from src.aloepri.private_inference import generate_private_text, load_release_profile, generate_mapped_ids
from src.transforms import restore_logits


@pytest.mark.parametrize("seed", [None, 314159])
def test_offline_sharded_model_to_release_and_private_generation(tmp_path, seed):
    torch.manual_seed(151)
    # Qwen2原生使用字节BPE；不能把WordLevel词表当作它的tokenizer。
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2,
             **{token: i + 3 for i, token in enumerate(sorted(ByteLevel.alphabet()))}}
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[], unk_token="[UNK]",
                               bos_token="[BOS]", eos_token="[EOS]", pad_token="[EOS]")
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         bos_token_id=1, eos_token_id=2, pad_token_id=2, tie_word_embeddings=False)
    baseline = Qwen2ForCausalLM(config).eval()
    original, changed, release = tmp_path / "original", tmp_path / "changed", tmp_path / "release"
    baseline.save_pretrained(original, max_shard_size="2KB")
    tokenizer.save_pretrained(original)
    assert (original / "model.safetensors.index.json").is_file()
    ids = tokenizer("word2 word4", return_tensors="pt")["input_ids"]
    assert ids.shape[1] > 0
    reloaded_ids = AutoTokenizer.from_pretrained(original, local_files_only=True)("word2 word4", return_tensors="pt")["input_ids"]
    assert torch.equal(reloaded_ids, ids)

    paths = export_qwen_paper_checkpoint(changed, model_dir=original, seed=seed,
                                         alpha_e=0., alpha_h=0., device="cpu", dtype="float32")
    manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["paper_algorithm1_reproduced"] is False
    export_release_bundle(release, profiles=[ReleaseProfile("default", str(changed), "小模型检查", "本地验证")],
                          materialize=True, reference_profile="default")
    loaded = load_release_profile(release, "default")
    assert torch.equal(loaded["tokenizer"]("word2 word4", return_tensors="pt")["input_ids"], ids)
    with torch.no_grad():
        mapped_ids = loaded["perm_vocab"][ids]
        raw = loaded["model"](mapped_ids).logits
        torch.testing.assert_close(restore_logits(raw, loaded["perm_vocab"]), baseline(ids).logits,
                                   atol=1e-6, rtol=1e-5)
        expected = baseline.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                     use_cache=True, max_new_tokens=8)[:, ids.shape[1]:]
        observed = generate_mapped_ids(loaded["model"], ids, attention_mask=torch.ones_like(ids),
                                      perm_vocab=loaded["perm_vocab"], inv_perm_vocab=loaded["inv_perm_vocab"],
                                      max_new_tokens=8)
    assert torch.equal(observed, expected)
    text = generate_private_text(bundle_dir=release, profile="default", prompt="word2 word4", max_new_tokens=8)
    assert text == tokenizer.decode(expected[0], skip_special_tokens=True)

    # 仅复制server到独立目录；框架加载和前向不需要客户端秘密。
    catalog = json.loads((release / "catalog.json").read_text(encoding="utf-8"))
    server_copy = tmp_path / "server_only"
    shutil.copytree(release / catalog["profiles"][0]["server_dir"], server_copy)
    assert not any(path.name == "client_secret.pt" for path in server_copy.rglob("*"))
    independent = Qwen2ForCausalLM.from_pretrained(server_copy, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(independent(mapped_ids).logits, raw, atol=1e-6, rtol=1e-5)
