"""SDA的因果Transformer实现，改编自桌面“攻击/attack_lab/sda.py”。

训练只需要公开明文。预测只接收替换后的符号，既不接收明文也不接收置换密钥。
first_occurrence和frequency_rank两种编码明确分开，不能用同一名称混称。
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def encode_symbols(sequence, mode="first_occurrence"):
    """重复符号用同一编号；同频符号按首次出现顺序编号，保持替换前后编码一致。"""
    first = dict.fromkeys(sequence)
    if mode == "first_occurrence":
        order = list(first)
    elif mode == "frequency_rank":
        counts = Counter(sequence)
        order = sorted(first, key=lambda item: -counts[item])  # Python稳定排序保留首次出现次序。
    else:
        raise ValueError("encoding must be first_occurrence or frequency_rank")
    ids = {item: i for i, item in enumerate(order)}
    return [ids[item] for item in sequence]


class SDATransformer(nn.Module):
    """用因果注意力读取编码和已经生成的明文，预测下一个token。"""
    def __init__(self, vocab_size, hidden=64, heads=4, layers=2, max_length=128):
        super().__init__()
        if hidden <= 0 or heads <= 0 or hidden % heads or layers <= 0 or max_length <= 0:
            raise ValueError("Invalid Transformer dimensions")
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.position = nn.Embedding(max_length, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden*4, dropout=0., batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, vocab_size)

    def forward(self, token_ids):
        length = token_ids.shape[1]
        positions = torch.arange(length, device=token_ids.device)[None, :]
        hidden = self.embedding(token_ids) + self.position(positions)
        mask = torch.ones(length, length, device=token_ids.device, dtype=torch.bool).triu(1)
        hidden = self.transformer(hidden, mask=mask, src_key_padding_mask=token_ids.eq(0))
        return self.output(self.norm(hidden))


def _validate_sequences(sequences, *, vocab_size=None, allow_empty=False):
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("Expected a nonempty list of token sequences")
    for sequence in sequences:
        if not isinstance(sequence, list) or (not sequence and not allow_empty):
            raise ValueError("Each sequence must be a nonempty list")
        if any(type(token) is not int or token < 0 or (vocab_size is not None and token >= vocab_size) for token in sequence):
            raise ValueError("Token IDs must be nonnegative integers within the vocabulary")


def _training_sequence(plain, n, mode):
    # 0用于补位；1..n用于编码；n+1分隔；n+2开始；n+3结束；明文从n+4开始。
    return [n+2] + [rank+1 for rank in encode_symbols(plain, mode)] + [n+1] + [n+4+t for t in plain] + [n+3]


def _collate(batch):
    """标签比输入向后移动一位；补位0不参与损失计算。"""
    length = max(map(len, batch))-1
    x = torch.zeros(len(batch), length, dtype=torch.long)
    y = torch.zeros_like(x)
    for i, seq in enumerate(batch):
        x[i, :len(seq)-1], y[i, :len(seq)-1] = torch.tensor(seq[:-1]), torch.tensor(seq[1:])
    return x, y


def train_sda_transformer(train_sequences, val_sequences, *, vocab_size, model_path,
                          epochs=12, batch_size=64, hidden=64, mode="first_occurrence",
                          seed=20261020, device="cpu", learning_rate=2e-3):
    """训练参数并按独立验证集损失选择轮次；拒绝覆盖已有模型文件。"""
    if any(type(x) is not int or x <= 0 for x in (vocab_size, epochs, batch_size, hidden)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Vocabulary size, epochs, batch size, hidden size and learning rate must be positive")
    encode_symbols([], mode)
    _validate_sequences(train_sequences, vocab_size=vocab_size)
    _validate_sequences(val_sequences, vocab_size=vocab_size)
    if set(map(tuple, train_sequences)) & set(map(tuple, val_sequences)):
        raise ValueError("Training and validation contain identical sentences")
    path = Path(model_path)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    torch.manual_seed(seed)
    max_length = 2*max(map(len, train_sequences+val_sequences))+3
    config = dict(vocab_size=2*vocab_size+4, hidden=hidden, heads=4, layers=2, max_length=max_length)
    model = SDATransformer(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    loaders = [DataLoader([_training_sequence(seq, vocab_size, mode) for seq in data],
               batch_size=batch_size, shuffle=i==0, collate_fn=_collate,
               generator=torch.Generator().manual_seed(seed)) for i, data in enumerate((train_sequences, val_sequences))]
    best, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        losses = []
        for training, loader in zip((True, False), loaders):
            model.train(training)
            total, count = 0., 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.set_grad_enabled(training):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, config["vocab_size"]), y.reshape(-1), ignore_index=0)
                    if not torch.isfinite(loss):
                        raise ValueError("SDA loss became non-finite")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                tokens = int(y.ne(0).sum())
                total += float(loss.detach())*tokens
                count += tokens
            losses.append(total/count)
        history.append(dict(epoch=epoch+1, train_loss=losses[0], validation_loss=losses[1]))
        if losses[1] < best:
            best = losses[1]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    # 只保存张量和基本类型，后续使用weights_only=True加载。
    with path.open("xb") as handle:
        torch.save(dict(config=config, state=best_state, plain_vocab_size=vocab_size, encoding=mode,
                        history=history, seed=seed, training_uses_client_secret=False), handle)
    return dict(history=history, best_validation_loss=best, encoding=mode, model_path=str(path))


@torch.inference_mode()
def recover_sda_transformer(model_path, observed_sequences, *, device="cpu", batch_size=32,mapping_scope="corpus"):
    """默认在整份观测中保持同一张一一映射；none保留旧的无约束批量解码。"""
    _validate_sequences(observed_sequences, allow_empty=True)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if mapping_scope not in ("corpus","sequence","none"):
        raise ValueError("mapping_scope must be corpus, sequence or none")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    c, n, mode = checkpoint["config"], checkpoint["plain_vocab_size"], checkpoint["encoding"]
    model = SDATransformer(**c).to(device).eval()
    model.load_state_dict(checkpoint["state"])
    if mapping_scope!="none":
        return _recover_consistent(model,observed_sequences,c,n,mode,device,mapping_scope)
    predictions = [[] for _ in observed_sequences]
    for length in sorted(set(map(len, observed_sequences))):
        if length == 0:
            continue
        if 2*length+2 > c["max_length"] or any(len(set(seq)) > n for seq in observed_sequences if len(seq)==length):
            raise ValueError("Observed sequence exceeds checkpoint context or symbol vocabulary")
        positions = [i for i, seq in enumerate(observed_sequences) if len(seq)==length]
        for start in range(0, len(positions), batch_size):
            indices = positions[start:start+batch_size]
            prefix = [[n+2]+[rank+1 for rank in encode_symbols(observed_sequences[i], mode)]+[n+1] for i in indices]
            current = torch.tensor(prefix, dtype=torch.long, device=device)
            generated = []
            for _ in range(length):
                logits = model(current)[:, -1, n+4:2*n+4]
                if not torch.isfinite(logits).all():
                    raise ValueError("SDA generation logits became non-finite")
                next_id = logits.argmax(-1)
                generated.append(next_id)
                current = torch.cat([current, (next_id+n+4)[:, None]], dim=1)
            for i, seq in zip(indices, torch.stack(generated, 1).cpu().tolist()):
                predictions[i] = seq
    return dict(attack="sda_transformer", predicted_sequences=predictions, encoding=mode,
                prediction_uses_client_secret=False, paper_attack_fully_reproduced=False,
                mapping_scope="none",scope="Unconstrained fixed-length decoding; no enforced global bijection")


def _recover_consistent(model,sequences,config,n,mode,device,mapping_scope):
    """按原输入顺序建立映射；重复符号强制相同明文，不同符号不能占用同一明文。

    首次遇到符号时使用神经网络分数选择尚未使用的词；这是受约束贪心解码，
    不是全局最优求解。它不接收正确答案，也不会用真实置换来补答案。
    """
    if mapping_scope=="corpus" and len({t for seq in sequences for t in seq})>n:
        raise ValueError("Observed symbol count exceeds plaintext vocabulary; no injective mapping exists")
    mapping,used,predictions,mappings={},set(),[],[]
    for sequence in sequences:
        if len(set(sequence))>n or 2*len(sequence)+2>config["max_length"]:
            raise ValueError("Observed sequence exceeds checkpoint context or symbol vocabulary")
        if mapping_scope=="sequence":
            mapping,used={},set()
        prefix=[n+2]+[rank+1 for rank in encode_symbols(sequence,mode)]+[n+1]
        current=torch.tensor([prefix],dtype=torch.long,device=device)
        generated=[]
        for symbol in sequence:
            if symbol not in mapping:
                scores=model(current)[0,-1,n+4:2*n+4].clone()
                if not torch.isfinite(scores).all():
                    raise ValueError("SDA generation logits became non-finite")
                if used:
                    scores[list(used)]=float("-inf")
                if not torch.isfinite(scores).any():
                    raise ValueError("No unused plaintext candidate remains")
                chosen=int(scores.argmax())
                mapping[symbol]=chosen
                used.add(chosen)
            chosen=mapping[symbol]
            generated.append(chosen)
            current=torch.cat([current,torch.tensor([[chosen+n+4]],device=device)],dim=1)
        predictions.append(generated)
        if mapping_scope=="sequence":
            mappings.append([{"observed_id":k,"predicted_plain_id":v} for k,v in mapping.items()])
    return dict(attack="sda_transformer",predicted_sequences=predictions,encoding=mode,mapping_scope=mapping_scope,
                observed_mapping=[{"observed_id":k,"predicted_plain_id":v} for k,v in mapping.items()] if mapping_scope=="corpus" else mappings,
                prediction_uses_client_secret=False,paper_attack_fully_reproduced=False,
                scope="Constrained greedy decoding; one-to-one mapping on observed symbols; consistency is enforced but correctness/global optimality is not guaranteed")
