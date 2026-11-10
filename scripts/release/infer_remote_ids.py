"""客户端入口：分词和置换都在本机，向服务端发送混淆编号。"""
from pathlib import Path
import sys
import argparse
import torch
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.aloepri.token_id_transport import remote_generate_ids

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/generate")
    parser.add_argument("--client-secret", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--auth-token-file")
    parser.add_argument("--ca-file")
    args = parser.parse_args()
    secret = torch.load(args.client_secret, map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True, trust_remote_code=False)
    # 此入口按原始文本分词，不隐式插入聊天模板；需要聊天格式时由用户提供对应文本。
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    result = remote_generate_ids(args.url, ids, perm_vocab=secret["perm_vocab"], inv_perm_vocab=secret["inv_perm_vocab"],
                    max_new_tokens=args.max_new_tokens, eos_token_ids=None if tokenizer.eos_token_id is None else [tokenizer.eos_token_id],
                    pad_token_id=tokenizer.pad_token_id,ca_file=args.ca_file,
                    auth_token=Path(args.auth_token_file).read_text(encoding="utf-8").strip() if args.auth_token_file else None)
    print(tokenizer.decode(result, skip_special_tokens=True))

if __name__ == "__main__":
    main()
