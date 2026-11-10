"""独立服务进程入口，模型端不接收客户端密钥。"""
from pathlib import Path
import sys
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.aloepri.token_id_transport import serve_model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ready-file")
    parser.add_argument("--auth-token-file")
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument("--max-batch-size",type=int,default=4)
    parser.add_argument("--max-queue-size",type=int,default=64)
    parser.add_argument("--batch-wait-ms",type=float,default=5)
    parser.add_argument("--max-connections",type=int,default=32)
    serve_model(**vars(parser.parse_args()))

if __name__ == "__main__":
    main()
