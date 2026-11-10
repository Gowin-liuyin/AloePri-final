"""生成本机HTTPS证书和访问令牌，客户端显式指定CA文件，无需关闭证书验证。"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.aloepri.transport_credentials import create_local_credentials


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--days",type=int,default=30)
    args=parser.parse_args()
    print(json.dumps(create_local_credentials(args.output_dir,days=args.days),indent=2))


if __name__=="__main__":main()
