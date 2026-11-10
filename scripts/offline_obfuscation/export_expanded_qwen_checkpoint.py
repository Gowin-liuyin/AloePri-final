"""导出扩维Qwen2实验模型；标量归一化是近似，不能据此宣称复现论文精度。"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.aloepri._implementation.export.expanded_qwen_checkpoint import export_expanded_qwen_checkpoint

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",required=True)
    parser.add_argument("--export-dir",required=True)
    parser.add_argument("--expansion-size",type=int,default=4)
    parser.add_argument("--lam",type=float,default=.1)
    parser.add_argument("--seed",type=int)
    parser.add_argument("--kappa",type=float)
    parser.add_argument("--alpha-e",type=float,default=0.)
    parser.add_argument("--alpha-h",type=float,default=0.)
    parser.add_argument("--attention-profile",choices=["simplified","rqk","rqk_hqk","rqk_hqk_block_taukv_taugroup"],default="rqk_hqk_block_taukv_taugroup")
    paths=export_expanded_qwen_checkpoint(**vars(parser.parse_args()))
    print(paths["manifest_path"])

if __name__=="__main__":
    main()
