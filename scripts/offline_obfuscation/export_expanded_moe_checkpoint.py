"""Qwen2-MoE/Mixtral扩维导出；路由向量归一化和标量RMS均需另测任务质量。"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.aloepri._implementation.export.expanded_qwen_checkpoint import export_expanded_decoder_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",required=True)
    parser.add_argument("--export-dir",required=True)
    parser.add_argument("--model-type",choices=["qwen2_moe","mixtral"],required=True,dest="expected_model_type")
    parser.add_argument("--expansion-size",type=int,default=4)
    parser.add_argument("--lam",type=float,default=.1)
    parser.add_argument("--seed",type=int)
    parser.add_argument("--kappa",type=float)
    parser.add_argument("--alpha-e",type=float,default=0.)
    parser.add_argument("--alpha-h",type=float,default=0.)
    parser.add_argument("--attention-profile",choices=["simplified","rqk_hqk","rqk_hqk_block_taukv_taugroup"],default="rqk_hqk_block_taukv_taugroup")
    print(export_expanded_decoder_checkpoint(**vars(parser.parse_args()))["manifest_path"])


if __name__=="__main__":main()
