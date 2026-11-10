"""读取本机校准数据，输出标量RMSNorm近似误差；文件中的P应留在客户端。"""
from pathlib import Path
import argparse
import json
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.aloepri._implementation.analysis.rms_calibration import report_scalar_rms_error

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",required=True,help="torch.save dict: P, calibration_inputs, evaluation_inputs")
    parser.add_argument("--epsilon",type=float,default=1e-6)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    data=torch.load(args.inputs,map_location="cpu",weights_only=True)
    result=report_scalar_rms_error(data["P"],data["calibration_inputs"],data["evaluation_inputs"],epsilon=args.epsilon)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)

if __name__=="__main__":
    main()
