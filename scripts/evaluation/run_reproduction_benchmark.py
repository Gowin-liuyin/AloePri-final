"""按同一数据比较原模型和多个导出模型；不会自动下载或伪造论文数据。"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.aloepri.evaluation.benchmark import benchmark_directories


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model-dir",required=True)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--checkpoint",action="append",default=[],help="alias=checkpoint_root (contains server/ and client/)")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--dtype",default="float32")
    parser.add_argument("--max-new-tokens",type=int,default=32)
    parser.add_argument("--length-normalize",action="store_true")
    parser.add_argument("--output",required=True)
    args=vars(parser.parse_args());output=Path(args.pop("output"))
    if output.exists():raise FileExistsError(output)
    checkpoints=[]
    for item in args.pop("checkpoint"):
        name,separator,path=item.partition("=")
        if not separator or not name or not path:parser.error("--checkpoint requires alias=path")
        checkpoints.append((name,path))
    result=benchmark_directories(checkpoints=checkpoints,**args)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as handle:json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(output)


if __name__=="__main__":main()
