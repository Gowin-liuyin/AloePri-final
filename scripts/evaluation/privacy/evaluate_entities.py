"""准备实体评分真值，或给已保存预测评分；该入口不训练、不生成攻击预测。"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from src.security_qwen.entity_metrics import prepare_entity_truth,evaluate_entities


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="mode",required=True)
    prepare=sub.add_parser("prepare");prepare.add_argument("--tokenizer-dir",required=True);prepare.add_argument("--annotations",required=True)
    score=sub.add_parser("score");score.add_argument("--predictions",required=True);score.add_argument("--truth",required=True)
    for child in (prepare,score):child.add_argument("--output",required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if args.mode=="prepare":
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_dir,local_files_only=True,trust_remote_code=False)
        result=prepare_entity_truth(tokenizer,read(args.annotations))
    else:result=evaluate_entities(read(args.predictions)["predicted_sequences"],read(args.truth))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as handle:json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(output)


if __name__=="__main__":main()
