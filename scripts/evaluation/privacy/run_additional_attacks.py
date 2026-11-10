"""新增攻击入口：预测和评分分开，预测命令不接收真实答案或客户端密钥。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    weights = sub.add_parser("weights")
    weights.add_argument("--attack", choices=["vma", "gate-ia", "attn-ia"], required=True)
    weights.add_argument("--baseline-model-dir", required=True)
    weights.add_argument("--server-dir", required=True)
    weights.add_argument("--layers", nargs="+", type=int, default=[0])
    weights.add_argument("--combinations", nargs="+", choices=["embed_head", "qk", "gate", "up", "down_head", "router"], default=["gate"])
    weights.add_argument("--candidate-ids", help="JSON integer list chosen independently of test answers")
    weights.add_argument("--observed-ids", help="JSON integer list; default full server vocabulary")
    weights.add_argument("--topk", type=int, default=5)
    weights.add_argument("--fuse-original-norms", action="store_true", help="Use only when target projections absorb original RMSNorm gamma")
    weights.add_argument("--max-feature-elements", type=int, default=16_000_000)
    train = sub.add_parser("sda-train")
    train.add_argument("--train-json", required=True)
    train.add_argument("--validation-json", required=True)
    train.add_argument("--vocab-size", type=int, required=True)
    train.add_argument("--model-path", required=True)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--hidden", type=int, default=64)
    train.add_argument("--mode", choices=["first_occurrence", "frequency_rank"], default="first_occurrence")
    train.add_argument("--seed", type=int, default=20261020)
    train.add_argument("--device", default="cpu")
    train.add_argument("--learning-rate", type=float, default=.002)
    predict = sub.add_parser("sda-predict")
    predict.add_argument("--model-path", required=True)
    predict.add_argument("--observed-json", required=True)
    predict.add_argument("--batch-size", type=int, default=32)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--mapping-scope",choices=["corpus","sequence","none"],default="corpus")
    isa = sub.add_parser("isa")
    isa.add_argument("--model-dir", required=True)
    isa.add_argument("--observed-states", required=True, help="torch tensor [batch,sequence,hidden] in public model coordinates")
    isa.add_argument("--candidate-ids")
    isa.add_argument("--layer", type=int, default=-1)
    isa.add_argument("--steps", type=int, default=200)
    isa.add_argument("--learning-rate", type=float, default=.05)
    isa.add_argument("--prior-weight", type=float, default=0.)
    isa.add_argument("--topk", type=int, default=5)
    isa.add_argument("--seed", type=int, default=20261020)
    isa.add_argument("--device", default="cpu")
    isa.add_argument("--observation-mode",choices=["states","cosine_gram"],default="states")
    isa.add_argument("--restarts",type=int,default=1)
    frequency = sub.add_parser("tfma")
    frequency.add_argument("--reference-json", required=True, help="Public reference token sequences")
    frequency.add_argument("--observed-json", required=True, help="Observed substituted token sequences")
    frequency.add_argument("--topk", type=int, default=5)
    score = sub.add_parser("evaluate")
    score.add_argument("--predictions", required=True)
    score.add_argument("--truth-json", required=True, help="Aligned true sequences, or true IDs for weight mapping")
    for child in (weights, train, predict, isa, frequency, score):
        child.add_argument("--output", required=True, help="New JSON file; existing files will not be overwritten")
    return parser


def evaluate_predictions(predictions, truth):
    """只为保存好的预测评分，不参与预测或选模型。"""
    if "predicted_sequences" in predictions:
        generated = predictions["predicted_sequences"]
    elif "predicted_plain_ids" in predictions:
        generated, truth = [predictions["predicted_plain_ids"]], [truth]
    else:
        raise ValueError("Unsupported prediction format")
    if not isinstance(truth, list) or not truth or len(truth)!=len(generated):
        raise ValueError("Truth must align with saved predictions")
    if any(not isinstance(t, list) or len(t)!=len(p) or any(type(x) is not int or x < 0 for x in t) for p,t in zip(generated, truth)):
        raise ValueError("Truth sequences must have matching lengths and valid IDs")
    count = sum(map(len, truth))
    if not count:
        raise ValueError("No tokens to evaluate")
    return dict(evaluated_tokens=count,
                token_accuracy=sum(a==b for p,t in zip(generated,truth) for a,b in zip(p,t))/count,
                sequence_accuracy=sum(p==t for p,t in zip(generated,truth))/len(truth),
                note="Accuracy for supplied data only; no full-paper or privacy guarantee claim")


def main():
    args = vars(build_parser().parse_args())
    command, output = args.pop("command"), Path(args.pop("output"))
    if output.exists():
        raise FileExistsError(output)
    if command == "weights":
        from src.security_qwen.weight_attacks import recover_from_directories
        for name in ("candidate_ids", "observed_ids"):
            if args[name] is not None:
                args[name] = read_json(args[name])
        result = recover_from_directories(**args)
    elif command == "sda-train":
        from src.security_qwen.sda_transformer import train_sda_transformer
        result = train_sda_transformer(read_json(args.pop("train_json")), read_json(args.pop("validation_json")), **args)
    elif command == "sda-predict":
        from src.security_qwen.sda_transformer import recover_sda_transformer
        result = recover_sda_transformer(observed_sequences=read_json(args.pop("observed_json")), **args)
    elif command == "isa":
        import torch
        from transformers import AutoModelForCausalLM
        from src.security_qwen.isa_optimization import run_isa_optimization
        model = AutoModelForCausalLM.from_pretrained(args.pop("model_dir"), local_files_only=True, trust_remote_code=False, dtype=torch.float32).to(args.pop("device"))
        states = torch.load(args.pop("observed_states"), map_location="cpu", weights_only=True)
        if args["candidate_ids"] is not None:
            args["candidate_ids"] = read_json(args["candidate_ids"])
        result = run_isa_optimization(model, states, **args)
    elif command == "tfma":
        from src.security_qwen.frequency_mapping import recover_frequency_mapping
        result = recover_frequency_mapping(read_json(args.pop("reference_json")), read_json(args.pop("observed_json")), **args)
    else:
        result = evaluate_predictions(read_json(args["predictions"]), read_json(args["truth_json"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
