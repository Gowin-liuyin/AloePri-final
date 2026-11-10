"""让新增命令真正处理文件，检查参数拼接与保存，防止只有库函数能用。"""
import json
import sys
import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from scripts.evaluation.privacy.run_additional_attacks import main,evaluate_predictions


def test_sda_tfma_and_evaluation_cli_files(tmp_path,monkeypatch):
    def save(name,value):
        path=tmp_path/name
        path.write_text(json.dumps(value),encoding="utf-8")
        return str(path)
    def run(*args):
        monkeypatch.setattr(sys,"argv",["run_additional_attacks.py",*map(str,args)])
        main()
    train=save("train.json",[[0,1,0],[1,2]])
    val=save("val.json",[[1,0,1]])
    observed=save("observed.json",[[8,9,8]])
    checkpoint=tmp_path/"sda.pt"
    run("sda-train","--train-json",train,"--validation-json",val,"--vocab-size",3,"--hidden",16,
        "--epochs",1,"--model-path",checkpoint,"--output",tmp_path/"train-result.json")
    run("sda-predict","--model-path",checkpoint,"--observed-json",observed,"--output",tmp_path/"pred.json")
    result=json.loads((tmp_path/"pred.json").read_text())
    truth=save("truth.json",result["predicted_sequences"])
    run("evaluate","--predictions",tmp_path/"pred.json","--truth-json",truth,"--output",tmp_path/"score.json")
    assert json.loads((tmp_path/"score.json").read_text())["token_accuracy"]==1
    run("tfma","--reference-json",train,"--observed-json",observed,"--output",tmp_path/"freq.json")
    assert json.loads((tmp_path/"freq.json").read_text())["prediction_uses_client_secret"] is False
    with pytest.raises(FileExistsError):
        run("tfma","--reference-json",train,"--observed-json",observed,"--output",tmp_path/"freq.json")
    with pytest.raises(ValueError):
        evaluate_predictions({},[[0]])
    with pytest.raises(ValueError):
        evaluate_predictions(dict(predicted_sequences=[[0,1]]),[[0]])


def test_weight_and_isa_cli_on_standard_model_files(tmp_path,monkeypatch):
    torch.manual_seed(17)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,head_dim=4)).eval()
    directory=tmp_path/"model"
    model.save_pretrained(directory)
    def run(*args):
        monkeypatch.setattr(sys,"argv",["run_additional_attacks.py",*map(str,args)])
        main()
    for attack in ("vma","gate-ia","attn-ia"):
        output=tmp_path/f"{attack}.json"
        run("weights","--attack",attack,"--baseline-model-dir",directory,"--server-dir",directory,"--output",output)
        assert json.loads(output.read_text())["predicted_plain_ids"]==list(range(16))
    with torch.no_grad():
        observed=model(torch.tensor([[3,4]]),output_hidden_states=True,use_cache=False).hidden_states[-1]
    tensor_file=tmp_path/"observed.pt"
    torch.save(observed,tensor_file)
    run("isa","--model-dir",directory,"--observed-states",tensor_file,"--steps",2,"--output",tmp_path/"isa.json")
    assert len(json.loads((tmp_path/"isa.json").read_text())["history"])==3
