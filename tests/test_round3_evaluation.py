"""任务与实体评分独立对照；这些是随机小模型用例，不是正式论文数据集成绩。"""
import json
import sys
import pytest
import torch
from src.aloepri.evaluation.benchmark import evaluate_model,benchmark_directories,load_records
from src.security_qwen.entity_metrics import prepare_entity_truth,evaluate_entities
from src.aloepri._implementation.export.standard_square_checkpoint import export_standard_square_checkpoint


def test_choice_likelihood_and_perplexity_against_native_model_loss():
    from tests.test_round3_models import llama_setup
    tokenizer,model=llama_setup()
    records=[dict(id="c",task="choice",prompt="t3 t7",choices={"A":"t9","B":"t4 t8"},answer="A"),
             dict(id="p",task="perplexity",text="t3 t7 t9 t4")]
    result=evaluate_model(model,tokenizer,records,max_new_tokens=3)
    ids=torch.tensor([[3,7,9,4]])
    with torch.no_grad():expected=float(model(ids,labels=ids).loss)
    assert result["summary"]["perplexity"]["mean_nll"]==pytest.approx(expected,abs=1e-6)
    with torch.no_grad():
        scores=model(torch.tensor([[3,7,9]])).logits[0,1].log_softmax(-1)
    assert result["records"][0]["scores"]["A"]==pytest.approx(float(scores[9]),abs=1e-6)
    changed=[dict(records[0],answer="B"),records[1]]
    other=evaluate_model(model,tokenizer,changed,max_new_tokens=3)
    assert result["records"][0]["predicted"]==other["records"][0]["predicted"]
    assert result["records"][0]["scores"]==other["records"][0]["scores"]


def test_model_directory_benchmark_and_actual_cli(tmp_path,monkeypatch):
    from tests.test_round3_models import llama_setup
    from scripts.evaluation.run_reproduction_benchmark import main
    tokenizer,model=llama_setup()
    original=tmp_path/"original";model.save_pretrained(original);tokenizer.save_pretrained(original)
    changed=tmp_path/"square"
    export_standard_square_checkpoint(changed,model_dir=original,expected_model_type="llama",seed=177)
    with torch.no_grad():
        ids=torch.tensor([[3,7]]);gen=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=3,do_sample=False)[0,2:]
    rows=[dict(id="choice",task="choice",prompt="t3 t7",choices={"A":"t9","B":"t4"},answer="A"),
          dict(id="text",task="perplexity",text="t3 t7 t9 t4"),
          dict(id="generation",task="generation",prompt="t3 t7",expected=tokenizer.decode(gen,skip_special_tokens=True))]
    data=tmp_path/"dataset.jsonl";data.write_text("\n".join(json.dumps(row) for row in rows),encoding="utf-8")
    output=tmp_path/"results.json"
    monkeypatch.setattr(sys,"argv",["benchmark","--baseline-model-dir",str(original),"--dataset",str(data),"--checkpoint",f"square={changed}","--max-new-tokens","3","--output",str(output)])
    main()
    result=json.loads(output.read_text());a,b=result["models"]["baseline"],result["models"]["square"]
    assert a["summary"]["generation"]["accuracy"]==1==b["summary"]["generation"]["accuracy"]
    assert a["records"][0]["predicted"]==b["records"][0]["predicted"]
    assert a["summary"]["perplexity"]["mean_nll"]==pytest.approx(b["summary"]["perplexity"]["mean_nll"],abs=1e-6)
    with pytest.raises(FileExistsError):main()


def test_dataset_validation_and_context_boundaries(tmp_path):
    from tests.test_round3_models import llama_setup
    tokenizer,model=llama_setup()
    file=tmp_path/"data.jsonl";file.write_text("")
    with pytest.raises(ValueError,match="empty"):load_records(file)
    row=dict(id="x",task="generation",prompt="t3",expected="t4")
    file.write_text(json.dumps(row)+"\n"+json.dumps(row))
    with pytest.raises(ValueError,match="unique"):load_records(file)
    result=evaluate_model(model,tokenizer,[dict(id="p",task="perplexity",text=" ".join(["t3"]*130))])
    assert result["summary"]["perplexity"]["scored_tokens"]==129
    with pytest.raises(ValueError,match="context"):evaluate_model(model,tokenizer,[row],max_new_tokens=100)


def test_complete_entity_recovery_and_unknown_token_denominator():
    truth=[dict(id="a",true_ids=[3,7,9],entities=[dict(start=0,end=1,label="NAME"),dict(start=1,end=3,label="PHONE")]),
           dict(id="b",true_ids=[4,5],entities=[dict(start=0,end=2,label="ADDRESS")])]
    result=evaluate_entities([[3,7,-1],[4,5]],truth)
    assert result["TTRSR"]==.8 and result["PIIRSR"]==pytest.approx(2/3)
    assert result["per_category"]["PHONE"]["recovery_rate"]==0
    assert evaluate_entities([[3]], [dict(id="a",true_ids=[3],entities=[])])["PIIRSR"] is None
    truth[0]["entities"].append(dict(start=0,end=1,label="NAME"))
    with pytest.raises(ValueError,match="Duplicate"):evaluate_entities([[3,7,9],[4,5]],truth)


def test_character_annotations_and_entity_cli(tmp_path,monkeypatch):
    from tests.test_round3_models import llama_setup
    from scripts.evaluation.privacy.evaluate_entities import main
    tokenizer,_=llama_setup();tokenizer.save_pretrained(tmp_path/"tokenizer")
    annotations=[dict(id="a",text="t3 t7 t9",entities=[dict(start=3,end=5,label="NAME")])]
    truth=prepare_entity_truth(tokenizer,annotations)
    assert truth[0]["entities"][0]["start"]==1 and truth[0]["entities"][0]["end"]==2
    ann=tmp_path/"annotations.json";ann.write_text(json.dumps(annotations))
    truth_path=tmp_path/"truth.json"
    monkeypatch.setattr(sys,"argv",["entities","prepare","--tokenizer-dir",str(tmp_path/"tokenizer"),"--annotations",str(ann),"--output",str(truth_path)])
    main()
    predicted=tmp_path/"pred.json";predicted.write_text(json.dumps(dict(predicted_sequences=[[3,7,-1]])))
    score=tmp_path/"score.json"
    monkeypatch.setattr(sys,"argv",["entities","score","--predictions",str(predicted),"--truth",str(truth_path),"--output",str(score)])
    main()
    assert json.loads(score.read_text())["PIIRSR"]==1
