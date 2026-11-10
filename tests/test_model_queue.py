import json
from pathlib import Path
import pytest
from scripts.operations.model_queue import create_plan, export_command, run_queue, source_fingerprint


def model(folder, family="llama"):
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"model_type": family, "hidden_size": 8, "num_hidden_layers": 1}))
    (folder / "model.safetensors").write_bytes(b"fixture")


def test_plan_covers_bert_and_decoders_without_modifying_inputs(tmp_path):
    source = tmp_path / "models"
    model(source / "llama")
    model(source / "bert", "bert")
    model(source / "unsupported", "other")
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    plan = create_plan(source, tmp_path / "run")
    assert len(plan["jobs"]) == 3
    assert [job["status"] for job in plan["jobs"]] == ["pending", "pending", "blocked"]
    bert = plan["jobs"][0]
    assert bert["requires_custom_runtime"] and bert["export_dtype"] == "float32"
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    with pytest.raises(ValueError, match="overlap"):
        create_plan(source, source / "outputs")
    with pytest.raises(FileExistsError):
        create_plan(source, tmp_path / "run")


def test_queue_continues_after_one_model_fails_and_preserves_attempts(tmp_path, monkeypatch):
    import scripts.operations.model_queue as queue
    source = tmp_path / "models"
    model(source / "first")
    model(source / "second")
    run = tmp_path / "run"
    create_plan(source, run)
    exports = []
    def step(command, log_file, environment):
        if log_file.name == "export.log":
            exports.append(command[command.index("--model-dir") + 1])
            output = Path(command[command.index("--export-dir") + 1])
            output.mkdir()
            return 1 if len(exports) == 1 else 0
        result = Path(command[command.index("--output-dir") + 1]) / "default.json"
        result.write_text(json.dumps({"summary": {"status": "pass"}}))
        return 0
    monkeypatch.setattr(queue, "run_step", step)
    plan = run_queue(run, "python")
    assert [job["status"] for job in plan["jobs"]] == ["export_failed", "complete"]
    failed_attempt = Path(plan["jobs"][0]["attempts"][0]["checkpoint"])
    failed_attempt.joinpath("keep.txt").write_text("preserve")
    second_run = run_queue(run, "python")
    assert second_run["status"] == "complete"
    assert len(second_run["jobs"][0]["attempts"]) == 2
    assert len(second_run["jobs"][1]["attempts"]) == 1
    assert failed_attempt.joinpath("keep.txt").read_text() == "preserve"
    assert not (run / "queue.lock").exists()


def test_queue_blocks_inputs_changed_since_inventory(tmp_path, monkeypatch):
    import scripts.operations.model_queue as queue
    source = tmp_path / "models"
    model(source / "changed")
    run = tmp_path / "run"
    create_plan(source, run)
    (source / "changed/model.safetensors").write_bytes(b"new weights")
    monkeypatch.setattr(queue, "run_step", lambda *_: pytest.fail("Changed input must not execute"))
    plan = run_queue(run, "python")
    assert plan["jobs"][0]["status"] == "blocked"
