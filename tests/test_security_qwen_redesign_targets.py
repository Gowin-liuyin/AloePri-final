from src.security_qwen.artifacts import get_security_target, resolve_security_target


def test_redesign_stage_j_target_resolves() -> None:
    spec = get_security_target("stage_j_redesign")
    assert spec.artifact_dir == "artifacts/stage_j_qwen_redesign"


def test_stage_k_reference_target_resolves(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "artifacts/stage_k_release"
    (root / "profiles/reference").mkdir(parents=True)
    (root / "catalog.json").write_text('{"profiles": []}', encoding="utf-8")
    resolved = resolve_security_target("stage_k_reference")
    assert resolved.resolved_root_dir.replace(chr(92), "/").endswith("artifacts/stage_k_release/profiles/reference")


def test_missing_standard_bridge_is_not_accepted(tmp_path) -> None:
    from src.aloepri._implementation.analysis.standard_weight_proof import build_stage_j_standard_weight_proof

    payload = build_stage_j_standard_weight_proof(tmp_path / "missing_server")
    assert payload["is_standard_weight_export"] is False
