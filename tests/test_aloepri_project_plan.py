from src.aloepri.project_plan import paper_aligned_project_layers, project_plan_payload


def test_project_plan_uses_paper_aligned_layers():
    payload = project_plan_payload()
    names = {layer["name"] for layer in payload["layers"]}
    assert "offline_obfuscation" in names
    assert "private_inference" in names
    assert "release_bundle" in names
    assert "evaluation" in names


def test_project_layers_have_modules():
    assert all(layer.modules for layer in paper_aligned_project_layers())
