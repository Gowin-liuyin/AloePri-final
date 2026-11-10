from pathlib import Path
import sys
import os
import pytest

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_local_assets(*paths): 需要用户另行提供的本地模型或历史产物")
    config.addinivalue_line("markers", "historical_repository_docs: 原大仓库的历史文档检查，不作为本项目的算法验证")


def pytest_collection_modifyitems(config, items):
    """缺少外部材料时明确跳过；跳过项不计入已验证算法。"""
    for item in items:
        if item.get_closest_marker("historical_repository_docs") and not (ROOT / "docs/history").is_dir():
            item.add_marker(pytest.mark.skip(reason="独立项目未包含原仓库的docs/history和阶段文档，历史文档检查未执行"))
        marker = item.get_closest_marker("requires_local_assets")
        if marker:
            missing = [str(path) for path in marker.args if not (ROOT / path).exists()]
            if missing:
                item.add_marker(pytest.mark.skip(reason="未提供本地实验材料：" + ", ".join(missing)))
