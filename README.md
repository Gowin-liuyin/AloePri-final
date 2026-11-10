# AloePri：模型混淆与本地推理

AloePri 在部署前改写模型权重和词表编号。服务器加载混淆权重，客户端保留编号映射，用它处理输入和输出。本仓库提供独立研究实现，不需要另外克隆 OSNIP 或 A+B。

## 先跑一个检查

使用 Python 3.11 或更高版本，建议新建虚拟环境。下面的测试无需另一个仓库；需要本地权重或历史材料的用例会跳过。

```bash
python -m venv .venv
# 激活虚拟环境后执行
python -m pip install -r requirements-tested.txt
python -m pytest -q
```

## 用自己的模型试跑

准备本地 Hugging Face 格式的 Llama 模型目录，包含配置、tokenizer 和权重。第一次先把噪声设为零，方便检查改写前后是否一致。每个输出目录使用新的名字。

```bash
python scripts/offline_obfuscation/export_llama_checkpoint.py --model-dir model/Llama-3.2-1B-Instruct --export-dir artifacts/llama/export --device cpu --dtype float32 --alpha-e 0 --alpha-h 0
python scripts/release/export_llama_bundle.py --source-dir artifacts/llama/export --export-dir artifacts/llama/bundle --materialize
python scripts/evaluation/run_llama_correctness.py --bundle-dir artifacts/llama/bundle --baseline-model-dir model/Llama-3.2-1B-Instruct --output-dir outputs/llama/check --device cpu --dtype float32
python scripts/release/infer_bundle.py --bundle-dir artifacts/llama/bundle --prompt "The capital of France is" --device cpu --dtype float32
```

Qwen2 对应 `export_qwen_checkpoint.py`、`export_qwen_bundle.py` 和 `run_qwen_correctness.py`，目录与 Llama 示例相同。各脚本的 `--help` 列出参数。分片、模型匹配与常见报错见[导出、验证与部署](docs/EXPORT_AND_DEPLOY.md)。

**只把发布包中选定的 `profiles/<name>/server/` 交给服务器。** 完整发布包包含客户端私有材料。默认导出使用系统随机源；显式传 `--seed` 的路线用于可复现实验。

## 如何理解结果

- **同宽导出**：Llama 和 Qwen2 的常规路线保持模型隐藏宽度，导出后可用普通 Transformers 加载。它是工程变体，不是论文算法 1 的完整扩维复现。
- **扩维和 BERT**：扩维实验使用近似计算，可能改变模型效果；BERT 扩展需要自定义 LayerNorm 恢复模块。详见[架构参考](docs/FULL_PROJECT_ARCHITECTURE.md)。
- **验证范围**：零噪声等价检查验证计算路径。带噪声后的任务效果和隐私需要另外评估，不能从导出成功或小样本一致推断。

## 更多文档

- [文档导航](docs/README.md) · [导出、验证与部署](docs/EXPORT_AND_DEPLOY.md)
- [维护与贡献](CONTRIBUTING.md) · [变更记录](CHANGELOG.md)
- [引用与来源](CITATION.md) · [安全问题报告](SECURITY.md)
