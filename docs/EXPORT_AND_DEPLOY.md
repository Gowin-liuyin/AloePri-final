# 导出、验证与部署

第一次使用请从 [README](../README.md) 的 Llama 或 Qwen2 示例开始。这里说明每一步的输入输出、如何判断通过，以及工件交付时的边界。

| 步骤 | 入口 | 检查 |
| --- | --- | --- |
| 准备底座 | 本地 Hugging Face 模型目录 | 模型类型、权重分片索引、tokenizer 和 revision 必须匹配 |
| 离线导出 | [Llama](../scripts/offline_obfuscation/export_llama_checkpoint.py) / [Qwen2](../scripts/offline_obfuscation/export_qwen_checkpoint.py) | 全层导出，使用新输出目录；首次做等价验证时将两种噪声都设为 0 |
| 组装发布包 | [Llama](../scripts/release/export_llama_bundle.py) / [Qwen2](../scripts/release/export_qwen_bundle.py) | 检查 profile、权重名、形状和分片；保留 manifest |
| 正确性对照 | [Llama](../scripts/evaluation/run_llama_correctness.py) / [Qwen2](../scripts/evaluation/run_qwen_correctness.py) | 使用同一底座和 tokenizer；记录 dtype、容差、生成一致性和误差 |
| 推理 | [infer_bundle.py](../scripts/release/infer_bundle.py) | 先在本地验证，分进程模式参见 [通信说明](archive/repair_round3.md) |

初次只想检查代码能加载，可先运行 `python -m pytest -q`。依赖模型权重或历史工件的测试可能跳过。Windows 的 1B 本机启动脚本另见 [LLAMA_1B.md](LLAMA_1B.md)，其中的机器路径是示例；通用命令请显式指定本机目录。

## 工件与验证结论

离线导出目录的 `server/` 保存混淆权重，`client/` 保存私有置换等材料。发布包包含各 profile 的服务器文件及本地使用的客户端文件。向服务器交付时只复制选定的 `profiles/<name>/server/`；完整 bundle、密钥、历史调试产物及秘密种子均留在本地。

以 manifest 的 `implementation_variant`、`paper_algorithm1_reproduced` 和 `requires_custom_runtime` 判断实际产物。标准同宽变体能由普通 Transformers 加载；扩维近似和 BERT 路线有额外限制，详见 [架构说明](FULL_PROJECT_ARCHITECTURE.md)。结构检查通过只说明文件可以按预期解释；零噪声等价通过只说明数值路径一致。带噪声的效用与隐私分别测量。

## 排错

- **找不到权重或 tokenizer**：指向模型根目录，检查索引中每个分片是否实际存在。不要将不同 revision 的配置和分片拼在一起。
- **导出目录已存在**：换一个目录。保留上次的工件和日志以便比较。
- **模型类型或运行时不匹配**：核对 Llama / Qwen2 / Qwen3.5 / BERT，选择对应入口。不要用标准同宽脚本处理扩维或其他架构。
- **数值误差超阈值**：先核对两侧 dtype、分词、模型与 profile，再检查噪声是否为零；记录失败，不靠放宽容差替代诊断。
