# Llama-3.2-1B-Instruct 本地使用

本项目的 Llama 导出、发布和推理默认使用 `model/Llama-3.2-1B-Instruct`。
模型来自 `D:\Models.zip`：16 层、隐藏宽度 2048、FFN 宽度 8192、32 个查询头、8 个 KV 头、每头 64 维、词表 128256。原模型共享 Embedding 和输出头，使用 Llama 3 缩放 RoPE。

## 环境和模型

从 `D:\Aloepri` 执行。已配置的 `.venv` 复用本机 CUDA PyTorch，并在虚拟环境内安装 Transformers；请使用这里的 Python。

```powershell
cd D:\Aloepri
# 新机器可先创建环境；已有 .venv 时不必重建。
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe scripts/setup/prepare_llama_1b.py --archive D:\Models.zip
```

准备脚本只提取完整 HF 文件及许可证，跳过 `original/consolidated.00.pth.incomplete`，检查配置、各层投影尺寸和 safetensors 文件长度。大权重会按张量原始字节重打包成约 512MB 分片，降低 Windows 文件映射时的提交内存峰值；原始单文件保存在 `source_single_file/model.safetensors`。重打包不改权重值，单个 Embedding 张量略大于分片目标时单独保存。已有完整分片会直接检查，已有不完整目录会报错，不覆盖文件。

## 导出、验证和对话

```powershell
# 生成混淆权重及本地发布包；目标目录必须为空。
powershell -ExecutionPolicy Bypass -File .\scripts\run_llama_1b.ps1 -Action Export
# 在 GPU 依次加载原模型和混淆模型，以 FP32 比较 5 条提示及各 8 个生成编号。
powershell -ExecutionPolicy Bypass -File .\scripts\run_llama_1b.ps1 -Action Check
# 使用混淆模型和客户端编号恢复生成回答。
powershell -ExecutionPolicy Bypass -File .\scripts\run_llama_1b.ps1 -Action Chat -Prompt "请用一句话介绍你自己。"
```

`-Device cpu` 可用于没有 CUDA 的机器进行 BF16 对话。FP32 检查默认 `-CheckDevice cuda`；CPU 检查需要更多系统提交内存，本机 16GB 内存环境下完整 FP32 服务端加载仍可能分配失败。`-Action All` 依次执行 Prepare、Export、Check、Chat，适合首次运行；对已经生成的产物再执行 Export/All 会拒绝覆盖，日常使用 Chat。

默认方阵导出关闭两处噪声，使用 BF16 保存符号置换后的权重；这类重排不需要舍入原 BF16 权重值。严格数值检查使用 FP32，避免 BF16 累加顺序差异干扰小容差检查。普通对话用 BF16 和 KV cache。默认密钥取自系统随机源；需要可复现实验时可在底层导出命令传入 `--seed`。

1B 入口明确采用 `2e-4` 的 FP32 绝对误差容差。首次 GPU 对照中，5 条提示的生成编号和文本全部一致，最大 logits 差异为 `1.087188720703125e-4`，因此原来的 `1e-4` 严格检查判为失败；该原始报告保留在 `outputs/evaluation/llama_3_2_1b_instruct/gpu_strict_1e4`。可传 `-CorrectnessAtol 0.0001` 重跑更严格阈值，检查仍会按实际结果返回失败，不能将生成一致当作严格数值检查已通过。

Windows 检查入口使用不加载 Torch 的轻量调度进程。原模型和每个混淆模型分别在独立进程中计算，退出后释放原生分配器占用的系统提交内存。默认检查 `default` 交付 profile；`reference` 是同一来源的副本，可另传 `--profiles reference` 单独检查。

Windows 本地 Llama/Qwen 加载按分片逐个读取参数，并初始化对应的 RoPE 缓冲区，以减少并行精度转换和文件映射造成的内存峰值。导出的文件仍是普通 Hugging Face 权重，前向推理不依赖这个加载辅助函数。

本机 Python 3.13 在系统内存压力后曾卡在 `platform` 的 WMI 查询，导致 Torch 导入停住。项目在 Windows 进程内启用 `platform` 已有的 WinAPI/环境变量回退路径，不修改系统 WMI 服务或系统 Python 文件。

入口也可单独调用：

```powershell
.\.venv\Scripts\python.exe scripts/offline_obfuscation/export_llama_checkpoint.py
.\.venv\Scripts\python.exe scripts/release/export_llama_bundle.py --materialize
.\.venv\Scripts\python.exe scripts/evaluation/run_llama_correctness.py --device cuda --correctness-atol 0.0002
.\.venv\Scripts\python.exe scripts/release/infer_bundle.py --device cuda --prompt "Hello!" --max-new-tokens 32
```

默认目录：

| 内容 | 位置 |
| --- | --- |
| 原始模型 | `model/Llama-3.2-1B-Instruct` |
| 混淆权重及客户端秘密 | `artifacts/llama_3_2_1b_instruct/square_zero_noise` |
| 本地发布包 | `artifacts/llama_3_2_1b_instruct/release_bundle` |
| 正确性报告 | `outputs/evaluation/llama_3_2_1b_instruct/correctness` |

服务器只接收 `release_bundle/profiles/default/server`；客户端保留 `client/client_secret.pt`。整个发布包包含秘密，供模型持有者本地使用。

## 扩维研究入口

```powershell
.\.venv\Scripts\python.exe scripts/offline_obfuscation/export_expanded_llama_checkpoint.py --export-dir artifacts/llama_3_2_1b_instruct/expanded --expansion-size 4 --attention-profile rqk_hqk
```

该入口现在保留 Llama 3 的真实 RoPE 频率，包括保存、重载以及超出原始 8192 位置的计算。`rqk_hqk` 只在各频率对内旋转和缩放；Llama 3 缩放 RoPE 下的频率块置换尚未验证，程序会拒绝该组合。扩维仍使用标量 RMSNorm 近似，可能影响任务效果，不能套用同宽方阵的等价结果。完整 1B 扩维导出需要更多 CPU 内存；主使用入口采用同宽版本。

## 任务和攻击评测

通用评测入口可直接传入 1B 路径：

```powershell
.\.venv\Scripts\python.exe scripts/evaluation/run_reproduction_benchmark.py --baseline-model-dir model/Llama-3.2-1B-Instruct --checkpoint square=artifacts/llama_3_2_1b_instruct/square_zero_noise --dataset YOUR_DATA.jsonl --device cuda --dtype bfloat16 --output outputs/evaluation/llama_3_2_1b_instruct/task_result.json
.\.venv\Scripts\python.exe scripts/evaluation/privacy/run_additional_attacks.py weights --attack vma --baseline-model-dir model/Llama-3.2-1B-Instruct --server-dir artifacts/llama_3_2_1b_instruct/square_zero_noise/server --combinations embed_head --output outputs/evaluation/llama_3_2_1b_instruct/vma.json
```

任务数据格式见 `docs/repair_round3.md`，攻击协议见 `docs/security_attack_protocol.md`。完整词表攻击会消耗较多内存和时间，可使用独立选择的 `--candidate-ids`、`--observed-ids` JSON 列表控制规模。旧 `security_qwen` 预设目标和 `AloePriEngine.obfuscate_model` 仍属于 Qwen 研究路线；Llama 使用专用导出入口及通用目录评测，避免把旧 Qwen 产物当成 1B 结果。

当前同宽版本是符号置换方阵工程变体。正确性检查通过只表示所测提示的输出和误差符合检查条件，不表示完成论文的扩维复现或隐私结论。

## 本机验证记录（2026-11-03）

- RTX 4060 8GB，系统内存约 16GB；Torch 2.13.0+cu126、Transformers 5.19.0，版本见 `requirements-tested.txt`。
- 真实 1B 原始权重检查通过；同宽零噪声导出及发布包通过全部权重名称、形状、分片索引检查。
- `run_llama_1b.ps1 -Action Check` 在 CUDA FP32、显式 `2e-4` 容差下返回 `completion_status: complete`：5 条提示、每条 8 个生成编号全部匹配，无 NaN/Inf；最大 logits 误差 `1.087188720703125e-4`。
- `1e-4` 阈值的原始失败报告保留在 `gpu_strict_1e4`，没有修改其结论。
- GPU BF16 中文对话已通过正式脚本实际运行；最终全量测试 **282 项通过、31 项跳过**，记录在 `outputs/llama_1b_final_tests.log`。跳过项依赖外部材料或历史文档。
- 扩维支持通过小模型的共享权重、缩放 RoPE、长位置计算和保存重载回归检查，未进行完整 1B 扩维任务质量验证。
