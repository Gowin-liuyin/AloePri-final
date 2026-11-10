# AloePri 研究代码

这个仓库包含论文机制的研究实现和可保存为普通 Hugging Face 权重的工程变体。两者需要分开理解：

- `AloePriEngine` 保留扩维隐藏向量和自定义运行模块，便于检查各项数学运算。
- Qwen2、Llama 的标准权重导出采用**不改变隐藏宽度的符号置换方阵**。它可以用普通 Transformers 模型加载，但不是论文算法 1 的完整扩维复现，也没有因此获得论文的隐私或准确率结论。
- 扩维实验入口支持 Qwen2、Llama、Qwen2-MoE 和 Mixtral，使用算法 1 构造残差矩阵，保留原注意力头宽，以普通 HF 权重保存。它使用标量 RMSNorm 近似；MoE 还会归一化路由向量，可能改变专家选择，不能等同于原模型或论文完整部署。见[最新修改与运行说明](docs/repair_round3.md)。
- BERT 是额外的本地研究扩展。它需要自定义 LayerNorm 恢复模块，不能作为无需额外计算的普通服务端发布。

部分函数和目录保留旧名 `paper_consistent`，以便旧脚本继续导入。判断产物实际做了什么，应查看 `manifest.json` 中的 `implementation_variant`、`paper_algorithm1_reproduced` 和 `requires_custom_runtime`，不能只看名称。

## 从哪里开始

- 首次验证可先运行下节的单元测试，再按 [Llama](#llama) 或 [Qwen2](#从原始-qwen2-生成标准权重) 示例使用已获授权的本地模型。
- 模型持有方负责离线导出，服务端只接收混淆权重；客户端保留词表置换和密钥。仓库中不包含模型、凭据或实验输出。
- [OSNIP-final](https://github.com/Gowin-liuyin/OSNIP-final) 维护 B 组件的独立训练与评估。A+B 运行时单独维护。

## 安装和检查

以下命令从仓库根目录执行。代码调用了 Transformers 内部接口，因此先使用已验证的版本：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m pytest
```

也可以安装项目及测试依赖：`.\.venv\Scripts\python.exe -m pip install -e ".[test]"`。`requirements-tested.txt` 记录的是本次本地小模型验证环境；它不意味着已经测过所有 CUDA、vLLM 或 SGLang 版本。

迁移来的历史文档、旧产物测试与当前独立仓库不相符，测试配置会明确说明相应跳过原因。数学、导出和评测测试仍应执行，不能把历史文件缺失当作数学检查通过。

## 从原始 Qwen2 生成标准权重

先准备本机原始模型目录，包括 config、tokenizer 和 safetensors 权重（可以分片）。下面先关闭噪声，检查改写前后的计算是否一致；它是正确性实验，不能用来宣称隐私保护有效。

```powershell
.\.venv\Scripts\python.exe scripts/offline_obfuscation/export_qwen_checkpoint.py --model-dir model/Qwen2.5-0.5B-Instruct --export-dir artifacts/qwen/square_zero_noise --dtype float32 --device cpu --alpha-e 0 --alpha-h 0
.\.venv\Scripts\python.exe scripts/release/export_qwen_bundle.py --source-dir artifacts/qwen/square_zero_noise --export-dir artifacts/qwen/release_zero_noise --materialize
.\.venv\Scripts\python.exe scripts/evaluation/run_qwen_correctness.py --bundle-dir artifacts/qwen/release_zero_noise --baseline-model-dir model/Qwen2.5-0.5B-Instruct --output-dir outputs/qwen/zero_noise --dtype float32 --device cpu --correctness-atol 0.0001
.\.venv\Scripts\python.exe scripts/release/infer_bundle.py --bundle-dir artifacts/qwen/release_zero_noise --profile default --prompt "请用一句话介绍你自己。" --max-new-tokens 16 --device cpu
```

输出目录需要为空。导出程序不会删除已有目录来覆盖结果。不传 `--seed` 的标准导出现在从系统取得256位主密钥，经 HMAC-SHA256 按用途、层和模块分开产生 ChaCha20 随机字节，用于矩阵、置换和噪声；主密钥只保存于客户端文件。`--seed 123` 保留 PyTorch 的可复现实验方式，元数据明确区分两者。随机源修复不等于证明整个混淆方案安全，旧研究运行程序也不自动获得这项保证。

输出中的文件这样使用：

```text
checkpoint/
  server/                    # 只把这里的普通推理文件交给服务器
  client/client_secret.pt    # 客户端保留，不能上传服务器
  stage_i_metadata.json      # 已去除秘密种子的公开说明
  manifest.json              # 实现范围和结构检查结果
```

release bundle 是模型持有者在本地使用的完整包，包含客户端秘密。**不能把整个包上传服务器**，只能交付 `profiles/<name>/server`。发布过程会检查全部权重名称、形状、分片索引和配置，并拒绝调试文件、需要自定义运行程序的 BERT 产物、非法 profile 名称及源目录重叠。结构检查不能代替数值、任务和隐私评测。

## Llama

同样只导出全部层。共享或不共享 Embedding/输出头的原模型都会分别处理输出头；两个位置可以采用不同的噪声参数。

```powershell
.\.venv\Scripts\python.exe scripts/offline_obfuscation/export_llama_checkpoint.py --model-dir model/Llama-3.2-1B-Instruct --export-dir artifacts/llama/square_zero_noise --dtype bfloat16 --device cpu --alpha-e 0 --alpha-h 0
.\.venv\Scripts\python.exe scripts/release/export_llama_bundle.py --source-dir artifacts/llama/square_zero_noise --export-dir artifacts/llama/release_zero_noise --materialize
.\.venv\Scripts\python.exe scripts/evaluation/run_llama_correctness.py --bundle-dir artifacts/llama/release_zero_noise --baseline-model-dir model/Llama-3.2-1B-Instruct --output-dir outputs/llama/zero_noise --dtype float32 --device cuda --correctness-atol 0.0002
```

`--num-layers` 不能用于部分层的标准导出。部分层的实验模块会额外恢复隐藏向量，保存后不再是普通模型的完整结构。

## 本地推理和检查结果如何理解

推理入口把输入 token 编号置换一次，随后由模型在混淆编号中生成，并使用 KV cache 保存已有计算、遇到 EOS 停止。最后客户端把输出编号恢复。可传 `--do-sample --temperature 0.65 --top-k 20 --top-p 0.95` 开启随机采样。

上面的入口在同一进程里模拟 token-ID 协议。`serve_token_ids.py` 与 `infer_remote_ids.py` 可分进程交换编号，新增 HTTPS、访问令牌、有界排队和把多个请求一起计算的功能。非本机监听必须同时使用 TLS 与令牌。客户端验证证书并拒绝地址跳转。尚未实现多机调度或 vLLM/SGLang 后端；普通文本 API 的“解码后再编码”仍不能保证保留任意混淆编号。具体命令见[最新修改与运行说明](docs/repair_round3.md)。

小样本等价检查必须满足：有测试样本，全部数值有限，所有决定性生成编号和文本相同，最大 logits 误差不超过明确给出的容差。通用检查函数的 float32 容差为 0.0001；Llama 1B 入口显式采用 0.0002，原始 0.0001 阈值未通过的对照报告也保留，见 [1B 使用说明](docs/LLAMA_1B.md)。检查程序会记录误差及容差；NaN、Inf、只匹配一个样本、调用者自行填入 status=pass 都不能使它通过。

加入噪声后本来就可能改变输出，严格等价检查失败不自动说明实现错误。应另用与论文相同的数据和协议测任务准确率，同时测攻击成功率；当前仓库没有完成论文的大模型、完整数据集和集群结果复现。

攻击评测需要什么数据、如何区分真实答案与攻击者可见内容，见 [攻击评测条件和评分说明](docs/security_attack_protocol.md)。

新增攻击入口：`scripts/evaluation/privacy/run_additional_attacks.py`。它提供 Gate-IA、完整行排序的 VMA、明确注明解释方式的 Attn-IA、Transformer SDA、迭代 ISA，以及独立 TFMA。预测与评分分开；新命令不接收客户端密钥。原有岭回归和统计基线保留，不能将旧结果重新标为新算法结果。[运行示例和数据格式](docs/repair_round2.md)。

第三次补充中，SDA 默认强制整个输入使用一致的一对一编号对应；VMA/Gate-IA 可读取 Qwen2-MoE/Mixtral；ISA 可在未知正交旋转或等比例扩维时比较位置之间的余弦相似度，仍不能处理一般未知扩维矩阵。新增 `run_reproduction_benchmark.py` 做相同数据的模型效果对照，`evaluate_entities.py` 给带实体标注的保存预测评分。真实 Qwen2.5-0.5B 的小规模公开数据对照与具体限制见[第三次修改说明](docs/repair_round3.md)。

这次实际对照中，SST-2固定前64条：原模型与同宽版本均正确61条，扩维近似版本正确31条。它使用h=4、lambda=0.1、关闭噪声、`rqk_hqk`和bfloat16；这是本地小规模协议，不能代替完整官方成绩。扩维版本明显掉分，当前仍须研究近似和数值精度的影响，不能把“能保存加载”写成“任务效果已经保持”。

## 历史研究产物

2026-10-28 使用真实Qwen权重新增了[攻击实现验证实验](docs/attack_verification_20261028.md)。IMA的两层网络训练现在默认采用仅由训练配对计算的逐通道标准化，词表匹配前恢复原尺度；`--disable-standardization`可重跑旧流程。受控恢复、自然文本效果和未见词条效果分别报告，不能把前者当成完整论文复现。

`--source-dir` 保留给已有中间产物的转换。它不能凭空生成仓库外的历史目录；一般使用上面的 `--model-dir` 路线。

历史调试格式保存了恢复矩阵、原始记录权重或重建种子。保存这些格式必须在 Python API 中显式设置 `allow_unsafe_debug_export=True`，仅用于本地研究，不能进入标准发布包。将它们改成普通文件名，并不会自动消除已有的秘密暴露或数学近似问题。

## 代码位置

| 内容 | 文件位置 |
| --- | --- |
| 扩维研究运行程序 | `src/aloepri/engine.py`、`layers/` |
| Qwen2/Llama 方阵导出 | `src/aloepri/_implementation/export/standard_square_checkpoint.py` |
| 完整权重结构检查 | `src/aloepri/_implementation/analysis/standard_weight_proof.py` |
| 单文件和分片读取 | `src/aloepri/_implementation/export/tensor_io.py` |
| 公开说明与秘密分离 | `src/aloepri/_implementation/export/export_safety.py`、`checkpoint_io.py` |
| 发布与本地推理 | `src/aloepri/release/`、`src/aloepri/private_inference.py` |
| 攻击评测 | `src/security_qwen/` |

旧架构文档保留了历史阶段名称，具体行为以当前代码和本 README 的限制说明为准。
