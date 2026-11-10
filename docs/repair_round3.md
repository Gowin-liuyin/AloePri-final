# 第三次补充修改与运行说明（2026-10-20）

这一轮继续完成前两轮汇总中仍能处理的工程工作。代码能运行、保存和通过检查，与完整复现论文是两件事。下面分别说明功能和限制。原来的 `repair_round2.md` 保留第二轮记录，当前状态以本文为准。

## 1. Llama 与 MoE 扩维导出

普通 Transformers 权重的扩维导出现在支持 `qwen2`、`llama`、`qwen2_moe`、`mixtral`，保留模型自己的类型，不通过改成 Qwen2 来绕过结构差异。需要 Transformers 5.19.0。

```powershell
python scripts/offline_obfuscation/export_expanded_llama_checkpoint.py --model-dir model/Llama --export-dir artifacts/llama/expanded --expansion-size 4 --lam 0.1 --attention-profile rqk_hqk
python scripts/offline_obfuscation/export_expanded_moe_checkpoint.py --model-dir model/Qwen2-MoE --model-type qwen2_moe --export-dir artifacts/moe/expanded --expansion-size 4 --attention-profile rqk_hqk
```

Mixtral 使用 `--model-type mixtral`。输入必须是完整本地模型目录，输出必须为空。注意：`expansion-size=h` 时，残差宽度是 `d+2h`，注意力每头的宽度保留原值。RoPE 只接受当前实现明确支持的形式；不支持的额外 Q/K 归一化仍拒绝。

MoE 由路由器选择几个专家参与计算。本实现改写各专家权重和顺序，也处理 Qwen2-MoE 的共享专家。按所采用的论文处理方式，先把路由器每个专家向量除以自身长度。这一步**可能改变专家选择**；加上标量 RMSNorm 近似，不能声称与原模型严格等价。元数据记录 `router_normalization`、`normalization_mode`。两个模型家族用随机小模型检查了前向、缓存生成和保存后普通 HF 加载，尚未测它们的真实预训练权重。它们也不能代表论文全部 MoE 架构，特别是 Qwen3-MoE 与 DeepSeek。

VMA / Gate-IA 的目录读取也支持上述 MoE，兼容打包专家和逐专家格式。VMA 将同层所有路由专家的通道一起比较，不假定两边专家编号相同；共享专家不并入该特征。结构检查兼容 HF 保存时转换的专家格式，仍检查全部名称和尺寸。

导出内存也作了调整：扩维模型先建立参数形状，再直接装入改写后的参数；同宽模型不再多复制整套模型，符号置换直接取行列。保存后先释放模型再验收，并按512MB目标大小保存分片（单个大张量仍可更大）。密码学正态样本也分块转换。没有为了节省内存改变矩阵运算的含义。

## 2. SDA 保持同一张编号对应表

```powershell
python scripts/evaluation/privacy/run_additional_attacks.py sda-predict --help
```

沿用第二轮训练、输入文件格式。恢复命令增加 `--mapping-scope corpus`，这是默认值。同一个观测编号在整个输入中必须猜成同一个明文编号，不同观测编号不能猜成同一个编号。`sequence` 每句话重新建立对应表，适合每句使用不同密钥的实验；`none` 保留旧的无约束生成，便于对照。

它按模型分数逐步选择符合限制的答案，没有读取真实明文或客户端密钥。这能避免自相矛盾的输出，**不能保证这张表猜对，也不是求全局最优解**。观测种类超过明文词表大小等不可能满足的情况会报错。

## 3. HTTPS、身份验证、排队与合批

身份验证令牌用于决定谁可以访问接口；TLS 用于加密网络连接。二者都不替代模型本身的隐私分析。

```powershell
python scripts/release/create_transport_credentials.py --output-dir credentials/local
python scripts/release/serve_token_ids.py --server-dir artifacts/llama/expanded/server --auth-token-file credentials/local/auth_token.txt --tls-cert credentials/local/server.crt --tls-key credentials/local/server.key --max-batch-size 4 --max-queue-size 64 --batch-wait-ms 5
```

在另一个终端执行：

```powershell
python scripts/release/infer_remote_ids.py --url https://127.0.0.1:8765/generate --client-secret artifacts/llama/expanded/client/client_secret.pt --tokenizer-dir model/Llama --auth-token-file credentials/local/auth_token.txt --ca-file credentials/local/server.crt --prompt "你好" --max-new-tokens 16
```

创建命令只接受空目录，生成30天有效、仅供本机示例使用的证书、私钥和令牌。客户端显式信任该证书文件，**没有关闭证书或主机名检查**；异机部署应提供包含正确主机名、受客户端信任的证书。私钥和令牌不能公开。非本机监听必须同时提供 TLS 与令牌。凭证目录和私钥已加入 git 忽略规则。

网络线程并发接收请求，单个模型线程将生成参数相同的请求短暂等待后放在一起算；不同长度的输入在左边补位，每条结果分别在结束符处停止。队列有容量限制，满时返回 HTTP 503。客户端拒绝地址跳转，不会带着令牌跟随跳转。测试包含真实 HTTPS 请求、错误令牌/证书、并发生成、队列恢复和跳转拒绝。

这仍是一台机器上的研究服务。没有实现 vLLM/SGLang 后端、多机调度或张量并行，也没有据此声称论文集群性能已复现。随机采样时合批会改变随机数消耗次序；逐条与合批完全相同的测试采用贪心生成。

## 4. 默认标准导出使用密码学随机源

安装要求增加 `cryptography==49.0.0`。不传 `--seed` 的标准同宽/扩维导出，从系统取得256位主密钥，经 HMAC-SHA256 按用途、层和模块分开产生 ChaCha20 字节流。置换使用 Fisher-Yates 与拒绝采样，正态样本使用 Box-Muller。各层不会因为整数种子偏移碰撞而复用同一条流。主密钥只放在 `client/client_secret.pt`。

传入 `--seed 123` 时保留 PyTorch 的可复现实验方式，元数据明确标为 `explicit_seed_experiment`。旧研究运行程序不自动改成密码学版本。随机来源的修复解决了以前“系统只产生一个种子，后面仍用 Torch 采样”的局限，**没有证明整个矩阵混淆方案具有密码学安全性**；有关分布、数值误差和攻击者能力仍须分析。

## 5. ISA 新增比较位置间关系的模式

迭代 ISA 增加 `--observation-mode cosine_gram --restarts 4`。它比较各位置隐藏向量之间的余弦相似度，而不是逐个比较坐标。因此，未知正交旋转或等比例扩维后仍可比较，观测宽度可以不同。至少需要两个位置。多个初始猜测按观测目标的误差选择，没有用真实答案挑结果。

默认 `states` 模式保留原坐标要求。`cosine_gram` **不适用于任意非等比例扩维矩阵**；一般的算法1矩阵可能改变夹角。多个不同输入也可能产生相似关系，所以误差低不表示文本必然恢复正确。这是有明确前提的补充实现，不能冒充论文未充分说明的一般攻击。

## 6. 同一数据对照模型效果

新增入口支持选择题、文本困惑度和生成精确匹配，逐个加载模型，减少显存占用：

```powershell
python scripts/evaluation/run_reproduction_benchmark.py --baseline-model-dir model/Qwen2.5-0.5B-Instruct --dataset outputs/round3_pretrained/evaluation.jsonl --checkpoint square=artifacts/round3_pretrained/square_zero_noise --checkpoint expanded=artifacts/round3_pretrained/expanded_zero_noise_sharded --device cuda --dtype bfloat16 --max-new-tokens 8 --output outputs/round3_pretrained/recheck.json
```

JSONL 每行是一条记录，`id` 必须唯一：

```json
{"id":"c1","task":"choice","prompt":"Question ... Answer:","choices":{"A":"yes","B":"no"},"answer":"A"}
{"id":"g1","task":"generation","prompt":"What is 2+2?","expected":"4"}
{"id":"p1","task":"perplexity","text":"Text to evaluate."}
```

选择题把提示词与各备选答案分别分词后拼接，累计答案的对数概率，取最高者。`--length-normalize` 改为按答案长度平均。正确选项只用于最终评分。聊天模板必须预先写入 prompt，程序不隐式添加。生成按去除首尾空白后的完整文字匹配。困惑度把文本切成符合上下文长度的片段，相邻片段共享一个边界 token，每个目标 token 只计一次。这些明确的本地约定不自动等于任何官方基准的全部规定。

真实验证使用固定版本的 Qwen2.5-0.5B-Instruct 和 SST-2 验证集前64条，并另加2条自写困惑度文本、2道算术生成题。完整参数、版本、摘要与结果位于桌面 `Aloepri/audit_work/repair_round3_20261020` 和本仓库 `outputs/round3_pretrained`。这只是公开数据的小规模对照，不能当成完整 SST-2 成绩。

本次结果必须同时报告：原模型与同宽版本均答对61/64条分类和2/2道算术，扩维近似版本为31/64和0/2。两条自写文本共37个目标token的困惑度分别为55.96、56.05、6421.41。同宽版本分类预测64/64与原模型相同，扩维版本仅30/64相同。参数为seed741、关闭噪声、h=4、lambda=0.1、`rqk_hqk`（没有跨RoPE频率块置换），bfloat16评测。这说明扩维版本的结构检查成功不能代替任务质量检查，当前设定明显掉分；不能据此宣称扩维近似已经修到等价。更高精度、校准和不同变换的影响还要分别核对。

初次导出曾遇到Windows页面文件不足，已保留日志。减少副本、分片保存、独立进程重试后，正式使用的是 `expanded_zero_noise_sharded`。早期 `expanded_zero_noise` / `expanded_zero_noise_v2` 未完成整个导出流程，不应作为本次成功产物使用。

进一步用第一条固定公开提示核对：CPU float32下，在原模型坐标中只把归一化换成同一标量近似，最高分的下一token已经改变，最大logits差约19.05。扩维文件与该独立近似参考的最高候选相同，最大logits差0.0758、平均0.0121，全部有限。第一层实际长度伸缩约1.04至3.89，而共用估计值约1.72。它说明近似本身就足以改变输出，不能把问题全部归因于bfloat16；这个单样本检查也不能证明所有样本与变换不存在其他误差。依据为桌面证据中的 `pretrained_norm_diagnosis.json`。

## 7. 有实体标注的隐私评分

实体评分入口与攻击预测分开。先用同一 tokenizer 将人工标注的字符区间转成 token 区间，再给保存好的预测评分：

```powershell
python scripts/evaluation/privacy/evaluate_entities.py prepare --tokenizer-dir model/Qwen2.5-0.5B-Instruct --annotations data/entities.json --output outputs/entity_truth.json
python scripts/evaluation/privacy/evaluate_entities.py score --predictions outputs/attack_predictions.json --truth outputs/entity_truth.json --output outputs/entity_scores.json
```

标注文件是 JSON 数组，如 `[{"id":"a","text":"Alice called.","entities":[{"start":0,"end":5,"label":"NAME"}]}]`。字符区间左含右不含。预测格式为 `{"predicted_sequences":[[...]]}`，顺序必须与真值一致，未知项填 `-1`。

TTRSR 用全部原文 token 作分母，未知和未恢复的位置都算失败。此处 PIIRSR 的明确约定是：某实体对应的全部 token 在原位置都正确，该实体才算恢复，随后按实体数量计算比例，并分别报告类别。没有实体时返回空值，不写成0。它支持真正带标注的实验，但本轮没有重跑作者的 PUPA 全套流程，不能把自己的约定写成作者已公开的精确实现。

## 8. 仍需另行解决的内容

- 一般扩维矩阵不保持所有方向的长度，单个 RMSNorm 系数不能对所有输入精确补偿。要精确恢复必须增加运算或限制矩阵；这涉及方法取舍。
- RoPE 不同频率之间怎样交换块，以及 Attn-IA 原文公式的方向，仍需明确解释。不能靠代码测试决定作者本意。
- IMA 的已知训练配对是额外实验条件；不能删掉提示参数后宣称不再依赖这些配对。ISA 也仍有上面的变换限制。
- Qwen3-MoE、DeepSeek、完整大模型任务/隐私数据与多机后端尚未验证。已支持两种 MoE 不能代替这些工作。
- BERT 的自定义 LayerNorm 路线仍是额外研究实现；部分层切换也需要恢复计算。不能同时声称任意矩阵、精确等价且标准模型无需额外操作。
- 缺失历史材料的检查仍按真实缺失原因跳过；新测试不伪造旧文件。

重新检查：`python -m pytest -q -ra`。本轮具体通过数、跳过原因和真实模型结果见桌面更新后的汇总 PDF 及对应日志，避免把不同时间的数字相加。
