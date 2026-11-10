# 攻击实验的条件与结果含义

本次修复让评测程序按明确的输入和评分规则运行。它没有补齐论文的全部攻击，也没有重新测出论文表格中的攻击成功率。

## 目前每种程序实际做什么

| 入口 | 实际算法与范围 |
| --- | --- |
| `run_vma_baseline` | 对每行排序后抽样、减均值、调整长度，再累计各来源分数。它是项目自己的 VMA 变体，尚未逐项实现论文表 9，也不是“各层第一候选多数票”。 |
| `run_ima_baseline` | 用岭回归预测原 Embedding。岭回归是一种带惩罚项的线性预测方法。 |
| `run_ima_paper_like` | 用两层、八头 Qwen2 预测原 Embedding；观测宽度不满足头宽要求时，先用线性层投到向上取整的网络宽度。该适配是本实现选择。训练数据配对仍需明确授权模拟。 |
| `run_isa_baseline` | 保留的岭回归基线。`run_isa_optimization` 冻结公开模型，反复调整输入向量；默认比较原坐标，新增 `cosine_gram` 模式允许未知正交旋转/等比例扩维，但不适用于一般未知矩阵。 |
| `run_tfma_baseline` | 比较公开参考语料与观测语料中每个 token 的出现比例。 |
| `run_sda_baseline` | 保留的单个/相邻 token 统计基线；`train_sda_transformer` / `recover_sda_transformer` 是训练 Transformer 的实现，恢复默认强制整个输入使用一致的一对一编号对应。两种结果不能混用。 |
| `build_ia_template` | 保留的未执行结果模板，空指标不是攻击测试结果。实际 Gate-IA 和 Attn-IA 解释版本在 `weight_attacks.py`。 |
| `recover_weight_mapping` / `recover_from_directories` | Gate-IA、完整行排序/L1匹配/投票的 VMA，以及采用二维列空间投影解释的 Attn-IA。目录支持 Qwen2/Llama/Qwen2-MoE/Mixtral；Attn-IA 的解释并非原文公式唯一确定的算法。 |
| `recover_frequency_mapping` | 新增不依赖目标目录注册表或客户端密钥的 TFMA；公开与观测编号序列由调用者分别提供。 |

初始命令与文件格式见 [第二轮说明](archive/repair_round2.md)，第三轮实现与限制见 [第三次补充](archive/repair_round3.md)。VMA 六类权重组合均有构造代码；MoE 支持打包/逐专家格式，集合特征不假定专家编号一致。词表平方大小的矩阵超出显式预算时会报错，不会悄悄退回抽样。

## IMA、ISA 为什么需要多写一个参数

这些实验现在需要显式传入 `allow_simulated_known_pairs=True`。命令行的对应参数为 `--allow-simulated-known-pairs`。

原因是：训练程序知道公开原文后，还用客户端秘密置换，生成同一目标密钥下的正确观测向量或隐藏状态。训练模型由此得到“这段原文对应这个观测结果”的配对。这可以用来研究**攻击者已经拿到一些正确配对时**的恢复能力，但不能直接当作“攻击者只有服务端权重时”的结果。结果 JSON 会写明 `training_pair_source=client_secret_simulator`、`deployment_only=false`。

例如，在已准备好本地模型和目标导出的前提下，从仓库根目录执行：

```powershell
python scripts/evaluation/privacy/run_ima.py --target stage_j_redesign --mode minimal --allow-simulated-known-pairs
python scripts/evaluation/privacy/run_isa.py --target stage_j_redesign --observable-layer 0 --allow-simulated-known-pairs
```

模型目录、目标导出不存在时，需要先准备这些文件；本次测试不会自动下载它们。`run_gate5_scan` 同样需要调用者显式启用这个条件。

## 候选、未知项与评分

- VMA、岭回归 IMA/ISA 的候选只按公开词表和公开种子独立选择，不能把测试答案塞进去。候选有限时，答案可能不在里面；结果同时报告 `candidate_coverage_rate`，即测试答案出现在候选中的比例。这个比例只用于评分，不用于生成预测。
- TFMA 和统计匹配版本 SDA 只用公开参考语料里最常见的 token 作候选。同分布设置现在使用同一模板语料池中的不相交子集，不再把私密原句倒序当成参考语料。
- SDA 没有预测的 token 保留为 `-1`，表示未知。程序不能用秘密置换补上正确答案。
- 频率实验同时报告按 token **种类**计算与按文本中**每次出现**计算的恢复率。旧的 `token_top1/10/100_recovery_rate` 的分母是实际参与匹配的 token 种类数，不是私密文本的全部种类；`token_occurrence_top1/10/100_recovery_rate` 的分母是私密文本中的全部出现次数，未攻击的位置也包含在内。频率方法的 `candidate_coverage_rate` 也只针对参与匹配的种类。
- Top-10 表示正确答案在十个候选里，不表示已经从十个中选对唯一答案。固定名称的 Top-1、Top-10、Top-100 指标始终按对应数量计算；`topk=1/5/10/100` 不会再产生字典键错误或把 Top-1 直接标为 Top-10。
- BLEU-4 只在同一句内截取相邻 token，再累计计分；不会把两句拼接成不存在的词组。代码输出范围为 0 至 1，并使用 `1e-8` 下限。这个具体计分设置不能直接视为与论文未公布的工具设置相同。
- `sensitive_token_recovery_rate` 仍只是公开示例提示词中一组 token 的恢复率，不是有隐私实体标注的 PIIRSR。`high/medium/low` 是项目自行用 30% 和 10% 阈值分档，不能解释成论文证明的安全等级。

另有 `evaluate_entities.py` 对保存预测作实体评分。TTRSR 分母为全部原文 token；此实现的 PIIRSR 按一个标注实体的全部 token 在原位置均正确才算恢复。未猜出的 `-1` 仍算失败。没有实体时返回空值。此约定已明确写入结果，不等于确认作者 PUPA 工具的精确细节，不能与旧的示例敏感词分数混用。

## VMA 读取权重时怎样避免比较对象弄错

原模型有独立的 `lm_head.weight` 时读取真实输出头；只有公开配置明确共享权重时，才允许用 Embedding 代替省略存储的输出头。所有读取支持单个 safetensors 文件与分片索引。

对于本仓库已知的带 buffer 导出，以及公开清单明确标为从这种格式转换的导出，投影权重已经乘入了原归一化层的逐通道权重。VMA 的公开原模型一侧现在同步乘入该权重。投影整体再乘一个正数会被“减均值再调整长度”的特征处理消去；这不代表每个通道不同的乘数也可以省略。其他未知导出方式不在这项自动适配的保证范围内。

## 怎样运行修复检查

```powershell
python -m pytest tests/test_audit_attacks.py -q
```

测试使用本地小矩阵、临时分片权重和简单 tokenizer，检查评分有没有偷看答案、参数边界、实际 Qwen2 反演网络的前向/反向计算，以及权重读取。它不代替真实大模型、真实文本和隐私实体的完整实验。
