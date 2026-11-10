# 第二轮代码修改与运行说明

> 本文保留第二轮结束时的记录。后续已经补充 Llama/MoE 扩维、SDA一致映射、HTTPS/身份验证/合批、密码学随机源和任务/实体评测入口；当前状态及命令见 [第三次补充修改](repair_round3.md)。下文写“尚未实现”的内容不应直接作为当前结论。

本轮代码改在 `D:\Aloepri`。复用了用户提供的桌面“攻击”项目的思路：SDA 因果 Transformer、两种符号编码、IMA 输入线性层、VMA 的排序/L1/投票，以及独立 TFMA。正式代码改用现有 PyTorch 计算，不新增 SciPy 依赖；桌面原教学项目未改动。旧结果不能直接当作本轮结果。

## 改了什么

| 原问题 | 现在的处理 | 仍需注意 |
| --- | --- | --- |
| IMA 要求观测维度整除16 | 观测向量先投到向上取整的网络维度，八个头的每头宽度为偶数；输出仍为原 embedding 宽度 | 这是我们的结构选择。已有 IMA 训练仍使用目标密钥模拟的已知配对，必须显式启用该条件 |
| SDA 只有统计基线 | 增加可训练、可保存和可加载的因果 Transformer | 只处理一对一、长度不变的 token 替换；未强制整段输出满足唯一词表映射 |
| Gate-IA 没有预测实现 | 用公开 embedding 与各层 gate 组合，每行求平均，按原模型公开统计量缩放后匹配 | 噪声和非均匀列缩放可能破坏不变量 |
| Attn-IA 只有模板且公式含糊 | 增加每个 RoPE 二维块的列空间投影对角值，再排序匹配；结果标明解释方式 | 不是唯一可从论文推出的实现，不代表确认了作者算法 |
| VMA 抽样后累计分数 | 新入口使用完整排序行、L1距离、每组权重第一候选投票 | 默认只选 gate；六类组合要显式选择。up/down的非均匀缩放可能破坏排序不变性 |
| ISA 只有岭回归 | 新入口冻结公开模型，用梯度反复优化输入向量，再找公开词表的近邻 | 观测须已经与公开模型指定层同坐标。没有实现未知服务器坐标变换的恢复 |
| 只有同进程编号模拟 | 新增HTTP编号接口与独立客户端，服务端不读取密钥 | 本机串行研究接口，无集群部署、TLS或身份验证 |
| 普通模型导出只做同宽变体 | 新增Qwen2扩维导出：残差D=d+2h，保留原head_dim | 标量RMSNorm仍近似；非平凡RoPE块置换也可能近似；其他框架/真实任务未验证 |
| 标量近似没有独立误差报告 | 增加校准样本估计和另一批样本误差统计 | 校准不能消除方向差异，也不能代替最终任务准确率 |

## 环境

从仓库根目录执行。使用 `requirements-tested.txt` 对应的 Python 环境；本机实际验证环境是 Python 3.13、PyTorch 2.13、Transformers 5.19。以下 `python` 应替换为你安装依赖的解释器。本机可用的现成解释器是：

```powershell
& 'C:\Users\lxl\Desktop\Aloepri\audit_work\venv\Scripts\python.exe' -m pytest
```

新增结果文件和模型文件拒绝覆盖。再次运行时请换一个输出文件名。测试会在临时目录生成随机小模型，不自动下载大模型；缺少的历史文档和用户模型测试会明确跳过。

## Gate-IA、VMA、Attn-IA

`--baseline-model-dir` 是公开原始模型目录；`--server-dir` 是混淆后的标准HF服务器目录。都需要 config 和 safetensors，可分片。这里只读权重，不读取客户端文件。默认候选是完整词表，含公开特殊编号；需要只评普通词时用 `--candidate-ids` 提供独立选定的 JSON 整数列表。`--observed-ids` 可选服务器行号列表。

```powershell
python scripts/evaluation/privacy/run_additional_attacks.py weights --attack gate-ia --baseline-model-dir model/original --server-dir artifacts/changed/server --layers 0 1 --output outputs/gate.json
python scripts/evaluation/privacy/run_additional_attacks.py weights --attack vma --baseline-model-dir model/original --server-dir artifacts/changed/server --layers 0 1 --combinations gate up down_head --output outputs/vma.json
python scripts/evaluation/privacy/run_additional_attacks.py weights --attack attn-ia --baseline-model-dir model/original --server-dir artifacts/changed/server --layers 0 1 --output outputs/attn.json
```

VMA 支持 `embed_head`、`qk`、`gate`、`up`、`down_head`、`router` 六类组合，对应论文表9。默认只选 gate。每个选择的层/组合给第一候选一票；qk先合并各注意力头的整行数据再排序，以免把头置换误当作对应不一致。并列时比较候选名次之和，再比较公开token编号。这是本实现补充的规则。此匹配允许多个服务器行预测到同一个明文，不强制全局一一匹配。

`embed_head` 和 `qk` 涉及词表平方大小的矩阵。默认单个特征表上限 `--max-feature-elements 16000000`；超出就报错，按实际内存显式调整。程序仍会把模型权重读入CPU，并保留各个选定来源的特征；该上限不是整个进程的内存上限。缺少router权重时报缺失；标准目录入口只接受Qwen2/Llama，尚未接入MoE模型布局。

若目标的线性权重已经吸收原始RMSNorm的gamma，给原侧使用 `--fuse-original-norms`。同宽导出保留norm时不要加；新增扩维导出已吸收gamma，需要加。该选项不能自动补偿标量近似、噪声或其他缩放。

Attn-IA采用 `diag(Q @ pinv(Q))`，其中Q是 `E @ Wq` 的一个RoPE二维块。用SVD取列空间，记录各块秩；Qwen/Llama的RoPE配对按前后半部，不按相邻列。列空间投影在可逆二维右变换下不变，排序还去除块/头顺序；这里必须先用完整词表求投影，再取候选行。秩判断有数值阈值，噪声可能改变秩。这段补全不能替代作者对原公式的解释。

## SDA Transformer

输入 JSON 是token编号的二维列表，例如 `[[0,1,0],[1,2]]`。编号必须由同一公开tokenizer定义，明文范围为0至 `vocab_size-1`。训练、验证必须独立，程序拒绝两者有完全相同的句子；近似重复和来源重叠还需准备数据时排除。观测文件只存替换后的符号，例如 `[[80,31,80]]`，不能附带测试明文。

```powershell
python scripts/evaluation/privacy/run_additional_attacks.py sda-train --train-json data/train.json --validation-json data/validation.json --vocab-size 100 --model-path outputs/sda.pt --mode first_occurrence --epochs 12 --output outputs/sda_train.json
python scripts/evaluation/privacy/run_additional_attacks.py sda-predict --model-path outputs/sda.pt --observed-json data/observed.json --output outputs/sda_prediction.json
```

`first_occurrence` 按首次出现顺序编码，例如 `[8,3,3,8,3,9]` 变为 `[0,1,1,0,1,2]`；`frequency_rank` 按次数排名编码，变为 `[1,0,0,1,0,2]`。两者来自不同的文献表述，不能混为一谈。模型文件保存选择。训练是“编码段 + 分隔标记 + 明文段”的下一token预测，0补位不计损失；验证损失决定保存哪个训练轮次。预测不读取测试答案，按原观测长度逐token生成。能运行不代表能读懂任意真实文本。

## ISA迭代优化

先准备 `torch.save(tensor, path)` 的观测张量，形状 `[批数,长度,隐藏维度]`。它必须已经处于公开模型所选隐藏层的坐标空间；即使宽度相同，也需要确认坐标相同。程序只检查形状，不能自动判断未知旋转是否存在。不要用客户端秘密逆变换观测后，又把实验称为只有服务器信息的攻击。

```powershell
python scripts/evaluation/privacy/run_additional_attacks.py isa --model-dir model/original --observed-states data/states.pt --layer -1 --steps 200 --learning-rate 0.05 --output outputs/isa.json
```

默认用公开embedding初始化猜测，冻结模型，用Adam降低内部状态均方误差，最后取词表余弦近邻。输出完整损失过程。`--layer -1` 是模型返回的最后隐藏状态；其他层按HF hidden_states编号。旧 `run_isa.py` 仍是已知配对岭回归，没有被自动改成迭代ISA。

## TFMA及单独评分

```powershell
python scripts/evaluation/privacy/run_additional_attacks.py tfma --reference-json data/public_reference.json --observed-json data/observed.json --output outputs/tfma.json
python scripts/evaluation/privacy/run_additional_attacks.py evaluate --predictions outputs/sda_prediction.json --truth-json data/truth.json --output outputs/sda_score.json
```

TFMA按两边频率排名配对；同频按编号决定结果，但同频本来无法唯一辨认。参考候选不够时第一猜测为-1。参考数据与真实客户原文需要独立准备，程序无法替你证明两份文件来源独立。

评分命令读已保存的预测和单独的真实答案。序列结果的答案是二维列表；权重映射结果的答案是一维列表，顺序对应结果 `observed_ids`，不是默认按服务器编号从小到大。报告token准确率和整句/整表准确率，不自动给出“安全”结论。

## 分进程编号接口

终端一只传服务器模型目录：

```powershell
python scripts/release/serve_token_ids.py --server-dir artifacts/changed/server --port 8765
```

终端二在客户端运行，分词、置换和恢复编号都在这里完成：

```powershell
python scripts/release/infer_remote_ids.py --url http://127.0.0.1:8765/generate --tokenizer-dir model/original --client-secret artifacts/changed/client/client_secret.pt --prompt "你好" --max-new-tokens 16
```

此客户端按原始文本分词，不自动插入聊天模板。服务器 `/generate` 的请求字段是混淆 `input_ids`、生成参数和混淆EOS/PAD编号，不包含置换表。一次处理一条序列，复用KV cache，并检查原始logits是否有限。服务端不把混淆编号解码成文字。`tokenizer_roundtrip` 可检查某个编号序列经文本往返是否改变；一次通过不保证所有序列都通过。不要把这个本机接口的时间当成论文集群延迟。

## Qwen2扩维实验导出和RMS误差

```powershell
python scripts/offline_obfuscation/export_expanded_qwen_checkpoint.py --model-dir model/original --export-dir artifacts/qwen/expanded --expansion-size 4 --alpha-e 0 --alpha-h 0
```

`--expansion-size h` 必须为正偶数，残差宽度D=d+2h。Q/K/V输出与原注意力头宽不变；输入投影接收D维，输出投影返回D维。各残差分支共享同一个P。各投影的Q可不同，只要PQ=I。FFN中间置换与缩放、注意力变换都离线融合为普通权重。所有RMSNorm共用标量κ；未指定时取 `||P||_F/sqrt(d)`，这是均方伸缩估计，不能说是论文中的期望比值或真实语料校准。系数 `c=κ sqrt(d/D)`，norm权重为c，epsilon改为 `c² epsilon`。原gamma融合进相邻权重。

默认包含RoPE块/头置换，可能另有近似。可选 `--attention-profile simplified` 或 `rqk_hqk` 做不含块置换的对照，但不能把这种对照当作全部论文变换。恢复矩阵和随机种子留在client文件；server只有标准模型文件。导出仍标记 `paper_algorithm1_reproduced=false`，因为使用算法1构造P不代表完整论文和部署已复现。

```powershell
python scripts/evaluation/report_rms_error.py --inputs data/rms_inputs.pt --output outputs/rms_error.json
```

`rms_inputs.pt` 是字典，键为 `P`、`calibration_inputs`、`evaluation_inputs`，后两项是原坐标激活，最后一维为d。文件在客户端使用。程序在校准样本估计κ，在另一批样本测平均、95%分位、最大相对误差；也报告P的奇异值范围。数据来源和独立性由调用者负责。它不改变既有模型参数，也不保证消除各方向的伸缩差异。

## 验证和尚未完成的部分

新增测试使用自建随机小模型：检查矩形扩维、头置换、不同候选、频率并列、Transformer因果性、实际训练/反演、CPU/CUDA、分片保存、标准加载、分进程HTTP、缓存和EOS。扩维导出还与原维度下独立写出的同一标量近似计算比较；这不是与原始模型严格等价的声明。

仍未完成论文大模型/原始任务、真实语料攻击成功率、vLLM/SGLang和集群性能评测。Llama扩维、MoE布局、未知坐标下ISA和Attn-IA原文解释仍需后续实现或确认。历史测试中缺少的26项文档检查和6项真实模型/历史产物检查仍明确跳过，没有伪造这些材料或把它们计为通过。
