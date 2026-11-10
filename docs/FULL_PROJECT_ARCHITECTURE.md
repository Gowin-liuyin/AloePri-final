# AloePri 论文对齐项目完整架构明细

本文档按当前 `paper_aligned_project/` 的实际文件组织说明每个文件的职责。它的目标不是复述旧的阶段 A-H/J/K 命名，而是从论文复现角度解释项目：离线混淆、发布包、私有推理、正确性验证、隐私攻击评测分别由哪些文件承担。

> 说明：`src/aloepri/_implementation/` 中有少量继承自原实验代码的函数名或元数据字段仍包含历史 `stage_*` 字样。这些不再作为项目架构入口使用，只是内部实现兼容层。用户应该从 `scripts/` 和 `src/aloepri/` 的论文语义接口进入。

## 1. 总体工作流

本项目围绕论文复现链路组织：

```text
原始模型 checkpoint
  -> offline_obfuscation：离线混淆权重、token 空间、隐藏空间、attention、FFN、norm
  -> paper checkpoint：论文一致的混淆模型权重与元数据
  -> release bundle：可搬运、可部署的 server / client / metadata 发布包
  -> private_inference：输入 token 混淆、server 推理、输出恢复
  -> evaluation：正确性回归与隐私攻击评测
```

核心分层如下：

```text
paper_aligned_project/
  README.md                         # 项目入口说明
  pyproject.toml                    # Python 项目与 pytest 配置
  environment.qwen-transformers.yml # Conda / pip 环境描述
  docs/                             # 架构、结构、Llama 使用说明
  scripts/                          # 命令行入口
  src/                              # 代码实现
    aloepri/                        # 论文语义的主 API
    security_qwen/                  # Qwen 隐私攻击评测
    *.py                            # 低层数学与原型组件
  tests/                            # 单元测试、集成测试、文档一致性测试
```

## 2. 根目录文件

### `README.md`

项目的第一入口。它说明这个目录是从原仓库拆出的独立项目，列出推荐的 Qwen 与 Llama-3.2-3B-Instruct 使用流程，并指向主要文档。它应该保持简短，适合快速上手。

### `.gitignore`

忽略运行时产物、缓存、模型权重、实验输出等不应进入版本管理的文件。这个文件保证项目可以反复运行导出与评测脚本，而不把大模型 checkpoint、artifact 或 Python 缓存误提交。

### `pyproject.toml`

定义 Python 项目元信息：

- 项目名：`aloepri-paper-aligned`
- Python 版本要求：`>=3.10`
- pytest 配置：测试路径为 `tests`，并把项目根目录加入 `pythonpath`

当前它没有声明完整 pip 依赖，环境依赖主要放在 `environment.qwen-transformers.yml` 中。

### `environment.qwen-transformers.yml`

Conda 环境文件。用于创建能运行 Qwen / Llama transformers 代码的环境：

- Python 3.11
- `torch`
- `transformers`
- `accelerate`
- `safetensors`
- `pytest`

如果你把项目搬到新机器，推荐先用这个文件建环境，再运行脚本。

## 3. 文档目录 `docs/`

### `docs/ARCHITECTURE.md`

短版架构说明。它说明项目主层级：

- `src/aloepri/config.py`
- `src/aloepri/adapters/`
- `src/aloepri/keys.py`
- `src/aloepri/layers/`
- `src/aloepri/engine.py`
- `scripts/offline_obfuscation/`
- `scripts/release/`
- `scripts/evaluation/`

这个文件适合作为架构原则备忘：新入口应使用论文语义命名，不再增加历史阶段命名的用户入口。

### `docs/PROJECT_STRUCTURE.md`

结构规划文件。它明确区分：

- clean surface：面向用户和脚本的干净接口
- internal implementation：`src/aloepri/_implementation/` 内部兼容实现层

它也列出未来维护规则：新增代码应进入 `token_space`、`offline_obfuscation`、`paper_checkpoint`、`release_bundle`、`private_inference`、`correctness`、`privacy` 这些语义区域。

### `docs/LLAMA_INSTRUCT_OBFUSCATION.md`

Llama-3.2-3B-Instruct 的操作说明。它面向你最关心的目标：生成 Llama-Instruct 混淆模型。通常包括：

- 原始模型目录放置方式
- 导出 paper checkpoint 的命令
- 导出 release bundle 的命令
- 推理和正确性验证入口
- 迁移项目时需要带走哪些目录

### `docs/FULL_PROJECT_ARCHITECTURE.md`

当前这份完整文件级架构明细。它负责把每个文件在复现链路中的位置说清楚。

## 4. 脚本入口 `scripts/`

脚本目录是你真正运行项目时最常用的入口。它们尽量只做参数解析和 API 调用，不直接塞复杂算法。

### `scripts/_project_root.py`

公共脚本辅助文件。它提供 `add_project_root()`：

- 解析当前项目根目录
- 把项目根目录加入 `sys.path`
- 让从任意脚本目录运行命令时都能导入 `src` 和 `src/aloepri`

所有命令行脚本都应该优先调用它，而不是手写路径处理。

### `scripts/offline_obfuscation/export_qwen_checkpoint.py`

Qwen 论文一致 checkpoint 导出入口。它的职责是：

- 解析 `--export-dir`、`--materialize` 等参数
- 调用 `src/aloepri/checkpoints/qwen.py`
- 生成 Qwen 的 paper-consistent 混淆 checkpoint

这个脚本对应论文流程中的离线混淆阶段：模型权重被变换，推理时不再暴露原始 token / hidden 表示。

### `scripts/offline_obfuscation/export_llama_checkpoint.py`

Llama-3.2-3B-Instruct 论文一致 checkpoint 导出入口。它的职责是：

- 接收 `--model-dir` 指向本地原始 Llama-Instruct 模型
- 接收 `--export-dir` 指向输出目录
- 接收 `--device`、`--dtype` 控制加载与导出精度
- 调用 `src/aloepri/checkpoints/llama.py`

如果你的目标是生成 Llama-Instruct 的混淆模型，这个脚本是第一步。

### `scripts/release/export_qwen_bundle.py`

Qwen 发布包导出入口。它的职责是：

- 从 paper checkpoint 或默认 Qwen 输出位置读取混淆模型
- 调用 release API 生成可部署 bundle
- 写出 server、client、profile、manifest 等发布元数据

它对应论文复现中的“可发布服务端模型 + 客户端恢复信息”打包步骤。

### `scripts/release/export_llama_bundle.py`

Llama 发布包导出入口。它的职责是：

- 读取 Llama paper checkpoint 输出
- 调用 Llama release profile 逻辑
- 生成 Llama-Instruct 可迁移发布包

生成 Llama 混淆模型后，通常第二步就是运行这个脚本。

### `scripts/release/infer_bundle.py`

发布包推理入口。它的职责是：

- 接收 `--bundle-dir`
- 接收 `--profile`
- 接收 `--prompt`
- 调用 `src/aloepri/private_inference.py`
- 完成私有推理链路：输入混淆、服务端生成、输出恢复

这个脚本用于验证 release bundle 是否真的能用。

### `scripts/evaluation/run_qwen_correctness.py`

Qwen 正确性验证入口。它对比原始模型与混淆模型在 prompts 上的输出差异，检查：

- logits 误差
- top-1 是否一致
- top-k overlap
- greedy generation 是否稳定

它调用 `src/aloepri/evaluation/correctness.py` 中的 Qwen correctness API。

### `scripts/evaluation/run_llama_correctness.py`

Llama 正确性验证入口。它服务于 Llama-Instruct 复现确认：

- 读取 Llama release 或 checkpoint
- 运行一组 prompts
- 输出正确性摘要

如果你要确认 Llama 混淆模型“是否按照论文方式复现”，这个脚本应在导出后运行。

### `scripts/evaluation/privacy/run_vma.py`

VMA 隐私攻击评测入口。VMA 通常指 vocabulary mapping attack，目标是评估攻击者能否从可见权重中恢复 token 映射信息。它调用 `src/security_qwen/vma.py`。

### `scripts/evaluation/privacy/run_ima.py`

IMA 隐私攻击评测入口。IMA 主要用于评估输入映射或内部映射相关的信息泄漏风险。它调用 `src/security_qwen/ima.py`。

### `scripts/evaluation/privacy/run_isa.py`

ISA 隐私攻击评测入口。ISA 关注中间状态或推理侧可观测信息是否泄漏。它调用 `src/security_qwen/isa.py`。

## 5. `src/` 根层公共基础模块

`src/` 根层保留了一批低层数学、模型加载、原型混淆组件。它们是 `src/aloepri/` 内部实现和测试经常复用的基础件。

### `src/__init__.py`

把 `src` 目录声明为 Python 包。它本身不承载业务逻辑。

### `src/defaults.py`

集中保存默认配置值。通常包括模型路径、随机种子、artifact 路径、默认 prompts 或默认超参数。它让脚本和测试不必重复硬编码常量。

### `src/model_loader.py`

模型和 tokenizer 加载工具。主要职责：

- 设置全局随机种子
- 把字符串 dtype 解析为 torch dtype
- 加载 Hugging Face 模型和 tokenizer
- 格式化 chat prompt
- tokenizer 编码
- 把 batch 移动到指定 device

Qwen / Llama 导出和 correctness 评测都会依赖它。

### `src/evaluator.py`

正确性评测基础工具。主要职责：

- 计算最大绝对误差 `max_abs_error`
- 计算平均绝对误差 `mean_abs_error`
- 比较 top-1 是否一致
- 统计 top-k overlap
- 安全地把 token id 转成字符串
- 收集 prompt 输出
- 写出 JSON 结果

它是 correctness 脚本下层的通用评估工具。

### `src/key_manager.py`

token 置换密钥管理。主要职责：

- 找出 ordinary token ids，避免特殊 token 被错误混淆
- 生成词表置换 `perm`
- 生成逆置换 `inv_perm`
- 验证置换是否合法
- 保存和加载置换文件

论文中的 token space obfuscation 依赖这里的工具。

### `src/transforms.py`

推理时张量变换工具。主要职责：

- 把输入 token id 映射到混淆 token 空间
- 把输出 token id 从混淆空间映射回原空间
- 还原 logits 顺序
- 对 hidden states 应用 hidden transform
- 对 hidden states 应用 inverse hidden transform

它对应 private inference 中客户端侧“输入混淆 / 输出恢复”的基础操作。

### `src/hidden_keys.py`

隐藏维度变换密钥。主要职责：

- 定义 `HiddenTransform`
- 生成 hidden permutation
- 生成 hidden scaling
- 组合 hidden transform
- 生成逆变换
- 校验 transform
- 构建 identity transform

这是 hidden state 维度置换和缩放的基础模块。

### `src/attention_keys.py`

attention 复杂混淆密钥。主要职责：

- 定义 intra-head QK 旋转 / 反旋转
- 定义 inter-head permutation
- 生成 Q/K/V 相关的块置换
- 支持 GQA 布局下的 KV head permutation 和 group permutation
- 构造 `AttentionComplexConfig`

它服务于 attention 层的结构保持混淆。

### `src/gqa_layout.py`

Grouped Query Attention 布局描述。它定义 `GQALayout`，用于记录：

- query heads 数量
- kv heads 数量
- 每组 query heads 如何对应一个 kv head
- head_dim 等形状信息

Qwen、Llama 这类 GQA 模型的 attention 混淆需要它。

### `src/keymat.py`

KeyMat 矩阵族核心实现。主要职责：

- 定义 `KeyMatBases`
- 定义 `KeyMatTransform`
- 采样高斯矩阵
- 采样正交矩阵
- 构造 nullspace basis
- 生成 key matrix 和 inverse key matrix
- 校验正逆矩阵是否互逆
- 对 hidden states 应用 KeyMat 和 inverse KeyMat

它是隐藏空间可逆线性混淆的重要数学核心。

### `src/keymat_embed_head.py`

embedding 和 lm head 的 KeyMat 混淆工具。主要职责：

- 对 embedding weight 加噪声
- 对 lm head weight 加噪声
- 对 embedding 应用 KeyMat 变换
- 对 lm head 应用 inverse KeyMat 适配
- 定义 `KeyMatEmbedHeadArtifacts`
- 提供 `KeyMatEmbeddingWrapper` 和 `KeyMatHeadWrapper`

它把 token 置换和 hidden KeyMat 变换接到模型输入输出两端。

### `src/keymat_attention_bridge.py`

attention 层的 KeyMat 桥接 wrapper。它把外部 hidden 表示和 attention 内部期望表示连接起来，保证混淆后的 hidden space 仍能通过 attention 模块。

### `src/keymat_ffn.py`

FFN 层的 KeyMat 桥接与包装。主要职责：

- 定义 `KeyMatFFNBridge`
- 包装 MLP / FFN 模块
- 在 FFN 输入输出之间应用 KeyMat 或 inverse KeyMat

### `src/keymat_norm.py`

RMSNorm 与 KeyMat 的桥接工具。主要职责：

- 估计 KeyMat 与 RMSNorm 之间的缩放修正 `kappa`
- 定义 `KeyMatRMSNormBridge`
- 构建 RMSNorm wrapper
- 尝试把 norm 变换融合到相邻线性层

### `src/obfuscate_embed_head.py`

早期但仍可复用的 embedding / lm head 词表置换工具。主要职责：

- 置换 embedding weight 行
- 置换 lm head weight 行
- 置换输出 bias
- 构建词表置换后的模型

它处理 token-level permutation，不处理完整 KeyMat。

### `src/obfuscate_attention_complex.py`

复杂 attention 混淆实现。主要职责：

- 定义带 trace 的 complex attention wrapper
- 应用 Q/K/V 头内和头间变换
- 支持 attention score / value 变换追踪
- 把部分 intra-head QK transform 融合到投影层

这是 attention 混淆的实验性低层实现，`aloepri` 内部实现可复用它。

### `src/obfuscate_ffn.py`

FFN 中间维度混淆实现。主要职责：

- 定义 `FFNTransform`
- 生成 FFN intermediate permutation
- 生成 FFN scaling
- 对 up/gate/down 路径做对应变换
- 定义 `ObfuscatedQwen2MLP`
- 提供 FFN block 混淆函数

### `src/obfuscate_rmsnorm.py`

RMSNorm 混淆实现。主要职责：

- 按 hidden transform 置换 norm weight
- 估计 RMSNorm 近似修正项 `kappa`
- 定义 `ObfuscatedRMSNorm`
- 对 RMSNorm 应用混淆 wrapper

### `src/llama_local_dev.py`

本地 Llama 开发辅助。主要职责：

- 从本地模型目录加载 Llama tokenizer
- 从本地模型目录加载 config
- 基于本地 metadata 构造 mock Llama 模型
- 输出 mock summary
- 对 Llama prompt 做 tokenizer 编码

它让没有完整大模型权重时，也能测试部分 Llama 配置和脚本行为。

## 6. 主 API 包 `src/aloepri/`

`src/aloepri/` 是新项目面向论文复现的主表面。新增用户入口原则上应该放在这里，而不是直接暴露 `_implementation`。

### `src/aloepri/__init__.py`

AloePri 包入口。它用于集中导出或标记主包，不承载具体混淆算法。

### `src/aloepri/config.py`

论文复现配置对象。它定义 `AloePriConfig`，用于记录：

- hidden size
- number of heads
- number of kv heads
- vocab size
- intermediate size
- seed
- key scale / noise scale 等混淆参数

它是 engine、keys、layers 的共同配置来源。

### `src/aloepri/keys.py`

AloePri 统一密钥构建层。主要职责：

- 定义 `AloePriKeys`
- 根据 `AloePriConfig` 和 tokenizer vocab size 生成所有密钥
- 生成 token permutation
- 生成 hidden transform
- 生成 attention complex config
- 生成 FFN transform
- 生成 KeyMat transform

它把分散在 `src/key_manager.py`、`src/hidden_keys.py`、`src/attention_keys.py`、`src/keymat.py` 的低层能力组合成论文复现需要的一组密钥。

### `src/aloepri/token_ops.py`

token 空间操作 API。它负责给上层提供语义更清楚的 token 映射能力，例如：

- plain token -> obfuscated token
- obfuscated token -> plain token
- logits restore

它是 private inference 和 token-space checkpoint 的轻量入口。

### `src/aloepri/engine.py`

离线混淆协调器。它定义 `AloePriEngine`，用于把配置、模型 adapter、密钥、层 wrapper 串起来。它应该承担“用论文机制改造一个模型”的总控职责。

### `src/aloepri/project_plan.py`

项目规划与状态说明模块。它以代码结构描述当前项目的论文链路、推荐入口和迁移边界。测试中会用它检查新项目是否遵守干净命名和独立结构。

### `src/aloepri/private_inference.py`

私有推理入口。它面向 release bundle 运行时，负责：

- 加载发布包 profile
- 加载 tokenizer / server 模型
- 对用户 prompt 做 token 混淆
- 调用 server 侧模型生成
- 把输出 token 或文本恢复到原始空间

`scripts/release/infer_bundle.py` 主要调用它。

## 7. 模型适配层 `src/aloepri/adapters/`

模型适配层负责屏蔽不同模型家族在模块命名、层结构、config 字段上的差异。

### `src/aloepri/adapters/__init__.py`

adapter 分发入口。主要职责：

- 根据模型类型选择 Qwen 或 Llama adapter
- 构建统一 architecture config

上层 engine 不应该直接判断模型内部字段，而应通过这里获得适配器。

### `src/aloepri/adapters/qwen.py`

Qwen adapter。它负责识别和访问 Qwen 模型结构，例如：

- embedding 层
- decoder layers
- attention 模块
- MLP / FFN 模块
- RMSNorm
- lm head
- Qwen config 中的 head / kv head / intermediate size

Qwen paper checkpoint、Qwen release、Qwen correctness 都依赖它。

### `src/aloepri/adapters/llama.py`

Llama adapter。它提供与 Qwen adapter 同样的统一访问方式，但适配 Llama / Llama-Instruct 模型结构。Llama-3.2-3B-Instruct 的 checkpoint 导出和 release 打包都通过它减少模型差异。

## 8. 组件层 `src/aloepri/layers/`

这一层对应论文中的模型组件：embedding、attention、FFN、norm、head。它们包装低层混淆操作，给 engine 使用。

### `src/aloepri/layers/base.py`

定义 `ObfuscatedLayer` 基类。它是各类混淆 wrapper 的共同父类，统一保存原始模块、密钥、配置或元数据。

### `src/aloepri/layers/embeddings.py`

embedding 和 lm head 混淆 wrapper。主要职责：

- 定义 `AloePriEmbedding`
- 定义 `AloePriHead`
- 包装原始 embedding
- 包装原始 lm head
- 在输入输出端接入 token permutation 与 hidden transform

### `src/aloepri/layers/attention.py`

attention 混淆 wrapper。主要职责：

- 定义 `AloePriAttention`
- 包装原始 attention 模块
- 接入 attention complex config
- 在 attention 前后处理 hidden representation 的混淆适配

### `src/aloepri/layers/ffn.py`

FFN / MLP 混淆 wrapper。主要职责：

- 定义 `AloePriMLP`
- 包装原始 MLP
- 接入 intermediate permutation / scaling
- 保持 FFN 输出与混淆 hidden space 一致

### `src/aloepri/layers/norm.py`

RMSNorm 混淆 wrapper。主要职责：

- 定义 `AloePriRMSNorm`
- 包装 RMSNorm
- 处理 hidden transform 对 norm weight 的影响
- 估计或应用 norm correction

## 9. Pipeline 层 `src/aloepri/pipelines/`

Pipeline 层把多个组件组合成可执行的导出流程。

### `src/aloepri/pipelines/__init__.py`

pipeline 包入口。用于标记目录为包，并可集中导出 pipeline API。

### `src/aloepri/pipelines/token_space.py`

token-space checkpoint pipeline。主要职责：

- 构建 token-space bundle
- 导出 token-space checkpoint
- 加载 token-space checkpoint

它聚焦词表置换，不代表完整论文混淆。

### `src/aloepri/pipelines/standard_shape.py`

标准形状完整 checkpoint pipeline。主要职责：

- 构建 standard-shape full bundle
- 导出 standard-shape full checkpoint
- 复用 `_implementation/standard_shape_obfuscation.py`

“standard shape” 的意思是导出模型仍保持标准 transformers 模型可加载形状，便于部署。

### `src/aloepri/pipelines/release.py`

release pipeline。它提供 `export_release_bundle()`，把混淆 checkpoint 整理成发布目录，供推理和评测使用。

## 10. Checkpoint 层 `src/aloepri/checkpoints/`

Checkpoint 层是论文一致模型导出的正式 API。

### `src/aloepri/checkpoints/__init__.py`

checkpoint API 包入口。用于集中暴露 Qwen / Llama 导出函数。

### `src/aloepri/checkpoints/qwen.py`

Qwen paper checkpoint 导出 API。它包装 `_implementation/qwen_paper_checkpoint.py`，对脚本提供语义清晰的函数：

- `export_qwen_paper_checkpoint(...)`

它隐藏内部历史字段和复杂 manifest 构造。

### `src/aloepri/checkpoints/llama.py`

Llama paper checkpoint 导出 API。它包装 `_implementation/llama_paper_checkpoint.py`，对脚本提供：

- `export_llama_paper_checkpoint(...)`

这是生成 Llama-Instruct 混淆模型的主 API。

## 11. Release 层 `src/aloepri/release/`

### `src/aloepri/release/__init__.py`

release API 包入口。

### `src/aloepri/release/bundle.py`

发布包导出 API。它把 `_implementation/release_bundle_impl.py` 等内部实现包装成干净接口，供 `scripts/release/export_*_bundle.py` 调用。

发布包通常应该包含：

- server 侧可加载模型
- tokenizer 或 tokenizer 配置
- client 侧 token 恢复信息
- profile manifest
- correctness / provenance metadata

## 12. Evaluation 层 `src/aloepri/evaluation/`

### `src/aloepri/evaluation/__init__.py`

evaluation API 包入口。

### `src/aloepri/evaluation/correctness.py`

正确性评测 API。它提供：

- `run_qwen_correctness(...)`
- `run_llama_correctness(...)`

内部会分发到 Qwen / Llama 对应 correctness 实现。脚本层不需要知道内部文件如何组织。

## 13. 内部实现层 `src/aloepri/_implementation/`

这个目录是项目独立化后的内部实现库。它保留从旧实验代码迁移来的成熟逻辑，但不作为新项目公开架构命名。外部脚本应通过 `src/aloepri/checkpoints`、`src/aloepri/release`、`src/aloepri/evaluation`、`src/aloepri/pipelines` 调用它。

### `src/aloepri/_implementation/__init__.py`

内部实现包标记文件。它不应承担业务入口职责。

### `token_space_runtime.py`

token-space 和 hidden permutation 运行时实验实现。主要包含：

- attention tracing wrapper
- token-space 模型准备
- block0 attention hidden transform 融合
- 单 prompt 运行与指标聚合

它支撑早期 token / hidden 混淆正确性验证。

### `static_attention_obfuscation.py`

静态化 attention 混淆实现。主要职责：

- 根据 GQA layout 构造 query / kv head 顺序
- 对 attention 投影权重应用输出变换
- 定义 `StaticizedQwen2Attention`
- 构建静态化 attention 模块

它让部分 attention 混淆从运行时 wrapper 转为权重级改写。

### `standard_weight_proof.py`

标准权重证明或检查报告生成。它读取 server 目录，构建关于导出权重是否处于标准可见形状的 proof payload。

### `standard_shape_obfuscation.py`

standard-shape 模型混淆核心。主要职责：

- 构造 square transform 对 embedding/head 的适配
- 包装 decoder layer handoff
- attach block hooks
- 原地适配线性层权重
- 构建 standard-shape full model 或 block0 model

它是“导出后仍像标准 transformers checkpoint”的关键逻辑。

### `standard_shape_noise.py`

standard-shape 噪声配置与排序。它定义噪声 case，并根据实验结果排序，用来选择较合适的混淆噪声强度。

### `standard_bridge_export.py`

把内部 buffered / wrapper 形式导出为标准可见 checkpoint 的桥接实现。主要职责：

- copy 或 link 源目录
- 解析 norm strategy
- 识别 buffered state
- 推断 layer indices
- 从 buffered state 构造标准 state dict
- 生成 manifest

它用于把实验模型变成更标准的部署模型。

### `square_transform.py`

square monomial transform 实现。主要职责：

- 定义 `SquareMonomialTransform`
- 构建置换 + 缩放的方阵变换
- 检查正逆变换
- 对 embedding/head 应用 square transform

### `release_bundle_impl.py`

通用 release bundle 内部实现。主要职责：

- 定义 release profile
- 读取 correctness summary
- 生成 profile summary
- copy/link 模型目录
- 导出 release bundle

Qwen release 包主要依赖它。

### `qwen_paper_checkpoint.py`

Qwen paper-consistent checkpoint 的内部实现。主要职责：

- 构建 Qwen 导出目标
- 读取源 manifest
- 构造 export-visible component metadata
- 导出 Qwen paper-consistent candidate
- 生成 attention / FFN / norm 可见性证明
- 生成 correctness regression status
- 汇总 evidence bundle

这是 Qwen 论文一致复现的核心导出文件。

### `qwen_correctness.py`

Qwen release correctness 内部实现。主要职责：

- 解析 profile 路径
- 汇总 prompt 结果
- greedy generate
- 构建 correctness summary
- 跑单个 profile correctness
- 跑完整 release correctness

### `pretrained_deployable.py`

把混淆模型保存为 Hugging Face 风格可加载目录。主要职责：

- 规范 tokenizer config
- 收集 tensor state
- 加载 tensor state
- 导出 pretrained 目录
- 从 pretrained 目录加载混淆模型

### `norm_gap_analysis.py`

RMSNorm 误差分析。它构建 norm gap report，帮助判断 hidden transform 与 RMSNorm 之间的近似误差是否可接受。

### `noise_calibration.py`

混淆噪声校准。它定义 noise calibration case，并根据结果排名，用于选择 correctness 与 privacy 的折中配置。

### `materialize_checkpoint.py`

checkpoint 物化导出。主要职责：

- 构建 redesign manifest
- 构造 component expression
- copy/link 源目录
- 导出 redesign checkpoint
- 构建 regression payload

### `llama_release_profiles.py`

Llama release profile 内部实现。主要职责：

- 定义默认 Llama release profiles
- 导出 Llama release bundle

### `llama_paper_checkpoint.py`

Llama paper checkpoint 内部实现。主要职责：

- 从本地 Llama 模型目录读取 config/tokenizer/weights
- 应用论文一致的 token 与权重混淆导出逻辑
- 写出 Llama paper checkpoint
- 生成 manifest 和 profile 所需元数据

它是 Llama-Instruct 混淆模型生成的核心文件。

### `llama_correctness.py`

Llama correctness 内部实现。主要职责：

- 加载 Llama checkpoint 或 release
- 运行 prompts
- 统计 logits / generation 差异
- 生成 correctness summary

### `layer_recovery.py`

逐层恢复 / 校准实验实现。主要职责：

- 定义 layer config
- attention tracing
- 估计 layer kappa
- attach hooks
- 校准每层 kappa
- 手动 greedy generation
- 聚合逐层结果

它用于检查多层混淆后是否还能恢复正确模型行为。

### `keymat_search.py`

KeyMat 参数搜索。它用于探索不同 KeyMat 构造或超参数对正确性、可逆性、norm gap 的影响。

### `keymat_norm_fusion.py`

KeyMat 与 norm 融合实现。它尝试把 RMSNorm 相关修正吸收到相邻线性层或 wrapper 中，减少运行时额外桥接。

### `keymat_grid.py`

KeyMat 网格实验。它定义或运行不同 KeyMat 参数组合，用于寻找更好的 correctness / privacy 配置。

### `keymat_fusion_runtime.py`

KeyMat 融合运行时。它把 KeyMat 变换尽量融合进模型组件，使 server 侧运行更接近普通 transformers 模型。

### `keymat_ffn_fusion.py`

KeyMat 与 FFN 融合实现。主要职责：

- 定义 norm-fused FFN bridge
- 定义 fused Qwen MLP
- 构建 fused FFN 模块

### `keymat_family.py`

KeyMat 家族构造工具。它提供对 RMSNorm 更友好的 KeyMat transform 构造方式，减少 norm correction 压力。

### `keymat_bridge_runtime.py`

KeyMat 跨层桥接运行时。主要职责：

- 定义每层 KeyMat config
- 校准 KeyMat kappa
- 构建 decoder layer handoff
- 构建 KeyMat 模型 wrapper
- 单 prompt 运行与指标汇总

### `keymat_attention_fusion.py`

KeyMat 与 attention 融合实现。它负责把 hidden-space KeyMat 变换与 attention 投影权重、输入输出桥接关系结合起来。

### `keymat_artifact.py`

KeyMat artifact 读写与元数据整理。它保存 KeyMat transform、校准结果、manifest 或可复现实验配置。

### `expression_audit.py`

表达式审计。它检查导出的 component expression / manifest 是否符合预期，帮助确认发布包没有遗漏关键变换说明。

### `deployable_obfuscation.py`

可部署混淆模型构造。它将低层混淆组件组织成可以保存、加载、推理的模型结构。

### `deployable_artifact.py`

可部署 artifact 管理。它负责整理部署所需文件、metadata、manifest 和路径关系。

### `component_gap_analysis.py`

组件级误差分析。它对 embedding、attention、FFN、norm、head 等组件分别统计差异，定位 correctness 误差来源。

### `checkpoint_io.py`

checkpoint 输入输出工具。它集中处理 checkpoint 读写、state dict 加载、JSON metadata 读写等基础操作。

### `bridge_strategy_scan.py`

bridge strategy 扫描。它尝试不同 bridge 策略，比较 correctness 和误差表现，为最终导出策略选择提供依据。

### `bridge_regression.py`

bridge 回归验证。它检查某个 bridge 策略在固定 prompts / 固定模型上的行为是否保持稳定。

### `block_component_recovery.py`

block 组件恢复实验。它针对单个 block 或局部组件挂 hook，验证混淆前后是否能恢复对应中间结果。

### `attention_gap_analysis.py`

attention 误差分析。它统计 attention 混淆前后的 score、value、output 等误差，帮助判断 attention 变换是否正确。

### `attention_component_recovery.py`

attention 组件恢复实验。它更细粒度地检查 Q/K/V/O projection、head permutation、GQA group 映射等 attention 子组件的恢复情况。

## 14. 隐私评测包 `src/security_qwen/`

这个包目前以 Qwen 为主要评测对象，负责复现 privacy attack 和安全矩阵。虽然目录名带 Qwen，但其中一些评测思想可迁移到 Llama。

### `src/security_qwen/__init__.py`

security_qwen 包入口。

### `src/security_qwen/artifacts.py`

安全评测目标与 artifact 解析。主要职责：

- 定义 security target spec
- 定义 resolved target
- 列出默认安全目标
- 根据目标名解析 artifact 路径
- 输出 security target payload

VMA / IMA / ISA 等评测都依赖它定位模型和元数据。

### `src/security_qwen/datasets.py`

安全评测数据集描述。主要职责：

- 定义 `DatasetSpec`
- 给出默认 phase0 数据集配置
- 输出 dataset payload

这里的 phase0 是历史评测数据命名，不是新项目架构入口。

### `src/security_qwen/gate5_scan.py`

安全 gate5 扫描。主要职责：

- 定义扫描 case
- 加载噪声 case
- 确保 artifact 存在
- 构建扫描结果 payload
- 运行 gate5 scan

### `src/security_qwen/gate6_enhancement.py`

安全 gate6 增强评测。它通常基于前序安全扫描结果，增加更强攻击或更严目标，形成增强报告。

### `src/security_qwen/ia.py`

基础 inference attack 或 identity attack 工具。它提供 IMA / ISA 等攻击可能复用的通用攻击逻辑。

### `src/security_qwen/ima.py`

IMA 攻击实现。主要职责：

- 定义 IMA baseline config
- 构建 IMA template
- 加载目标 bundle
- 抽取攻击特征
- 运行 baseline attack
- 生成 comparison payload

### `src/security_qwen/isa.py`

ISA 攻击实现。主要职责：

- 定义 ISA baseline config
- 构建 ISA template
- 采样 plain sequences
- 抽取 hidden state 或 attention score 观测
- 加载模型 bundle
- 运行 ISA baseline
- 生成 comparison payload

### `src/security_qwen/matrix.py`

安全矩阵 payload 构造。它把各类攻击、目标、指标汇总成矩阵形式，便于报告展示。

### `src/security_qwen/metrics.py`

安全评测指标工具。它定义 attack success rate、top-k 命中、rank、泄漏分数等通用指标计算。

### `src/security_qwen/schema.py`

安全评测 JSON schema 或 payload schema。它保证不同攻击输出的报告结构一致，便于测试和文档引用。

### `src/security_qwen/sda.py`

SDA 攻击或安全分析实现。它用于补充 token / state 泄漏方向的评测。

### `src/security_qwen/summary.py`

安全评测汇总。它把 VMA / IMA / ISA / gate scan 等结果整合成可读摘要。

### `src/security_qwen/tfma.py`

TFMA / frequency mapping attack 实现。主要职责：

- 定义 frequency attack config
- 构建语料模板
- tokenization 和混淆序列统计
- 根据频率尝试恢复 token 映射

### `src/security_qwen/vma.py`

VMA / vocabulary mapping attack 实现。主要职责：

- 定义 VMA baseline config
- 加载标准或 buffered server 权重
- 推断可攻击投影层
- 构造权重特征
- 采样候选 token 集
- 运行 top-k 恢复攻击
- 构建 comparison、source attribution、layer ablation payload

这是 privacy 评测中最重要的文件之一。

## 15. 测试目录 `tests/`

测试目录验证数学工具、组件混淆、adapter、security payload、文档一致性等。它保留了一些历史测试名，但已经移除了直接以阶段文件命名的测试入口。

### `tests/conftest.py`

pytest 公共配置。通常负责：

- 设置导入路径
- 提供共享 fixtures
- 准备临时目录或测试数据

### `tests/test_transforms.py`

测试 token 映射、logits restore、hidden transform 等基础变换是否正确。

### `tests/test_key_manager.py`

测试 token permutation 的生成、保存、加载、逆置换和合法性校验。

### `tests/test_hidden_keys.py`

测试 hidden permutation / scaling / inverse transform / identity transform。

### `tests/test_attention_keys.py`

测试 attention keys、QK rotation、head permutation、GQA group permutation 的构造。

### `tests/test_gqa_layout.py`

测试 GQA layout 的形状关系和 query/kv head 分组逻辑。

### `tests/test_keymat.py`

测试 KeyMat 生成、正逆关系、hidden 应用和恢复。

### `tests/test_keymat_embed_head.py`

测试 KeyMat 在 embedding 和 lm head 两端的变换是否保持 logits 恢复关系。

### `tests/test_obfuscate_embed_head.py`

测试 embedding/head 词表置换逻辑。

### `tests/test_integration.py`

集成测试。它把多个基础组件串起来，验证局部端到端行为。

### `tests/test_aloepri_qwen_adapter.py`

测试 Qwen adapter 能否正确识别模型结构、读取配置、暴露统一组件访问接口。

### `tests/test_aloepri_llama_adapter.py`

测试 Llama adapter 能否正确解析 Llama / mock Llama 结构。

### `tests/test_aloepri_project_plan.py`

测试项目规划与命名规则。它通常检查：

- 新项目是否存在预期目录
- 是否不再暴露旧阶段式用户入口
- 独立项目结构是否完整

### `tests/test_mainline_docs_history.py`

文档历史一致性测试。它用于确保主线文档没有重新引入不希望的阶段式入口或过时说明。

### `tests/test_qwen_paper_consistent_docs.py`

Qwen 论文一致文档测试。它检查文档中描述的 Qwen 导出、发布和验证流程是否与代码入口一致。

### `tests/test_security_qwen_schema.py`

测试安全评测 schema 结构是否稳定。

### `tests/test_security_qwen_matrix.py`

测试安全矩阵 payload 是否包含预期字段和目标。

### `tests/test_security_qwen_summary.py`

测试安全评测汇总输出。

### `tests/test_security_qwen_frequency.py`

测试 frequency attack / TFMA 相关语料、统计或恢复逻辑。

### `tests/test_security_qwen_vma.py`

测试 VMA 基础流程。

### `tests/test_security_qwen_vma_analysis.py`

测试 VMA 分析 payload，例如 source attribution 或 layer ablation。

### `tests/test_security_qwen_vma_comparison.py`

测试 VMA 不同目标或不同配置之间的 comparison payload。

### `tests/test_security_qwen_vma_redesign.py`

测试 redesign 后的 VMA 目标和输出结构。

### `tests/test_security_qwen_ima.py`

测试 IMA 基础攻击流程。

### `tests/test_security_qwen_ima_redesign.py`

测试 redesign 后的 IMA 目标和输出结构。

### `tests/test_security_qwen_isa.py`

测试 ISA 基础攻击流程。

### `tests/test_security_qwen_isa_redesign.py`

测试 redesign 后的 ISA 目标和输出结构。

### `tests/test_security_qwen_gate5.py`

测试 gate5 安全扫描。

### `tests/test_security_qwen_gate6.py`

测试 gate6 增强安全评测。

### `tests/test_security_qwen_redesign_targets.py`

测试 redesign 后 security targets 的路径、字段和目标集合。

## 16. 运行时目录和输出文件

这些目录不一定在干净仓库中存在，但运行项目时会出现。

### `model/`

建议放原始大模型目录，例如：

```text
model/Llama-3.2-3B-Instruct/
```

这个目录通常很大，不应提交。Llama checkpoint 导出需要 `--model-dir` 指向这里。

### `artifacts/`

建议作为所有导出产物根目录。常见结构：

```text
artifacts/
  qwen/
    paper_consistent_checkpoint/
    release_bundle/
  llama_3_2_3b_instruct/
    paper_consistent_checkpoint/
    release_bundle/
```

### `outputs/` 或评测结果目录

用于保存 correctness 和 privacy 评测 JSON 报告。具体路径由脚本参数决定。

## 17. Llama-Instruct 混淆模型生成链路

如果目标是生成 Llama-3.2-3B-Instruct 的混淆模型，文件调用关系是：

```text
scripts/offline_obfuscation/export_llama_checkpoint.py
  -> src/aloepri/checkpoints/llama.py
    -> src/aloepri/_implementation/llama_paper_checkpoint.py
      -> src/model_loader.py
      -> src/aloepri/adapters/llama.py
      -> src/aloepri/keys.py
      -> src/aloepri/layers/*

scripts/release/export_llama_bundle.py
  -> src/aloepri/release/bundle.py
    -> src/aloepri/_implementation/llama_release_profiles.py
    -> src/aloepri/_implementation/release_bundle_impl.py

scripts/release/infer_bundle.py
  -> src/aloepri/private_inference.py
```

推荐命令：

```bash
python scripts/offline_obfuscation/export_llama_checkpoint.py \
  --model-dir model/Llama-3.2-3B-Instruct \
  --export-dir artifacts/llama_3_2_3b_instruct/paper_consistent_checkpoint \
  --device cpu \
  --dtype bfloat16

python scripts/release/export_llama_bundle.py \
  --export-dir artifacts/llama_3_2_3b_instruct/release_bundle \
  --materialize

python scripts/release/infer_bundle.py \
  --bundle-dir artifacts/llama_3_2_3b_instruct/release_bundle \
  --profile default \
  --prompt "请用一句话介绍你自己。"
```

导出后应再运行：

```bash
python scripts/evaluation/run_llama_correctness.py \
  --release-dir artifacts/llama_3_2_3b_instruct/release_bundle
```

这样才能从“能导出”进一步验证到“行为与论文复现目标一致”。

## 18. 维护原则

后续你继续开发时，建议遵守：

1. 新的用户入口放在 `scripts/offline_obfuscation/`、`scripts/release/`、`scripts/evaluation/`。
2. 新的公开 Python API 放在 `src/aloepri/` 的语义目录下。
3. 不再新增 `stage_*.py`、`phase_*.py` 这类用户可见文件。
4. `_implementation/` 可以继续承载复杂内部逻辑，但不要让用户直接调用它。
5. 每新增一个导出流程，都补一条 correctness 验证路径。
6. 每新增一个 privacy 目标，都在 `src/security_qwen/artifacts.py` 或对应安全包里登记，并加 schema 测试。
