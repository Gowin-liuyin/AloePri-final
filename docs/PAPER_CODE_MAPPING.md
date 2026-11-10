# AloePri 论文复现与代码对应说明

本文档用于说明本仓库如何对应论文
`Towards Privacy-Preserving LLM Inference via Covariant Obfuscation (Technical Report)`。

论文核心思想可以概括为一句话：**不要只混淆用户输入，而是同时变换输入 token、模型权重和输出恢复规则，使混淆空间里的推理结果能映射回明文空间，同时服务端只能看到混淆后的 prompt、模型和 response。**

在论文符号里，这是 covariant obfuscation：

- `phi_X`：输入数据混淆，主要是 token permutation。
- `phi_Theta`：模型参数混淆，覆盖 embedding、head、attention、FFN、RMSNorm。
- `phi_Y`：输出标签/token 空间混淆。
- `psi_Y`：客户端输出恢复。
- `f_tilde`：服务端在混淆模型上执行的普通 LLM 推理。

在当前代码里，这条链路对应为：

```text
baseline Hugging Face model
  -> scripts/offline_obfuscation/*        # 离线模型混淆
  -> src/aloepri/checkpoints/*            # paper-consistent checkpoint API
  -> scripts/release/*                    # release bundle 打包和推理
  -> src/aloepri/private_inference.py     # 在线 token 混淆与输出恢复
  -> scripts/evaluation/*                 # correctness / privacy 验证
```

## 1. 论文工作流与项目入口

论文把 AloePri 分成两大阶段：

1. Offline Model Obfuscation：客户端离线生成密钥和混淆模型。
2. Online Private Inference：客户端混淆 prompt，服务端用混淆模型推理，客户端恢复输出。

当前项目的入口与它一一对应：

| 论文阶段 | 当前入口 | 说明 |
| --- | --- | --- |
| 离线模型混淆 | `scripts/offline_obfuscation/export_qwen_checkpoint.py` | 导出 Qwen paper-consistent checkpoint。 |
| 离线模型混淆 | `scripts/offline_obfuscation/export_llama_checkpoint.py` | 导出 Llama-3.2-3B-Instruct paper-consistent checkpoint。 |
| checkpoint API | `src/aloepri/checkpoints/qwen.py` | Qwen checkpoint 的公开 API 包装。 |
| checkpoint API | `src/aloepri/checkpoints/llama.py` | Llama checkpoint 的公开 API 包装。 |
| release bundle | `scripts/release/export_qwen_bundle.py` | 将 Qwen 混淆 checkpoint 打包成 server/client bundle。 |
| release bundle | `scripts/release/export_llama_bundle.py` | 将 Llama 混淆 checkpoint 打包成 server/client bundle。 |
| 在线私有推理 | `scripts/release/infer_bundle.py` | 从 release bundle 加载模型，执行 token 混淆、推理、输出恢复。 |
| correctness 评测 | `scripts/evaluation/run_qwen_correctness.py` | 验证 Qwen release surface 的输出一致性。 |
| correctness 评测 | `scripts/evaluation/run_llama_correctness.py` | 验证 Llama release surface 的输出一致性。 |
| privacy 评测 | `scripts/evaluation/privacy/run_vma.py` | VMA 攻击评测。 |
| privacy 评测 | `scripts/evaluation/privacy/run_ima.py` | IMA 攻击评测。 |
| privacy 评测 | `scripts/evaluation/privacy/run_isa.py` | ISA 攻击评测。 |

## 2. Covariant Obfuscation 总体抽象

论文 Section 4 先定义 covariant obfuscation，再用 sequential、parallel、summation composition 说明复杂 Transformer 可以由组件级混淆组合起来。

当前代码没有把这些定理写成形式化证明器，而是把它们落成工程结构：

| 论文概念 | 当前代码 | 技术含义 |
| --- | --- | --- |
| 模型配置空间 | `src/aloepri/config.py` | 记录 hidden size、head 数、KV head 数、RoPE 参数、噪声强度、KeyMat 参数等。 |
| 模型族适配 | `src/aloepri/adapters/qwen.py`, `src/aloepri/adapters/llama.py` | 把 Qwen/Llama 的结构统一成 embedding、layers、attention、MLP、norm、head。 |
| 统一密钥生成 | `src/aloepri/keys.py` | 生成 token permutation、KeyMat、attention keys、FFN transform。 |
| 组件级混淆 | `src/aloepri/layers/*` | embedding、attention、FFN、norm、head 分别包装成可组合模块。 |
| 离线总控 | `src/aloepri/engine.py` | 把 adapter、keys、layer wrappers 串成一个混淆模型。 |
| 内部兼容实现 | `src/aloepri/_implementation/*` | 承载历史实验代码和当前 checkpoint/release 的底层逻辑。 |

可以这样理解：论文的组合定理对应代码里的“分组件 wrapper + 统一 hidden/token 空间约定”。每个组件都保证输入/输出空间能和相邻组件接上，最后组合成完整模型。

## 3. Token-Level Permutation

论文中，客户端采样一个秘密词表置换 `tau`。输入 token 先映射到混淆 token，模型输出的混淆 token 再由客户端映射回原 token。

当前代码对应：

| 文件 | 职责 |
| --- | --- |
| `src/key_manager.py` | 生成、反转、保存、加载 vocab permutation；跳过 special tokens。 |
| `src/transforms.py` | `map_input_ids`、`unmap_output_ids`、`restore_logits`。 |
| `src/aloepri/token_ops.py` | 给上层暴露语义更清楚的 token-space API。 |
| `src/aloepri/_implementation/token_space_runtime.py` | 构建 token-space obfuscated model，生成 `perm_vocab` / `inv_perm_vocab`。 |
| `src/aloepri/private_inference.py` | 在线推理时对 prompt token 做映射，并用 `restore_logits` 恢复输出空间。 |

技术要点：

- `ordinary_token_ids(tokenizer)` 会排除 special token，避免 BOS/EOS/PAD 等控制 token 被错误置换。
- `perm_vocab[input_ids]` 实现 `phi_X`。
- `inv_perm_vocab[output_ids]` 或 `restore_logits(logits, perm_vocab)` 实现 `psi_Y`。
- `restore_logits` 不只是恢复 token id，也恢复 logits 的词表维度顺序，因此可以继续做 greedy decoding 或 top-k 比较。

## 4. Key Matrix / Hidden-Space Transform

论文 Algorithm 1 生成 key matrix 和 inverse key matrix。它的作用是把 hidden space 变到混淆空间，同时保证相邻组件之间可以通过 inverse 抵消，维持推理正确性。

当前代码对应：

| 文件 | 职责 |
| --- | --- |
| `src/keymat.py` | `KeyMatBases`、`KeyMatTransform`、key/inverse 生成、逆关系检查。 |
| `src/aloepri/keys.py` | 根据 `AloePriConfig` 构建全局 `keymat_transform`。 |
| `src/keymat_embed_head.py` | embedding/head 与 KeyMat 的连接。 |
| `src/keymat_attention_bridge.py` | attention 层输入输出空间桥接。 |
| `src/keymat_ffn.py` | FFN 层输入输出空间桥接。 |
| `src/keymat_norm.py` | RMSNorm 与 KeyMat 的缩放修正。 |

技术要点：

- `h` 对应论文里的 expansion size。
- `lam` 对应论文里控制矩阵扰动强度和数值范围的系数。
- `check_keymat_inverse()` 用 `key @ inverse` 检查是否接近单位矩阵。
- 代码里的 KeyMat 是保护 hidden state 和组件权重结构的重要基础；没有 KeyMat，token permutation 更容易通过 embedding/head 权重关系被攻击恢复。

## 5. Embedding 与 Model Head 混淆

论文 Section 5.2.2 对 embedding/head 使用三类技术：

1. token permutation；
2. Gaussian noise；
3. key matrix / inverse key matrix。

当前代码对应：

| 文件 | 职责 |
| --- | --- |
| `src/keymat_embed_head.py` | embedding/head 加噪、KeyMat 变换、wrapper。 |
| `src/aloepri/layers/embeddings.py` | `AloePriEmbedding` 和 `AloePriHead`。 |
| `src/obfuscate_embed_head.py` | 早期 token permutation 版 embedding/head 工具。 |
| `src/aloepri/_implementation/standard_shape_obfuscation.py` | 当前 standard-shape checkpoint 路线中的 embedding/head 物化实现。 |

技术要点：

- `alpha_e` 控制 embedding noise 强度。
- `alpha_h` 控制 lm head noise 强度。
- embedding 侧负责让混淆 token 进入混淆 hidden space。
- head 侧负责让混淆 hidden space 的输出 logits 对齐到混淆 token space。
- 如果模型 tie word embeddings，导出时需要小心 untie 或保持一致的权重关系，Llama 导出路径里已经显式处理这一点。

## 6. Attention Obfuscation

论文 Section 5.2.3 针对 attention 设计两层混淆：

1. intra-head transformation：Q/K/V/O 权重内的旋转、缩放、block permutation；
2. inter-head permutation：KV head 和 group head 的置换，兼容 MHA/MQA/GQA。

当前代码对应：

| 文件 | 职责 |
| --- | --- |
| `src/attention_keys.py` | 生成 Q/K 的 RoPE 兼容旋转、缩放、block permutation、KV/group head permutation。 |
| `src/obfuscate_attention_complex.py` | 早期复杂 attention wrapper。 |
| `src/aloepri/_implementation/static_attention_obfuscation.py` | 当前 staticized/fused attention 实现。 |
| `src/aloepri/_implementation/keymat_attention_fusion.py` | KeyMat 与 attention 权重融合。 |
| `src/aloepri/layers/attention.py` | 面向 `AloePriEngine` 的 attention wrapper。 |
| `src/gqa_layout.py` | GQA head/group 布局描述。 |

技术要点：

- `generate_r_qk()` 生成 RoPE 兼容的 2x2 rotary block。
- `generate_h_qk()` 生成 Q/K 成对缩放矩阵。
- `generate_block_perm()` 用 `beta` 和 `gamma` 控制 RoPE block 的局部置换窗口。
- `generate_tau_kv()` 和 `generate_tau_group()` 对应论文里的 KV head permutation 和 group permutation。
- 当前实现重点覆盖 Qwen/Llama 这类 GQA decoder-only 模型；MLA/MoE 的完整论文扩展没有作为主线入口实现。

## 7. FFN Obfuscation

论文 Section 5.2.4 对 FFN 的 `Wgate`、`Wup`、`Wdown` 使用 intermediate dimension 上的 permutation 和 scaling，原因是 FFN 中有 SiLU 和 Hadamard product，不能随意使用破坏非线性结构的线性变换。

当前代码对应：

| 文件 | 职责 |
| --- | --- |
| `src/obfuscate_ffn.py` | `FFNTransform`、FFN permutation、scaling、基础 obfuscation。 |
| `src/keymat_ffn.py` | KeyMat FFN bridge。 |
| `src/aloepri/_implementation/keymat_ffn_fusion.py` | KeyMat 与 FFN 权重融合。 |
| `src/aloepri/layers/ffn.py` | `AloePriMLP` wrapper。 |
| `src/aloepri/_implementation/standard_shape_obfuscation.py` | standard-shape 路线中对 gate/up/down 权重的适配。 |

技术要点：

- `Z_ffn` 对应 intermediate channel permutation。
- `H_ffn` 对应 intermediate channel scaling。
- gate/up/down 必须协同变换，才能保持 `(SiLU(xWgate) * xWup) Wdown` 的结构一致。
- 当前主线覆盖 dense FFN；MoE router/expert permutation 只在论文层面说明，没有作为当前主线的完整实现目标。

## 8. RMSNorm / Normalization

论文 Section 5.2.5 讨论 RMSNorm 在 hidden transform 下的近似修正，用 `kappa` 之类的系数补偿变换带来的范数偏差，并尽量融合到相邻线性层。

当前代码对应：

| 文件 | 职责 |
| --- | --- |
| `src/keymat_norm.py` | 估计 `kappa`，构建 RMSNorm bridge。 |
| `src/obfuscate_rmsnorm.py` | 早期 RMSNorm obfuscation。 |
| `src/aloepri/_implementation/keymat_norm_fusion.py` | KeyMat-fused RMSNorm。 |
| `src/aloepri/layers/norm.py` | `AloePriRMSNorm` wrapper。 |
| `src/aloepri/_implementation/standard_shape_obfuscation.py` | square transform 路线中的 norm weight permutation。 |

技术要点：

- RMSNorm 不是普通线性层，不能简单左右乘矩阵后完全等价。
- 当前代码通过 `estimate_kappa_for_keymat()` 和 fused norm wrapper 降低误差。
- 在 standard-shape 导出路径中，norm 会以保持 Hugging Face 标准图结构的方式被物化。

## 9. Online Private Inference

论文 Section 5.3 中，客户端本地完成 prompt token 混淆和 response 恢复，服务端只执行标准推理。

当前代码对应：

| 文件 | 职责 |
| --- | --- |
| `src/aloepri/private_inference.py` | 加载 release profile，执行私有推理。 |
| `scripts/release/infer_bundle.py` | 命令行入口。 |
| `src/transforms.py` | 输入 token 映射和 logits 恢复。 |
| `src/aloepri/_implementation/checkpoint_io.py` | 加载 server 模型和 client secret。 |

当前在线推理逻辑：

```text
prompt
  -> tokenizer
  -> map_input_ids(current_ids, perm_vocab)
  -> obfuscated model forward
  -> restore_logits(logits_perm, perm_vocab)
  -> choose next plaintext token
  -> append plaintext token locally
  -> repeat
  -> decode plaintext generated ids
```

这里保留了论文的核心客户端/服务端边界：服务端拿到的是混淆 token 和混淆模型，客户端掌握 `perm_vocab` / `inv_perm_vocab`。

## 10. Checkpoint 与 Release Bundle

论文强调兼容现有 LMaaS / vLLM / SGLang 等推理基础设施。代码里对应的是 standard-shape checkpoint 和 release bundle。

| 文件 | 职责 |
| --- | --- |
| `src/aloepri/checkpoints/qwen.py` | Qwen paper checkpoint API。 |
| `src/aloepri/checkpoints/llama.py` | Llama paper checkpoint API。 |
| `src/aloepri/_implementation/qwen_paper_checkpoint.py` | Qwen paper-consistent manifest/evidence 生成。 |
| `src/aloepri/_implementation/llama_paper_checkpoint.py` | Llama paper-consistent checkpoint 导出。 |
| `src/aloepri/release/bundle.py` | release bundle 公开 API。 |
| `src/aloepri/_implementation/release_bundle_impl.py` | server/client/profile/catalog 具体打包。 |
| `src/aloepri/_implementation/checkpoint_io.py` | checkpoint 读写、client secret、manifest 等。 |

技术要点：

- `server/` 保存可部署的混淆模型权重和 tokenizer。
- `client/` 保存客户端恢复所需 secret，例如 `perm_vocab` / `inv_perm_vocab`。
- `catalog.json` 记录 `default`、`reference` 等 profile。
- Qwen 路线偏向从历史 redesign artifact 导出 paper-consistent candidate。
- Llama 路线当前使用 `square_monomial_standard_shape` 作为部署安全、标准图可见的复现路径。

## 11. Correctness Evaluation

论文要求混淆后模型保持推理准确性。当前代码用 correctness regression 验证 release surface 的 logits 和 generation 行为。

| 文件 | 职责 |
| --- | --- |
| `src/evaluator.py` | `max_abs_error`、`mean_abs_error`、top-k overlap、JSON 输出。 |
| `src/aloepri/evaluation/correctness.py` | Qwen/Llama correctness 公开 API。 |
| `src/aloepri/_implementation/qwen_correctness.py` | Qwen release correctness。 |
| `src/aloepri/_implementation/llama_correctness.py` | Llama release correctness。 |
| `scripts/evaluation/run_qwen_correctness.py` | Qwen correctness CLI。 |
| `scripts/evaluation/run_llama_correctness.py` | Llama correctness CLI。 |

主要指标：

- last-token logits max/mean absolute error；
- greedy first token 是否一致；
- generated ids/text 是否一致；
- profile 级 correctness summary。

这些指标对应论文中的 accuracy preservation 目标，但当前项目更偏工程回归验证，不等同于完整 benchmark 复现实验。

## 12. Privacy Attack Evaluation

论文 Appendix D 和实验部分讨论了多种攻击：VMA、IA、ISA、IMA、TFMA、SDA。当前项目主要放在 `src/security_qwen/` 下。

| 论文攻击 | 当前代码 | 说明 |
| --- | --- | --- |
| VMA, Vocabulary-Matching Attack | `src/security_qwen/vma.py` | 从可见权重关系中尝试恢复 token permutation。 |
| IA, Invariant Attack | `src/security_qwen/ia.py` | 构建 invariant attack 模板，支撑 Gate/Attn 类分析。 |
| IMA, Inversion Model Attack | `src/security_qwen/ima.py` | 训练或拟合 inversion 模型，从 obfuscated embedding 恢复 token。 |
| ISA, Internal State Attack | `src/security_qwen/isa.py` | 利用 hidden states / attention observations 做恢复攻击。 |
| TFMA, Token Frequency Mapping Attack | `src/security_qwen/tfma.py` | 利用 token frequency 做映射恢复。 |
| SDA, Substitution Deciphering Attack | `src/security_qwen/sda.py` | 把 token permutation 看作替换密码做 deciphering。 |
| 安全汇总 | `src/security_qwen/summary.py`, `matrix.py`, `schema.py` | 统一 payload、schema、matrix 和 summary。 |

技术要点：

- VMA 对应论文里 `Y = Z1 X Z2` 类结构恢复攻击。
- IMA 对应“知道混淆机制后训练反演模型”的威胁。
- ISA 对应“观测中间 hidden state/attention state 后优化输入恢复”的威胁。
- TFMA/SDA 对应 deterministic token permutation 的频率泄漏风险。
- 当前 security 包名仍是 `security_qwen`，说明安全评测主线首先围绕 Qwen；其中部分攻击思想可以迁移到 Llama。

## 13. Qwen 与 Llama 复现路线差异

当前仓库同时支持 Qwen 和 Llama，但两条线的成熟度和实现方式不同。

| 模型线 | 当前状态 | 关键文件 |
| --- | --- | --- |
| Qwen | 更完整，包含历史 redesign、paper-consistent candidate、release correctness、privacy attack 评测。 | `src/aloepri/_implementation/qwen_paper_checkpoint.py`, `src/security_qwen/*` |
| Llama-3.2-3B-Instruct | 已有 paper checkpoint、release bundle、correctness 路线；隐私攻击评测尚未像 Qwen 一样系统化。 | `src/aloepri/_implementation/llama_paper_checkpoint.py`, `src/aloepri/_implementation/llama_correctness.py` |

需要特别说明的是：Llama 当前导出路径使用 `square_monomial_standard_shape`，它是为了标准 checkpoint 形状和部署兼容做的工程实现。它保留了论文的 token permutation、embedding/head noise、hidden-space transform、standard graph compatibility 等关键思想，但不是论文所有 attention/FFN/MLA/MoE 扩展机制的完整覆盖。

## 14. 当前复现完成度

| 论文技术点 | 当前完成度 | 说明 |
| --- | --- | --- |
| token-level permutation | 已实现 | `key_manager.py`, `transforms.py`, `token_ops.py`。 |
| embedding/head noise | 已实现 | `keymat_embed_head.py`, `standard_shape_obfuscation.py`。 |
| KeyMat generation | 已实现 | `keymat.py`。 |
| hidden-space transform / inverse | 已实现 | KeyMat 路线和 square transform 路线均存在。 |
| attention block/head obfuscation | 部分实现 | Qwen/GQA 路线有 `attention_keys.py` 和 staticized attention；MLA 未作为主线覆盖。 |
| FFN permutation/scaling | 已实现 | dense FFN 路线已实现；MoE router/expert 未作为主线覆盖。 |
| RMSNorm correction | 已实现/近似实现 | 通过 `kappa`、fused norm、square norm permutation 处理。 |
| release bundle | 已实现 | server/client/profile/catalog。 |
| online private inference | 已实现 | 当前是 greedy generation 流程。 |
| correctness evaluation | 已实现 | Qwen/Llama 均有 release correctness。 |
| privacy attacks | Qwen 较完整 | VMA/IMA/ISA/TFMA/SDA 等主要在 `security_qwen`。 |
| full industrial benchmark | 未完整复现 | 论文的大规模 benchmark、Deepseek-V3.1-Terminus、完整多数据集实验不是当前仓库主线。 |

## 15. 推荐阅读代码顺序

如果要向别人讲清楚这个项目，建议按下面顺序读代码：

1. `README.md`：先看项目入口和运行命令。
2. `docs/ARCHITECTURE.md`：看干净架构分层。
3. `src/aloepri/config.py`：理解论文超参数如何进入代码。
4. `src/aloepri/adapters/qwen.py` 和 `src/aloepri/adapters/llama.py`：理解模型结构如何统一。
5. `src/key_manager.py`、`src/transforms.py`：理解 token permutation。
6. `src/keymat.py`：理解 KeyMat。
7. `src/aloepri/layers/*`：理解 embedding/attention/FFN/norm/head 怎么被包装。
8. `src/aloepri/_implementation/standard_shape_obfuscation.py`：理解当前可导出 checkpoint 的标准图路线。
9. `src/aloepri/_implementation/llama_paper_checkpoint.py` 或 `qwen_paper_checkpoint.py`：理解导出产物。
10. `src/aloepri/private_inference.py`：理解在线推理。
11. `src/security_qwen/*`：理解隐私攻击评测。

## 16. 一句话讲解版

这个项目复现的是 AloePri 的核心路线：客户端离线生成秘密 token permutation 和 hidden-space 变换，把原始 LLM 权重改造成混淆 checkpoint；上线时客户端只发送混淆 token，服务端用普通推理框架跑混淆模型，返回混淆输出，客户端再用 secret 恢复明文结果。代码中的 `key_manager/transforms` 对应输入输出 token 混淆，`keymat/attention/ffn/norm` 对应模型权重协变变换，`checkpoints/release/private_inference` 对应论文部署链路，`evaluation/security_qwen` 对应正确性与隐私攻击验证。
