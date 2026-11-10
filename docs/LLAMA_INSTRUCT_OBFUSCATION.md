# Llama-3.2-3B-Instruct 混淆模型生成说明

本文档只面向独立的新项目目录 `paper_aligned_project/`。

## 1. 前置条件

把项目移动到任意新位置后，目录内至少需要有：

```text
paper_aligned_project/
  src/
  scripts/
  docs/
  environment.qwen-transformers.yml
  model/Llama-3.2-3B-Instruct/
```

`model/Llama-3.2-3B-Instruct/` 应该是 Hugging Face 格式模型目录，至少包含：

- `config.json`
- tokenizer 文件
- 模型权重文件
- generation config 或可由 config 生成的配置

环境建议使用：

```bash
conda env create -f environment.qwen-transformers.yml
conda activate qwen-transformers
```

如果环境已存在：

```bash
conda activate qwen-transformers
```

## 2. 生成 Llama-Instruct 论文对齐混淆 checkpoint

在 `paper_aligned_project/` 根目录执行：

```bash
python scripts/offline_obfuscation/export_llama_checkpoint.py \
  --model-dir model/Llama-3.2-3B-Instruct \
  --export-dir artifacts/llama_3_2_3b_instruct/paper_consistent_checkpoint \
  --device cpu \
  --dtype bfloat16 \
  --alpha-e 0.02 \
  --alpha-h 0.01
```

输出目录：

```text
artifacts/llama_3_2_3b_instruct/paper_consistent_checkpoint/
  server/
  client/
  manifest.json
  paper_consistent_target.json
  stage_i_metadata.json
```

其中：

- `server/` 给服务端加载；
- `client/client_secret.pt` 给客户端做 token 映射和输出恢复；
- `manifest.json` 记录 checkpoint 契约；
- `stage_i_metadata.json` 是当前导出格式沿用的 metadata 文件名。

当前默认噪声工作点为：

```text
alpha_e = 0.02
alpha_h = 0.01
```

这是当前 Llama-Instruct 路线的可部署工作点，不等同于论文默认强度参数。

## 3. 打包 release bundle

```bash
python scripts/release/export_llama_bundle.py \
  --source-dir artifacts/llama_3_2_3b_instruct/paper_consistent_checkpoint \
  --export-dir artifacts/llama_3_2_3b_instruct/release_bundle \
  --materialize
```

输出目录：

```text
artifacts/llama_3_2_3b_instruct/release_bundle/
  catalog.json
  deployment_contract.json
  README.md
  profiles/
    default/
    reference/
```

`default` 和 `reference` 当前指向同一个论文对齐 checkpoint，但承担不同入口语义：

- `default`: 默认部署入口；
- `reference`: 审计和复核入口。

## 4. 运行混淆推理

```bash
python scripts/release/infer_bundle.py \
  --bundle-dir artifacts/llama_3_2_3b_instruct/release_bundle \
  --profile default \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 16 \
  --dtype bfloat16
```

这条命令执行的是：

1. 加载 release bundle；
2. 读取 profile；
3. 加载 server 混淆模型；
4. 读取 client secret；
5. 把 prompt token 映射到混淆 token 空间；
6. 用混淆模型推理；
7. 恢复输出 logits；
8. decode 明文 token。

## 5. correctness 复核

如果有 GPU，推荐：

```bash
python scripts/evaluation/run_llama_correctness.py \
  --bundle-dir artifacts/llama_3_2_3b_instruct/release_bundle \
  --baseline-model-dir model/Llama-3.2-3B-Instruct \
  --output-dir outputs/evaluation/llama_3_2_3b_instruct/correctness \
  --profiles default reference \
  --device cuda \
  --dtype bfloat16
```

如果只能 CPU 验证，可改成：

```bash
python scripts/evaluation/run_llama_correctness.py \
  --bundle-dir artifacts/llama_3_2_3b_instruct/release_bundle \
  --baseline-model-dir model/Llama-3.2-3B-Instruct \
  --output-dir outputs/evaluation/llama_3_2_3b_instruct/correctness \
  --profiles default reference \
  --device cpu \
  --dtype float32
```

输出：

```text
outputs/evaluation/llama_3_2_3b_instruct/correctness/
  default.json
  reference.json
../correctness_summary.json
```

## 6. 独立性确认

新项目不依赖原仓库目录。需要一起移动的是：

```text
paper_aligned_project/
```

以及你自己的模型目录：

```text
paper_aligned_project/model/Llama-3.2-3B-Instruct/
```

项目内部已经不需要：

- `legacy_scripts/`
- `historical_implementation/`
- 原仓库 `src/`
- 原仓库 `scripts/`
- 原仓库 `docs/`

## 7. 当前边界

当前 Llama-Instruct 路线是标准形状可部署混淆路线：

- 保持标准 Hugging Face checkpoint 形状；
- 使用 token permutation、embedding/head 噪声、标准 hidden-space square transform；
- 能用 release bundle 方式做私有推理。

它还不是论文全部复杂机制的完全同态版本。尤其是：

- 复杂 attention 扰动没有达到 Qwen 当前安全评测强度；
- VMA / IMA / ISA 尚未补齐到 Qwen 同等级闭环；
- 默认参数不是论文推荐的 `alpha_e = 1.0`, `alpha_h = 0.2`。
