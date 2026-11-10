# 多模型混淆实验队列

入口为 `scripts/operations/model_queue.py`。先扫描本地模型配置，生成计划，再按模型逐个导出和检查。一个模型失败后继续处理后面的模型；重跑保留旧 attempt，并跳过已完成的任务。运行过程中使用排他锁，避免两份队列同时写同一实验目录。

```bash
python scripts/operations/model_queue.py plan \
  --models-root /path/to/models --run-dir /path/to/experiments/run_001 --threads 8
python scripts/operations/model_queue.py run --run-dir /path/to/experiments/run_001
python scripts/operations/model_queue.py status --run-dir /path/to/experiments/run_001
```

计划目录必须为空，且不能与输入模型目录相互包含。配置与权重文件的大小、修改时间会登记并在运行前后核对。发现源文件变化、量化配置或不支持的架构时记录原因，不假装完成。

当前这条队列采用已验证的同宽方阵工程路线：

| 项目 | 设置 |
| --- | --- |
| 架构 | Llama、Qwen2、BERT |
| 噪声 | Embedding / head 均为 0 |
| 解码模型导出 | CPU、BF16、全部层 |
| BERT 导出 | CPU、FP32、全部层、需要自定义 LayerNorm bridge |
| 解码模型检查 | 原模型与产物分进程运行；CUDA FP32，5 条提示，每条 8 个生成编号，容差 2e-4 |
| BERT 检查 | 现有 BERT 隐藏状态、CLS 和 pooler 对照，FP32，容差 1e-4 |
| 随机源 | 解码模型使用现有系统随机源；不传可复现实验种子 |

`checkpoint/catalog.json` 只是对本次 server/client 目录的本地视图，不额外复制模型文件。BERT 的 `requires_custom_runtime` 为 true，需要现有 BERT 编码与恢复接口；其文件不能当作普通无需 bridge 的推理模型。

每次 attempt 下保存：

```text
export.log
verify.log
checkpoint/server/                 # 模型文件
checkpoint/client/client_secret.pt # 客户端恢复资料
checkpoint/manifest.json
correctness/default.json
```

`complete` 表示导出和本次小样本数值检查均通过；`exported_verification_failed` 表示产物已经生成，但检查未通过，原因及测量值保留在日志和报告中。它们都不代表完整任务质量或隐私实验结论。

2026-11-03 的 nss-m 部署：

- 代码：`/home/nss-marker/lxl/Aloepri_20261103_models`
- 输入：`/home/nss-marker/dcy/code/ModelSplit/models`
- 队列：`experiments/all_models_20261103/queue.json`
- 日志：`experiments/all_models_20261103/queue.log`
- 汇总：`experiments/all_models_20261103/RESULTS.md`
- 环境检查：`runtime_probe.json`、`preflight_status.json`、`preflight_tests.log`

队列由该目录内的独立 Python 环境执行，后台 supervisor 等环境安装及预检通过后自动启动任务。普通运行失败可直接重新执行 `run`；若进程被强制终止留下锁，应先确认记录的 PID 已退出，再处理该实验自己的锁文件。
