# 攻击实现验证（2026-10-28）

> 历史记录。机器路径、当时的测试结论和运行参数仅用于追溯；当前入口见 [文档导航](../README.md)。

这次检验 SDA、IMA、Gate-IA、VMA、TFMA，并补查 Attn-IA 解释版本与 ISA。使用本地 Qwen2.5-0.5B-Instruct 的真实权重和 tokenizer；ISA 还实际加载完整24层模型前向。Docker 可执行程序仍在，但 Desktop Linux Engine 未运行，因此采用 Transformers 在本机执行，没有通过聊天接口假装完成权重攻击。

## 运行

从 `D:\Aloepri` 运行，Python 环境采用 `requirements-tested.txt`；自然文本读取还需要 pyarrow（本机已经安装）。先准备完整本地模型，以及第三次修改中的同宽与扩维零噪声产物。输入数据还使用 `outputs/round3_pretrained/sst2_validation.parquet`。

下面的 `--out` 应选择新的实验目录，避免与已保存的检查点或CLI结果冲突。各阶段分进程执行，降低内存占用：

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python scripts/evaluation/verify_attack_implementations.py prepare --out outputs/attack_check_new
python scripts/evaluation/verify_attack_implementations.py weights --out outputs/attack_check_new
python scripts/evaluation/verify_attack_implementations.py sequences --device cuda --out outputs/attack_check_new
python scripts/evaluation/verify_attack_implementations.py ima --device cuda --ima-standardization none --out outputs/attack_check_new
python scripts/evaluation/verify_attack_implementations.py ima --device cuda --ima-standardization train_only --out outputs/attack_check_new
python scripts/evaluation/verify_attack_implementations.py isa --device cuda --out outputs/attack_check_new
python -m pytest -q -ra
```

无GPU时可使用 `--device cpu`，训练和完整模型迭代会更慢。已有本次结果在 `outputs/attack_verification_20261028`；具体恢复数和限制以桌面实验报告及该目录的JSON为准。

## 对照怎样建立

- 权重攻击：公开种子先选128个真实词条，使用第0、11、23层的完整隐藏宽度和前馈宽度。已知变换包括算法1扩维、专家之外的普通FFN通道置换、GQA头置换、Q/K二维可逆变换。该正对照不加噪声或FFN非均匀缩放；预测函数不接收变换密钥。输入目录里的权重文件只是攻击样本，README已标明不是可部署的完整模型。
- 同宽/扩维产物：使用已经正式导出的模型文件，取上述固定128词条，分别运行攻击。为构造有答案的封闭对照，准备者使用客户端置换选取相应观测行；预测程序仍只收到两边权重。不能把这说成未知全词表攻击成功率，Attn-IA投影也只对应这个受限行集合。
- SDA：六种符号的规则文本用160/32/32条互不重复序列训练、验证、测试，检查重命名不影响预测、映射一致和实际恢复。自然文本另用SST-2不相交的128/32/32条，Qwen真实分词、最多12个token、完整151643词表，训练3轮。
- TFMA：先用已知频率排名检查匹配、同频标记及未知项，再对上面的自然文本做独立公开/观测统计，分别报告按种类和按出现次数的恢复率。
- IMA：使用原项目的两层、八头反演网络、维度适配和完整词表候选匹配。32个训练词条组成96条训练、8条验证、16条测试序列，每条8个token且序列互不重复；主测试仍是见过的词条组成新序列，另测32个未见词条。用已知目标密钥生成训练配对，不能称为不需要配对的攻击。仅为节省内存，把网络未使用的token查表缩为64行；实际候选仍为完整151643词表。
- ISA：完整预训练模型、2条各4个token的公开文本、零输入初值，比较第0/1层观测；另检查未知等比例正交扩维的Gram模式。模型参数运行前后逐张量SHA256必须一致。

## 本次修正的IMA训练处理

原始数值流程即使没有混淆，在本次20轮对照里也只恢复9/128个位置；扩维组为0/128，单凭损失下降不能验收。

现在 `run_ima_paper_like` 默认只用训练配对计算输入与目标的逐通道均值、标准差，并在匹配公开词表前把预测还原到原Embedding尺度。验证、测试不参与统计。命令行 `--disable-standardization` 可重跑旧方式。统计还原、零方差通道、拒绝非法输入和CLI开关都有专门检查；验证脚本还保存统计并重新加载模型核对输出。

这个处理是为了改善训练数值尺度的实现选择，不是论文已经明确给出的固定细节。它没有改变已知配对这一攻击条件。两组新训练与旧训练使用同一公开样本、网络规模、种子及20轮设置；测试答案只在保存预测后评分，错配训练标签作为反向检查。

## 如何读结果

受控恢复通过表示当前实现可以在所列条件下计算、训练、保存加载并恢复对应关系，不能证明所有输入、所有模型都正确，也不能等同于完整论文攻击复现。特别注意自然文本上的SDA结果、IMA未见词条结果和ISA内部层结果：它们与容易核对的正对照分开报告，不能合并成一个“攻击成功率”。

本次没有改C、D采样尺度、RoPE顺序或模型的混淆参数；实际发现并调整的是IMA训练数值处理。
