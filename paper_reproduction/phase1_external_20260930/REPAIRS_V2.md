# 三项复现代码修复记录（V2）

日期：2026-09-30。优先修复作者实现，保留主要逻辑、功能和分类结构。三项均通过合成数据技术检查；未接入 CVS 冻结物理样本划分，未进行真实 WiSig 性能复现。

| 方法 | 保留内容 | 修复内容 |
|---|---|---|
| ASKNet | 原模型、CE、Adam、调度器、source validation loss 选模、早停 | 无效 IQ 归一化；可配置输入/输出/日志；只测试时不加载训练数据；安全 state_dict 读回；实际配置和详细日志 |
| WaveMLP | 原模型及 wavelet 结构、分组学习率、CE、Adam、source validation loss 选模、早停 | 调度器参数兼容性、padding 缺少 F 导入、无效 IQ 归一化；同样补齐目录配置和日志 |
| 域不变特征学习（内部简称 DIFL） | 原分类层与参数形状、教师 FFT 相位/NLL/Adam、学生 CE/KD/CORAL 公式、source validation accuracy 选模、固定顺序域均衡采样 | 空 CORAL 分支、训练未调用、硬编码路径/域数/批量、采样器长度、统计、IQ/标签错配、单样本测试、教师设备与 checkpoint、可视化 |

## DIFL 最小修复

原 `fc2` 特征只有 256 维，训练却要求前 256 维蒸馏、后 256 维 CORAL。修复将返回特征的位置前移至**已有 fc1 的 ReLU 输出（512 维）**。原 fc2 和分类头继续按原顺序计算，未增加、扩宽或删除层。相同权重与输入下，分类 logits 逐值相同，state_dict 参数形状全部一致。

保留原损失 `(CE + lam * KD + beta * CORAL) / (1 + lam + beta)`。KD 对接原教师 256 维输出，CORAL 使用后 256 维，按实际 receiver ID 分组并保留原域对平均公式。教师只加载一次并冻结，参数及 BN 状态在学生训练中不变。特征取出位置是明确记录的修复，尚未核实论文精确位置，不声称已完全复现论文。

采样器保留原固定顺序、域均衡和丢弃尾部行为；batch 数取最短域的完整 batch 数。损失和准确率按实际样本统计。IQ 与标签使用同一行掩码选择，保留从 1 开始的标签转为从 0 开始。原 `run_DIFEX.sh` 保留为历史网格示例，它未提供新增的必需路径参数；当前 Windows 使用下述 Python 入口，不直接运行旧脚本。

## 日志与 checkpoint

三个方法均输出 `resolved_config.json`、详细 `training.log`、`steps.jsonl`、紧凑 `epochs.jsonl` 和 `epochs.csv`，记录实际损失分量/权重、学习率、梯度、方法状态、source validation 和耗时。未测量项不虚构。

checkpoint 保存为 state_dict，默认 `weights_only=True` 读回。DIFL 显式 `--allow-legacy-checkpoint` 可兼容可信旧完整模型；技术兼容不证明数据来源合规。此次未加载作者权重，正式使用前仍须检查初始化、教师和继承状态的训练数据契约。

## 运行入口

下面路径大写部分是待替换参数，未执行真实数据训练。切换到相应工作目录，使用 ssr-gpu Python。

ASKNet 目录：`E:/type10-7/code/external_baselines/phase1_20260930/asknet/WiSig/`。

```text
python -X utf8 main.py --code_state only_train --dataset_name ManySig --data-root DATA --output-root NEW_OUTPUT --log-root NEW_LOGS
```

WaveMLP 目录：`E:/type10-7/code/external_baselines/phase1_20260930/wavemlp/`。

```text
python -X utf8 main.py --code_state only_train --dataset_name ManySig --data-root DATA --output-root NEW_OUTPUT --log-root NEW_LOGS
```

WaveMLP 的 ManyRx 配置需显式 `--wavelet_levels 5 --wavelet_kernel_size 32`；ManySig 默认保持 4 和 16。两者测试使用 `--code_state only_test`，训练后测试用 `train_test`。原 receiver/date 和模型超参数继续可配置。

DIFL 目录：`E:/type10-7/code/external_baselines/phase1_20260930/difl/`。

```text
python -X utf8 pretrain.py --source-files RX1.npy RX2.npy --label-path LABELS.npy --output-dir NEW_TEACHER_OUTPUT --log-dir NEW_TEACHER_LOGS --device cpu
python -X utf8 train_DIFEX1.py --source-files RX1.npy RX2.npy --label-path LABELS.npy --teacher-checkpoint NEW_TEACHER_OUTPUT/pretrain.pth --output-dir NEW_STUDENT_OUTPUT --log-dir NEW_STUDENT_LOGS --code-state only_train --device cpu --lam 0.05 --beta 0.01
```

`--batch-size` 是每个 receiver 的批量，CORAL 要求每域至少 2 个样本且至少 2 个域。测试时显式提供 `--test-data`、`--test-labels`；`--plot` 保存 t-SNE PDF。原作者全监督划分不等于 CVS `L/U/V=0.07/0.63/0.30`；上述默认入口不能直接作为匹配协议对比结果。

## 验证与保留证据

环境：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`，PyTorch `2.10.0+cu128`，CPU。输入与 checkpoint 全为本次生成的合成 fixture，没有访问真实 IQ、目标真值或作者权重。

- ASKNet、WaveMLP：有限 forward/backward、参数更新、eval 状态不变、原生单轮训练及 checkpoint 读回通过；实际 train_test CLI、目录配置和完整日志通过；12/32 receiver 四折及合成 source train/validation 分离检查通过。
- DIFL：分类输出逐值相同、参数形状不变、损失和 CORAL 平均公式一致、KD/CORAL 有效、教师冻结、不同域长度、单样本边界、交错标签配对和 checkpoint 兼容检查通过。
- DIFL 实际教师和学生 CLI 各训练 2 epochs，保存/读回、only_test CLI、预测文件和 t-SNE PDF 通过。
- 所有工作源码 AST 解析通过。机器证据见 [repair_results_v2.json](repair_results_v2.json)；原始日志与合成产物位于 `E:/type10-7/local_artifacts/phase1_external_20260930/repair_smoke_v3/`。

不可变作者克隆位于 `E:/type10-7/local_artifacts/phase1_external_20260930/upstream/`，受版本管理的原文件未修改。V1 工作副本保留在同级 `working_v1/`。[AUDIT_V1.md](AUDIT_V1.md) 与 [audit_results.json](audit_results.json) 保留历史缺陷证据，其中 BLOCKED 属于旧状态。

下一步是固定数据划分适配与正式实验预登记。此次未启动 N607、真实 WiSig 训练或正式评分；论文准确率与真实数据计算开销未测量。
