# PAMF八种子原源域验证集发射机召回率诊断

- run_id：`20261019-phase1-pamf-source-class-manysig-m8-r02`
- group_id：`phase1-pamf-frozen-source-class`；类别：`diagnostic`；阶段：`Phase1 source diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen historical residual_fusion E200 checkpoints, original default head and original source V; CPU-only no adaptation/training/target access.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## Fixed diagnostic plan and technical recovery

Original source V only: 27000 packets per model, 4500 per class; all8 fixed E200 default-head checkpoints, CPU4 threads, no training/optimizer/target access. Stop on first input, smoke, coverage, nonfinite, state-mutation or historical accuracy reproduction mismatch; preserve partial artifacts; no performance-based rerun.

r01 failed before validation data access: historical model.build accepts variant only. Remote historical API read verified; code now uses the exact one-argument call. Six metadata protocol-negative checks and compile pass. Independent P0/P1 review plus one targeted signature-fix review PASS. r01 and its evidence remain unchanged. r02 is a new output root.

## 完整源域逐类结果（2026-10-19）

VERIFIED。8/8 个历史 residual_fusion E200 默认分类头冻结推理完成，全部 CPU 执行。每个 seed 使用原 source V 的 27000 包，每类4500包；无新训练、换头、适配或目标访问。完整预测CSV、逐seed 6×6混淆矩阵与结果JSON保留在本run；独立读回并从8份CSV重新计数核验。全部seed重现历史末轮源V准确率；模型参数/buffer保持不变。

| 发射机 | 源 V 召回率（%） | 历史 clean 测试召回率（%） | 差距（百分点） |
|---|---:|---:|---:|
| 14-7 | 97.4611 ± 0.5020 | 48.9469 ± 5.6601 | 48.5142 |
| 20-19 | 95.7278 ± 0.2485 | 66.5219 ± 1.8006 | 29.2059 |

均值及样本标准差（ddof=1）覆盖全部8个model seed，原split seed392005不变。单类准确率即召回率，不是precision。历史target仅引用已有评分，不重新预测/评分、不作参数或候选反馈。

源V相互混淆：14-7->20-19=1.4222%, 20-19->14-7=2.4500%。源域这两类可区分，未见RX测试显著恶化支持跨接收机泛化落差；本诊断不能单独确定具体物理或网络机制的因果作用。

实际执行代码提交：7d05cac80cd96fbce72ee88ccca0b3148f84c2aa。r01接口失败保留，本次r02完成。结果与读回：local_artifacts/pamf_source_class_20261019_r02/results.json、readback.json、verification.json。

| model_seed | 源V TX14-7（%） | 源V TX20-19（%） | 源V overall（%） |
|---|---:|---:|---:|
| 2026092701 | 97.6444 | 95.5333 | 98.0852 |
| 2026092702 | 97.1556 | 96.2222 | 98.1889 |
| 2026092703 | 96.7778 | 95.6444 | 97.9481 |
| 2026092704 | 97.0444 | 95.8222 | 98.0074 |
| 2026092705 | 97.9111 | 95.4222 | 98.1000 |
| 2026092706 | 98.0222 | 95.6667 | 98.1556 |
| 2026092707 | 97.0889 | 95.8889 | 98.0667 |
| 2026092708 | 98.0444 | 95.6222 | 98.1519 |
