# SatIQ 原版 160 原型权重 0.25 首轮配对

[配置](experiment.json) · [事件](events.jsonl) · [原始权重 0.50](../20261119-phase1-satiq-ccs2023-native160-sup100-s2026092701-r01/report.md) · [梯度复核](../20261120-diagnostic-satiq-native160-aux-gradient-s2026112007-r01/report.md)

原型项是第 10 轮编码器端范数最大的单一辅助项，与主 CE 的平均余弦为 0.0721。其加权梯度在 17/64 个真实训练批次上投影为负。本轮把等权五分支留一原型 CE 的系数从 0.50 降为 0.25。保持原始 160 维、五分支、原生 Sinc、8,000 点同步字段和合成增强关闭。其余损失为主 CE、0.20 同星 CE、0.15 分支 CE 和 0.05 分支 KL。

与原版用相同数据划分、标签、模型与采样种子，从零训练一个完整实际 epoch，28,225 步。预定 5,000、20,000 和 28,225 步对比训练主 CE、验证主 CE、全类 Top-1、K16 关联及同星验证 AUC。固定步数和数据角色不受结果影响。该试验独立于四组 200 轮正式对照。

N607 GPU0、PID 2180270。2026-11-20T01:03:29+08:00 核对真实 CWD、启动命令、GPU UUID 和生效配置，已经完成 300 步。**VERIFIED 已启动**，验证结果待出。独立测试留出。

- 生效配置和逐步日志：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-native160-proto25-source-e1-s2026092701-r01/PAMF_SUP/resolved_config.json`、`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-native160-proto25-source-e1-s2026092701-r01/PAMF_SUP/steps.jsonl`
- 过程日志：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-native160-proto25-source-e1-s2026092701-r01/train.log`

## 第 5,000 步同预算结果

| 原型权重 | 源 V 主 CE | 源 V 全类 Top-1 | 源 V K16 Top-1 | 源 V 同星 K16 AUC |
|---:|---:|---:|---:|---:|
| 0.50 | 8.0584 | 0.0803% | 0.1486% | 0.52945 |
| 0.25 | 8.0448 | 0.0662% | 0.1408% | 0.51152 |

到 5,000 步，减权的主 CE 略低，两个 Top-1 和同星 AUC 均略降。该检查点位于首轮内，继续既定 20,000 步和首轮读出。
