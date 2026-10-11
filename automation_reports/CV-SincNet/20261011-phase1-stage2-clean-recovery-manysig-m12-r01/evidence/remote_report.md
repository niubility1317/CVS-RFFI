# 第二阶段 clean 测试技术恢复

12 行身份训练均完成 E40/8,880 更新，原自动测试在首个批次调用 `torch.from_numpy` 时失败，未生成 prediction 或连接 truth。N607 使用 PyTorch 2.1.0+cu121 / NumPy 2.2.5，已复现相同 TypeError；替换为拥有独立可写存储的 float32 缓冲区复制，与原始数值完全相同。

本次仅恢复固定矩阵的 clean 测试。原权重、训练配置、样本、seed、预测规则、指标均不变；旧失败目录和日志保留。新恢复 run 单独保存调度记录与测试日志，最终 prediction/score 写回原 row 的既定位置供完整矩阵汇总。每卡总进程不超过 4，逐行先固定 prediction，再由独立 CPU scorer 连接 truth。

本地检查通过：1/64/256 批次完全数值一致、禁用旧 NumPy 桥时仍通过、错误 dtype 拒绝、复制后不修改输入。远端只读复现与缓冲区核验通过。恢复发布状态待独立读回。


# Fixed-parent 40-epoch multi-disentanglement: clean test

Each row uses 168000 fixed physical queries across 7 receivers and 4 days. Predictions precede independent truth access.
P0 predictions are reused when registered. Results do not feed model, hyperparameter, or rerun selection.

| Arm | Seeds | Mean accuracy (%) | SD (pp) | Minimum (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|---:|
| P0 | 3 | 80.526 | 1.320 | 79.052 | 79.950 |
| C0 | 3 | 80.875 | 1.536 | 79.101 | 80.362 |
| C1 | 3 | 80.756 | 1.313 | 79.246 | 80.281 |
| C2 | 3 | 79.633 | 0.341 | 79.324 | 79.277 |
| C3 | 3 | 79.670 | 0.353 | 79.349 | 79.321 |

## Every seed

| Arm | Model seed | Accuracy (%) |
|---|---:|---:|
| C0 | 2026092701 | 81.766 |
| C1 | 2026092701 | 81.393 |
| C2 | 2026092701 | 79.576 |
| C3 | 2026092701 | 79.612 |
| C0 | 2026092702 | 79.101 |
| C1 | 2026092702 | 79.246 |
| C2 | 2026092702 | 79.324 |
| C3 | 2026092702 | 79.349 |
| C0 | 2026092703 | 81.757 |
| C1 | 2026092703 | 81.628 |
| C2 | 2026092703 | 79.998 |
| C3 | 2026092703 | 80.048 |
| P0 | 2026092701 | 80.928 |
| P0 | 2026092702 | 79.052 |
| P0 | 2026092703 | 81.599 |

## Paired differences

| Comparison | Seeds | Mean (pp) | Positive seeds |
|---|---:|---:|---:|
| C0 − P0 | 3 | 0.348 | 3 |
| C1 − C0 | 3 | -0.119 | 1 |
| C2 − C1 | 3 | -1.123 | 1 |
| C3 − C2 | 3 | 0.037 | 3 |
| C3 − C0 | 3 | -1.205 | 1 |
| C3 − P0 | 3 | -0.857 | 1 |

Primary success requires all three C3 seeds and clean mean ≥84%; no best-arm replacement.
C0−P0 includes the changed optimizer, schedule, sampling, and additional training. C3−C0 estimates the fixed mechanism increment.
Three seed differences are descriptive evidence, not an automatic statistical significance claim.
Parent training, shared preparation, incremental training, and test costs are referenced separately in resources.json.
