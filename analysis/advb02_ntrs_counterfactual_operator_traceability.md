# ADVB02 NTRS-V4指导可追溯表

| 指导要求 | 实现位置 | 验证 | 状态 |
|---|---|---|---|
| 保留成熟D1并冻结raw backbone/head | `model_dual_cvsincnet.py`、`ntrs_training.py` | 参数冻结、raw漂移与checkpoint加载测试 | 代码完成，真实smoke待发布 |
| 分组归一化IQ描述符q | `ntrs.py`、`train_ssdg.py` | 从精确V_cal一次拟合median/IQR并写入buffer | 完成 |
| metadata教师监督q，推理不依赖metadata | `ntrs.py`、`ntrs_training.py` | 训练教师与IQ-only eval测试 | 完成 |
| constant/shuffled/random-feature对照 | launcher/model | 8个profile命令快照测试 | 完成 |
| q条件低秩算子 | `ntrs.py` | z依赖、零初始化、梯度隔离测试 | 完成 |
| paired shift和pair cosine | `ntrs_training.py` | 精确数值与梯度测试 | 完成 |
| harm优先、rescue分层目标 | `ntrs_training.py` | protect/rescue掩码与权重约束测试 | 完成 |
| clean尾部约束或exact bypass | `ntrs_training.py`、launcher | relative correction hinge测试 | 完成 |
| B0 PCA/连续oracle/交互诊断 | `export_phase1_source_leo_pair_features.py`、`ntrs_b0_diagnostics.py`、`run_ntrs_b0_from_pair_exports.py` | 精确V_cal、物理ID/checkpoint核对、rank/oracle/TX×场景交互测试 | 完成 |
| 仅LEO_WEAK训练和测试 | launcher、协议负测 | 命令快照无`mixed_orbit`且包含三场景 | 完成 |
| seed392034矩阵 | launcher、运行报告 | 命令快照与N607 readback | 本地完成，远端发布待执行 |
| clean及三种LEO_WEAK最终测试 | launcher/report | 独立评测命令和artifact闭合 | 命令完成，结果待E200 |

本地最终验证：Python语法编译通过，NTRS模型、训练损失、协议负测、launcher、B0诊断与独立评测共77项测试通过。

## 非采纳项

- 外部分析中的正式`mixed_orbit`轨：与当前`项目.md`冲突，按现行Phase1默认LEO_WEAK执行。
- B4 learned gate：B3-R未满足正净救回前不发布。
- B5-RX：属于Phase2 receiver calibration，不扩入本次Phase1矩阵。
