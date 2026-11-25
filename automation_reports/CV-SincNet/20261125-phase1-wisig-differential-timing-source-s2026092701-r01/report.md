# WiSig 短延迟复乘积时标的双路线冻结对照

固定 WiSig Main split Tri-SIG-TIME SSL seed 2026092701、原冻结 DURA/R69、同一 1,800 个源 V 包和两路线 clean 与六景。比较直通、原平滑前导时标、1／2／4 点归一化复乘积时标、两组分数等权组合和原冻结 R69。三组工程读出共享名义窗候选、粗频偏、可见范围、响应外推以及仅补齐缺失边界的规则。有效重叠的实收 IQ 逐点复制，短延迟相位差只参与选位。 [源 U 时标证据](../../../docs/analysis/WISIG_HELD_PREAMBLE_20261124.md) · [接收处理](../../../experiments/pamf_unrolled_restore/wisig_differential_timing.py)

三个触发门分别仅由同一批 4,096 个盲源 U 包上取最大余量并加固定边际。F、TX、V、目标或仿真轨迹不参与拟合或定门。候选零新增训练参数及零优化步。真实权重无目标检查在冻结门后执行。源预测与 R69 对照均使用同一个冻结识别器，两路 14 个矩阵闭合后独立 CPU 才读源真值。记录 clean、低城区、六景、各视图和源接收机正确数，及同物理包的纠错改错、净变化和 95% 配对区间。目标矩阵及任何后续候选训练不在此源工程对照中。

N607 普通账号，命令 `python -m experiments.pamf_unrolled_restore.wisig_differential_timing_source publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config>`。发布与输出根为对应运行名的 `-release` 和 `runs` 子目录，提交由回读发布记录固定。计算优先使用实测较空闲 GPU。数据契约、门限、真实权重检查、预测类型和矩阵闭合出错时停止本链并保留结果。预期 `threshold.json`、`smoke.json`、两路线预测及完整性记录和独立 `source-scores.json`。
