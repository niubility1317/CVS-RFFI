# SatIQ 四组 200 轮独立预测与真值后置评分

[登记](experiment.json) · [事件](events.jsonl) · [200 轮训练](../20261120-phase1-satiq-ccs2023-sup200-s2026092701-r01/report.md) · [测试口径](../../../docs/PAMF_SATIQ_CCS2023_SUPERVISED_20261119.md)

## 比较与数据角色

同一 SatIQ 划分的四个固定 E200 检查点，分别为参考 SatIQ、PAMF-SUP 原版 160、融合 256 和原型监督 320。所有组完成 5,645,000 步和源验证后进入评估。使用相同 8,192 个源 L 物理包拟合每个模型的部分白化，源 V 固定 EER 阈值。再对同一 50,000 个无标签测试包冻结原始和部分白化表示，PAMF 另保存全类头的首选波束。

评分在独立 CPU 进程中启动，先检查四组预测文件、完成记录和种子，再按不透明 ID 接真值与日期。比较随机和跨日 K1/K16 的 Top-1、同星异波束及跨星 AUC/EER，以及源阈值下 FAR/FRR。另报 PAMF 全类头 Top-1 和测试覆盖。

## 验证及实时状态

**VERIFIED，N607 调度进程已进入等待。** 读回 PID 2230670、实际命令、工作目录和上游续训 PID 2159557。本地检查覆盖 E200 缺失、预测失败和 owner 错配。N607 用真实 E10 模型与源 L 波形验证原始、白化向量均为 256 维且有限。启动时无测试预测和评分产物。调度冻结了四组检查点的选择规则，E200 未到前没有独立测试成绩。

- 调度日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/20261120-phase1-satiq-ccs2023-sup200-truthlast-s2026092701-r01/dispatch.log`
- 预测日志 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-phase1-satiq-ccs2023-sup200-truthlast-s2026092701-r01/prediction.log`
- 评分日志 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-phase1-satiq-ccs2023-sup200-truthlast-s2026092701-r01/scoring.log`
- 最终矩阵 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-phase1-satiq-ccs2023-sup200-truthlast-s2026092701-r01/scores/matrix_scores.json`

## 100 轮预算调整

**VERIFIED 已撤下等待任务。** 2026-11-20T12:01:02+08:00 先核对 PID 2230670 的命令、工作目录、独占输出和无子进程等待状态，再结束调度。当前步骤尚未开始训练或测试预测，原日志与已有产物保留。
