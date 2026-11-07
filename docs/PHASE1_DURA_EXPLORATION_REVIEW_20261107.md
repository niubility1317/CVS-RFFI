# Phase1 恢复网络近期结果复核

截至 2026-11-07，对本对话完成的 22 个 DURA 系列 F-free 版本，按同一 DURA 基准和六个受损场景逐项比较：**132 项场景比较均低于 DURA**。这限定于下表版本和固定单种子测试，不代表所有可能方法都无效。部分版本在 clean 的小幅正差只有 1 至 3 条记录，不能解释为整体正收益。

这些恢复器训练不使用 F、TX 标签、V 或目标；评估统一连接冻结的 Tri-SIG-TIME SSL 及原默认头，Phase1 Main split 每景 168000 条，同记录配对，全部预测固定后独立 truth-last。测试已反复用于研发观察，不是独立盲测或真实卫星结果。逐版本 source 训练预算和祖先不同，不能把方法间差异都归因于单一机制；测试结果不回流选模、调参或选择性重跑。

raw clean 为 86.538690%。六景按下降不超过 3 个百分点须各自达到 83.538690%；若按相对下降不超过 3%，须各自达到 83.942530%。固定 DURA 六景依次为 84.3435%、83.0161%、81.3220%、83.0494%、75.7060%、68.8012%，两种口径均只有 1/6 达标。

下表为准确率相对 DURA 的变化，单位均为百分点；负数表示低于 DURA。各数值由原始逐版报告复制，并用同报告中的修复数减损伤数按 168000 条复算。

| 已完成版本 | clean | 高郊区 | 中郊区 | 低郊区 | 高城区 | 中城区 | 低城区 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [频段保护](../automation_reports/CV-SincNet/20261106-phase1-dura-band-guard-ground-test-s2026092701-r01/analysis.md) | -0.1536 | -0.2946 | -0.3107 | -0.3268 | -0.3351 | -0.4107 | -0.4226 |
| [交叉半窗自监督](../automation_reports/CV-SincNet/20261106-phase1-dura-crosshalf-iq-ssl-ground-test-s2026092701-r01/analysis.md) | +0.0018 | -0.5542 | -0.4220 | -0.4381 | -0.5464 | -0.3482 | -0.3881 |
| [跨视图保护](../automation_reports/CV-SincNet/20261106-phase1-dura-crossview-guard-ground-test-s2026092701-r01/analysis.md) | -0.1458 | -0.5780 | -0.4750 | -0.4655 | -0.5613 | -0.4851 | -0.4405 |
| [三专家混合](../automation_reports/CV-SincNet/20261106-phase1-dura-expert-mixture-ground-test-s2026092701-r01/analysis.md) | +0.0006 | -1.6964 | -1.7726 | -1.7804 | -1.7518 | -1.6595 | -1.5768 |
| [固定自监督编码器](../automation_reports/CV-SincNet/20261106-phase1-dura-fixed-iq-ssl-ground-test-s2026092701-r01/analysis.md) | +0.0006 | -0.7685 | -0.6673 | -0.6488 | -0.7804 | -0.5768 | -0.5226 |
| [DURA 内部联合更新](../automation_reports/CV-SincNet/20261106-phase1-dura-internal-guard-ground-test-s2026092701-r02/analysis.md) | -0.0476 | -0.5583 | -0.7071 | -0.6893 | -0.5732 | -0.7554 | -0.9333 |
| [源 IQ 几何](../automation_reports/CV-SincNet/20261106-phase1-dura-iq-geometry-ground-test-s2026092701-r01/analysis.md) | +0.0006 | -0.5494 | -0.4881 | -0.4702 | -0.5780 | -0.4917 | -0.4577 |
| [联合自监督编码器](../automation_reports/CV-SincNet/20261106-phase1-dura-joint-iq-ssl-ground-test-s2026092701-r01/analysis.md) | +0.0006 | -0.6554 | -0.5196 | -0.5345 | -0.6708 | -0.5524 | -0.5750 |
| [多时滞关系](../automation_reports/CV-SincNet/20261106-phase1-dura-moment-relation-ground-test-s2026092701-r01/analysis.md) | +0.0006 | -0.4774 | -0.4196 | -0.3923 | -0.5119 | -0.4506 | -0.4601 |
| [多尺度频域修正](../automation_reports/CV-SincNet/20261106-phase1-dura-multiscale-spectral-ground-test-s2026092701-r01/analysis.md) | +0.0006 | -0.5780 | -0.4738 | -0.4744 | -0.5792 | -0.4738 | -0.4940 |
| [噪声门控修正](../automation_reports/CV-SincNet/20261106-phase1-dura-noise-gated-ground-test-s2026092701-r01/analysis.md) | +0.0006 | -0.4917 | -0.4423 | -0.4214 | -0.5137 | -0.4452 | -0.4375 |
| [非线性幅度保护](../automation_reports/CV-SincNet/20261106-phase1-dura-nonlinear-guard-ground-test-s2026092701-r01/analysis.md) | -0.1429 | -0.3524 | -0.3262 | -0.3554 | -0.3690 | -0.4399 | -0.4548 |
| [物理创新正交修正](../automation_reports/CV-SincNet/20261106-phase1-dura-orthogonal-ground-test-s2026092701-r01/analysis.md) | -0.2125 | -0.6655 | -0.5292 | -0.4893 | -0.6208 | -0.5220 | -0.4607 |
| [正交修正加保护](../automation_reports/CV-SincNet/20261106-phase1-dura-orthogonal-guard-ground-test-s2026092701-r01/analysis.md) | -0.1500 | -0.3940 | -0.3577 | -0.3798 | -0.4065 | -0.4631 | -0.4524 |
| [源理想后验教师](../automation_reports/CV-SincNet/20261106-phase1-dura-posterior-teacher-ground-test-s2026092701-r01/analysis.md) | +0.0012 | -0.6012 | -0.5149 | -0.4643 | -0.5726 | -0.4470 | -0.4006 |
| [源模拟信道作用监督](../automation_reports/CV-SincNet/20261106-phase1-dura-sim-action-ground-test-s2026092701-r01/analysis.md) | -0.0357 | -0.3036 | -0.4637 | -0.4589 | -0.3149 | -0.5577 | -0.8220 |
| [源尾部误差保护](../automation_reports/CV-SincNet/20261106-phase1-dura-tail-excess-ground-test-s2026092701-r01/analysis.md) | +0.0012 | -0.3869 | -0.3155 | -0.3333 | -0.4333 | -0.3185 | -0.3625 |
| [时变滤波修正](../automation_reports/CV-SincNet/20261106-phase1-dura-timevarying-ground-test-s2026092701-r01/analysis.md) | -0.0155 | -0.3690 | -0.3173 | -0.2976 | -0.3589 | -0.3006 | -0.3589 |
| [扩大源数据覆盖](../automation_reports/CV-SincNet/20261106-phase1-dura-wide-source-ground-test-s2026092701-r01/analysis.md) | +0.0000 | -1.4500 | -1.4946 | -1.4839 | -1.4821 | -1.5244 | -1.4512 |
| [三专家蒸馏](../automation_reports/CV-SincNet/20261107-phase1-dura-expert-distill-ground-test-s2026092701-r01/analysis.md) | +0.0000 | -1.3167 | -1.3506 | -1.4542 | -1.4821 | -1.4649 | -1.3000 |
| [频域回归](../automation_reports/CV-SincNet/20261107-phase1-dura-frequency-ground-test-s2026092701-r01/analysis.md) | -0.0560 | -0.5167 | -0.4196 | -0.4214 | -0.4720 | -0.3649 | -0.3167 |
| [可逆耦合](../automation_reports/CV-SincNet/20261107-phase1-dura-reversible-ground-test-s2026092701-r01/analysis.md) | -0.2232 | -0.1857 | -0.1560 | -0.1756 | -0.1857 | -0.1667 | -0.1435 |

## 局部正收益的边界

独立时频学生加复数 FIR 修正后，相比其父学生，低郊区、中城区、低城区分别提高 0.0667、0.1339、0.3036 个百分点；clean 和另三个受损场景下降。该版新增五轮训练，而且六个受损场景均低于 DURA 和各自 raw 输入，因此是弱父模型上的局部正差，不能作为优于 DURA 或目标达标的证据。[完整配对结果](../automation_reports/CV-SincNet/20261107-phase1-direct-student-fir-ground-test-s2026092701-r01/analysis.md)

可逆耦合与频域回归都曾在源 pilot 的 12 个受损分组降低波形 NMSE，但正式七景识别率全部低于 DURA。这说明本批证据中，源 IQ 重建误差下降并不足以证明身份识别改善；它不证明其他源域训练目标或尚在运行的方案必然失败。
