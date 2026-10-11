# 当前交接

本轮12/12训练和自动clean完成，480epoch/106560steps完整核对。主目标84%未达成，仍active。B3=80.9429%，B0=80.9133%，增量+0.0296pp且1/3正。

执行commit24abe2a8a3824068db42949193c7a54da8f188a2；run 20261011-phase1-stage2-geometry-manysig-m12-r01。

完整结果位于 automation_reports/CV-SincNet/20261011-phase1-stage2-geometry-manysig-m12-r01/report.md。原始数据、checkpoint、logs保留N607，训练进程已自然退出，无重启。

下一步只按源证据研究可辨识方向/跨TX不迁移，不把target结果用于权重或候选筛选。不重复本轮测试。没有已发布后续新实验。


## 后续条件 R 源域诊断（已完成）

Run：`20261011-phase1-stage2-conditional-geometry-source-manysig-m3-r01`，执行提交 `3076dfd7076e0451919a5d535345dddc12035f33`，三种子完成，CPU 3.794 秒，身份更新 0，target 访问 false。报告：`automation_reports/CV-SincNet/20261011-phase1-stage2-conditional-geometry-source-manysig-m3-r01/report.md`。

原 R 平均误差比 0.984028；全维 R 1.291381；受界状态 Jacobian 0.990868；受界环境 Jacobian 0.991955；源 RX/day 共享全局偏移 0.946627（9/9 折 <1）。条件候选均未满足“每个种子优于原 R”的预登记推进规则，未启动真实 IQ 检查或新身份训练。

无界 oracle 0.620294，整体缩小后 0.969904；后者不是最优盒约束解。下一步：受界 oracle 与跨日稳定 RX 成分/包估计噪声诊断，先固定设计再运行。不得用既有目标分数选择新机制。目标保持 active，clean 三种子 84% 未达成。


## 受界 oracle 与跨日 RX 成分（已完成）

Run `20261011-phase1-stage2-relation-identifiability-source-manysig-m3-r01`，执行提交 `a0cac7f30f6c0ed49de96480bbc74821857cddce`，三种子完成，CPU 2.965 秒，无身份更新或 target 访问。报告 `automation_reports/CV-SincNet/20261011-phase1-stage2-relation-identifiability-source-manysig-m3-r01/report.md`。

固定收缩 0.4 的 RX-only 跨 TX＋留一日误差比分别 0.958587、0.939026、0.929107，27/27 条件优于零预测，满足预登记研究推进规则。稳定 RX 能量 22.79%/25.75%/26.67%，TX×RX 能量 64.26%/61.63%/58.39%；双半包可复现，但不证明独立噪声假设。系数 1 大多不如零预测，不能全量减除。

直接盒约束 oracle 误差比 0.735660/0.676194/0.716290，能解释 26.43%/32.38%/28.37% 的局部组差；统一缩放明显保守。oracle 不得变成训练 target 或预测置信度。数值凸误差界全部通过。

下一步：只用 L_s 拟合稳定 RX 分量，检查源 V 身份保持；固定源 RX 关系只约束训练、推理保持单身份网络。再冻结具体 R 目标/权重后发布从对应 E200 开始的 E40 三种子变体，保留同预算 CE 和自动 clean。不要整体删除 TX×RX 耦合。目标仍 active、84% 未达成，没有新身份训练发布。
