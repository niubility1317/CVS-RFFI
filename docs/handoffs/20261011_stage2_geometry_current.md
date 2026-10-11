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


## 稳定 RX S3 实现与发布准备

源 V 检查 `20261011-phase1-stage2-stable-rx-identity-source-manysig-m3-r01` 三种子全部通过，执行 f9a461c539495a19eeb450fa7518438c3b732653，CPU 2.661 秒。alpha .25 准确率变化 −.011111/+ .003704/+ .003704 pp；source-only，无新身份训练或target。已保存完整报告。

新 run `20261011-phase1-stage2-stable-rx-manysig-m3-r01`：S3×3从对应原始E200重新开始E40/8880，替换共享RX预测器，其他B3目标/日程/LT不变。其他4TX＋其他2日拟合，shrink=.4、alpha≤.25、9折标量全160 c，完整有梯度有限总体修正。复用旧geometry全部B0/B2/B3同预算控制，自动逐row clean后truth-last；4总进程/GPU。9项本地checks和共享训练真实图/完成存档回归通过，独立P0/P1审查通过。

新代码 `experiments/cvs_stage2_stable_rx`；shared geometry/source.py提供design_module与receiver_payload注入，旧配置行为保持。发布器先CPU拟合R+真实父前反向不更新参数，再启动唯一dispatcher。当前准备Git交付与正式发布，必须以远端readback确认启动。目标仍active，未达84%。源 V 筛查使用全部六 TX 的 leave-day 拟合；训练使用四 TX 的 leave-day 拟合。筛查与训练不是完全同一变换，不把筛查结果作为训练后身份保持的保证。


### S3 正式运行 VERIFIED

执行提交5c9465846a49a90b5cd008584ebff839f9647595，dispatcher3913667；S3-s2026092701 GPU3 PID3913802，seed02 GPU5 PID3913807，seed03 GPU6 PID3913818。最新快照各E3/40，无失败，resolved确认无teacher/U/satellite。新R27折c=.0262至.1021，所有alpha=.25，CPU实际parent梯度预检通过。

远端run `/home/szu2070436088/2510044040/CV-SincNet/runs/20261011-phase1-stage2-stable-rx-manysig-m3-r01`，不可变release `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_stage2_stable_rx_20261011_r01`；已有唯一owner和自动clean链路，不要重新启动。收集脚本 `.codex_tmp/collect_stable_rx_training.py`，发布读回 `local_artifacts/stable_rx_training_20261011_r01/readback.json`。等待E40/8880固定、各row自动prediction和truth-last评分；完成后收集全部epoch/step日志和完整clean结果，报告与Git登记。目标仍未达成，不能依据逐row测试干预健康训练。


## S3 已完成

`20261011-phase1-stage2-stable-rx-manysig-m3-r01` 三行E40/8880和自动clean全部完成，完整120epoch/26640step核验。S3均值80.946627%，相对B0 +0.033333pp（1/3正），性能目标状态NOT_MET。全部按预登记完成，不因结果重跑。报告 automation_reports/CV-SincNet/20261011-phase1-stage2-stable-rx-manysig-m3-r01/report.md，原始权重和日志仍在N607原run；禁止重复发布或预测。报告与测试均为证据，不是之后调参的输入。下一步须由源证据决定，目标未完成时保持active。

源端补充：全部30个审计点R梯度非零，全部关系活跃；八视图固定源V压力日志已完整聚合，S3相比B0在temporal/LTL约+1.08/+1.53pp。下一步应核查真实源组差目标实现率和R锚定分量，而非依据target改权重。目标仍active；未发布新的后续训练或诊断。
