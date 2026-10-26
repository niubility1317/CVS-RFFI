# 单视图 REPL clean/H 三种子正式确认

源域探索已全部完成，三个方法都通过原门槛。按预先固定的 clean 优先规则选中 REPL_H2_LU_Pair（source raw26554/27000、H362190/81000）；不会按目标测试结果改选其他变体。

本轮只训练 seed2026092702/03，从各自同契约普通 PL 的单 SIG_TIME scratchE20 独立 +10+20 到 E50，复用胜出方法 seed01 成品。完整 PAMF、单 SIG_TIME、原默认头联合训练，保留全 L 真标签和高置信 U 跨 RX；训练仅 clean 和直接叠加星地 H。没有 DURA 权重、恢复 IQ 或恢复分数。

三种子每行测原 cross_rx126000、cross_both42000、cross_date30000 的 raw clean＋六 practical。主168000按正确数加权，cross_date另报。已完成的六个单视图监督/普通 PL 对照不重训、不重做主168000预测，只用原权重补缺失的 cross_date 七景。全矩阵为2训练、9预测、9独立CPU评分和1真实源权重smoke，新增5418000类别决策。每行全部人口和视图预测固定后才评分。

目标保持 raw clean≥85.98174603174603%、raw六景>72.0961970899471%，且同 seed 配对较监督和普通PL clean≥1pp、六景>1.5pp。源增益不是目标增益；当前未启动、没有新目标成绩，不宣称达到目标。报告三 F seed 均值±样本SD、逐seed配对差与逐RX/day/TX/scene；SD不是显著性检验。

唯一矩阵、祖先和source选择、命令、代码commit、环境/CWD、独占目录、资源和停止规则见 [experiment.json](experiment.json)。沿用已有VALIDATED_ONCE物理数据，只读核实了三份handle与两个scratchE20来源，没有读取目标IQ/真值。22项focused检查通过；仅一次独立P0/P1审查，真实权重联合L/U backward及冻结原头L2评测smoke为launcher第一步，PASS立即继续。无新审批、签名或额外hash门槛。

GPU0至7，总CUDA与pending每卡最多2且空余至少10000MiB，计入其他owner；独立CPU scorer最多1。系统技术失败只停止新派发，健康已启动任务和全部产物保留，不自动重跑或操作其他任务。发布后按PID/CWD/命令/GPU/日志增长及产物独立核实状态。

本地验证已完成：22 项 focused 检查与编译 PASS，唯一独立 P0/P1 审查 PASS（0 P0、0 P1）。[验证范围](local_verification.json)。此时未启动；真实权重 smoke 的结果将在发布后独立核实，不能把本地检查写成实际运行成功。


## 实际发布与两种子训练启动

VERIFIED_RUNNING。实际执行 commit `68b6338d20423498c110284c0c40b65a50d37de7`，发布时本地与上游 OID 一致；单 archive 传输比较 VERIFIED，远端编译 PASS。真实权重联合 L/U backward 和冻结 E50 原头 source L2 的 raw 评测 smoke 均 PASS，没有 optimizer 更新、目标或 U 真值读取、恢复权重读取。低置信同包 H 项对 SIG_TIME/head/other 的真实梯度范数约 0.123956/0.116774/0.255524，有限且非零。

seed02/03 在 GPU0/3 独立续训，controller 与两个 worker 的 PID/CWD/完整命令/CVD/onGPU 匹配，跨两次观察真实训练记录增长。独立检查当时全部完整可读 213 条新训练记录（107/106）：每步 L768、U6、H2与pair计数、As KL0、全 L 真标签、关闭 style、无非有限损失或梯度跳步均一致。随后已核实首轮实际完成（累计 E21），首轮约 215.05/215.93 秒；最新完整记录 323/322，stdout 日志也已增长。首轮与这些记录不能冒充完成 30 轮或 E50 收敛。

复用 seed01 的三个声明人口 raw 七景预测已全部固定并独立评分；六个历史严格对照的缺失 cross_date 七景预测和独立评分全部完成。本次没有对照训练或主168000重复预测。当前共 15 个 job 完成、2 个训练在运行、4 个新种子的预测/评分待训练冻结；目标分数没有回流改变配方、候选、seed或预算。当前没有完整三种子新方法成绩，性能目标未完成。每行训练完成后继续由同一 controller 自动冻结、预测和独立评分，无需另行指令。

独立证据见 [启动与增长核实](launch_evidence.json)。
