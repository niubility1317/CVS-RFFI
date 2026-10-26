# 单视图 REPL clean/H 三种子正式确认

源域探索已全部完成，三个方法都通过原门槛。按预先固定的 clean 优先规则选中 REPL_H2_LU_Pair（source raw26554/27000、H362190/81000）；不会按目标测试结果改选其他变体。

本轮只训练 seed2026092702/03，从各自同契约普通 PL 的单 SIG_TIME scratchE20 独立 +10+20 到 E50，复用胜出方法 seed01 成品。完整 PAMF、单 SIG_TIME、原默认头联合训练，保留全 L 真标签和高置信 U 跨 RX；训练仅 clean 和直接叠加星地 H。没有 DURA 权重、恢复 IQ 或恢复分数。

三种子每行测原 cross_rx126000、cross_both42000、cross_date30000 的 raw clean＋六 practical。主168000按正确数加权，cross_date另报。已完成的六个单视图监督/普通 PL 对照不重训、不重做主168000预测，只用原权重补缺失的 cross_date 七景。全矩阵为2训练、9预测、9独立CPU评分和1真实源权重smoke，新增5418000类别决策。每行全部人口和视图预测固定后才评分。

目标保持 raw clean≥85.98174603174603%、raw六景>72.0961970899471%，且同 seed 配对较监督和普通PL clean≥1pp、六景>1.5pp。源增益不是目标增益；当前未启动、没有新目标成绩，不宣称达到目标。报告三 F seed 均值±样本SD、逐seed配对差与逐RX/day/TX/scene；SD不是显著性检验。

唯一矩阵、祖先和source选择、命令、代码commit、环境/CWD、独占目录、资源和停止规则见 [experiment.json](experiment.json)。沿用已有VALIDATED_ONCE物理数据，只读核实了三份handle与两个scratchE20来源，没有读取目标IQ/真值。22项focused检查通过；仅一次独立P0/P1审查，真实权重联合L/U backward及冻结原头L2评测smoke为launcher第一步，PASS立即继续。无新审批、签名或额外hash门槛。

GPU0至7，总CUDA与pending每卡最多2且空余至少10000MiB，计入其他owner；独立CPU scorer最多1。系统技术失败只停止新派发，健康已启动任务和全部产物保留，不自动重跑或操作其他任务。发布后按PID/CWD/命令/GPU/日志增长及产物独立核实状态。

本地验证已完成：22 项 focused 检查与编译 PASS，唯一独立 P0/P1 审查 PASS（0 P0、0 P1）。[验证范围](local_verification.json)。此时未启动；真实权重 smoke 的结果将在发布后独立核实，不能把本地检查写成实际运行成功。
