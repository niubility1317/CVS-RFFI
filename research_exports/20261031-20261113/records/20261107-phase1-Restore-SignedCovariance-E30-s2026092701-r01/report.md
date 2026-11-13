# 双向源协方差谱适配恢复器，从零单阶段30轮

- run_id：`20261107-phase1-Restore-SignedCovariance-E30-s2026092701-r01`
- group_id：`source-signed-spectral-covariance-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定源协方差top64方向，观测条件化预测有正有负的softsign系数，初始保持原C。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## [计算环境] 实际队列已核实

VERIFIED / QUEUED。双向协方差版已发布N607；实际控制器PID/CWD/命令/CUDA空/等待日志已核实，子进程0、新增GPU训练进程0。等待当前ConditionedCovariance r02最后GPU阶段后再按GPU6容量准入。代码4项针对性测试、一次独立P0/P1、归档SHA一致与远端编译通过；真实源2检查仍待其训练阶段执行，不将排队或单元测试写成真实模型已通过。模型1340293参数，固定源top64方向，C+Bdiag(0.9softsign(raw))B^T允许增减且初始严格为原C；实际FP32模式因子的白化谱界保证正定余量。从零单阶段30轮/7410更新、原L_s/U_s与源F0连续特征权限保持，final30后自动同一原TriSIGTIME SSL基座Main两人群七景及独立评分。现有健康训练和CPU Main继续，新版尚无识别成绩。

队列和验证摘要

## 未训练队列已由资源双路径 r02 接替

VERIFIED / REPLACED。原控制器没有孩子或已提交GPU阶段，已在新归档编译后精确退出。模型、数据、seed及30轮不变，旧release/日志/输出保留。替代运行。
