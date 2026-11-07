# 源协方差方向耦合恢复器，从零单阶段 30 轮

- run_id：`20261107-phase1-Restore-CoupledCovariance-E30-s2026092701-r01`
- group_id：`source-coupled-covariance-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

观测条件化的 64x64 正对角三角因子允许源协方差方向共同改变，初始严格保持原先验。

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

## 实际训练与源检查

VERIFIED / RUNNING。CoupledCovariance 已在 GPU 1 实际启动，准入已有2个计算进程、剩余20889 MiB，使用原单卡总数最多3个限制；独立读回本 run 一个 GPU 训练子进程，父控制器不使用 CUDA。真实源2条、原 F0、完整后验、K=I初始恒等、正定余量、梯度隔离、F0冻结和严格 E0 重载全部通过。6项针对性检查与唯一一次独立 P0/P1 审查通过，发布包 SHA 对比和远端编译通过。当前尚无完整epoch和Main成绩。1471333个参数全从零，单阶段30轮/7410更新；原 L_s/U_s、连续源 F0 特征和全部Main双总体七景保持，final30固定后自动预测并独立评分。

[启动与源检查证据](launch_evidence.json)

## 前3轮完整记录与实际学习核实

VERIFIED：耦合协方差版已完成3/30轮、741/7410次更新，原63000条源记录完整曝光3次，累计189000次，其中L_s18900、U_s170100；冻结F0连续特征在线前向189000条。逐一核实全部3轮日志/源顺序排列，E03保存模型与优化器有限、AdamW步数741，两个分支和输出头确实更新，原mean/C/modes缓冲不变。方向耦合RMS在三轮均非零，正对角映射平均斜率约0.35至0.71，未出现正方差版那种近零增量/梯度现象。以上仅说明该分支已参与学习，不能当作Main识别收益。

[训练与梯度分析](training_progress_E03.md)
