# 主路相位独立监督的恢复器，从零单阶段 30 轮

- run_id：`20261107-phase1-Restore-PhaseSupervisedCovariance-E30-s2026092701-r01`
- group_id：`explicit-source-canonical-main-phase-recovery`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在源均值定义的 FIR 相位坐标中，独立预测包级主路相位，用原源模拟真值监督圆周角差。

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


对原源训练缓存固定 512 条 × 20 次的统计显示，相干参考归一化重合度中位数 0.0441，最小值 0.0009144。原首 tap 只有 1526/10240 次是最强 tap。使用相干参考固定 FIR 相位，并以圆周误差监督主路。 [源训练模拟统计](source_reference_geometry.json)

## N607 主路相位监督版排队已核实

VERIFIED / QUEUED。实际代码 36b13adad5b091bf1f8f4aa7934c403e3b50d8eb，控制器 PID 3933764 在线，实际工作目录、命令、CUDA 空环境和日志已核对。发布包 SHA 一致，远端编译通过。等待接收调制版最后一项 GPU 阶段终止，再按 GPU4/6 原准入条件启动。真实源检查、训练和 Main 均待执行。
