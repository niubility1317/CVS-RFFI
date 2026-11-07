# 物理接收调制估计与分量监督，从零单阶段 30 轮

- run_id：`20261107-phase1-Restore-ModulationCovariance-E30-s2026092701-r01`
- group_id：`receiver-modulation-source-operator-recovery`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

包级主路/镜像复增益，加独立线性与局部相位。用原源模拟真值监督调制、FIR、完整算子与噪声。独立冻结 Main 七景。

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


已读原 L_s/U_s 512 条固定源记录在 20 个缓存模拟中的调制系数。主路包内相对振幅极差最大值 1.19e-07，非零镜像最大值 1.18e-07。这支持包级复增益的源结构约束。只读缓存模拟系数，无新模型、TX、V 或目标读取。 [源系数摘要](source_trace_shape.json)

## N607 接收调制版排队已核实

VERIFIED / QUEUED。实际代码 3fc9201628e7602ea58c6a5fb13d6ec934513e99，控制器 PID 3869088 在线，实际工作目录、命令与 CUDA 空环境已核对。发布包 SHA 一致，远端编译通过。等待整体相位版最后一个 GPU 阶段终止，然后按原 GPU4/6 准入条件开始。当前真实源两条检查、训练和 Main 均待执行。
