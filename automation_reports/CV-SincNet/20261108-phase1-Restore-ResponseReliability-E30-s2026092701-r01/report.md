# 源变化可靠度加权的响应保持恢复器，从零单阶段 30 轮

- run_id：`20261108-phase1-Restore-ResponseReliability-E30-s2026092701-r01`
- group_id：`source-reliability-weighted-response-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留原响应恢复器与 dense 后验，利用源模拟 Action/噪声代理连续降低低可观察性扰动的响应损失权重。

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
# 实际启动

VERIFIED / RUNNING。N607 GPU 0 启动前有 2 个计算进程、可用显存 20889 MiB；加入本任务后总计 3 个。已独立核对实际进程、命令、工作目录、GPU 及日志增长。原 F0、真实源样本、加权响应到均值／协方差两头的梯度、物理分支隔离和 E0 严格重载检查通过；TX、V、OE 和目标输入读取均为 0。

本版与等权 dense 版本分别从零单阶段训练 30 轮。每版固定最终轮后自动完成两个 Main 总体各七视图，全部预测固定后独立评分；匹配对照在两版完整预测固定后进行。当前没有正式识别性能结论。实际启动证据位于远端本 run 的 `runtime_observation.json`。
