# Phase1 锚定流对原恢复器的源样本配对诊断

- run_id：`20261101-diagnostic-phase1-anchor-source-paired-s2026092701-r01`
- group_id：`phase1-anchor-source-paired-diagnostic`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在相同源训练样本上比较固定 DURA 与已训练锚定流对干净基座判断的一致性及波形误差。

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

## 技术失败与复现

状态 FAILED，无诊断结果。CPU 进程 PID 2320138 进入源分组统计时，`torch.from_numpy(mask)` 在 N607 环境抛出 `TypeError: expected np.ndarray (got numpy.ndarray)`。模型已执行源样本推理，未读取 TX 标签、V 或测试 IQ；独立核对进程已退出，完整日志保留在 `/home/szu2070436088/2510044040/CV-SincNet/logs/20261101-diagnostic-phase1-anchor-source-paired-s2026092701-r01/stdout.log`。

在同一 Python、发布目录与环境中使用独立的布尔 NumPy 数组复现：加载 Tri-SIG-TIME 前后该接口均失败，而从布尔列表构建 PyTorch 张量成功。原因定位于当前环境 NumPy／PyTorch 转换边界，而非识别基座或模型权重。r02 仅更改该分组掩码转换并使用新 run ID；不修改模型、样本、统计指标或诊断设计。
