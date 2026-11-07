# 源域可信度门控输出诊断

- run_id：`20261108-phase1-signal-gate-source-probe-s2026092701-r01`
- group_id：`phase1-signal-gate-source-probe`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结训练完成的信号门控，检查源 pilot 的门控幅度与配对波形误差，不读取 TX 或目标。

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

## 方案与运行

固定 DURA v1、源域第20轮父修正器和可信度门控；对 672 条已探索源 pilot 计算波形 NMSE，并只在修正能量大于 anchor 能量 1e-10 的记录上通过向量投影恢复门控权重。pilot 与门控及父模型拟合 ID 不相交，DURA 历史重叠未知；不读 F 权重或输出、TX 标签、U_s 真值、V、目标，也不更新模型。

代码 `198e5cac82bcfddf33a1239cbe17455adc723356`，聚焦测试、独立 P0/P1 审查、远端归档校验及编译通过；N607 CPU 两线程、隐藏 GPU、PID 991150，首批 8 条真实源记录 smoke 通过。源域波形与门控统计不是身份准确率，不影响已启动的七景测试。

- 已完成 672 条固定源记录诊断；576 条受损记录的门控权重全部达到 0.999 以上，门控与父模型波形 NMSE 在该样本逐条相同。详细结果见 [analysis.md](analysis.md)；这不是目标识别准确率。
