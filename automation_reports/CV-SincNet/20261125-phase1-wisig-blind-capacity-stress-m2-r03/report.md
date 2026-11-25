# WiSig 默认及增强盲恢复正式测试

- run_id：`20261125-phase1-wisig-blind-capacity-stress-m2-r03`
- group_id：`wisig-blind-capacity-stress`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用 r02 两份 40,000 步冻结恢复器，修复冻结识别器冒烟后运行同一组交叉测试和独立评分。

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

## 测试续跑

复用 [r02](../20261125-phase1-wisig-blind-capacity-stress-m2-r02/report.md) 中默认与增强两份 40,000 步冻结权重，沿用相同物理包、模拟器强度、识别器和八片测试矩阵。评估使用冻结识别器兼容的无梯度模式，并保留张量版本计数。六项聚焦检查通过，其中两项覆盖冻结识别器的初始化路径。默认视图预测与增强 IQ 生成并行。八片盲预测闭合后执行单独的真值评分。

发布提交 `3ca0fd8f88319ed2c97b7cc5d6f851158e34f56d`。独立核实控制器和 GPU 0 检查进程已启动。
