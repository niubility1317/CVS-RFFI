# 三个独立预测并行与共享预测的源域等价和速度验证

- run_id：`20261103-phase1-shared-physical-parallel-source-perf-s2026092701-r01`
- group_id：`phase1-shared-physical-equivalence-performance`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：FAILED（实际状态按 events.jsonl 及独立证据更新）

## 目的与对照

相同112条Phase1源样本，三个原预测进程并行对比共享进程；预热8条，逐项检查104条波形门控和分类，不读目标与真值。

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

## 技术结果

并发源域检查在 bool DLPack 转换处失败，未读取目标；修复后由 r02 完成。
