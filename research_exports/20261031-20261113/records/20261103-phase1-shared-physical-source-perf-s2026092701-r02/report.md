# 三个物理融合模型共享前向的源域等价和速度验证

- run_id：`20261103-phase1-shared-physical-source-perf-s2026092701-r02`
- group_id：`phase1-shared-physical-equivalence-performance`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：ANALYZED（实际状态按 events.jsonl 及独立证据更新）

## 目的与对照

在固定的112条Phase1源记录上比较三个原有checkpoint的独立与共享前向，保持原批次和精度；不接触测试IQ或真值。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

真实 checkpoint、CPU 两线程、FP32、batch 8；固定源域 112 条样本覆盖两组接收机与七个视图。三种模型的恢复波形、门控与分类逐项完全一致，且与已有源域固定预测一致。

首批不计时，余下 104 条样本三模型独立计算用时 27.7773 秒，共享计算用时 9.7471 秒，观察到 2.8498 倍计算阶段加速。独立计算在本次计时中按三个模型依次执行；现有三个正式预测则并行运行，因此 2.8498 倍不能直接推算正式任务的墙钟时间收益。数据量有限且受当时 CPU 负载影响，尚未验证并行场景的端到端加速。

未读取目标样本或 TX 真值，未修改模型；尚未证明全部目标样本逐项一致，也未接入现有正式预测。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
