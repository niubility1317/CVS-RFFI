# 源波形多假设融合与细节恢复器，单阶段30轮

- run_id：`20261108-phase1-Restore-MixtureFusion-E30-s2026092701-r01`
- group_id：`source-mixture-fusion-scratch30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

单次物理后验解生成四个源波形假设，观测条件融合后接不限幅多尺度 IQ 细节网络；源端连续 F0 指导。

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

## 运行状态（2026-11-08）

VERIFIED：远端控制器 PID 999379 已按登记命令启动，CUDA 为空，GPU 3 尚未满足预留一条进程余量的入场条件，当前等待训练。程序包一次传输核对及远端编译通过。真实源样本、原 F0、双分支梯度与严格重载检查尚未执行；通过后自动完成 30 轮和原 Main 两总体七视图、独立评分。默认 DURA 不变。
