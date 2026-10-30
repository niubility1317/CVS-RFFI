# DURA 保持功率与噪声的源采集响应扰动诊断

- run_id：`20261030-diagnostic-DURA-paired-acquisition-stress-s2026092701-r01`
- group_id：`phase1-dura-paired-acquisition-stress`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结 noVar 源 final；同源 512×2 缓存视图比较原始及精确 A 配对采集响应扰动；CPU2、无更新或目标读取。

本次固定两抽头因果滤波与小幅 IQ 镜像扰动，并保持每包原始 RMS。输入使用 `y+A(Sx-x)`，因此模拟信道 A 和原噪声不变。固定 noVar E20/4940 源权重与原 F50；F 只描述源特征和选择保持情况，不参与资格阈值，也不读取 TX 真值。

两个缓存视图均须满足：扰动后的共同功率归一化 MSE 至少增加 10%，每包功率相对误差不超过 2e-6，噪声重放误差不超过原功率的 1e-10。条件不全满足则不据此启动训练。若通过，另行登记从零 E20/4940、每次 0.5 概率一次扰动、一次 G 前向的单阶段对照；本次没有新训练权重或目标预测。

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


## 实际结果

VERIFIED / ANALYZED_SOURCE_ONLY。执行 commit `6328fe60db02e99c1ea806f0c27b6fd70c6b5e3b`，CPU2/0GPU 耗时 8.427 秒，真实 source2Ls G/F smoke 通过，32 个 B32 批次完成。PID269042 存活时已独立核实 release CWD、完整命令、空 CUDA 绑定与日志增长；最终产物完整。唯一 P0/P1 审查 PASS，3 项聚焦配对测试 PASS，一次 preflight、归档两端 SHA 和远端 compile 通过。F/G 状态不变、grad None、0 更新/直接 TX/V/target。

独立复算 1080 个 role/scene/state/lock 聚合标量，并核对原始分支与上一 source-stage 对照的相同位置、角色和 IQ/F 结果一致。源数组与位置仅保留私有 artifact，公开汇总见 results.json。

|缓存视图|原 MSE|扰动 MSE|相对增加|功率相对误差最大值|噪声回放误差最大值|训练资格|
|---|---:|---:|---:|---:|---:|---|
|1|0.02235236|0.02381189|6.5297%|2.980e-07|3.748e-14|False|
|20|0.02112201|0.02263714|7.1732%|2.980e-07|4.033e-14|False|

两视图 MSE 增幅均小于预定 10%，因此不据此启动 paired-style 训练，不改变扰动强度或放宽门槛。信号功率与原噪声配对均通过。源 clean/style F 选择保持 511/512；恢复后 teacher 一致数分别变化 −2/+1，仅为源特征描述，不能代替 TX 真值或迁移性能。默认 DURA、全七景正式验收和训练效率目标均保持。
