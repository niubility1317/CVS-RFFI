# LoRa R17 与物理残差实增益融合、冻结 E104 源域指导

- run_id：`20261128-phase1-lora-r17-realgain-e104-s2026112861-r01`
- group_id：`phase1-lora-r17-real-gain-supervised-fusion`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；未通过预登记晋级条件。

## 目的与对照

首轮 source V 筛查未通过后，固定新候选：R17 和 R07 伴随残差双方向频率实增益，源域真实标签及冻结 E104 高置信 U 伪标签训练，独立物理 V 子集筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
发布归档传输校验与远端编译通过。控制器 PID 3535905，首个训练 worker PID 3535917，物理 GPU 5；实际进程、CWD、显存与日志增长已读回。真实 checkpoint 首步确认冻结 E104 的监督损失对两条新增方向均有有限非零梯度。

## 结果与覆盖

完成 4000 步训练，源 L 和 U 各曝光 16000 次，U 伪标签接受 14731 次；训练耗时 1622.77 秒，峰值 CUDA 分配 303761920 字节，新增头 11394 参数。未读取 U 真值、V 或目标数据进行训练。

与第一轮 60 包互斥的 300 包源 V 预测冻结后，独立 CPU 评分。clean 中直通、LoRa R17、实增益及融合正确数分别为 298、297、297、297；low urban 分别为 146、73、184、178。融合相对实增益改对 3 包、改错 9 包，净减 6 包，两个预定半区各净减 3 包，配对 bootstrap 95% 区间为净变化 [−13, +1] 包。相对直通净增 32 包，相对 LoRa R17 净增 105 包；clean 相对三个对照依次为 −1、0、0 包。未达到相对全部对照各半区均净增至少 2 包的门槛，故不晋级完整测试。

这属于研发筛查：E104 历史选模曾使用源 V，既往完整目标成绩也已曝光。目标集七景及固定 DURA v1 的 LoRa 匹配对照仍待完成，以上结果不用于最终达标判断。

源域低城区按 TX 类别 0 至 9 分组，相对实增益的净变化依次为 −1、0、−1、−1、−2、0、0、−1、−2、+2 包；每类仅 20 至 35 包。300 包的日期索引均为 0，仅有一个日期分组。分组原始统计位于 `local_artifacts/20261128-phase1-dura-realgain-crossdata-s2026112845-r02/readback/new-source-groups-with-day.json`，读取时下一候选已固定并启动。

## 交接

保留本候选及负结果。后续候选应另行固定数据范围、结构和门槛，并排除已用于筛查的物理包。
