# WiSig 完整 R17 与实增益源域联合微调

- run_id：`20261128-phase1-wisig-r17-joint-realgain-source-s2026112869-r01`
- group_id：`phase1-wisig-r17-joint-real-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；未通过预登记晋级条件。

## 目的与对照

前两项冻结 R17 后附加恢复头均未通过筛查；固定 R02/R07 并联合微调 R17 三个 prior 与双方向实增益头，通过同契约原 R17 IQ 教师约束偏移，筛查未用过的源 RX4。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
发布归档传输校验及远端编译通过。控制器 PID 3562183，首个训练 worker PID 3562201，物理 GPU 7；已读回进程、CWD、首步及后续日志增长。首步检查 R17 prior 与两条增益方向梯度有限且非零，R02/R07 保持冻结。

## 结果与覆盖

训练完成 1500 步、U 曝光 24000 次，耗时 787.18 秒，峰值 CUDA 分配 335901184 字节；未读取 F、TX 标签、V 或目标数据参与训练。完整 R17 有 4665073 参数，新增实增益头 11394 参数。

固定源 RX 4 的 360 条 clean 和 low urban 预测后独立评分。low urban 直通、原 R17、实增益、第一版融合、第二版融合和联合微调的正确数分别为 213、312、184、299、292、246；clean 依次为 351、351、351、351、350、350。联合微调相对原 R17 改对 11、改错 77，净减 66 包，两半区分别净减 28 和 38 包，配对 bootstrap 95% 区间 [−84, −49] 包；clean 净减 1 包。相对第一、第二融合分别净减 53、46 包，未通过预登记门槛。

RX 4 与前两次筛查的 RX 1、RX 3 物理记录互斥；源 V 和历史目标基准已有曝光。完整目标七景与 DURA v1 配对比较仍待完成。

独立评分后的源域分组中，TX 类别 0 至 5 的净变化分别为 −9、−9、−5、−33、−7、−3 包；类别 3 为 TX `20-19`，63 包中原 R17 正确 57、联合微调正确 24。日期 1、2、3 各 120 包，净变化分别为 −20、−31、−15 包。分组原始统计位于 `local_artifacts/20261128-phase1-dura-realgain-crossdata-s2026112845-r02/readback/new-source-groups-with-day.json`，读取时下一候选已固定并启动。

## 交接

保留本候选与负结果；后续研发应重新固定候选和源域样本，避免重复使用筛查包冒充独立验证。
