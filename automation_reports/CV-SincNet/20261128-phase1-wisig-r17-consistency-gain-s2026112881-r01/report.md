# WiSig 冻结 R17 clean 教师一致性与实增益筛查

- run_id：`20261128-phase1-wisig-r17-consistency-gain-s2026112881-r01`
- group_id：`phase1-wisig-r17-consistency-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；未达到预登记晋级条件。

## 目的与对照

冻结完整 R17，以其 clean 输出为 IQ 教师训练受扰输入的实频谱增益，源域 RX8 筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验和远端编译通过；控制器 PID 3605401，训练进程 PID 3605415。实际 GPU、命令、真实权重首步及日志增长已核实；执行提交 `ff6ff3a184e481f266d792c8155975842ae6ff14`。登记文件和事件已写入，索引重建遇到其他任务持有的锁，作为非阻断记录保留。

## 结果与覆盖

训练完成 2000 步，源 U 曝光 48000 次，耗时 323.00 秒，峰值 CUDA 分配 57982976 字节，新增实增益头 11233 参数；训练未读取 F、TX 标签、V 或目标数据。

固定 RX8 的 360 包 clean 和 low urban 预测后独立评分。直通、R17、WiSig 实增益、第一版融合和本候选在 low urban 的正确数分别为 236、318、205、311、282；clean 依次为 357、355、354、355、355。相对 R17 改对 10 包、改错 46 包，净减 36 包，两个半区各净减 18 包，配对 bootstrap 95% 区间 [−50, −22] 包；clean 与 R17 持平。相对第一版融合净减 29 包，未达到晋级门槛。

此前 RX1、RX3、RX4、RX6 已用于筛查，RX8 与其物理记录互斥。历史源 V 和目标成绩已曝光；固定 DURA v1 对照与目标七景仍待完成。

## 交接

保留本候选和负结果；继续研究时须重新说明源域样本曝光与选择范围。
