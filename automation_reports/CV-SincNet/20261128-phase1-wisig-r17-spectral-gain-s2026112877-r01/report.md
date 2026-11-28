# WiSig R17 保持约束实频谱增益源域筛查

- run_id：`20261128-phase1-wisig-r17-spectral-gain-s2026112877-r01`
- group_id：`phase1-wisig-r17-spectral-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；未达到预登记晋级条件。

## 目的与对照

冻结完整 R17，学习接收信号与其输出差值的实频谱增益，使用源域波形和 R17 保持损失，在未筛查源 RX6 上评估。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验与远端编译通过。控制器 PID 3591851，训练进程 PID 3591867；已核实实际命令、GPU、首步及日志增长。真实 R17 checkpoint 与合法源 U 的无查询检查确认零初始增益恒等且梯度有限非零。执行提交 `2a9eaba95d648a37bbaff5c2315972b0ef75b886`。

## 结果与覆盖

训练完成 2000 步，源 U 曝光 48000 次，耗时 343.96 秒，峰值 CUDA 分配 57982976 字节，新增头 11233 参数；训练未读取 F、TX 标签、V 或目标数据。

固定 RX6 的 360 包 clean 和 low urban 预测后，独立 CPU 评分。直通、R17、实增益、第一版融合、新候选在 low urban 的正确数分别为 228、320、197、313、308；clean 五者均为 360。相对 R17 改对 9 包、改错 21 包，净减 12 包，两个半区各净减 6 包，配对 bootstrap 95% 区间 [−23, −1] 包。相对实增益净增 111 包，相对第一版融合净减 5 包；未达到逐半区相对所有对照净增至少 2 包的门槛。

RX6 与前次 RX1、RX3、RX4 筛查互斥；历史源 V 和目标基准已曝光。完整七景及固定 DURA v1 的匹配正式对照仍待完成。

## 交接

保留权重及负结果；后续候选另行预登记并避开已用于方案选择的物理样本。
