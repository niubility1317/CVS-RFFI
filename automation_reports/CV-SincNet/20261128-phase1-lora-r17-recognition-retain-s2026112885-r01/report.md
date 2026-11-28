# LoRa R17 实增益与源识别保持约束

- run_id：`20261128-phase1-lora-r17-recognition-retain-s2026112885-r01`
- group_id：`phase1-lora-r17-recognition-retention`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；未达到预登记晋级条件。

## 目的与对照

冻结 LoRa 实增益、R02/R07/R17 与 E104，训练双方向实增益头；使用合法源 L 与 U 伪标签，约束分类器相对实增益基线的损伤。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验及远端编译通过，控制器 PID 3677838，训练进程 PID 3677967；已核实命令、CWD、GPU、首步和后续日志增长。执行提交 `b9b99c508bc7c4203ac41ab85320d90090350a20`。

## 结果与覆盖

训练完成 1500 步，源 L／U 各曝光 12000 次，接受 11040 次 U 伪标签，耗时 1063.33 秒，峰值 CUDA 分配 759153152 字节，新增头 11394 参数；训练未读取 U 真值、V 或目标数据。

对排除前 660 包后的源 RX b200_1 固定 300 包 clean／low urban 预测，再由独立 CPU 评分。low urban 直通、LoRa R17、实增益与候选正确数分别为 142、73、186、186，clean 分别为 293、291、295、295。候选相对实增益改对 0、改错 0，两个预定半区净变化均为 0，配对 bootstrap 区间 [0, 0] 包；clean 也与实增益相同。低城区未达到逐半区净增 2 包的门槛，未晋级。

历史源 V 已用于 E104 选模，历史目标基准已曝光。固定 DURA v1 的 LoRa 匹配对照与目标七景仍待完成。

## 交接

保留权重和筛查结果，后续候选需固定新的源样本和方案。
