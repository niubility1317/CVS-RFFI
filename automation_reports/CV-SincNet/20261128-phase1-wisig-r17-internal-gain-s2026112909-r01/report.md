# WiSig R17 每轮物理求解内部实增益

- run_id：`20261128-phase1-wisig-r17-internal-gain-s2026112909-r01`
- group_id：`phase1-wisig-r17-internal-real-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；未达到预登记晋级条件。

## 目的与对照

冻结 WiSig R17 和实增益先验，将三方向实增益插入 R17 每轮先验与两步加权求解之间，源 U 低城区 Full 配对训练。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验及远端编译通过；控制器 PID 3750044，训练进程 PID 3750122；已核实 GPU、命令、真实 checkpoint 首步与日志增长。执行提交 `a3e38fb86dbd86abbf4403e38628383cf040f3bd`。状态事件已写入；索引重建遇共享锁，登记及事件保留。

## 结果与覆盖

训练完成 1500 步，源 U 低城区 Full 池 7517 条、曝光 24000 次；耗时 632.96 秒，峰值 CUDA 分配 235031552 字节，新增头 11555 参数。训练未读取 F、TX 标签、V 或目标数据。

固定 RX8 三日期各 100 条 clean／low urban 预测后独立评分。low urban 直通、R17、实增益、第一版融合、低城区版和内部融合正确数分别为 173、264、138、264、264、266；clean 六者均为 299。相对 R17 改对 2 包、改错 0 包，净增 2 包，两个半区为 0、+2 包，配对 bootstrap 95% 区间 [0, +5] 包；相对第一版融合两个半区为 +5、−3 包。未达到对所有对照逐半区净增至少 2 包的预登记门槛。

前 3000 条筛查记录和 1 条源 V 信道探针已排除；源 V 历史及目标基准均已有曝光。固定 DURA v1 与目标七景对照待完成。

## 交接

保留此版本及筛查结果；后续版本另行固定训练和源物理样本范围。
