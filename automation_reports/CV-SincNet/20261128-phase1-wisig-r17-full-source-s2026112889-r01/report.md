# WiSig R17 实增益使用原生 full 信道源训练

- run_id：`20261128-phase1-wisig-r17-full-source-s2026112889-r01`
- group_id：`phase1-wisig-r17-real-gain-full-source`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；未达到预登记晋级条件。

## 目的与对照

冻结 R17，使用已登记源 U full 信道与 clean 配对训练实增益，测试排除既有源筛查物理包的 RX1 新样本。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验及远端编译通过；控制器 PID 3687557，训练进程 PID 3687742，已核实 GPU、命令、真实 checkpoint 首步与后续日志增长。执行提交 `d7208489d750d6609eeaa13c390058d2f291939a`。

## 结果与覆盖

训练完成 2000 步，合法 Full 源 U 池 44773 条，曝光 48000 次，耗时 320.19 秒，峰值 CUDA 分配 57982976 字节，新增头 11233 参数；训练未读取 F、TX 标签、V 或目标数据。

按原 Full 仿真配置固定 RX1 三个日期各 100 包的 clean／low urban 预测后独立评分。low urban 直通、R17、实增益、第一版融合及本候选正确数分别为 171、272、155、265、272；clean 五者均为 300。相对 R17 改对和改错均为 0，两个半区净变化均为 0，配对 bootstrap 区间为 [0, 0] 包。相对第一版融合改对 8、改错 1，净增 7 包，但两半区为 +7、0；故未达到相对全部对照逐半区净增至少 2 包的门槛。

评分后在 128 条源 U full 观测上的波形诊断中，候选相对 R17 修改的均方根幅度与 R17 输出之比，均值 0.2368%、中位数 0.2024%、最大值 0.4668%；诊断未读取识别器、TX 标签、V 或目标数据。结果位于 `local_artifacts/20261128-phase1-wisig-r17-full-source-s2026112889-r01/readback/source-gain-diagnostic.json`，仅用于解释该候选的源域行为。

已排除前 1800 个筛查包。技术检查额外读取的源 V ID `5fa0fd382eb8af0fbc4cc08b` 未读取标签或识别结果，并从本轮与今后未曝光筛查中排除；完整源 V 和历史目标成绩曾曝光。DURA v1 对照及目标七景待完成。

## 交接

保留此版本及筛查结果；后续候选另行固定训练条件与物理样本。
