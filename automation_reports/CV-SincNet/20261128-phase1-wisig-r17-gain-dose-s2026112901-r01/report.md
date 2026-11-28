# WiSig R17 源域实增益固定两倍修正

- run_id：`20261128-phase1-wisig-r17-gain-dose-s2026112901-r01`
- group_id：`phase1-wisig-r17-real-gain-dose`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；未达到预登记晋级条件。

## 目的与对照

复用源 U 合规训练的低城区实增益头，将其相对 R17 的复数修正固定乘 2，与原幅度同包比较。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验和远端编译通过，控制器 PID 3722023；远端状态及日志确认 GPU 7 完成无查询 smoke、两视图预测与独立 CPU 评分。执行提交 `ae312f1d89b046922ffc890fc168030802c0b5fa`。本版无需新增训练，沿用父版权重的 2000 步源 U 训练成本。

## 结果与覆盖

在 RX4 三日期各 100 条源 V 上，固定全部预测后独立评分。low urban 直通、R17、实增益、第一版融合、未放大父版、两倍候选正确数分别为 177、252、150、249、251、250；clean 依次为 295、295、296、295、295、295。相对 R17，两倍候选改对 2 包、改错 4 包，净减 2 包，两半区为 −2、0，配对 bootstrap 95% 区间 [−7, +3] 包；相对未放大父版净减 1 包。clean 相对 R17 和父版持平，相对实增益净减 1 包。未达到预登记晋级条件。

父版本身在此新源 RX4 相对 R17 净减 1 包，先前源 RX3 的小幅净收益未在本筛查中复现。父版由已曝光的源 V 筛查结果触发选择，因而本项属于持续研发；目标基准历史已曝光，DURA v1 匹配对照及目标七景待完成。

## 交接

保留本版本及负结果；后续候选须另行固定来源和未在先前筛查中使用的物理包。
