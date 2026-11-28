# WiSig 一致性实增益源筛查补充 DURA v1 同包对照

- run_id：`20261128-diagnostic-wisig-r69-consistency-s2026112887-r01`
- group_id：`phase1-wisig-r17-real-gain-consistency`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：ANALYZED，源域筛查未晋级。

## 目的与对照

在一致性实增益候选源训练期间固定 RX8 源 V 的 360 个物理包；候选与历史 R69 的 clean 和 low urban 预测全部冻结后独立同包评分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

候选和历史 R69 的两视图预测均冻结后，独立 CPU 审计核对 RX8 物理 ID 和两景直通预测，之后连接源 V 标签。预登记在训练期间完成，代码提交 `2acc7c22b7492c9f54d100308c60857ffd4ba8f8`，登记提交 `bc5428ba6828369990c8513630a75dd2be41dab7`。

| 视图 | 候选正确 | 固定 R69 正确 | 改对 | 改错 | 净变化 | 95% 配对区间 | 两半净变化 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| clean | 355/360 | 357/360 | 0 | 2 | -2 | [-5, 0] | -1、-1 |
| practical_low_urban | 282/360 | 318/360 | 19 | 55 | -36 | [-52, -20] | -16、-20 |

低城区未满足两半各净增至少 2 包的预登记规则；clean 的净减 2 包达到容忍边界。原筛查中 R17 同为 318/360，候选相对 R17 改对 10、改错 46、净减 36，95% 区间 [-50, -22]。候选未晋级，不启动本版本完整测试。源 U 训练 2,000 步、48,000 次曝光，未读 F、TX 标签、V 或目标；新增模块 11,233 参数，训练 323.00 秒，峰值分配 57,982,976 字节，均未包括父模型成本。

本结果仅限已曝光源 V 的单 RX、两个视图。目标七景预测及正式验收仍待完成，结果不用于目标集结论。详细配对见 `r69_source_audit.json`；原候选评分见其远端 run 的 `screen-score.json`。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
