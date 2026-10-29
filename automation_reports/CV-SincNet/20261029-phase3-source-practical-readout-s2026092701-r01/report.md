# PAMF第三阶段：源端星地域读出与跨RX冻结诊断

- run_id：`20261029-phase3-source-practical-readout-s2026092701-r01`
- group_id：`pamf-phase3-source-practical-rejection-readout`；类别：`diagnostic`；阶段：`Phase3-source-practical-readout-and-frozen-development`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen E51/clean-L geometry/DURA; compare original-readout four-view recalibration against L/OE four-view membership readout fitting, then fixed C0/P1 development inference with historical original control.

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

## 本次冻结预登记

完整方法/角色/矩阵/stop rule见`docs/PAMF_PHASE3_SOURCE_PRACTICAL_READOUT_PLAN_20261029.md`与科学配置；实际代码/config commit为`ca60bdf20090f664fbcf27682138d63cfd2751dc`。runtime在本run local_artifacts/runtime.json，发布后位于新release根；启动命令/CWD/环境/路径/GPU/owner逐行在本experiment.json，不使用历史query拟合。

源端四角色6300/4200/27000/3150，共40650物理记录，clean+3practical→DURA四view共162600计算曝光。主候选augmented_readout::mlp16_oe，calibration_only同V/风险规则作为控制，原F/G/head/clean-L几何冻结。known-only阈值保护FRR5%/原F总损伤2pp，未知校准0。源科学失败仍执行固定C0/P1开发Q864；两条件同864原记录，目标U0/正式RX24-18读取0。开发三未知曾源评，结果不称新TX盲测；不声明当前run适应/注册。

聚焦29tests/24subtests PASS；唯一一次独立P0/P1审查发现compact旧API与truth字典解包两个直接错误，已RED复现、修复并定向确认PASS。N607一次preflight数据句柄/旧D0/源OE VALIDATED_ONCE与显存PASS；实际运行GPU按当时free选择，单新增worker、不停其他owner。实际checkpoint无query smoke为执行首步，PASS后立即继续，尚未用启动返回值证明运行。
