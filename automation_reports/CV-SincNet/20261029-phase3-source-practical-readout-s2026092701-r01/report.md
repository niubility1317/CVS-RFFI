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

## 已完成同row分析（2026-10-29）

实际为VERIFIED/ANALYZED_DEVELOPMENT_ONLY，9stage全部COMPLETE、4登记行均ANALYZED，owned退出，无技术失败、无foreign停止。主候选两条件均NO_PROMOTION，不扩seed、不调目标阈值、不接触正式RX24-18；完整Phase3目标ACTIVE。

新增source四view训练提高known接收与分数排序，但未同时解决unknown拒识。clean original→calibration_only→augMLP：unknown99.3056→96.1806→96.8750%，known接收80.7292→86.8056→90.4514%，known open62.6736→67.8819→71.5278%。aug较原F closed损伤8.1597pp，clean未知单项过95而known保护失败。

星地+DURA三状态unknown57.2917→34.7222→38.8889%，known接收72.9167→89.4097→93.0556%，known open53.2986→66.3194→68.5764%。aug较原F closed损伤4.1667pp，unknown未过90，known保护也失败。相对calibration_only相同源校准规则，训练将星地AUROC70.9262→86.1075%，是排序收益，不是90%拒识成功。中/低城市AUC84.3967/82.1994低于95%接收与90%拒识同时成立所需85.5%的必要下界，不能只挪当前τ解决逐景目标，不扫描开发oracle阈值。

本次source两个arm的known calibration均可行/保护通过，不是accept-all退化；source未知主指标仍失败，校准V成绩不称独立确认。dev历史/本次默认head closed差异C0/P1均0，两个currentarm严格共forward。目标U0/新F、head、G更新0/匿名注册0；source成员监督和E51历史OE祖先照实披露。

一次preflight/单归档比较/compile VERIFIED，真实checkpoint source四view合成smoke在sourceIQ前，dev各在Q前；一次P0/P1及定向修复、29tests/24subtests PASS。独立捕获driver2624334/export2624355 PID/CWD/cmdline/CUDA_VISIBLE_DEVICES、ownedGPU与日志539→1206→4879B；900MB是采样不是峰值。实际GPU分配{"export": 3, "C0_predict": 3, "P1_predict": 3}，source导出437.2247s、fit stage6.0401s，全pipeline472.1788s含source生成/模型加载/CPU处理/评分，不是逐包推理延迟。原始完整日志、200步训练loss、评分和post_state在本run local_artifacts，实际预测/权重保留N607原不可覆盖root。

最终完整报告见`docs/PAMF_PHASE3_SOURCE_PRACTICAL_READOUT_RESULTS_20261029.md`；下一优先是合法source星地/G输入下编码器与原默认头的联合增量控制，保留clean身份与source-only选择，不把开发Q回收为训练。
