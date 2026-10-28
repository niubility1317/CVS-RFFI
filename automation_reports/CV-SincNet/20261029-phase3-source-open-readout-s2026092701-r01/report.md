# PAMF第三阶段：完整六类基座源OE拒识

- run_id：`20261029-phase3-source-open-readout-s2026092701-r01`
- group_id：`pamf-phase3-source-open-readout`；类别：`cvs`；阶段：`Phase3-source-open-rejection-current-full-PAMF`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；实际产物为VERIFIED/ANALYZED_SOURCE_ONLY，科学NO_PROMOTION，正式target未读。

## 目的与对照

当前六类TriSSL F冻结，L6300源几何+OE4200训练linear/MLP读出，knownV27000校阈值，TX隔离novel3150独立评分；5静态控制不触发OE晋级。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

本row四阶段和独立truth-last评分已完成。完整结果见[源OE拒识报告](E:/type10-7/docs/PAMF_PHASE3_SOURCE_OE_REJECTION_RESULTS_20261029.md)；机器摘要见[analysis_summary.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/analysis_summary.json)。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 版本与本地验证

实际code/config commit：`76951634c0901175a6d78c88caffbf49fdce8d04`。64项测试和35个subtests通过；一次独立P0/P1审查PASS，审查者独立同64项通过。源F6/defaulthead冻结，2新读出各200CPUfeature步，几何L-only、normalizer L+OEtrain、knownV-only99%阈值，novel source验证truth-last。5静态控制不能推动OE候选，源过门不构成完整Phase3成功。已验证OE数据随同一release archive复制一次，保持bytes/ID/句柄，原机E/F路径不驱动N607读取。真实2L checkpoint smoke为export首步。N607一次preflight通过，GPU2当时空闲，实际容量选择不停止任何其他用户任务。

## 真实源端结果（2026-10-29）

技术完成，科学NO_PROMOTION；linear_oe未知TX宏拒识68.4762%、mlp16_oe76.6032%，已知FRR均0.9963%，旧类开放宏准确率97.9148%／97.8852%，相对原闭集98.4667%下降0.5519／0.5815pp。旧类保护整体及5个RX通过，只有RX1未知宏拒识达到95%，RX3/4/6/8未达；无选中方法。两读出200步损失分别0.693147→0.186006、1.140048→0.116537，完整400记录有限且单调下降。四日志完整26行、2119B无技术异常；真实2Lsmoke PASS、F40652条冻结前向、encoder/head更新0、target0、原U_s0、foreignstop0。pipeline16.732910s，仅export GPU2。

这是当前完整六类冻结PAMF的source clean辅助新TX泛化，不是跨RX／星地达标或无标签适应／注册成功。不改阈值、不延长训练、不扩seed、不读正式Full pool；下一步建议在独立source候选预登记后执行，当前无新增实验。
