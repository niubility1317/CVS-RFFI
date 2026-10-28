# PAMF第三阶段：源已知风险约束拒识校准

- run_id：`20261029-phase3-source-risk-calibration-s2026092701-r01`
- group_id：`pamf-phase3-source-risk-calibration`；类别：`cvs`；阶段：`Phase3-source-known-risk-calibration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；VERIFIED/ANALYZED_SOURCE_ONLY，科学NO_PROMOTION。

## 目的与对照

复用完整六类PAMF及两冻结OE读出；仅knownV27000标签校准单全局threshold，宏FRR4%/正确拒绝损失1.5pp同时约束总体和各源RX；novel3150重复源开发独立truth-last评分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

两阶段和独立评分已完成，结果见[风险校准结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_RISK_CALIBRATION_RESULTS_20261029.md)。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 代码和发布依据

实际 code/config commit：`ba2b4ed63a03751b8934dd7385f16a05aa75bcf3`。13 项相关测试通过；本候选一次独立 P0/P1 审查 PASS。复用 parent 完整 F6 导出与固定 source 几何／拒识器，首步真实拒识 checkpoint 加载后打分两条合法 L 特征；不冒充重新前向 F。不改变 parent、F、默认头或拒识器；known V 标签只校准单全局阈值。预算事前固定为总体及各 RX 宏误拒4%、正确拒绝损失1.5pp。源 novel3150 是重复开发评估，不是独立新确认；正式 RX24 U/Q 读取为0。N607 CPU-only，新增 GPU worker0，其他用户任务保持。源95%未知／5%误拒／2pp旧类目标不降。

## 实际完成结果（2026-10-29）

MLP未知宏拒识81.4603%（父76.6032%），known FRR1.3667%、开放宏acc97.6185%、drop0.8481pp；linear未知72.9206%。整体与全部RX已知风险约束合格，但未知整体及RX3/4/6/8不足95%，NO_PROMOTION，selected=null。两完整日志5行无故障，真实2L特征拒识checkpoint smoke PASS；0新训练／Fforward／GPU，target0、foreignstop0，pipeline2.425367s。同knownV经验风险和同novel重复源开发，不是新独立确认。完整[机器摘要](E:/type10-7/local_artifacts/pamf_phase3_source_risk_calibration_20261029_r01/analysis_summary.json)。不继续从未知调整known预算。
