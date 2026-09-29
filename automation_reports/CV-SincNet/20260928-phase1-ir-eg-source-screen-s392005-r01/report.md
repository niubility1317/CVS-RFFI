# IR-EG源侧筛查已发布

发布状态：**VERIFIED**。N607六行训练的PID、CWD、argv、CUDA绑定、实际配置及接受步日志增长已独立核实。该段为2026-09-28启动记录；当前SIM/EG已完成、四个IR保持停止，见末尾更新。

固定model seed392005，split/data/augmentation/evaluation seed392005；support不适用。六行均从零训练，L/U/V为6300/56700/27000，同一既有物理ID契约，启用原生DAOT/RC4核心实现的联合配置（并非旧原生入口全部配方）。200epoch/44400接受步是固定曝光预算，不代表收敛。

|行|GPU|PID|读回接受步|状态|
|---|---:|---:|---:|---|
|SIM_s392005|1|1946779|99|RUNNING|
|EG_s392005|3|1946780|68|RUNNING|
|IR_s392005|4|1946781|63|RUNNING|
|IR_G0_s392005|5|1946782|58|RUNNING|
|OR_EG_s392005|6|1946783|64|RUNNING|
|IR_ENCODER_OFF_s392005|7|1946784|62|RUNNING|

读回时间：2026-09-28T00:43:56.158891+00:00。发布commit：`92f40367d61c45b9d16a9718bcd68df17af507c2`。归档SHA：`adacb95e57cc53835b71b433bb7ba370833170f0dd3bddd68cd43ac961f99e8b`。

## 验证

- 本地旧PyTorch接口兼容专项2项通过；独立P0/P1审查发现的autocast签名问题已修复并定点闭合。
- N607实际torch2.1.0+cu121 CUDA环境全部112项IR测试通过，包含active IR路径；单次归档校验及远端编译通过。
- 原生scratch模型checkpoint保存、CPU反序列化、严格恢复及4个合法L_s样本前向检查PASS。该诊断checkpoint没有被训练继承。
- 新建run/log根目录，六个GPU各一行，GPU0/2原有任务保持运行。

## 环境准备与边界

初次测试调用因N607环境没有pytest失败；远端PyPI超时，随后使用本地下载的pytest8.3.5及依赖离线安装到独立/tmp测试目录。未修改现有Conda环境，训练不使用该测试PYTHONPATH。

本次仅启动六行单seed源侧筛查。三seed确认、BR效率和目标预测尚未启动，依设计待源侧证据和候选冻结后推进；收敛延长配置仍未冻结。本次发布不形成性能、收敛或目标泛化结论。技术异常只影响所属行，不自动重试、跳过坏batch或按性能停止。

## 证据

- [发布登记](experiment.json)
- [112项远端测试](evidence/remote_acceptance.log)
- [归档与编译](evidence/publish.log)
- [提交进程](evidence/launch.log)
- [最新独立读回](evidence/readback_1790556295011623100.log)
- [发布审查](../../../experiments/ir_eg_v1/docs/release_review_20260928.md)

## 2026-09-28日志反馈修订

用户指出stdout只有epoch和总loss，并明确要求现有训练继续，后续默认提供详细日志及适合AI查看的额外结构化格式。

完整读取本次六行截至快照的actions.jsonl、logs.jsonl、ir_steps.jsonl（适用行）和stdout后，确认已有学习率、损失分量、权重、梯度、DAOT/RC4状态、IR/CG响应和source V指标；六行无无效完整JSON记录。源侧训练accuracy未记录，不从loss反推；target未评估。详见[字段与记录数审计](evidence/logging_audit_summary_20260928.json)。

根因是IR专用runtime保留开发期单行摘要，没有接入原生CVS的多段日志展示。已为后续默认增加详细stdout及持久training.log、training_config.json、紧凑epoch training_metrics.jsonl和training_metrics.csv。现有逐步原始JSONL格式保持。说明见[日志字段与口径](../../../experiments/ir_eg_v1/docs/logging.md)。已在工作区和Git承载面的AGENTS.md登记用户默认要求。

修订只在本地Git实现，未向当前N607发布目录写入、未增设日志旁路进程、未停止或重启任何训练。原训练commit仍为92f40367d61c45b9d16a9718bcd68df17af507c2；后续新发布使用更新后的日志实现。

验证覆盖文本/JSONL/CSV一致性、真实测量与缺失区分、IR晚出现字段及恢复计时、原始输入不变，以及原生训练入口连续/恢复模型、优化器、EMA、prototype、solver和RNG一致性。相关测试4项通过。

## 2026-09-29停电核实

SIM、EG完成E200；IR、OR_EG保留E164；IR_G0、IR_ENCODER_OFF保留E163。checkpoint可读取且来源契约一致。用户要求先不恢复，四行保持停止。SIM、EG固定最终权重目标评估记录：20260929-phase1-sim-eg-target-eval-s392005-r01。历史RUNNING/PID为启动时证据，不能代表当前运行。

## 2026-09-29完整诊断

用户要求先不续跑，全面分析性能与速度。六行全量234265个完整主步及全部结构化文件已扫描；没有数值失败，发现P路由整体失活、固定V诊断面板覆盖不足及事务全量复制成本。详见[完整诊断报告](analysis_20260929/report.md)，提供逐epoch CSV/JSONL、曲线和CPU只读证据。训练代码与远端任务均未修改。
