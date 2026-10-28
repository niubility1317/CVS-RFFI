# PAMF第三阶段：跨分支同类支持拒识

- run_id：`20261029-phase3-source-consensus-s2026092701-r01`
- group_id：`pamf-phase3-source-consensus`；类别：`cvs`；阶段：`Phase3-source-cross-branch-head-aligned-rejection`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；VERIFIED/ANALYZED_SOURCE_ONLY，科学NO_PROMOTION。

## 目的与对照

完整F6和L几何冻结，100D描述符加入默认头概率对齐的分支统计，保持101/1633参数与200CPU步；V-only固定风险阈值；source novel重复开发truth-last。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

三个阶段及独立评分完成，完整[结果报告](E:/type10-7/docs/PAMF_PHASE3_SOURCE_CONSENSUS_RESULTS_20261029.md)和[机器摘要](E:/type10-7/local_artifacts/pamf_phase3_source_consensus_20261029_r01/analysis_summary.json)已落盘。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 代码与发布依据

实际 code/config commit：`951fbddea7cafb4deb5e0e18a4d404edd7d02b74`。10项相关测试通过；本新候选一次独立P0/P1审查PASS。只改变100D描述符为head_aligned_branch_summary_v1，同101/1633参数和每读出200CPU全批量步。F/默认头/L几何冻结，source数据复用，首步实际父拒识checkpoint在2条合法L特征smoke，不能说重前向F。新读出仅L/OEtrain，knownV仅既定4%FRR/1.5pp正确拒绝损失的单global风险阈值，novel在阈值冻结后才读。新state明示descriptor_schema，不混用旧100D语义。source三novel重复开发非新确认，正式target0，CPU-only新增GPU0，其他进程保持。

## 实际完成结果（2026-10-29）

linear未知82.6984%、FRR1.6185%、drop0.6963pp；MLP未知85.7460%、FRR1.4444%、drop0.7815pp。MLP较匹配risk父方法81.4603%提高4.2857pp，但整体及RX3/4/6/8仍未达95%，已知保护全过；NO_PROMOTION、selected=null。两新读出各200CPUstep，400loss有限且单调下降，终值0.190165／0.109273；实际父checkpoint2L和新100D descriptor smoke PASS，新state回读完成预测。三日志8行564B无故障，F/geometry冻结，新增Fforward/GPU/target/foreignstop0。重复source clean开发非新独立确认，未证明接收机不变，不改预算重跑。
