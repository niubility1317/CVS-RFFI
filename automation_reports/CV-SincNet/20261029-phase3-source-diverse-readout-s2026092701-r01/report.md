# PAMF第三阶段：21台源辅助设备域均衡拒识

- run_id：`20261029-phase3-source-diverse-readout-s2026092701-r01`
- group_id：`pamf-phase3-source-diverse-readout`；类别：`cvs`；阶段：`Phase3-source-diverse-OE-domain-balanced-rejection`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；真实流水线 VERIFIED / ANALYZED_SOURCE_ONLY，NO_PROMOTION

## 目的与对照

冻结完整F6/源几何/100D同类支持描述符，新21设备6134OE按role内15域均衡配重，同101/1633读出各200CPU步；V-only固定风险阈值和同novel3150重复源开发。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

四阶段完整、无技术故障。linear未知宏拒识61.4286%、known宏FRR1.5630%、drop0.5778pp；MLP未知75.0794%、FRR1.5185%、drop0.5852pp，较consensus85.7460%下降10.6667pp。两个方法总体及每RX已知保护通过，MLP未知整体及RX3/4/6/8未达95%，eligible=[]、selected=null、NO_PROMOTION。MLPRX1/3/4/6/8未知99.5238/64.2857/87.1429/85.7143/38.7302%。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 版本与本地验证

实际 code/config commit：`70c8346339f5d2f5bcbe14aa8863eca545fa167a`。新候选34项测试和3个subtests通过，无warning；一次独立P0/P1审查PASS并同34项独立通过。源21数据已VALIDATED_ONCE6134，归档仅complete与public3文件，不传TXtruth/builder私有分配，N607显式root拼接取数。首步原L2条IQ真实F6 smoke，随后只OE6134前向、其余旧features复用。F/head/L几何冻结，consensus100D同前；新domain-balanced regime双标记，normalizer.6/.4、loss.5/.5、每role15域各1/15，200CPU步/101和1633参数。knownV仅.04FRR/.015正确拒绝风险阈值，novel在freeze后读；source重复开发非新确认，目标0。一次N607preflightPASS、源parent/权重/缓存在位、卡可用19–24GB；实际启动按容量新增export GPUworker1（每卡≤2），后三CPU，不停止其他进程。

## 实际完成与分析

GPU2一个export worker先真实L2 IQ smoke PASS，随后OE6134，共6136新F前向；旧L/V/novel特征复用。两个拒识器各200CPU步，linear损失.693147→.152570、MLP1.080209→.108692，400点有限且单调下降。F/head/geometry更新、原U_s/target读取、foreignstop均0。四完整日志、smoke、fit、calibration、predict、score、pipeline和独立poststate齐全；pipeline11.032秒。knownV固定4%宏FRR／1.5pp正确拒绝损伤校准全部通过，novel不入阈值选择。21设备＋domain weighting＋weighted normalizer为组合改动，不拆因；同knownV经验风险和同novel重复source开发，非新盲测／跨RX／星地／适应或注册成功。不调预算重跑。

[结果报告](E:/type10-7/docs/PAMF_PHASE3_SOURCE_DIVERSE_READOUT_RESULTS_20261029.md)；[完整机器摘要、400loss及证据入口](E:/type10-7/local_artifacts/pamf_phase3_source_diverse_readout_20261029_r01/analysis_summary.json)。下一方向为独立小预算source OE embedding微调与同预算CE对照，尚未实施。
