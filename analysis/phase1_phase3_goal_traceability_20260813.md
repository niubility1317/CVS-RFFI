# Phase1—Phase3目标域泛化与真实未知拒识追踪

日期：2026-08-13

目标来源：`E:\codex\home\attachments\c75febfd-60b9-42bb-9825-a0b3b9eda0bb\goal-objective.md`

协议权威：根目录`项目.md`（2026-08-07，`p2_min_v1`）

本表用于防止把技术工件完成、单节点目标确认或局部指标误写成Phase1晋级或Phase3完成。状态只依据当前Git实现、N607不可变工件及实验报告更新。

|ID|来源章节|可验收要求|目标文件／工件|状态|验证|备注|
|---|---|---|---|---|---|---|
|P1-TECH|目标一、2.6(1)、2.6(7)|12臂训练技术闭合，真实checkpoint可导出C描述器与G deployment bundle|`phase1_clic12_20260812_v5`；`phase1_clic_predictor_artifacts_20260812_v2`；`phase1_clic_g_bundles_20260812_v3_safe_pack`|verified|训练12／12 checkpoint+terminal；C6／6；G6／6 production verify/reload|只证明技术与部署工件，不证明性能|
|P1-SPLIT|目标2.1|每fold的source train／validation／proxy TX身份互斥，训练与阈值冻结只使用source|v5 checkpoint／terminal；clean v4；PAIR v3|verified|PAIR raw 14项逐SHA重开；proxy／source-V fit与threshold rows均0|target结果不得用于候选重排、阈值或重训|
|P1-SRC-CLEAN|目标2.3(1)、2.6(2)|同一checkpoint报告source-known clean的overall／macro／class／RX／day及floor|待建`phase1_clic_source_metrics_20260813_v1`|pending|需从clean v4不可变NPZ及checkpoint class-order重算并封存|必须区分source-L拟合行与source-V未拟合行|
|P1-SRC-LEO|目标2.3(1)、2.6(2)|同一checkpoint报告三种source LEO weak结果及逐scene／RX／class／day floor|待建source-V单物理样本单LEO观测缓存与source metrics receipt|pending|需在读取target结果前冻结source-V物理行、scene／seed分配和口径|现有source-L LEO只承担tail calibration，不能冒充held source-V DG|
|P1-SRC-PROXY|目标2.3(2)、2.6(3)|TX互斥fixed400 source-proxy unknown报告连续正向研发信号|PAIR v3中12份`proxy_diagnostic.json`|implemented|已封`AUROC_unknown`、`u_gap`、fit／threshold=0；尚待同一行汇总|只能是source研发信号，不能替代真实unknown|
|P1-TARGET-DATA|目标2.2|target-known／real-unknown使用同一规则的固定单观测LEO weak IQ，三scene物理ID互斥，TX集合互斥|`phase1_clic_target_confirmation_20260812_v2`|verified|3120行；三scene各1040；registered240／unknown800；production loader闭合|WiSig／LEO模拟只属于目标接收机／星地压力代理|
|P1-PREDICT|目标2.1|12臂预测先封存，零训练／适配／更新／选择，truth在预测期间未打开|`phase1_clic_target_prediction_20260812_v1`|verified|12／12 prediction，每份3120 forward；truth_sidecar_opened=false；fit/update/retry/selection=0|C／G共用同一IQ-only package|
|P1-TARGET-METRICS|目标2.3(3)(4)、2.4、2.5|独立truth-side scorer封存known DG、unknown rejection、open-set、scene／RX／class／day结果|`phase1_clic_target_metrics_20260812_v1`|verified|12／12receipt、6／6日志；每臂同一行known／unknown／open-set／DG指标；commit`ed26c9df`|不使用ADV；全部固定`ADV_COMPARISON_PENDING`|
|P1-U70|目标2.5、2.6(5)(6)|真实unknown显式拒识率global及clear／low-elev／rain分别≥0.70；defer不计分子；known拒绝按错|12份target metrics receipt|verified|12／12unknown gate均FAIL；global显式拒识率范围C=`0.0192–0.0800`、G=`0.0129–0.0612`，所有scene均低于0.70|有效性能失败；禁止调阈值、候选选择或重跑反馈|
|P1-ADV-CONFIG|目标2.4|ADV3B02与候选训练配置、known测试配置和指标定义逐字段等价；不要求同一capsule字节|待建6fold ADV baseline train／known config|pending|旧ADV原件为0.10／0.70／0.20与`tx_rx_day_1_7_2`，不能复用|必须新训，不得把旧checkpoint改名|
|P1-ADV-REF|目标2.4|每foldADV reference含三scene overall／macro／min、class、RX、day及class×RX／class×day交叉单元|待建ADV训练／评测／reference ingest入口|pending|现有combined scorer保持缺reference fail-closed|同foldC／G可共享一个配置等价reference，但比较各自local4|
|P1-NI|目标2.4、2.6(4)|候选在overall、macro、三scene、min-RX、min-class及预注册细分上逐项严格不弱于ADV|12份combined score receipt|pending|只有合法ADV reference闭合后才能评分|平均提升不可补偿任何关键slice下降|
|P1-ROW|目标2.3、2.6|每个冻结checkpoint同一行同时报告source clean／LEO、source proxy、target known DG、target real-unknown|Phase1完成报告|pending|需合并同一candidate的source与target不可变receipt，不得拼极值|ADV comparison可用配置等价不同target包|
|P1-VERDICT|目标2.6、九|七项不可补偿门逐项审计；仅全过才声明单节点目标确认完成|本追踪表及Phase1完成报告|pending|当前不可作晋级结论|即使通过也不是Phase3协同／运营unknown生命周期|
|P3-LOCAL|目标三、四|每个接收节点先输出不可变`z_id/z_dom/q/d_class/e_unknown/p_local`和registered／unknown／defer|现有bundle／target prediction是单接收节点前置证据|implemented|C／G单节点prediction已封存|尚无多节点same-event证据|
|P3-MULTI|目标四、六、八|实现显式处理节点差异、缺失、冲突和相关性的多节点协同；`N_sat∈{1,2,3,4,5}`|待建Phase3协同方法、缓存和矩阵|pending|无当前Git运行工件|平均／投票／最高置信只能作为基线|
|P3-U95|目标5.1|协同unknown FAR≤5％、safe rejection≥95％，registered拒绝／defer按错|待建Phase3 truth-blind prediction与独立scorer|pending|无当前实验|不得用Phase1单节点70％门替代|
|P3-ADAPT|目标5.2|比较独立适应、共享平均、质量加权、完整协同；K10后old≥92％、floor≥88％|待建与Stage2-B／C合法support对接矩阵|pending|无当前同输入四路线证据|不得读取query真值或把unknown query转support|
|P3-ENTITY|目标5.3、5.4|跨节点／过境形成anonymous entity并结合外部证据输出可信确权与`registration_authorized`|待建事件关联和确权工件|pending|无当前实现／数据|不能输出真实运营身份完成声明|
|P3-HANDOFF|目标5.5|授权后重新采集独立K-shot support并交Stage2-C；历史unknown query不回写|待建授权／新split交接合同|pending|无当前实现|新support需新的`split_id`和`VALIDATED_ONCE`|
|P3-ABCD|目标七|同输入完成A原基座单节点、B新基座单节点、C原基座协同、D新基座协同及差分归因|待建A／B／C／D矩阵与报告|pending|无当前完整矩阵|不得只比较A与D|
|P3-CLAIM|目标六、九；`项目.md`7.3|非同步接收机数据只能称“多接收节点代理协同”|未来Phase3报告|pending|待数据事件绑定审计|不得称真实在轨同步多星验证|

## 当前最短执行顺序

1.`phase1_clic_target_metrics_20260812_v1`已完成12臂同一行target-known DG与真实unknown封存；12臂真实unknown门均失败，不调参、不重跑。
2.继续执行在读取target结果前已冻结的source-V clean／单观测LEO weak口径和6fold ADV3B02配置等价训练矩阵；不得依据本次target数值更改它们。
3.生成source四组receipt与ADV rich reference，运行原严格combined scorer，完成Phase1七项不可补偿门审计。
4.当前12臂已因U70门失败而不得晋级。后续若设计新候选，只能依据source侧信号，必须重新完整冻结后再做一次新的target确认，不能用本表排序或调阈值。
5.仅当未来某个Phase1候选完整通过全部门后，才能进入Phase3多接收节点代理协同、`N_sat=1..5`及A／B／C／D矩阵。
