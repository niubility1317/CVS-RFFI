# Phase1—Phase3目标域泛化与真实未知拒识追踪

日期：2026-08-13

目标来源：`E:\codex\home\attachments\c75febfd-60b9-42bb-9825-a0b3b9eda0bb\goal-objective.md`

协议权威：根目录`项目.md`（2026-08-07，`p2_min_v1`）

本表用于防止把技术工件完成、单节点目标确认或局部指标误写成Phase1晋级或Phase3完成。状态只依据当前Git实现、N607不可变工件及实验报告更新。

|ID|来源章节|可验收要求|目标文件／工件|状态|验证|备注|
|---|---|---|---|---|---|---|
|P1-TECH|目标一、2.6(1)、2.6(7)|12臂训练技术闭合，真实checkpoint可导出C描述器与G deployment bundle|`phase1_clic12_20260812_v5`；`phase1_clic_predictor_artifacts_20260812_v2`；`phase1_clic_g_bundles_20260812_v3_safe_pack`|verified|训练12／12 checkpoint+terminal；C6／6；G6／6 production verify/reload|只证明技术与部署工件，不证明性能|
|P1-SPLIT|目标2.1|每fold的source train／validation／proxy TX身份互斥，训练与阈值冻结只使用source|v5 checkpoint／terminal；clean v4；PAIR v3|verified|PAIR raw 14项逐SHA重开；proxy／source-V fit与threshold rows均0|target结果不得用于候选重排、阈值或重训|
|P1-SRC-CLEAN|目标2.3(1)、2.6(2)|同一checkpoint报告source-known clean的overall／macro／class／RX／day及floor|`phase1_clic_source_metrics_20260816_v4`|verified|6fold×C／G clean同row指标已封存|技术闭合；六fold非补偿aggregate为false|
|P1-SRC-LEO|目标2.3(1)、2.6(2)|同一checkpoint报告三种source LEO weak结果及逐scene／RX／class／day floor|`phase1_clic_source_metrics_20260816_v4`|verified|三scene逐fold／arm指标完整；仅F1 verdict=true，F2—F6=false|source-known门有效失败|
|P1-SRC-PROXY|目标2.3(2)、2.6(3)|TX互斥fixed400 source-proxy unknown报告连续正向研发信号|`phase1_clic_source_metrics_20260816_v4`|verified|同row AUROC_unknown／u_gap及零fit／threshold闭合；仅F1 proxy gate=true|六foldaggregate失败；不能替代真实unknown|
|P1-TARGET-DATA|目标2.2|target-known／real-unknown使用同一规则的固定单观测LEO weak IQ，三scene物理ID互斥，TX集合互斥|`phase1_clic_target_confirmation_20260812_v2`|verified|3120行；三scene各1040；registered240／unknown800；production loader闭合|WiSig／LEO模拟只属于目标接收机／星地压力代理|
|P1-PREDICT|目标2.1|12臂预测先封存，零训练／适配／更新／选择，truth在预测期间未打开|`phase1_clic_target_prediction_20260812_v1`|verified|12／12 prediction，每份3120 forward；truth_sidecar_opened=false；fit/update/retry/selection=0|C／G共用同一IQ-only package|
|P1-TARGET-METRICS|目标2.3(3)(4)、2.4、2.5|独立truth-side scorer封存known DG、unknown rejection、open-set、scene／RX／class／day结果|`phase1_clic_target_metrics_20260812_v1`|verified|12／12receipt、6／6日志；每臂同一行known／unknown／open-set／DG指标；commit`ed26c9df`|不使用ADV；全部固定`ADV_COMPARISON_PENDING`|
|P1-U70|目标2.5、2.6(5)(6)|真实unknown显式拒识率global及clear／low-elev／rain分别≥0.70；defer不计分子；known拒绝按错|12份target metrics receipt|verified|12／12unknown gate均FAIL；global显式拒识率范围C=`0.0192–0.0800`、G=`0.0129–0.0612`，所有scene均低于0.70|有效性能失败；禁止调阈值、候选选择或重跑反馈|
|P1-ADV-CONFIG|目标2.4|ADV3B02与候选训练配置、known测试配置和指标定义逐字段等价；不要求同一capsule字节|ADV v1—v4技术smoke|stopped|v1—v4均在正式盲预测前技术停止；formal=0|不再追加修复；不能生成合法reference|
|P1-ADV-PREDICT|目标2.4|从精确baseline terminal tuple与checkpoint-bound WiSig物理轴封train-config；3120个opaque target row逐行一次forward，零fit／update／retry／selection|ADV v1—v4技术smoke|stopped|合法blind prediction=0|不是ADV性能失败|
|P1-ADV-REF|目标2.4|每foldADV reference含三scene overall／macro／min、class、RX、day及class×RX／class×day交叉单元|未生成|cannot_establish|blind prediction缺失，reference未构建|combined scorer保持fail-closed|
|P1-NI|目标2.4、2.6(4)|候选在overall、macro、三scene、min-RX、min-class及预注册细分上逐项严格不弱于ADV|未生成|cannot_establish|没有合法ADV reference或combined score|不得写成通过或性能失败|
|P1-ROW|目标2.3、2.6|每个冻结checkpoint同一行同时报告source clean／LEO、source proxy、target known DG、target real-unknown|`phase1_clic_final_gate_20260816_v1`|verified|source与target均按fold／arm同row汇总，无跨候选拼极值|ADV门单独标记CANNOT_ESTABLISH|
|P1-VERDICT|目标2.6、九|七项不可补偿门逐项审计；仅全过才声明单节点目标确认完成|本追踪表及`phase1_clic_final_gate_20260816_v1`|verified|source门FAIL、unknown70％门FAIL、ADV门CANNOT_ESTABLISH|`NOT_PROMOTED / PHASE1_GATE_FAIL`；Phase3仍pending|
|P3-LOCAL|目标三、四|每个接收节点先输出不可变`z_id/z_dom/q/d_class/e_unknown/p_local`和registered／unknown／defer|现有bundle／target prediction是单接收节点前置证据|implemented|C／G单节点prediction已封存|尚无多节点same-event证据|
|P3-MULTI|目标四、六、八|实现显式处理节点差异、缺失、冲突和相关性的多节点协同；`N_sat∈{1,2,3,4,5}`|待建Phase3协同方法、缓存和矩阵|pending|无当前Git运行工件|平均／投票／最高置信只能作为基线|
|P3-U95|目标5.1|协同unknown FAR≤5％、safe rejection≥95％，registered拒绝／defer按错|待建Phase3 truth-blind prediction与独立scorer|pending|无当前实验|不得用Phase1单节点70％门替代|
|P3-ADAPT|目标5.2|比较独立适应、共享平均、质量加权、完整协同；K10后old≥92％、floor≥88％|待建与Stage2-B／C合法support对接矩阵|pending|无当前同输入四路线证据|不得读取query真值或把unknown query转support|
|P3-ENTITY|目标5.3、5.4|跨节点／过境形成anonymous entity并结合外部证据输出可信确权与`registration_authorized`|待建事件关联和确权工件|pending|无当前实现／数据|不能输出真实运营身份完成声明|
|P3-HANDOFF|目标5.5|授权后重新采集独立K-shot support并交Stage2-C；历史unknown query不回写|待建授权／新split交接合同|pending|无当前实现|新support需新的`split_id`和`VALIDATED_ONCE`|
|P3-ABCD|目标七|同输入完成A原基座单节点、B新基座单节点、C原基座协同、D新基座协同及差分归因|待建A／B／C／D矩阵与报告|pending|无当前完整矩阵|不得只比较A与D|
|P3-CLAIM|目标六、九；`项目.md`7.3|非同步接收机数据只能称“多接收节点代理协同”|未来Phase3报告|pending|待数据事件绑定审计|不得称真实在轨同步多星验证|

## 当前最短执行顺序

1.`phase1_clic_final_gate_20260816_v1`已完成七门终裁：当前候选`NOT_PROMOTED / PHASE1_GATE_FAIL`。
2.本目标内停止ADV修复、阈值调整、候选选择与重跑；ADV非劣关系永久记为`CANNOT_ESTABLISH`。
3.未来若提出新候选，必须从新的source-only冻结实验开始；只有完整通过Phase1全部门后，才能进入Phase3多接收节点代理协同与A／B／C／D矩阵。
