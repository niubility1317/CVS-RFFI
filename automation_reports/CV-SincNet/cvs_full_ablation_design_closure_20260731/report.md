# CVS-RFFI全量消融设计闭环追踪报告

|字段|值|
|---|---|
|追踪ID|`cvs_full_ablation_design_closure_20260731`|
|时间|2026-07-31|
|operator|Codex主代理|
|设计来源|`CVS-RFFI_全部消融实验设计_Phase1_Phase2_20260728.md`|
|目标|逐项完成设计报告中的Phase1、Phase2和联合消融，在N607上使用8张GPU×每卡2进程执行正式GPU矩阵|
|当前总状态|`IN_PROGRESS / COMPLETED_MATRICES_ANALYZED / NEW_RELEASE_PAUSED_BY_USER`|

## 执行边界

- 复用已经完成且可证明完整的训练、prediction、score和deployment artifact。
- 不重新审计或重建已标记`VALIDATED_ONCE`且输入条件未改变的数据。
- 不要求不同启动批次的数据hash一致，不进行跨批次hash对齐。
- 新的N607正式矩阵仍须本地实现、测试、独立审查、Git提交和报告预登记后，由唯一发布代理使用普通`N607`账号发布。
- GPU正式矩阵固定使用8张GPU×每卡2个进程，共16个并发worker；不干预无关任务。
- T2内部arm只在T1同row结果证明对应模块整体作用稳定后执行；未触发的T2 arm记录为条件未满足，不用参数扫描挽救主张。

## 已封存矩阵事实

|矩阵|计划维度|技术终态|当前证据|
|---|---|---|---|
|Phase1 T1|6个arm×5个train seed=30次训练|`ARTIFACTS_COMPLETE / ANALYSIS_PENDING`；30/30成功，失败0|10条direct reuse、1条reexport、19条new train；逐行receipt有效，P0=0|
|Phase1标签率|新增14次训练，另复用ρ=0.10的5个`P1-FULL`行|`ARTIFACTS_COMPLETE / ANALYSIS_PENDING`；19/19逻辑闭合，失败0|14条新训练成功；ρ=0.10五条引用均由T1最终artifact证明闭合|
|Phase2 States|25行Stage2-A+300行Stage2-B=325行|`ARTIFACTS_COMPLETE / ANALYSIS_PENDING`；325/325，失败0|5个receiver、5个method/query seed；Stage2-B覆盖K1/K2/K5/K10|
|Stage2-C T1 screening|19个arm×75个identity=1425 logical；1350 physical|`ARTIFACTS_COMPLETE / READY_FOR_ANALYSIS`；1350/1350 physical、1425/1425 logical、失败0|5个receiver、3个method/support/query seed、1个new-class draw；K∈{1,2,5,10}，新类数∈{5,20}|

Stage2-C的19个arm为：`P2-FULL`、7个同权限基线、`P2-A0/P2-B0/P2-C3/P2-D0/P2-D1/P2-D2/P2-E0/P2-F0/P2-F1/P2-F2/P2-F3`。`P2-F3`与`P2-FULL`共享物理prediction，因此1425个logical row对应1350个physical row。

## 设计要求追踪

状态只使用`pending/implemented/verified/deferred/rejected/blocked`。`verified`表示已有与要求同范围的当前证据；技术artifact闭合不自动等于性能结论闭合。

|ID|设计章节|要求|目标实现/运行|状态|验证证据|备注|
|---|---|---|---|---|---|---|
|T0-01|9.1、13|单因素diff、参数匹配、query不可达、K1/K2 fallback、逐样本argmax、量化无FP32 sidecar、prediction/scorer分离、paired manifest|每个新release的本地测试与独立审查|implemented|Stage2-C v5相邻回归180 passed、core 109 passed、真实smoke 3/3，独立审查P0=0/P1=0|只证明已发布T1实现；后续新arm仍须重复对应窄测试|
|P1-T1|6.1、9.2、14.1|`P1-FULL/P1-SUP/P1-A0/P1-B0/P1-C0/P1-D0`，6×5=30次完整训练|既有Phase1 T1 v5 reuse run|verified|30/30完成，30成功，失败0；逐行receipt有效，P0=0|性能同row表待分析|
|P1-LABEL|4.1、6.7、9.4、14.2|ρ至少覆盖0.01、0.05、0.10；设计全网格为5个标签率|既有Phase1 label v2+ρ=0.10引用|verified|14/14新训练成功；ρ=0.10五条引用闭合；19/19逻辑行完整|ρ=0.01使用5 seed；ρ=0.05当前3 seed；ρ=0.10使用5 seed|
|P1-A|6.2、9.3|`P1-A1–A13`内部表征消融|T1触发后冻结I/S/D矩阵|pending|缺少正式矩阵与结果|A级子arm为条件执行|
|P1-B|6.3、9.3|`P1-B1–B14`伪标签内部消融，B1必须matched coverage|T1触发后冻结I/S/D矩阵|pending|缺少正式矩阵与结果|不得用历史不同模型替代|
|P1-C|6.4、9.3|`P1-C1–C14`角几何与尾部风险内部消融|T1触发后冻结I/S/D矩阵|pending|缺少正式矩阵与结果|需保留C1→C2→FULL累加链|
|P1-D|6.5、9.3|`P1-D1–D13`反事实外推内部消融|T1触发后冻结I/S/D矩阵|pending|缺少正式矩阵与结果|D4为receiver挑战×LEO压力2×2|
|P1-H|6.5.1、9.3|`P1-H0–H6`source侧信道机制拆解|冻结source训练压力矩阵|pending|缺少正式矩阵与结果|不在同一target IQ上制造额外正式观测|
|P1-E|6.6、9.3/9.4|`P1-E0–E6`稳定器、调度和权重敏感性|T1触发后冻结I/S/D矩阵|pending|缺少正式矩阵与结果||
|P1-BASE|6.8|5类Phase1同权限基线|同split、seed、参数预算的正式矩阵|pending|当前T1仅含`P1-SUP`和`P1-A0`等因果arm|历史异质结果不能代替|
|P2-STATES|7.1、14.3–14.4|Stage2-A零标签和Stage2-B旧类适配，K1/K2/K5/K10|既有Phase2 States v14|verified|325/325 artifact完整、失败0；5receiver×5seed|性能同row表仍待分析|
|P2-BASE|7.2、9.2、14.5|7个同权限Stage2-C基线|既有Stage2-C T1 v5|verified|7个基线各75个logical row，完整矩阵artifact闭合|当前为screening证据|
|P2-A-T1|7.3、9.2|联合特征整体消融`P2-A0`|既有Stage2-C T1 v5|verified|75/75 logical artifact闭合|性能作用待same-row分析|
|P2-A-T2|7.3、9.3/9.4|`P2-A1–A8`内部与βaux敏感性|T1触发后冻结矩阵|pending|缺少正式矩阵与结果||
|P2-B-T1|7.4、9.2|稳健中心整体消融`P2-B0`|既有Stage2-C T1 v5|verified|75/75 logical artifact闭合|性能作用待same-row分析|
|P2-B-T2|7.4、9.3/9.4|`P2-B1–B11`稳健形式、谱秩和量化先验消融|T1触发后冻结矩阵|pending|缺少正式矩阵与结果||
|P2-C-T1|7.5、9.2|任务均衡对照`P2-C3`|既有Stage2-C T1 v5|verified|75/75 logical artifact闭合|需与完整`P2-C4`同row比较|
|P2-C-T2|7.5、9.3/9.4|`P2-C0–C2/C4–C10`协方差内部与权重敏感性|T1触发后冻结矩阵|pending|缺少正式矩阵与结果||
|P2-D-T1|7.6、9.2|`P2-D0/D1/D2`整体几何与固定融合|既有Stage2-C T1 v5|verified|每个arm 75/75 logical artifact闭合|需与完整D5同row比较|
|P2-D-T2|7.6、9.3/9.4|`P2-D3–D8`可靠性、cross-fit和标尺内部消融|T1触发后冻结矩阵|pending|缺少正式矩阵与结果||
|P2-E-T1|7.7、9.2|关闭Fisher residual的`P2-E0`|既有Stage2-C T1 v5|verified|75/75 logical artifact闭合|性能作用待same-row分析|
|P2-E-T2|7.7、9.3/9.4|`P2-E1–E9`Fisher门控与秩/增益消融|T1触发后冻结矩阵|pending|缺少正式矩阵与结果|E1/E2/E4/E5/E7必须标记诊断|
|P2-F-MAIN|7.8、9.2、14.8|`P2-F0–F3`的FP32/FP16/单层INT8/双层INT8|既有Stage2-C T1 v5|verified|四个logical arm各75/75闭合|资源—精度同row表和目标硬件时延仍待汇总|
|P2-F-T2|7.8、9.3/9.4|`P2-F4–F8`量化粒度、截距和整数kernel|T1触发后冻结矩阵并执行resource profiler|pending|缺少正式矩阵与结果||
|P2-K|7.9|K1/K2精确fallback，K5/K10模块激活|Stage2-C T1+States|implemented|矩阵覆盖K1/K2/K5/K10|逐logit fallback闭合与性能解释仍待统一审计|
|P2-G|7.10、9.4、14.9|一次15类、5+5+5三session、至少3种到达顺序、持久状态、增量更新、rollback和receiver切换|连续注册正式矩阵|pending|尚无设计对应正式run||
|P2-R|7.11、9.5|`P2-R0–R6`鲁棒性和安全补充|独立诊断manifest/capsule或分层结果|pending|尚未建立完整设计对应矩阵|改变support/IQ的诊断不混入主排名|
|JOINT-2X2|8.1、14.7|`P1-SUP/P1-FULL`×`P2-PROTO/P2-FULL`四cell|各Phase1模型使用自身合法bundle|pending|尚无四cell同矩阵正式证据||
|JOINT-P1|8.2|`P1-A0/B0/C0/D0`各自bundle进入固定`P2-FULL`|4个上游bundle×同一Phase2 screening|pending|尚无完整正式证据||
|JOINT-P2|8.3|所有Phase2核心消融使用同一fresh-confirmed`P1-FULL`bundle|confirmation矩阵|implemented|现有screening统一使用同一上游bundle|fresh confirmation未完成|
|CONFIRM|4.3、9.2、14.6|核心贡献arm使用5个fresh seed、至少3个new-class draw完成confirmation|根据T1同row结果冻结核心arm confirmation|pending|当前Stage2-C只有3个seed和1个draw||
|STATS|5.3、10、14.10|paired CI、per-receiver、per-class、per-scenario、失败/fallback、资源与多重比较|全量same-row分析和表格生成器|pending|现有artifact已闭合但尚未完整分析||
|ARTIFACT|11|每run保存设计列出的配置、seed、预测、评分、逐类、fallback、量化和资源字段|统一artifact终态审计|pending|现有T1已证明核心prediction/score/receipt闭合|尚未逐项证明全部资源与逐类字段|

## 当前缺口判定

现有执行已经证明Phase2 States和Stage2-C screening的技术artifact完整，但整份设计尚未完成。下一步必须先读取完整T1同row性能artifact，按设计预登记规则判断哪些I级内部模块被触发；该分析不得按单receiver、单seed、单K或单draw挑选arm。随后同时准备：

1. 被T1稳定作用触发的Phase1/Phase2内部arm；
2. 核心贡献arm的5 fresh seed×至少3 new-class draw confirmation；
3. Phase1×Phase2最小2×2与上游模块传递；
4. 5+5+5连续注册、3种到达顺序及rollback；
5. 量化粒度与完整目标硬件resource profiler。

## N607发布模板

|字段|冻结要求|
|---|---|
|Conda/Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|工作目录|N607普通账号下的CV-SincNet release目录|
|GPU|GPU0–7，每卡slot0–1，共16个worker|
|输出|fresh、不可覆盖的run/request/log/driver根|
|停止规则|仅P0协议/安全违规，或至少两个不同row在prediction前产生相同确定性异常指纹|
|成功标准|冻结矩阵全部row终态、prediction、score、completion和runner summary闭合，失败0、P0=0|
|监控|启动、首row/首wave、25%/50%/75%/100%或真实异常；不高频轮询|

## 2026-07-31已完成矩阵分析结论

|矩阵|闭合状态|主要结论|设计判定|
|---|---|---|---|
|Phase1 T1|30/30成功|A/B/C未形成稳定独立收益；D对LEO稳定性贡献明确|仅D达到后续触发证据|
|Phase1标签率|19/19逻辑闭合|rho从0.005提高到0.10时总体、严格、最差类和卫星指标单调改善|标签率是强主效应|
|Phase2 States|325/325成功|原型是主体；对角度量仅提供小增益；K≥2后状态更新收益明显|状态矩阵分析完成|
|Stage2-C|1350/1350物理、1425/1425逻辑闭合|A为决定性模块；B无独立贡献；C/E小效应；D混杂；F仅证明存储压缩|筛选分析完成，不能外推为确认性结论|

用户已明确要求暂停发布新实验。后续设计缺口和确认性矩阵保留为未完成项，但在收到新的明确指令前，不执行同步、发布或启动。
