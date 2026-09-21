> 2026-09-11后续修复：本报告保留历史审查状态；R1—R4及phase时序问题的当前结论见[修复与独立复核报告](CORE90_CROSS_RESPONSE_V2_REPAIR_20260911.md)。科学待完成项仍保留。

# CORE90交叉响应V2再次审查：设计一致性与实际启用

日期：2026-09-11。被审查代码：`6f46498633d0e2e236768d6656379b47ddcb5889`；依据用户设计附件第1—11节及[22项实现计划](CORE90_CROSS_RESPONSE_V2_IMPLEMENTATION_PLAN_20260911.md)。本次是代码复审和有界CPU反例，没有执行生产代码修复、正式训练、checkpoint加载、N607发布或target再评分。

**结论：当前不能按“严格按设计全部实现、相关机制均正确启用”验收。确认1项P1、2项P2；P0主体修复和优先U3基础路径已有证据，但源机制闭环仍缺接线，部分条件接口此前被过宽地称为“条件通过”。本报告收紧上一版验收结论。**

## 1.确认的问题

### R1／P1：三条件门未接入训练，冻结参数也不能从scratch开启身份响应

[integration.py](../code/cvsrffi/cross_response/integration.py)第237—240行的`joint_open`只读取V2的`mechanism_gate.last_result`；第242行的`observe_source_mechanism()`才会向该门提交证据。但真实训练只在[train_ssdg.py](../code/SSDG/train_ssdg.py)第11490行调用`evaluate_source()`，后者第535—551行保存V2源审计、更新旧`self.gate`，没有调用新门。全`code/`Python AST扫描找到0个`observe_source_mechanism()`调用；根目录tests的手工调用不算生产接线。

CPU反例配置了新门，同时令旧门打开，结果仍为`new_gate_last_result=null, joint_open=false`。因此当前U2/U4/U5从scratch正常训练只能保留域侧及辅助响应，不能实现预期的身份响应，更无法自动完成单模块扩层。

这与合理的“源阈值尚未冻结所以关闭”是两件事。缺失的是源必要性、真实基础更新反事实风险、困难源组/LEO保护的证据汇总与训练消费通路。[source_eval.py](../code/cvsrffi/cross_response/source_eval.py)第327—332行也明确把身份风险标为N/A。不能直接把响应MSE结果转换成三条件通过，或恢复旧门来绕过。

### R2／P2：增强反馈没有绑定当前有效联合目标

[scheduler.py](../code/cvsrffi/cross_response/scheduler.py)第59—79行只校验外置声明及三个子条件的`passed/sample_count/metrics`，不检查整体授权、稳定观察资格及当前源契约/响应目标/参数范围。[integration.py](../code/cvsrffi/cross_response/integration.py)第148—154行直接将该配置交给scheduler。

主Agent的CPU反例得到`gain_strategy=reliable_evidence, joint_open=false, mechanism_gate=null`，而增强概率审计已有1条。独立审查另验证：无响应的U1也可接受该配置；门结果显式`authorized=false`但三个子条件为true时仍生成增强概率。三个子条件曾通过，不代表稳定门已授权，更不能证明当前运行就是被验证的联合目标。

修复应使证据与同一源契约、同一联合目标和参数范围对应，检查最终授权，并明确门关闭期间的采样语义。不能仅补一个`joint_objective_validated=true`。本轮缺省`evidence_config=null`仍会拒绝真正构造U5_reliable；发现的是补入格式合法证据后的漏洞，并非当前默认已经偷偷开启。

### R3／P2：delta_pairs缺少类对产物时静默降为delta

[integration.py](../code/cvsrffi/cross_response/integration.py)第117—124行没有要求`delta_pairs`产物必须含competitors，第369行用`.get('competitors')`取值；[decision_calibration.py](../code/cvsrffi/cross_response/decision_calibration.py)将None合法解释为全部竞争类。

CPU反例提供同源契约、冻结noise和V验证字段，但不提供competitors：`U3_delta_pairs`运行时仍初始化成功，4类样本各启用全部3个竞争类。配置名声称关键类对保护，实际只执行delta约束。应在delta_pairs模式初始化时拒绝缺失/不合规的类对产物，保留delta模式允许None的语义。反例字段均为合成接口数据，不是合法真实源资格。

## 2.当前参数与开启状态

|机制/分支|当前配置与实际状态|判断|
|---|---|---|
|V2角色与反馈事务|36种角色；候选独立进度；collect后每步finish一次|已有修复及恢复/顺序不变证据|
|U1与U1_mask_off|无新增loss；相同完整块与角色，mask不同|有效组织对照；U0仅桥接|
|U3|vectorized；lambda_dec=0.01；基础delta=0；全部竞争类|已在既有合成真实CUDA入口产生决策梯度；尚未升级为源噪声/关键类对保护|
|Ux_normalized|独立候选，lambda_cross=0.01|单位范数loss有真实入口证据，不能据此宣称身份收益|
|head_only|lambda_resp=0.01；独立辅助optimizer/scaler|已有FP32/AMP/NaN/Inf隔离及恢复证据|
|U2/U4/U5身份响应|lambda_resp=0.01，gradient_cap=0.1；mechanism_gate=null|当前身份响应关闭；R1说明补阈值仍不足以接通|
|域侧响应|非head_only允许域参数接收响应；共享参数排除直接响应|符合设计允许的域侧诊断，不能把域梯度记成身份启用|
|V2必要性/等规模复用审计|source_audit_enabled=false；fit_steps/lr=null|默认不执行；预算须明确。启用后的合成入口已有证据，真实源检验未完成|
|U3_delta/U3_delta_pairs|decision_calibration=null|默认拒绝启动属合理未冻结；另有R3需修复，且缺真实源校准/验证产物生成闭环|
|U4_decomposed|response_decomposition=null；mechanism_gate=null|默认拒绝启动；分组autograd已测，真实身份分解路径受R1阻断|
|U5_reliable|evidence_config=null|默认构造scheduler时拒绝；增强策略还存在R2|
|方向收缩|direction_shrinkage=0；默认legacy_gain|方向历史已保存；默认概率仍用候选历史，不能称已启用可靠方向收缩采样|
|纯诊断|diagnostic_interval=20|训练loss未减频；强制审计目前只检测非有限z_id，不覆盖全部loss/梯度异常，V220仍有缺项|
|正式预算|E200、130+70、batch128、49step/epoch；data_seed392005|模型seed清单/独立确认边界未冻结；launch=false；不是已完成正式矩阵|

完整16个变体的配置解析结果见[机器可读反例](CORE90_CROSS_RESPONSE_V2_REAUDIT_20260911.json)。`ACCEPTED_CONFIG_ONLY`只表示一级validator接受；不代表后续runtime构造成功、条件满足或已启动训练。

## 3.22项逐条追踪

状态按完整要求重判。verified仅指列明工程要求已有实现证据，不表示真实科学收益；implemented表示已有代码但验收闭环不完整；pending表示存在明确未完成要求；deferred为授权范围之外的后续正式实验。统计：**10verified、6implemented、5pending、1deferred、0rejected、0blocked。**

|ID／设计章节|要求与目标文件|状态|验证与限制|
|---|---|---|---|
|V201／1、11|matrix/config组织对照|verified|同角色mask配对；U0桥接边界明确|
|V202／2.2—2.3|gradient_audit/replay共同参数与实际更新|implemented|有梯度审计/反事实接口；完整实际基础更新证据依赖V205，不能算全闭环|
|V203／3.2—3.3|source_eval/source_baselines必要性对照|implemented|匹配小头fit和双向打乱已测；默认关闭，真实源必要性未检验|
|V204／3.3|source_baselines等规模复用|verified|6TX与5RX单查询构造及互斥有合成入口证据；不声称真实源复用收益|
|V205／3.3、8.3|replay/source_update同状态实际基础更新|pending|现脚本仅合成IQ的CE+orth；省略domain CE/GRL/pseudo/EMA等，尚非完整基础一步|
|V206／4|roles/sampler/schema完整二分|verified|36角色独立推进、恢复及理论/实际候选全集区分|
|V207／4—5|coverage三类覆盖|verified|观测/查询/定向分开，不能用450观测冒充查询|
|V208／5.1|scheduler每步一次反馈|verified|顺序反例均0.025；一次flush、失败隔离、pending恢复|
|V209／5.2|scheduler方向历史与稀疏回退|verified|方向主键/计数/回退有测试；默认legacy_gain未消费增强方向估计|
|V210／5.3|coverage支持计数|verified|K2矩形8曝光；唯一记录/累计/任务数分开|
|V211／5.3、11|scheduler可靠新证据反馈|pending|R2：不能保证作用于获授权的同一有效联合目标|
|V212／6|tensor_ops单位范数几何|verified|逐记录归一后均值、尺度不变、近零mask；候选效果未证明|
|V213／7.3|decision_calibration源delta与关键类对|pending|R3及真实源产物流程缺项；不是全部实现完只差数字|
|V214／7.3|decision/gradient_audit详细证据|implemented|有效/正缺口/类对参数梯度已实现；真实源及LEO风险保护尚无闭环|
|V215／8.2|tensor_ops/training分解响应|implemented|能量/梯度分组单测通过；完整身份训练消费受R1阻断|
|V216／8.3|gate_v2/integration三条件与扩层|pending|R1：生产调用缺失；接口测试不能代替训练门启用|
|V217／8.4|source_update间接正交耦合|implemented|合成CE+orth探针已显示间接变化；完整基础项重放未完成|
|V218／9|replay/独立事务与历史分叉|pending|新隔离/真实短入口通过；历史首次分叉根因仍UNRESOLVED|
|V219／10.1|decision向量化等价|verified|既有CUDA FP32/FP16 loss、mask、梯度对齐及计时|
|V220／10.2|integration低频诊断|implemented|20步与训练频率分离；仅z_id非有限强制审计，异常覆盖未完整|
|V221／10.3|cache固定统计缓存|verified|只缓存原始固定统计，配置key失效及U1无统计证据成立|
|V222／11|matrix多seed独立确认|deferred|注册接口已有；正式seed/确认范围与训练未开展，本次复审不授权启动|

## 4.本次验证与后续顺序

本次重新读取原设计全文与计划，检查实际配置、调用图和路由，另由独立审查者只读核查R1/R2。主Agent运行`ssr-gpu/python.exe -X utf8 analysis/audit_cross_response_v2_reaudit.py analysis/CORE90_CROSS_RESPONSE_V2_REAUDIT_20260911.json`，退出0。脚本只使用临时合成源fixture和CPU路径，不更新模型、不读取真实checkpoint/target。[脚本](audit_cross_response_v2_reaudit.py)与JSON可复查。

未重复运行上一轮174项套件。角色、事务、CUDA入口和向量化等表述引用[既有原始证据](core90_v2_acceptance_evidence_20260911/index.json)，不称为本轮重测，也不以这些通过项抵消新发现。当前结论是**部分设计一致，未达严格完整实施**。

修复顺序应为：补完整基础更新的源反事实与证据汇总，接入新门并验证失败回退；修R2/R3的条件校验；补异常诊断；再在源侧明确冻结参数并逐分支证明非零实际梯度/概率消费。历史首次分叉继续保留未解决状态；不得为让计数变正而调大lambda/cap、恢复弱门或引用已有target结果。

本次只交付复审报告、可复现反例及旧验收勘误；上述生产缺陷尚未修复。没有把本次复审扩大为完整源实验或正式重跑。
