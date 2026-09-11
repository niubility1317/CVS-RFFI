# CORE90博弈优化V2修复与实现计划

> 已完成本轮本地实施验收：本文件保留计划交付时的勾选状态，当前实现、验证及未完成项以[实施与验收记录](CORE90_GAME_V2_IMPLEMENTATION_ACCEPTANCE_20260911.md)为准。

> 执行说明：本文件是计划交付，尚未实施下列代码改动或发布新实验。后续按任务依赖逐项执行，可使用executing-plans工作流；是否派生Agent按当时授权与运行环境判断，不设固定角色数或重复审查。

**目标：**先建立可信的测量和无副作用负对照，再降低B8计算冗余，最后用小型配对实验区分求解器、对抗交互与课程暴露的贡献。

**架构：**保留现有CORE90网络、目标和事务求解器，以独立证据对象拆开经验优化滞后、跨TX域读出和身份能力；控制器只消费满足各自语义的证据。B8先做首轮head-only梯度计算，再独立评估图复用；课程直接控制实际场景混合，不再用三桶映射掩盖level变化。

**技术环境：**Python/PyTorch；本地ssr-gpu为PyTorch2.10+cu128，既有N607为2.1.0+cu121，执行时复核环境。Windows原生Python/powershell.exe，pwsh及WSL转接保持禁用。

**设计依据：**[用户设计报告原文](design_refs/CORE90_GAME_V2_DESIGN_REVIEW_20260911.txt)，原附件`4a3d0e47-87b9-4d7a-a153-0d0641d647f0/pasted-text.txt`；[本轮综合证据报告](CORE90_OPTIMIZATION_REPORT_392005_20260911.md)。基线Git提交`4c892d90`，正式旧训练版本`300ff4a9`。

## 1.先明确哪些结论不能写进新方法假设

B8/B5确实改变了训练，并出现单seed收益；这不证明恢复探针可靠，也不证明控制器识别了有害旋转。普通对抗B1当前不如关闭身份对抗B2，不能预设“原对抗已经有效”。C2困难LEO暴露不足是现行课程的实际结果，不能把它当作与机制无关的外部借口。

新版本不能以追赶次数必须大于0、LEO必须超过B8、课程必须到最高级作为工程通过条件。测量可能合法地报告无法确定；可靠信号仍未触发动作，是允许的实验结论。拒绝把未知状态改成健康，只为让矩阵跑出预期故事。

本计划严格落实设计的估计对象分离、状态隔离、负对照、计算优化、课程和因子实验要求。设计未规定的探针预算、连续映射和种子数在下文明确为本计划的实现选择；它们尚未被证明有效。B8图复用不预先声明等价或加速，隐式响应保留原有条件近似命名。

## 2.全局约束与权限

- 当前科学权限以本机`项目.md`为准。source RX=1/3/4/6/8、day=1/2/3，6个TX，15类RX×day；L/U/V=6300/56700/27000，标注率以L/(L+U)计为0.1。保持数据契约，不因为方法或报告变化重复验证数据。
- 所有新正式训练从本次契约scratch开始；不继承旧checkpoint、teacher、EMA或目标接触的选模状态。工程回放可使用合规源域诊断快照，必须标为功能验证，不能混入新正式模型来源。
- 所有probe拟合仅使用合法L_s及其派生特征；U隐藏TX，V不反传或更新持久状态，target不进入控制、调参、课程及阈值选择。
- source审计子集是训练池的观测视图，不新建V_cal/V_select，也不改变builder角色。TX×RX×day容器不能随机拆窗后冒充独立采集验证。
- 本轮不扩展卫星/U直接身份域对抗，不新增融合分支，不改变E80/E131阶段或其他loss权重；这些会改变研究问题，不能混入“等价修复”。
- 已观察的target结果只作回顾性分析。未来新方法可在合法source上选择；若继续评估同一target，标为已接触基准复核。全新确认集及其协议属于独立设计，不能凭改seed消除目标接触历史。
- 本轮仅写计划。后续正式训练默认E200；先实施和短程验证，再决定已授权范围内的派发。多seed建议不等于当前已获新增seed授权。默认每GPU最多2个训练实验，过去针对旧run的超额许可不自动沿用。
- 保留原日志/权重/分数及无关修改；本工作树已有`code/scripts/report_core90_target_test.py`未提交改动，本计划不接管。只提交本计划及设计原文，不合并PR。

## 3.现有缺口与实施优先级

|优先级|已核对位置|问题|修复目的|
|---|---|---|---|
|P0|source_audit.py:fit_probe/SourceAuditor.run|恢复monitor严重恶化仍可valid，负差截成0|未知不能进入健康/稀疏化路径|
|P0|data.py:audit_indices；runtime.py:audit|跨TX恢复被用于解释优化滞后|分开经验优化空间与跨TX泛化|
|P0|runtime.py:audit、apply_response_tracking调用|直接使用`[:32]`，覆盖偏置|显式覆盖全部源RX/day和多个TX|
|P0|runtime.py:calibrate；controller.py:decide|probe/capability共用valid，失败可影响两条链|独立有效性、独立定标与时钟|
|P0|state.py/solvers.py/runtime.py|长程无动作对照存在差异，根因未知|定位首个差异，不先归因随机噪声|
|P1|solvers.py:GameSolver.step|B8首轮计算全部梯度，仅使用head部分|先省无用梯度，再研究图复用|
|P1|step_context.py:satellite_stage；curriculum.py:update|level改变但输入三桶不变，仍失效缓存|每次有效升级对应实际训练分布改变|
|P1|capability.py及runtime审计|只用固定晴空特征余弦决定难度|加入当前/下一难度的身份风险|
|P1|matrix/replay/analysis|强对照、严格预算、TX弱项不足|建立可证伪的收益归因|

## 4.任务A：证据对象与探针估计对象分离

修改：`code/cvsrffi/game_tracking/source_audit.py`、`data.py`、`runtime.py`、`config.py`。新增：`code/cvsrffi/game_tracking/audit_evidence.py`。测试：新增`code/tests/test_game_tracking_evidence_v2.py`，扩展`test_game_tracking_audit_control.py`。

- [ ] A1.定义版本化证据对象，删除决策对聚合valid的依赖。旧valid可以作为兼容展示字段，旧审计日志只能按v1读取，禁止静默作为v2控制输入。新增schema不兼容时拒绝恢复控制状态，不重解释旧阈值。

```python
# 计划接口；nullable指标用None，不能用0代替测量缺失。
EvidenceV2 = {
    'schema': 'game_audit_v2', 'observation_id': str,
    'step': int, 'encoder_version': int,
    'data_valid': bool, 'coverage_valid': bool,
    'lag': dict, 'cross_tx_readout': dict,
    'gradient': dict, 'capability': dict,
}
# lag至少包含：estimand、status、fit_before/after、monitor_before/after、
# gap_raw、gap_normalized、quality_pass、budget、trajectory、reason_codes。
# status枚举：OPTIMIZATION_FAILURE / BUDGET_INCONCLUSIVE /
# TRANSFER_FAILURE / RELIABLE_HIGH_GAP / RELIABLE_LOW_GAP / UNAVAILABLE。
```

- [ ] A2.新增优化滞后探针。冻结当前编码器，用覆盖全部6TX×5RX×3day的L_s样本形成固定头目标；当前每容器4条，共360条，是有界诊断子集，不声称全训练分布精确最优。按实际训练分布权重计算原头和恢复头的同目标CE差，记录`estimand=bounded_empirical_head_optimization_gap`。保持当前身份对抗使用的主视图、头尺度和forward模式语义；如果只能采用eval-clean代理，显式标`eval_clean_surrogate`，先只记录，不直接替代实际训练目标的触发证据。
- [ ] A3.经验gap在同一固定目标上评价，不虚构独立monitor。用同TX×RX×day内额外随机窗口作“独立验证”被禁止。没有独立重复容器时，承认外推不确定；经验低gap不等于全分布或全非线性头最优反应。
- [ ] A4.保留跨TX域读出为第二探针。现有3TX拟合/3TX监测可以保留，并按固定种子轮换互换两折；每折完整采集容器只在一侧。分别输出15域和5RX的linear/MLP读出，不将跨TX gap重命名为优化滞后。特征标准化如启用，仅在fit侧估计并固定；不能把原在线头直接套到未经对应变换的标准化特征上进行不公平比较。
- [ ] A5.恢复过程保存每步fit CE/梯度范数、step0及固定检查点monitor CE、特征范数分位数、head参数范数与logit尺度、优化器重置/复制模式、实际步数与停止原因。monitor只记录和质量判断，不用于反传或从多个恢复step择优；默认选固定终点或fit侧预定收敛规则。

source开发的有界候选：恢复lr={2e-4,2e-3}、步数={40,120}，保留旧lr=0.02/40步作为失稳诊断参考而非默认。只在固定source诊断面比较，选择规则先写入配置，不能运行中按目标分数切换。head的AdamW状态cold-reset与在线状态副本是两个不同估计设置，主方案先保留cold-reset，副本仅用于诊断其影响。

质量判断不能简单规定“CE必须提升才有效”：已经接近最优的解析/合成负例应允许低gap；出现未完成预算、非有限、fit不稳定则未知。拟合成功但跨TX恶化记TRANSFER_FAILURE，只否定跨TX读出，不抹除独立经验gap。只有固定预算/收敛诊断对低gap有支撑时才发出RELIABLE_LOW_GAP；所有阈值由source开发冻结，记录固定值与数据定标值的来源。

验收案例：近最优头→可靠低gap；可恢复欠拟合头→可靠高gap；高lr非有限/大幅失稳→OPTIMIZATION_FAILURE；预算不足→BUDGET_INCONCLUSIVE；fit下降而跨TX升高→两个估计对象分别记录。比较原头和恢复头使用完全相同样本、权重、尺度和模式。

## 5.任务B：代表性梯度、控制器与独立时钟

修改：`data.py`、`runtime.py`、`gradient_audit.py`、`controller.py`、`curriculum.py`、`budget.py`。测试：新增`code/tests/test_game_tracking_reference_coverage.py`，扩展`test_game_tracking_audit_control.py`和`test_game_tracking_resume.py`。

- [ ] B1.实现`balanced_reference_indices(records, samples_per_group, seed)`，输入只含合法L_s的TX/RX/day/容器/opaque ID，输出确定性索引及覆盖表。方向参考默认每个TX×RX×day容器1条，共90条，显存不足分块累加同一均值梯度；不能退回排序前32条。H1现有fit/monitor各45组，至少各1条共45条/侧，各自保持组隔离，不强求每侧同时具有6TX。
- [ ] B2.梯度诊断区分`TX_CE`、完整当前labeled非对抗目标、完整当前训练目标三种scope。先复用固定StepContext计算主视图/卫星/U各项对身份参数的贡献；U只用原伪标签与mask。因预算未计算完整目标时记UNAVAILABLE，不能把clean诊断写成完整目标或自动用于完整目标控制。
- [ ] B3.控制器仅从合法且新鲜的lag和gradient证据选择动作。失败恢复头不能参与方向校正，也不能触发健康稀疏化；数据无效/恢复未知时主训练普通更新，原因独立记录。capability即使有效也不能让坏lag变好；坏lag也不能让有效capability自动失效。

|证据状态|博弈动作|审计时钟|课程|
|---|---|---|---|
|高gap可信＋当前头目标可读＋身份保护通过|有限head追赶|动作后证据过期，按预算重新测量|独立能力链决定|
|低gap可信＋恢复方向可信且代表性通过|可申请一次EG校正|一次观测默认最多授权一次校正|独立能力链决定|
|优化失败/预算不足/方向覆盖不足|普通更新|保持基础间隔，预算不足明确记录|有效能力仍可推进|
|仅跨TX读出失败|不用于宣称低域信息|不因该失败稀疏化|不影响独立能力|
|身份测量无效或塌缩|禁止身份保护未通过的额外动作|保留诊断|保持当前课程|

- [ ] B4.原始一条审计可驱动11步动作簇；v2默认“一条独立观测只授权一次动作事件”，有限k步追赶视作一个事件，记录k个已提交step。事件完成后不能继续消费同一observation_id，重复调用、resume和超预算均不得重复提交。该改变属于新控制策略，不能称纯实现等价修复。审计间隔初始保留250步，失败不增至500；可靠健康的稀疏化先关闭，作为以后独立选项。过期数据不靠延长TTL变新鲜。
- [ ] B5.拆为`next_game_audit_step`与`next_capability_step`，初始各250步，分别保存观测、确认序列和校准。能力定标不再要求lag有效。定标未完成时博弈保持普通更新；课程只在自己的有效证据满足后变化。方向阈值0.5/0.3如仍保留必须标fixed，不能写成全部source自适应定标。

验收：构造失败恢复、极大方向差和低截断gap，仍不能CORRECT/稀疏化；同时独立有效能力可按自身条件升级。同一观测重复100次最多一次EG；过期/未来版本不能动作。预算耗尽是“没有执行”，不记为动作成功。控制与课程的连续/恢复日程逐步一致。

## 6.任务C：三组短程负对照先于新矩阵

修改：仅针对发现的具体副作用修改`state.py`、`solvers.py`、`runtime.py`、`step_context.py`；新增`code/tests/test_game_tracking_negative_controls.py`，复用`test_game_tracking_resume.py`、`test_game_tracking_integration.py`。

- [ ] C1.普通无审计vs只读审计；普通vs所有额外动作禁用的控制框架；完整保存同一初态和batch序列，逐步比较参数、BN、optimizer矩/step、EMA、原型、Python/NumPy/Torch CPU/CUDA RNG、增强generator、伪标签mask和数据消费位置。
- [ ] C2.B8零对抗退化测试：显式关闭所有依赖adv_head的训练贡献，并核对参数依赖，不能只看到`lambda_adv=0`就假设完整关闭。保留与普通组一致的前向/RNG与持久状态语义。验证实际身份参数更新及全部非头持久状态相同；无梯度头的weight decay/step处理也要相同。
- [ ] C3.在E1、E40→41、E79→80、E130→131和伪标签mask变化处运行固定上下文回放。阶段边界可以用合成受控状态作功能检查，不伪装成已经从E1正式训练到该epoch。另做真实source短程回放，完整记录首个分歧。

同平台确定性参考路径尽量逐位一致；GPU存在已识别的不确定算子时先以CPU/可确定路径定位，再预登记容差。建议float32梯度/参数`atol=1e-6, rtol=1e-5`为候选验收线，整数step、mask、RNG必须精确一致；不能看到失败后不断放宽容差。无动作差异未定位前，不把1个百分点差异当机制收益或“噪声标准差”。

## 7.任务D：B8等价提速与实际更新遥测

修改：`solvers.py`、`step_context.py`；必要时新增`code/cvsrffi/game_tracking/head_lookahead.py`，只封装B8特定路径。新增`code/tests/test_game_tracking_b8_fastpath.py`与`code/scripts/benchmark_core90_b8.py`。

- [ ] D1.保留当前B8为reference实现。第一阶段只将原点autograd请求限制为adv_head参数；虚拟预测仍执行原来的head-only裁剪、AdamW矩/weight decay语义，第二次完整前向及正式更新照旧。避免为了取head梯度改变当前clean+satellite拼接、Dropout、BN或强视图的随机消耗。
- [ ] D2.逐参数比较原始梯度、clip前后梯度、clip系数、虚拟头位移、正式AdamW参数位移、optimizer状态、BN/EMA/proto及伪标签选择。新增遥测包括各参数角色的update范数、online/predictor对抗梯度余弦、非对抗梯度变化、总前向和backward/HVP次数。数据默认低频记录，独立计量日志开销。
- [ ] D3.只有D1等价后，再做计算图复用路径：原点编码器图保留，虚拟头使用隔离参数副本，重新计算其对原特征的对抗反馈；替换原对抗项，复用不依赖头的目标。不在活跃图上原地修改/恢复在线参数，避免autograd版本污染。虚拟头构造detach，不引入展开高阶求导；正式虚拟头参数梯度按参数角色映射回在线头。
- [ ] D4.复用时必须保持头的Dropout掩码、共享BN统计、完整拼接批次语义、U强视图mask、原型状态以及FISHR等已有高阶项。若不满足，图复用记录为未通过，保留已等价的D1；不能删掉loss或切eval来获得速度。新增高阶项禁止，原目标已有高阶项不能被误删。
- [ ] D5.固定同设备、相同batch和E1/E80/E131目标，预热20步、计时100步×3次；CUDA同步，报告中位数/范围、峰值allocated/reserved、模型forward、backward及头计算。共享负载无法固定时记录UNKNOWN的纯算法墙钟归因，仍可报告算子计数。等价是硬条件，速度没有预设必达百分比；若无稳定节省或显存恶化，reference仍保留。

建议接口`GameSolver.step(..., predictor_grad_scope='all'|'head_only')`与`Core90Objective.prepare_reusable_graph(ctx)`；后者只在图复用通过时接入，不能让不支持的目标静默走近似路径。CLI新增`--game_b8_impl reference|head_grad_only|graph_reuse`，resolved_config、checkpoint及日志都保存实际实现值。

## 8.任务E：直接控制课程输入，并与测量解耦

修改：`curriculum.py`、`capability.py`、`step_context.py`、`runtime.py`、`config.py`。新增`code/tests/test_game_tracking_curriculum_v2.py`。

- [ ] E1.以连续场景混合替代原三桶level。为保持单轴可解释性，先固定各场景物理参数，仅改变混合与扰动概率。计划映射：`p=0.30+0.50*l`，`w_clear=1-2*l/3`，`w_low=w_rain=l/3`，`l∈[0,1]`。每次level变化会改变实际概率分布，l=1对应三场景等权/p=0.8。此为新课程定义，不声称等价旧C2。
- [ ] E2.场景由独立可恢复generator按混合分布采样，保留逐样本mask和真实扰动计数。保持原batch级场景生成接口也可实现混合，但日志必须区分配置概率、抽到的场景、实际mask和累计暴露；单个batch未抽到困难场景不是映射失效。
- [ ] E3.定义`effective_policy_signature=(p,w_clear,w_low,w_rain,physical_config_version)`；只有signature变化才增加课程版本、重置历史/缓存。内部计数和未导致有效policy变化的事件不能重置Optimistic。
- [ ] E4.能力读出保留fit days1/2、monitor day3的合法L_s组隔离；同一固定身份读出头评价clean、当前与下一候选难度的TX准确率/CE/margin、最弱source TX和RX×TX表现。拟合只在fit组；monitor仅观察。平均特征余弦保留为描述性辅证，不再单独决定升级。
- [ ] E5.升级要求身份不塌缩、下一难度的直接判别能力通过source预定进入门槛、连续3个新能力观测确认和250步冷却，每次level上限0.1。能力时钟独立于恢复probe；不能将缺失G_lag补0来绕过。可靠高lag可单独记录，但本阶段课程主规则不依赖lag，从而可单独检验“身份能力课程”。不以epoch强制升级凑困难暴露。

验收：level0→0.1使有效policy实际变化；固定generator续跑与resume一致；初始化余弦很高但身份接近随机不能升级；probe失败且身份能力可靠时课程可独立判断；最弱身份退化被记录，不用总体准确率掩盖。阈值仅由source开发冻结，不能由TX1/TX3目标错误比例反推新loss权重。

## 9.任务F：实验按问题拆开，不一次重开25行

以下是拟议矩阵与执行依赖，尚未发布。种子建议为392005/392006/392007，保持训练内配对；只有当前任务明确授权的seed可派发，新增seed未授权则完成配置并标记未派发。旧392005结果不能充当新代码的配对组。

### F0：工程与source诊断

任务A/B/C完成后，先运行有界source probe及负对照；D可在相同冻结上下文验证，无需先做完整E200。机制诊断允许使用明确标识的合成退化/失稳案例与合法source回放，不访问target。通过对象是数据/数值/等价性，不能要求一定测出有害滞后。

### F1：核心求解器×对抗因子矩阵

固定旧固定课程、相同数据与初态、普通共同超参数；审计先统一关闭以隔离求解器，独立只读评测方式相同。开启审计的机制子实验另列，不能只给部分row增加诊断副作用。

|新行|求解器|身份域对抗|研究目的|
|---|---|---|---|
|V2_A|普通|关闭|非对抗基线|
|V2_B|普通|原固定系数|普通对抗效应|
|V2_C|B8已验收实现|关闭|零对抗退化负对照|
|V2_D|B8已验收实现|原固定系数|头前瞻对抗交互|
|V2_E|完整EG|关闭|EG对非对抗目标的贡献|
|V2_F|完整EG|原固定系数|EG对抗交互|

每seed分别计算`(D-C)-(B-A)`及`(F-E)-(B-A)`，同时报D−B、D−强普通对照。建议3个配对seed，共18行；3seed仍为有限精度确认，完整列出每seed差值和均值/离散程度，不夸大统计确定性。如果C与A功能上等价，可以研究报告中说明冗余，但首轮保留为负对照。

强普通候选单独以合法source选择：lr={1e-4,2e-4,4e-4}，先只动lr；再研究head lr ratio={0.5,1,2}或固定补步，不做无边界笛卡尔积。候选规则、预算和选择指标先冻结，source选中的一个强普通配置应用到全部配对seed，不能每个target结果挑不同强对照。增加强对照最多3条正式确认行，主因子矩阵仍保持原固定参数。

### F2：修复控制器的因果对照

固定课程及普通主solver，比较无动作审计、仅追赶、仅校正、组合。只有测量可信后才比较控制策略收益；天然0动作保留为无激活结果，不强制触发。

严格时间表来自独立source donor；固定/随机/跨seed replay全部在recipient训练前冻结，动作类型和head k的多重集合相同。已有旧U缺陷donor不得使用。若没有额外seed授权或完整donor，严格对照标deferred，不偷偷用recipient未来日程。

“同动作数”和“同成本”拆成两种分析：真实状态控制会按recipient拒绝不合法动作，不保证与donor执行数量相同；必须报告请求/接受/拒绝数。要研究纯时间位置效应，另建预冻结动作日程的mechanistic schedule实验，各组统一安全规则；实际计数未匹配则不下严格等动作结论。等计算实验另固定训练步数、审计/probe预算及field/head/HVP总上限，记录预算截断；同计数仍不等于同墙钟，不能通过无意义空转制造同成本。

### F3：课程与暴露对照

先固定普通solver比较旧固定课程、v2身份能力课程，避免立即与控制器叠加。课程完成后，从独立source donor取得完整扰动访问日程；recipient在相同样本流上应用预冻结的均匀/随机重排。匹配实际被扰动样本访问次数及clear/low/rain分类计数，尽可能固定样本ID与信道随机实现；若只匹配期望概率，必须降格为期望暴露匹配。

自适应recipient的实际暴露依赖自身学习状态，不能预先保证与另一个seed严格相同。故分别报告“自适应端到端效果”和“冻结日程的顺序效应”，不混为一个因果结论。保留E80前BN暴露和E80后直接卫星CE暴露两套计数。初期不同时更改物理严重度、loss、epoch和课程规则。

## 10.任务G：评价、异常诊断与发布

修改：`code/scripts/analyze_core90_game.py`、`build_core90_game_matrix.py`、`build_core90_game_replay.py`及独立报告脚本；新增`code/scripts/analyze_core90_conditional_robustness.py`。测试扩展`test_game_tracking_analysis.py`、`test_game_tracking_matrix.py`和`test_game_tracking_replay.py`。

- [ ] G1.输出clean/三个LEO、Macro-F1、最弱RX、最弱TX、最弱RX×TX、完整混淆与预测直方图、训练墙钟/峰值显存/动作计数、probe质量分布与原因、effective-policy变化及实际暴露。所有指标按row/seed独立保留，不跨模型拼最佳项。
- [ ] G2.既有预测可作回顾性条件鲁棒性复核：按opaque ID对齐模型A/B的clean与LEO，取“两模型clean都对”的共同集合，只比较其中LEO错误转移率；同时报告集合规模、占比及各TX/RX覆盖。集合使用truth只能发生在固定预测后的独立分析，不回流模型；该条件指标不替代总体LEO表现。
- [ ] G3.先做B4完整日志首次塌缩定位和B7历史梯度预测价值诊断，必要时记录训练预测熵、类直方图、TX/adv梯度比、原始与乐观方向余弦及阶段边界。没有定位根因前不增加B4/B7正式修复矩阵。
- [ ] G4.H1仅修复共同参考覆盖和质量状态，不扩展隐式模块；J1继续为局部诊断，记录无效次数与原因，不作为性能增强模块。当前阶段不提高其预算以追求更好结果。
- [ ] G5.后续获得执行授权后，本地聚焦测试→提交/push及远端OID读回→新release→一次真实source完整目标smoke→唯一owner派发；核实PID/PPID/CWD/argv/GPU UUID、scratch和实际模块动作。独立P0/P1实验审查按现有八项流程执行，不增加其他流程门槛。
- [ ] G6.新run保留E200权重、配置、完整日志和四场景固定预测；预测完整后独立评分，分别标技术完成、测量有效、实际激活和性能结论。测试目标接触历史写入run报告。无动作、无提升和失败产物均保留。

## 11.设计追踪清单

表中状态指后续实现完成度，不是计划文档完成度。G表示`code/cvsrffi/game_tracking/`，T表示`code/tests/`。

|ID|设计章节|要求|目标文件/任务|状态|验收证据|
|---|---|---|---|---|---|
|R01|§2、§10.6|契约、来源与目标接触边界|config/data/runtime；§2|pending|scratch负测、配置与来源读回|
|R02|§5.1/5.2、§10.1|失败与低滞后分离|G/audit_evidence、source_audit；A1|pending|失稳案例不输出健康|
|R03|§5.3、§10.2|经验优化滞后估计对象|G/source_audit/data；A2/A3|pending|同目标对比和scope记录|
|R04|§5.3、§10.2|跨TX域/RX读出单列|G/source_audit/data；A4|pending|跨折组不交、分任务输出|
|R05|§5.4、§10.2|替换方向/H1的前32条|G/data/runtime；B1|pending|15域、多个TX及H1两侧覆盖|
|R06|§5.5、§10.1|完整probe优化与尺度记录|G/source_audit；A5|pending|fit/monitor/尺度/停止原因齐全|
|R07|§6.2、§10.1|坏恢复不校正/不稀疏化|G/controller/runtime；B3|pending|失败恢复负测|
|R08|§6.3、§10.5|时钟/有效性拆分与动作去重|G/runtime/controller/curriculum；B4/B5|pending|单观测一次动作、恢复一致|
|R09|§6.4、§10.3|审计/无动作等价|G/state/runtime；C1/C3|pending|首分歧定位、状态逐步比对|
|R10|§10.3|B8零对抗退化|G/solvers/step_context；C2|pending|全部依赖关闭后非头更新一致|
|R11|§10.4|B8提速并保持语义|G/solvers/head_lookahead；D1/D3/D4|pending|reference逐步对齐；图复用单列|
|R12|§3.1、§10.4|原始梯度与实际更新/成本|G/solvers；benchmark；D2/D5|pending|裁剪/AdamW位移与受控成本|
|R13|§7.3、§10.5|有效课程映射与正确重置|G/curriculum/step_context；E1/E2/E3|pending|每次有效升级改变概率分布|
|R14|§7.2、§10.5|直接身份能力与独立时钟|G/capability/runtime；E4/E5|pending|随机身份不升级、probe失败解耦|
|R15|§7.4、§10.5|困难暴露匹配|matrix/replay/analysis；F3|pending|实际访问计数、先后顺序分开|
|R16|§2、§10.6|对抗×求解器及强对照|matrix；F1|pending|6行因子和配对交互差|
|R17|§9、§10.6|严格动作/计算/跨seed对照|replay/budget；F2|pending|合法donor及请求/接受计数|
|R18|§8.1/8.2/8.3|TX弱项、混淆、条件鲁棒性|analysis；G1/G2|pending|固定预测对齐、完整分组|
|R19|§9|B4/B7新增修复实验|G3后续实施|deferred|先定位塌缩/历史失效，不猜根因重跑|
|R20|§9、最终判断|H1/J1复杂度扩展|G4后续研究|deferred|先修共同测量；保留诊断/条件近似|
|R21|§10.3/10.6及项目流程|集成、聚焦验证、发布闭合|T及release/report；G5/G6|pending|本地证据、远端独立状态、固定评分|

实现状态计数：pending=19、verified=0、implemented=0、deferred=2、rejected=0、blocked=0。R17依赖donor与seed授权，作为未来实验依赖记录，不阻止A—E本地实现。最高风险是R03：把有界经验优化差重新包装成真实最优反应差；其次R11图复用在Dropout/BN/FISHR下未保持原事务语义。

## 12.执行顺序、检查命令与交付边界

顺序：A→B→C；C通过后D；A/B稳定后E；A—E局部验收后生成F1配置。F2依赖可信测量与完整donor，F3依赖有效课程与独立日程，均不要求先跑完整25行。G1/G2可基于固定产物独立实施。

每个任务先添加能复现具体缺陷的测试，再做最小实现，执行该任务相关测试；通过后进入下一任务，不机械重复完整测试集。以下为计划命令，当前尚未运行，因为对应新测试/实现还未创建：

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_game_tracking_evidence_v2.py code/tests/test_game_tracking_reference_coverage.py code/tests/test_game_tracking_audit_control.py -q
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_game_tracking_negative_controls.py code/tests/test_game_tracking_resume.py -q
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_game_tracking_b8_fastpath.py code/tests/test_game_tracking_solvers.py code/tests/test_game_tracking_field.py -q
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_game_tracking_curriculum_v2.py code/tests/test_game_tracking_replay.py code/tests/test_game_tracking_matrix.py -q
```

新benchmark脚本应提供`--implementation`、`--stage`、`--warmup 20`、`--steps 100`、`--repeats 3`参数，分别输出reference/head_grad_only/graph_reuse的相同上下文结果；支持失败即保留差异产物，不能根据失败自动改变算法配置。

计划交付本身只验证：设计原文UTF-8完整保存、21项覆盖、当前代码位置存在、链接可读、明确未实现项、Git显式提交及远端OID一致。它不等于上述实现已验收。预计交付分为测量修复、负对照、B8提速、课程、实验矩阵五个可独立审阅的变更组，性能目标与GPU时间在真实source microbenchmark后再估计，不依据旧共享GPU墙钟承诺加速比。

### 本次计划核对记录

已实际执行`git status -sb -uno`；用`rg -n`及完整函数读取核对source_audit、runtime、data、controller、curriculum、step_context和solvers；用`F:/App/miniconda3/python.exe -X utf8 -c`执行设计原文逐字节复制比对、21项ID唯一连续检查、状态计数和Markdown本地链接存在性检查，均通过。原文19577字符，计划正文含21项要求（19 pending、2 deferred），未运行计划中的新增pytest或任何新训练。提交前执行`git diff --cached --check`，发布后独立`git ls-remote`核对本地HEAD。
