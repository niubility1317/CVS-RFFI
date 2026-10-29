# PAMF Phase3：跨接收机星地无标签开放世界识别

2026-10-28 启动，2026-10-29 按用户最新要求更新指标。跨接收机 clean 未知拒识至少95%，跨接收机叠加星地后至少90%，已知类接收率在这些约束下尽可能提高。默认N607，不停止或修改其他用户/owner任务；每张卡最多新增两个本任务训练或预测进程。本文同时承载目标prompt、设计和分步计划，不另设设计审批。

## 最新目标修订（2026-10-29，用户明确指定）

用户明确“95%拒识正确率是指跨接收机的，同时叠加星地的跨接收机拒识正确率在90%以上，还有已知类接收率越高越好”。据此分开验收 clean 跨RX≥95% 与星地跨RX≥90%，后者各预登记主场景也≥90%；不再把星地95%作为当前硬门槛。已知类接收率定义为1−宏误拒率，在未知拒识约束下尽可能提高，保留宏误拒≤5%和同输入原F闭集到开放准确率损失≤2pp保护；错误接受为另一旧类不算成功识别。未来持续未知也按相应输入条件采用95%/90%。历史run的原预登记和NO_PROMOTION结论不追溯修改。机读口径见`configs/pamf_phase3_targets_20261029.json`。总目标仍为完整拒识、零目标标签适应、发现及未来匿名注册，未宣称完成。

## 目标prompt

在当前三视图PAMF/Tri-SIG-TIME SSL基础上，完成跨接收机开放世界RFFI：clean条件未知TX宏拒识≥95%，叠加practical residual星地后总体及每主场景≥90%；在这些约束下最大化已知类接收率，同时保留FRR≤5%及同输入原F闭集到开放损失≤2pp。依次实现冻结模型拒识、零目标TX标签适应、未知类别发现和匿名持续注册；适应须有旧类正收益，注册须在未来独立观测新类宏识别≥80%、旧类较注册前损失≤2pp、后续未知维持相应95%/90%拒识。目标端不提供TX标签、known/unknown角色、未知类别数或类别配额；允许的无标签发现数据必须与最终query物理互斥。真实身份命名/业务授权与匿名设备注册分开。预测先固定、独立truth-last评分；不根据目标成绩调阈值、选模或选择性重跑。保留Phase1/2原产物，先做单模型seed小规模可证伪实验，源侧冻结规则且达到科学门槛后扩展3个模型seed确认。指标未达时保留负结果，不擅自降低门槛或宣称完成。

## 验收口径

|任务|量化目标|评价限制|
|---|---|---|
|未知拒识|按真实未知TX等权，clean跨RX≥95%；星地跨RX总体及各预登记主场景≥90%|unknown仅在scorer连接；不能用defer冒充成功|
|旧类接收与保护|最大化已注册类宏接收率；宏误拒≤5%；含误拒的正确率相对相同IQ/default-head闭集基线下降≤2个百分点|接收率与正确识别分开，同时报告逐类、RX、scene及95%区间|
|无标签适应|保持上述拒识/旧类约束，并报告同query旧类正收益|不适应、普通伪标签、可靠旧类筛选作明确控制；未证实则不得声称有效|
|匿名注册|未来独立观测新类宏平均识别≥80%；旧类相对注册前下降≤2个百分点|不在建簇样本上回测；真实TX映射只可评分端计算|
|持续开放|后续未注册设备clean跨RX拒识≥95%、星地跨RX拒识≥90%|注册后的新unknown集合独立；错误合并/拆分/重复注册分别报告|

目标针对点估计；95%统计区间另列，不将“95%拒识”写成“95%置信下界”。纯噪声/证据不足可暂缓，但全部已注册类暂缓/拒识均计错误。缺失已知或未知的组返回不适用，不能判成功。

## 数据与方法设计

默认仍为跨RX泛化：source RX与target RX不交。单星被动监听是首轮主场景，多节点协同不是前置条件。分别评价clean跨RX及固定practical residual/post_sync/noEQ/25MHz星地跨RX；每记录仍仅一份固定practical观测，known与unknown使用同一物理规则。clean为独立条件验收，不作为星地算法可读的逐条参考或监督。源训练保持当前合法旧6TX契约；严格未知TX必须未进入其监督/无标签/教师/恢复器祖先。曾无标签见过但未注册设备单列，不混称严格未见。

新Phase3独立划分source L/U/V、target无标签适应/发现池、future query及后续未注册设备。历史Phase1/2 query不转为训练/注册。首轮只使用冻结PAMF，不加载DURA，不更换或另训Phase1默认头。源类原型仅由L_s形成，V只用于冻结阈值。部署预测进程不读取source IQ/源loader、truth、角色或真实类别构成；源侧校准与truth侧评分分为独立命令。

首轮冻结比较MSP、maxlogit、margin、energy、合法L_s cosine中心距离；所有分数高值代表known，各自最终接受分数在同一source V上以固定99% retention校准一个全局阈值。初始公式/retention不得看target后改变；不按真实scene或TX路由阈值。所有拒识分数共享原default-head的闭集类别决策，隔离拒识贡献。融合/适配/注册属于后续独立候选，仅在合法source开发证据基础上固定并预登记。

## 分步与分工

1. 数据与基座核对：数据代理定位checkpoint全部祖先、旧/未知TX、RX/day/物理角色及未使用记录。主Agent决定最小可运行切片；只核对改变的数据事实，不重复验证既有capsule。
2. 拒识基线：主Agent实现source校准、truth-blind frozen predictor和独占run launcher。评分代理独立实现truth-last scorer，防止全拒绝、配额/角色泄漏和大样本二次内存。
3. 目标无标签适应：可靠已知候选与不确定/未知候选隔离；源锚点不可被target回写；先判决状态，再小adapter。每版独立run、同输入匹配比较。
4. 新类发现与注册：在独立无标签池建立候选簇，检查旧类污染、跨条件稳定与足够独立观测；以匿名ID及非参数代表注册，在future query验证。
5. 持续确认：新设备分轮到来，报告旧类保持、未来新类识别及后续unknown；达到预先门槛才扩展3seed，不提前铺满矩阵。
6. 一名独立审查者只做一次下次真实run所需P0/P1审查；主Agent整合、提交/push/OID读回、N607归档传输/编译/真实checkpoint无query smoke、启动后PID/CWD/argv/GPU/log增长核实及后续评分交付。

所有worker只拥有明确指定的新模块；共享工作区现有未提交/已暂存修改不回退、不混入提交。唯一launch owner为本聊天主Agent。

## 首轮实现计划

Tech stack：既有verified Python/PyTorch/NumPy；不安装新依赖。隔离在新包`experiments/pamf_phase3_open_world/`，不修改历史训练器。

- [x] `scores.py`：`knownness_scores(features, logits, prototypes)`及source-only quantile冻结；有限值、形状、类别置换一致性和组合阈值语义负测。
- [x] `data.py`：新单观测capsule的public received/opaque IDs与独立truth；验证source/target、discovery/query、场景物理互斥和无角色公开。输入复用现有合法构建器，禁止旧七视图loader改名冒充。
- [x] `runner.py`：源L原型/V阈值独立校准；严格加载三视图E50，批量导出无真值knownness/closed_prediction/accepted/prediction；全部文件写完才产生`ALL_PREDICTIONS_COMPLETE`。
- [x] `scoring.py`：ID一一对齐、closed baseline/known误拒/unknown拒识、逐TX/RX/scene、O(NlogN) AUROC/OSCR、完整预测后才连接truth；全拒绝/漏ID/重复ID/错shape/提前truth负测。
- [x] `launch.py`：不可覆盖root、每卡本run新增进程≤2、显存容量检查、单owner，不杀其他进程；固定技术停止条件。
- [x] `tests/test_pamf_phase3_*.py`：仅相关协议负测和算法行为测试；真实checkpoint no-query smoke为launcher第一步，PASS自动继续。
- [x] 按实际路径登记`20261028-phase3-pamf-open-set-s2026092701-r01`；实际矩阵在`experiment.json`维护，本文不伪造尚未核实的样本数/运行状态。

## 发布和停止

单一code/config commit；只stage相关文件，push当前upstream并比较远端OID。镜像保持相关新文件精确一致。N607一份不可覆盖release归档、一次本地/远端SHA比较、一次remote compile。每GPU本run最多两个训练/预测worker，其他用户进程不作为整卡禁用条件；启动前仍检查实际显存余量，优先较空闲卡，容量不足则排队。

仅停止经精确PID/CWD/cmdline/root验证属于本run的技术失败进程。协议越界、错误split/checkout、覆盖冲突、无法合法预测或两row相同pre-prediction异常属于技术停止；低拒识率或负收益只记录科学结果。可能已完成的远端动作先独立读回再决定是否重试。相同技术指纹经一次针对性修复仍复现则保留产物并请求必要信息。

## 当前状态

目标模式ACTIVE。首轮`20261028-phase3-pamf-open-set-s2026092701-r01`已在N607 GPU2完成真实模型smoke、source校准、1440条冻结预测与独立truth-last评分，产物状态VERIFIED/ANALYZED。五种简单拒识基线全部未达95%及旧类保护门槛；完整结果见`docs/PAMF_PHASE3_OPEN_SET_PILOT_20261028.md`。不按该target表调阈值/选方法，继续source-only类别/RX留出开发。新的目标无标签适应/发现池与后续query已经单独构建，历史query不转训练。其他用户进程保持，目标未完成。

Full eq下载已于2026-10-29由独立owner完成158PKL/8767867条全量验证，见docs/PAMF_PHASE2_FULL_EQ_VALIDATION_20261029.md；本Phase3复用该原始验证，不重扫20ZIP。RX `24-18`不在现有ManySig/ManyTx/ManyRx/SingleDay四子集RX列表内（四个header有界只读证据已保存），也与本轮Phase2原7RX隔离。新run`20261029-phase3-full-pool-rx24-r01`已完成VERIFIED/VALIDATED_ONCE：旧6加发现2类D1每TX每景32条无标签适应/发现（768条），D3上述8类加后续未知2类每TX每景64条query（1920条），总2688唯一物理接收记录、三景与两角色互斥。三景固定low suburban/mid urban/low urban压力分层，不当作实际被动监听概率。两public各仅3白名单文件，truth独立，零模型/预测/训练，正式Q未推理或评分；code commit e4ebaa2682d8c2e87182a8b8ce36ae5ea5eec4d9，详细数据句柄见`docs/PAMF_PHASE3_FULL_POOL_20261029.md`。source含D3地面数据，该切片不声明严格未来日期外推；不按Full index>=1000或容器前缀判断新记录，不改其他owner的下载/暂停Phase2任务。

source run`20261029-phase3-source-holdout-s2026092701-r01`已在N607 GPU2完成，VERIFIED/ANALYZED_SOURCE_ONLY。四类scratch232673参数、160维，TX0/1与RX8不进入训练，E20完整540步；source split392005、model seed2026092701。73项Phase3测试、一次本source候选独立P0/P1与真实metadata兼容检查通过。完整20epoch及87行stdout解析显示训练最终98.75%、已见RX V已知98.6528%、留出RX8为66.4722%；8规则全部未达95%推进，NO_PROMOTION，不扩三fold，详细见`docs/PAMF_PHASE3_SOURCE_HOLDOUT_RESULTS_20261029.md`。

物理分支run`20261029-phase3-source-branch-s2026092701-r01`也已VERIFIED/ANALYZED_SOURCE_ONLY，95项测试及一次新候选P0/P1通过；N607 GPU2零新增训练，一次导出30360条8分支，完整16规则均NO_PROMOTION。全V最高未知拒识3.378%、RX8最高7.278%，RX8最高AUROC0.7493，不从中挑正式方法。执行commit df34171b8dc9942ec236e4504cac61768d70997a，结果见`docs/PAMF_PHASE3_BRANCH_EVIDENCE_RESULTS_20261029.md`。无标签代理的合法输入和未来评分边界保持；下一步优先联合域偏移校正、unknown成簇和anonymous注册。正式无标签适应、未知分类与持续注册仍未完成，目标保持完整ACTIVE。

联合source run`20261029-phase3-source-joint-s2026092701-r01`已VERIFIED/ANALYZED_SOURCE_ONLY，执行commit6570e891982d72bc317c9219ead3e97221d93951。模型、数据、主Agent及review分工完成纯几何/源角色导出/四阶段流水线/独立truth-last评分；初始68项＋P1修复相关48项（23项未变）和独立4复核通过。仅D1R1可代表完整联合晋级，R0/D0控制不能替代注册/适应成功。真实2Lsmoke PASS，N607 GPU2只新增一条导出worker，零神经训练/原U_s/正式target读取；源代理U1050与未来V5400完整处理。D0/D1均首轮触及32组件上限、合格0，D1旧类开放准确率63.9167%较D0下降1.2222pp，后续未知拒识2.1111%，新类未来识别0，NO_PROMOTION；详细见`docs/PAMF_PHASE3_SOURCE_JOINT_RESULTS_20261029.md`。该失败不外推完整PAMF能力，不增cap/改阈值挽救旧run。

下一优先路线直接利用当前满意的完整六类PAMF，使用独立source开放开发数据，不再只依赖预算弱得多的4CE代理。元数据确认Full174类比四compact并集150多24类，其中7类在原source5RX×D1D2D3均≥70；按容量与字典序事前分为辅助暴露4类4200条、独立源新设备验证3类3150条，排除旧6和正式4unknown。科学边界已写入PROJECT_PROTOCOL，额外source已知/未知成员监督明确披露，原L/U/V和六类身份头不变；正式target仍零标签。数据run`20261029-phase3-source-oe-data-r01`已VERIFIED/VALIDATED_ONCE：7350唯一source记录、15成员、每TX/RX/day70、公开无辅助TX、17行完整日志无异常，原L/U/V/模型/target/channel均0，零GPU、不停止他人。6项测试/10subtests及一次独立P0/P1通过，执行commit795e4a9604f391cb53e19ba24c2d2e8d5f24188a，数据句柄见`docs/PAMF_PHASE3_SOURCE_OPEN_DATA_20261029.md`。

完整六类source拒识候选`20261029-phase3-source-open-readout-s2026092701-r01`已完成，详细在`docs/PAMF_PHASE3_SOURCE_OE_REJECTION_PLAN_20261029.md`及对应config。F与默认身份头冻结，8物理分支导出后由L拟合类置换不变的100维描述符；仅额外源成员监督训练101参数linear或1633参数MLP16，各200次CPU feature更新，known V仅99%阈值校准，辅助新设备3TX只在全部预测后独立评分。5静态控制不能推进OE候选，source门槛95%未知/5%误拒/2pp旧类保护含各源RX，双过时优先linear；这不是全Phase3验收。实际64项测试＋35个subtests与独立P0/P1通过，真实导出、200步读出训练及独立评分均已完成，科学NO_PROMOTION；实际结果见下节，正式RX24 U/Q仍未模型访问。

## 完整六类源OE拒识实际结果（2026-10-29）

`20261029-phase3-source-open-readout-s2026092701-r01`已VERIFIED/ANALYZED_SOURCE_ONLY，执行commit`76951634c0901175a6d78c88caffbf49fdce8d04`；4完整日志、真实2Lsmoke和四阶段产物齐全。当前完整6类F与身份头冻结，8分支完成40652条前向，两个源成员监督读出各完成200次CPU更新；原U_s／正式RX24 U/Q读取0，encoder／身份head更新0。linear_oe／mlp16_oe的3个辅助未知TX宏拒识68.4762%／76.6032%，known宏FRR均0.9963%，旧类开放宏准确率97.9148%／97.8852%（原closed98.4667%）。整体及全部sourceRX旧类保护通过，但unknown整体和RX3/4/6/8未达95%，仅RX1通过；eligible空、selected=null、NO_PROMOTION。详情见[源OE结果报告](E:/type10-7/docs/PAMF_PHASE3_SOURCE_OE_REJECTION_RESULTS_20261029.md)。这是source clean已见RX的新TX开发，非跨RX／星地／无标签适应／注册成功；本run不调参重跑、不扩seed、不访问正式池。后续仍需独立source候选，当前无新实验启动；总目标仍未完成。

## 下一独立源风险校准候选状态（2026-10-29）

source-known risk calibrated scalar 已由 `20261029-phase3-source-risk-calibration-s2026092701-r01` 真实完成VERIFIED/ANALYZED_SOURCE_ONLY，执行commit ba2b4ed63a03751b8934dd7385f16a05aa75bcf3。它复用冻结F／OE状态，只由known V标签校准总体及每RX宏FRR≤4%、原正确样本拒绝损伤≤1.5pp，所有源known约束通过；MLP未知宏拒识81.4603%（父76.6032%）、FRR1.3667%、known开放97.6185%、drop0.8481pp，未知整体和RX3/4/6/8仍未达95%，NO_PROMOTION。0新训练／Fforward／GPU，2L特征拒识checkpoint smoke PASS，正式target0；同knownV经验风险与同novel重复源开发，不是新盲测／跨RX／星地／适应或注册成功。见[风险校准结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_RISK_CALIBRATION_RESULTS_20261029.md)。不根据未知提高预算或重跑，Phase3总目标仍未完成。

## 下一跨分支类别对应候选（2026-10-29）

`20261029-phase3-source-consensus-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，执行commit951fbddea7cafb4deb5e0e18a4d404edd7d02b74。仅100D跨分支同类支持描述符改变，F／L-only几何和数据冻结；新linear／MLP各200CPU步，真实父checkpoint2L与新descriptor smoke PASS，新增Fforward／GPU／target／foreignstop0。linear未知宏拒识82.6984%、FRR1.6185%、drop0.6963pp；MLP未知85.7460%、FRR1.4444%、drop0.7815pp，较risk父方法81.4603%提升4.2857pp。known保护全过，unknown整体及RX3/4/6/8仍未达95%，NO_PROMOTION，eligible空、selected=null。同knownV经验风险／同novel重复source clean开发非新盲测，未证实接收机不变，也非跨RX／星地／适应或注册成功。见[consensus结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_CONSENSUS_RESULTS_20261029.md)。不调参重跑或访问正式池，总目标仍未完成。

## 源辅助设备多样性扩展（2026-10-29）

下一步先构建21台源辅助训练数据，保留3台novel和正式4unknown，原L/U/V与F保持。`20261029-phase3-source-oe-diverse-data-r01`已登记，builder5项测试/5subtests及一次独立P0/P1通过，待本机CPU真实构建。按原source5RX×3day每非空TXcell最多32、少量全取，预期6134唯一记录（原4台1920＋新增17台4214），不按IQ/score挑选。后续源读出拟按role内15域均匀权重，归一化保持旧6300/4200的known.6/OE.4先验；这是新的源监督与域配重候选，不宣称只改了样本数或已经有效，目标端标签和访问仍0。

后续若需要训练开放空间embedding，必须沿原`pamf_repl_view_shuffle.source.train`训练构造器重建三视图/固定排列/六类默认头后strict加载核实E50payload，不能对`frozen_inference.reader_from_payload`直接解冻：推理reader把Sinc._filters换为含no_grad的frozen_filters。合法增量训练需保留原CENTER_UNITS及标签权限、单列原11100步＋新增预算；当前只做只读可行性调查，没有开启该训练。

## 多设备源OE数据实际完成（2026-10-29）

`20261029-phase3-source-oe-diverse-data-r01` 已VERIFIED/VALIDATED_ONCE、ANALYZED，执行commit2f41fc2835798f583c54900ab4c791d762d0101c。6134条唯一source记录来自21台训练暴露设备，220非空／95空cell，保留novel19-15／19-16／7-1不转train；15个source容器读取、完整18行1467B日志及独立poststate齐全。额外source成员监督明确，原L/U/V／novel验证IQ／模型／channel／正式target／GPU调用0，不停止其他任务；尚无模型性能结果。见[数据结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_OE_DIVERSE_DATA_20261029.md)。

## 多设备源域均衡拒识实际结果（2026-10-29）

`20261029-phase3-source-diverse-readout-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，执行commit70c8346339f5d2f5bcbe14aa8863eca545fa167a，NO_PROMOTION。冻结完整六类F／身份头／L几何／100D consensus描述符，新21台OE6134同时采用role内15域均匀及normalizer .6/.4固定先验；linear／MLP各200CPU步。linear未知61.4286%、FRR1.5630%、drop0.5778pp；MLP未知75.0794%、FRR1.5185%、drop0.5852pp，较consensus85.7460%下降10.6667pp。known保护总体和每RX均过，unknown整体及RX3/4/6/8仍未达95%，eligible空、selected=null。真实2L IQ smoke与OE6134共6136新F前向，旧L/V/novel特征复用，GPU2只新增一导出worker；F/head/geometry更新、原U_s／正式target读取、foreignstop均0。34项测试＋3subtests及唯一P0/P1 PASS，4完整日志与400loss已归档。组合改动不能拆因，也不能断言更多设备有效或特征绝对上限；同knownV经验风险和同novel重复source开发，非跨RX／星地／适应／注册成功。见[结果报告](E:/type10-7/docs/PAMF_PHASE3_SOURCE_DIVERSE_READOUT_RESULTS_20261029.md)。不调阈值预算或重跑；该后续小预算source OE embedding微调＋同预算原生L监督增量对照已由下文r02完成，结果NO_PROMOTION。Phase3总目标仍未完成。

## 原生开放集特征增量候选（2026-10-29）

`20261029-phase3-source-embedding-increment-s2026092701-r01` 已登记并实现，实际发布后在优化前技术失败；同矩阵r02已完成，见下文。supervised_only保留原native KL/CRIC/As监督骨架，不是纯CE/PAMF-CE；uniform_oe加λ.1均匀六类KL，两行均从原F6 E50独立初始化，cold CENTER_UNITS恒lr1e-4、新增222步/累计11322，原头和骨干joint、U关闭。77项测试/52subtests及一次完整独立P0/P1 PASS、一次N607 preflight PASS；先原生重建并strictload避免冻结Sinc梯度，真实2L梯度smoke(和OE行2OE)无step且恢复state/RNG后继续。source已知保护独立对照原F6同输入closed，源门槛保持95%unknown/5%FRR/2pp total保护，各RX均验，额外closed损伤预算.5pp；仅uniform_oe::msp可晋级，两行完整才判。source21远端public数据复用，无新数据重传/重验；2行并发每卡≤2新增，不停他人，单行技术失败保留另一健康行。正式RX24仍不读。详细见`docs/PAMF_PHASE3_SOURCE_EMBEDDING_INCREMENT_PLAN_20261029.md`；总目标保持ACTIVE。

r01实际发布后两行在新增父兼容检查失败，0梯度smoke／优化／评估，未产生科学性能；原native祖先检查已过。实际source contract存split_seed392005、cache复用路径不同，新增校验错误要求config字段和路径相等。现仅定向修复该层，保留native E50与物理role_ids约束；2项／3subtests与一次本P1定向复核PASS，无raw重验／额外hash。r01已记FAILED并保留，r02同科学矩阵／222／种子／数据恢复，独立新root、一次release preflight PASS，现已实际完整训练、预测及评分，科学NO_PROMOTION。目标和95%门槛不降，当前仍无达标正式target证据。

## 原生特征增量 r02 实际结果（2026-10-29）

`20261029-phase3-source-embedding-increment-s2026092701-r02` 已VERIFIED/ANALYZED_SOURCE_ONLY，commit28d53ea9b38c19f642c59e31f178a3f65f40894d。两个独立原F6 E50各新增222步、累计11322，原native L监督／原头联合保持，uniform_oe额外λ.1均匀KL，原U_s关闭；真实原L2与OE2梯度和最终模型smoke均PASS。预指定MSP监督未知47.8095%、closed98.5074%、open97.9222%、FRR1.4815%、原F总损伤0.5444pp；OE未知59.9683%、closed98.4963%、open97.9593%、FRR1.5185%、总损伤0.5074pp，配对未知+12.1587pp。全部5分数总体/每RX编码器与开放损伤保护通过，unknown各RX仍未95，只有uniform_oe::msp可晋级且实际未过，NO_PROMOTION。每行170496 L包视图/28416 L采样/14208 As，OE另14208；GPU2监督/GPU1OE，owned在跑各1496MB观察转录有来源，最终均退出，foreignstop/target/V训练0。444步完整trace有随机As波动，不宣称收敛；r01原FAILED为0优化技术失败，无新性能，定向修复未改变科学矩阵。见[完整结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_EMBEDDING_INCREMENT_RESULTS_20261029.md)。同knownV经验风险及同novel3重复source开发，不是跨RX／星地／无标签适应或注册成功。下一独立固定E51＋原4OE4200／consensus100D组合已按下节完成；不改本轮MSP/95%门槛或重跑，总目标未完成。

## 固定 E51 与读出组合实际结果（2026-10-29）

`20261029-phase3-source-embedding-readout-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，执行commit806dec253b7556ee3d4af842bf9a2d8fd714009e。两编码器各自重导40652样本、L-only新几何＋原4OE4200读出，各linear/MLP200CPU步，历史222不变、新encoder/head0。MLP监督86.3492%、OE90.7619%，配对+4.4127pp，较原E50同novel参考85.7460%为+.6032/+5.0159pp。OE FRR1.5556%、open97.7407%、原F总损伤.7259pp；7方法known总体/每RX保护均过，两个MLP仅RX1未知达95%，故NO_PROMOTION。30tests/2subtests与一次P0/P1PASS、800loss/8logs/2Lsmoke、原始GPU2/1 PID和退出证据完整，foreign/target0。见[完整结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_EMBEDDING_READOUT_RESULTS_20261029.md)。仍是同novel重复source开发，目标未完成。

下一独立候选`20261029-phase3-source-baseline-risk-s2026092701-r01`已登记、当前实现中：冻结上述全部状态，无新F/head/readout更新，只用knownV和原Fbaseline直接校准原目标中的总open损伤≤2pp与FRR≤5%（总体/每RX）。此前自身损伤1.5pp/FRR4%的安全预留不追改；新候选仍须95%unknown/5%FRR/.5ppencoder/2pp总损伤全部通过，unknown不校准。正式RX24 U/Q仍不读，Phase3总目标ACTIVE。

## 原模型参照风险实际结果（2026-10-29）

`20261029-phase3-source-baseline-risk-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，执行commita18daf5080bca7e76c3837907dcfdefd97fbc5bd。两固定E51/读出只重校known-only阈值，F/head/readout新增0；MLP监督88.0635%／OE92.6667%，OE较父+1.9048pp，FRR2.0296%、open97.3852%、原F总损伤1.0815pp。总体及每RX known保护通过，监督RX6/OERX3总损伤2pp预算恰绑定，两个MLP仍仅RX1未知过95，故NO_PROMOTION。9tests/4subtests与一次P0/P1PASS，4完整logs、2Lreadout smoke、阈值最大合法及完成产物齐全；工作极短首次post即完成，未捕获liveCWD/GPU观察、不伪造/重跑。见[结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_BASELINE_RISK_RESULTS_20261029.md)。同novel重复source，不是跨RX/星地/DA/注册成功。

下一独立`20261029-phase3-source-embedding-budget-s2026092701-r01`已登记并实现中：同原F6 E50独立冷启动的监督/OE两行固定新增2220步（E51..60），累计13320，非E51挑选后续训；native L/CRIC/As、OE21 λ.1、Uoff保持，自己的8分支/newL几何＋原4OE读出200，knownV仍原F总损伤2pp/FRR5%。固定末步、无V/novel早停，95与旧类保护不降，每卡≤2新增、不停foreign；正式RX24U/Q仍未访问，Phase3总目标ACTIVE。

## 十轮源增量实际运行（2026-10-29）

`20261029-phase3-source-embedding-budget-s2026092701-r01`已VERIFIED_RUNNING，执行commit6ffccc4ac1bb80d2d47a77e08a8a288012fa1353，36tests/38subtests PASS、一次新候选P0/P1 PASS。N607一次preflight/单归档比较/compile后，真实原L2/OE2梯度smoke PASS，GPU1监督/GPU2 OE已独立核实PID/CWD/argv/CUDA与日志增长；均各一新增worker、不停foreign。固定两行各2220步，当前尚无终态性能；训练后自动source导出/读出/known-only校准/独立评分。正式RX24U/Q仍不访问。

状态工具在本次用户更新指令后读回总目标为`blocked`，未提供原因；该读回不等于科研指标完成。用户原Phase3研发授权继续适用，本轮已授权实验照常执行；没有将目标工具写成ACTIVE、没有修改其状态或新建重复目标。当前Phase3整体尚未完成。

## 十轮源增量实际结果（2026-10-29）

`20261029-phase3-source-embedding-budget-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY/NO_PROMOTION，执行commit6ffccc4ac1bb80d2d47a77e08a8a288012fa1353。两原E50独立分叉各2220步，累计13320；4440步trace、800loss、10logs、实际Sinc/原head梯度与own8branch/readout smoke齐全，五stage全部完成。MLP监督90.0635%、OE92.0635%，OE FRR1.7778%、open97.7222%、原F总损伤.7444pp；仅RX1未知达95，各eligible空。与E51已完成参考92.6667%相比，十轮低.6032pp，未证实加轮数进一步改善未知拒识。正式RX24U/Q仍未模型访问；其他用户保持、owned GPU1/2均已退出。见[完整结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_EMBEDDING_BUDGET_RESULTS_20261029.md)。

用户当前请求总结报告，正在汇总方法/训练验证测试角色/跨RX与未知设置；此时不启动新的科学候选。后续局部聚合几何、连续拒识证据到未知发现和future验证的机制仅记研究笔记，不冒充已实现。总目标工具读回blocked且未被改写，Phase3整体未完成；源best92.6667%是同RX clean开发成绩，不是正式跨RX星地95%。

## 最新跨接收机实证与下一步（2026-10-29）

用户最新指标已固定：clean跨RX未知拒识≥95%，叠加星地后总体及各主景≥90%；已知类接收尽可能高，保留FRR≤5%与原F同输入闭集到开放损伤≤2pp。历史各run原门槛/NO_PROMOTION不追改，完整目标的零标签适应、发现、未来匿名注册与持续未知保护仍保留。

RX24-1独立开发池已实际完成：U384（D1）/V864（D2），旧六类＋开发发现19-15/19-16、后续未知7-1；三景low suburban/mid urban/low urban。原source五RX、正式24-18与开发24-1互斥，原Full/compact元数据复用。star code26c3e79eb6d078fc492c6b885ae8831e4358fb6c，clean控制code959a1a68e76a7e2cfba6700c2a251da8330e59b9；两条件同1248原始物理记录/同序opaque ID，不计更多独立样本，模型端clean与star隔离。两个真实CPU构建有起始live PID/CWD/命令及日志增长、完成产物和退出证据。正式RX24-18 U768/Q1920仍未模型访问。

已完成两独立冻结迁移诊断。统一固定source E51 uniformOE/MLP/原knownV阈值，原F6同输入closed作保护基线，不根据开发成绩选endpoint/分数/调阈值；各自全部V864预测后独立truth-last评分，U0/sourceIQ0/逐条sourcebank0/formal0/更新0。P1/MLP为星地主行，P0和其余6分数仅诊断。

|条件/主MLP|未知拒识|已知接收|含拒识已知识别|原F同输入闭集|总损伤|
|---|---:|---:|---:|---:|---:|
|C0 clean跨RX|99.3056%|80.7292%|62.6736%|79.6875%|17.0139pp|
|P0 星地received|75.0000%|44.0972%|30.5556%|53.9931%|23.4375pp|
|P1 同received＋DURA v1|57.2917%|72.9167%|53.2986%|72.7431%|19.4444pp|

C0未知单项过95，但旧类FRR/总损伤失败；P0/P1均未过90且known保护失败，所有实际状态NO_PROMOTION。DURA改善known，却在当前source拒识器下多接受未知，不能仅报known收益当开放集成功。C0/P0/P1是配对物理记录的不同条件，不把sameRX source92.6667%与此新RX跨条件结果当同输入提升。完整报告分别见`docs/PAMF_PHASE3_DEVELOPMENT_TRANSFER_RESULTS_20261029.md`、`docs/PAMF_PHASE3_DEVELOPMENT_CLEAN_TRANSFER_RESULTS_20261029.md`。

实用景codea3b8bb24b6b9e357bf37678209cf0c21d666f7df，10tests/16subtests及一次P0/P1 PASS；clean生产算法未改，code/config1e1876db7d0f7b8f3724fe95acf3c8c8d5b64fef补3个C0 tests/8subtests及一次新C0配置边界P0/P1 PASS，未重复旧代码全审。两run各完成N607 preflight/单归档一次比较/compile、实际checkpoint合成2IQ no-query smoke后立即预测。前台任务很短，首次独立post已终态，未捕获运行期独立PID/CWD/GPU/log增长；实际CLI身份、完整日志/markers及已退出状态有保存，不冒充live证据/峰值/部署延迟，不为补观察重跑。GPU0实用景与GPU6 clean各只新增1worker，无foreign停止；后续GPU表其他任务保持。

下一研究重点直接转向合法U_dev上的无标签域偏移处理及身份保持的未知发现：先提高未见RX的已知接收/正确识别，同时守住95%/90%拒识。当前结果显示未见RX的已知样本被大量拒绝，支持域变化影响拒识的判断；单纯提高拒绝强度或只恢复known不足以达标。下一候选须另行固定算法/源校准/更新预算与D0/D1及注册控制，U只读公共无标签输入、不读真实构成，V不回收训练，正式RX24-18继续隔离。目标保持ACTIVE；当前没有成功无标签适应或未来匿名注册证据，不标完成。

## 首个真实无标签U适应结果（2026-10-29）

`20261029-phase3-development-alignment-s2026092701-r01`已VERIFIED/ANALYZED_DEVELOPMENT_ONLY，code`cdaeccaf201f8c242ac00400b5c499f4dc1e63a8`。当前F使用GN/LN，无可直接更新的BN统计；新方法为源锚点/初始可靠旧伪类约束的正交多分支特征adapter。源MSP/cos门槛＋head/fused近邻一致＋time/frequency/PA2票，≥8条/旧伪类且≥3类；不用MLP已接受或unknown比例/类数/配额。一次求R=polar(I+.5M.TP/K)，reference27固定0，std raw右乘后交原默认head重算logits/descriptor。它保留std空间欧氏距离/cos，不保留whitened距离，也不把latent坐标称真实PA恢复。原F/G/head/MLP/source几何/阈值不变，但确实拟合了U特定R状态，每condition234825个FP64系数/1878600B，不把0神经optimizer步数写成未适应。

C0/P1从同一source独立用U384（RX24-1 D1）拟合R并写冻结checkpoint后，再对D2 V864逐样本预测；D0只在R冻结后加载旧完成预测作匹配控制，不重复旧row；真值只独立scorer连接。两condition同1248原物理记录，不共享clean参考。两行均实际applied，C0合格core157，P1初选150/合格147；支持5个旧伪类0/2/3/4/5，不等于5个真实TX或已知纯度。U中的两个未注册设备已无标签见过，第三个later未进入U；这些开发TX此前source评过，不称source新TX盲测。正式RX24-18 U768/Q1920依然未模型访问。

|主MLP条件|旧open D0→D1|已知接收D0→D1|未知拒识D0→D1|D1原F总损伤|
|---|---:|---:|---:|---:|
|C0 clean跨RX|62.6736→63.1944%|80.7292→81.2500%|99.3056→98.9583%|16.4931pp|
|P1 星地+DURA跨RX|53.2986→55.0347%|72.9167→74.8264%|57.2917→57.2917%|17.7083pp|

旧open分别多3/576与10/576，+0.5208/+1.7361pp；都有正收益，但known FRR/原F2pp保护仍失败，P1总体/scene/later未知保护仍失败，均NO_PROMOTION，不扩seed、不按V改η或阈值。clean unknown单项仍过95，不代表整体成功。已完成真实无标签适应实验，但尚无通过验收的适应方法；匿名注册仍0，完整目标ACTIVE。

两行各真实checkpoint合成2IQ smoke先于U，所有模型与R在Q冻结；sourceIQ/逐源样本bank/formal/新阈值0。原pure+outer联合22tests/39subtests、launcher3tests PASS；唯一review的两个P1（storage同根与Torch2.1/NumPy2桥）已定点修复，新2回归＋1邻近3tests/14subtests PASS并一次定向复核PASS，未重复全审/数据验证。一次N607 preflight/单归档比较/compile PASS。真实driver/C0worker live PID/CWD/cmd和C0日志增长已捕获，模型运行时ownedCUDA采样未捕获，GPU3分配来自driver，不报告峰值或逐包时延；完成后owned均退出，foreignstop0。实际pipeline13.7489s包含模型载入/两条件U与Q/评分，不是部署延迟。完整报告见`docs/PAMF_PHASE3_DEVELOPMENT_ALIGNMENT_RESULTS_20261029.md`。

下一优先改进源训练的拒识分数/表征对星地域和冻结恢复后输入的覆盖。当前P1 D1主分数AUROC约71.6363%；本等量开发池若单global阈值同时已知接收≥95%且未知拒识≥90%，至少95%×90%=85.5%的known/unknown配对应正确排序，因此仅移动当前阈值无法满足两项。此为当前单分数/本数据的必要条件，不推导所有模型不可能，也不扫描V寻找oracle阈值。新独立候选可让合法source L与既有辅助OE通过同规则星地/G输入来训练拒识器、源known-only校准后再验证；不把开发V回收训练，不更换source基座/默认恢复器或擅改数据权限，不读取正式Q。未知身份聚类与未来匿名注册≥80%的完整目标继续保留。

## 源端星地读出匹配结果（2026-10-29）

`20261029-phase3-source-practical-readout-s2026092701-r01`已VERIFIED/ANALYZED_DEVELOPMENT_ONLY，代码/config为`ca60bdf20090f664fbcf27682138d63cfd2751dc`。原E51/F0/G/六类默认头/clean-L几何全部冻结。本次复用source5RX、三日的L6300/OE4共4200/V27000/novel3共3150，40650物理记录，每条clean＋三practical→DURA，共162600计算view，不新增独立记录。新channel seed2026102931；各角色同scene/opaqueID规则，不读OE TX或novel truth生成。源端成员监督额外存在，不称全流程无标签。

calibration_only精确复用原linear/MLP和mean/std；augmented_readout在L25200/OE16800视图上各固定200CPU AdamW步，拟合101/1633参数及训练mean/std。两arm同四view known V108000校准，整体/5RX/4view/20RX×view共30组FRR5%/原F同输入closed−open2pp，ties整组，一个global τ/方法。源未知不校准；两arm状态/阈值全部冻结后才source novel预测，8组预测固定后独立truth-last。两arm均known校准可行，但source未知拒识不达标。主候选augMLP事前固定，不按source或target切换。

开发仅复用RX24-1、D2的Q864（576known/288unknown）配对C0/P1，两条件共864原始物理记录，targetU0/正式RX24-18读取0。两个current arm共享本次F/G前向和六类head；历史original D0及原F baseline精确复用。两条件历史/本次closed差异均0。source负结果仍按预登记完成全部开发预测和独立评分，不重跑、不调目标阈值。

|条件/主MLP状态|未知拒识|已知接收|已知open正确率|原F closed−open损伤|AUROC|
|---|---:|---:|---:|---:|---:|
|clean original|99.3056%|80.7292%|62.6736%|17.0139pp|97.8709%|
|clean calibration_only|96.1806%|86.8056%|67.8819%|11.8056pp|97.8709%|
|clean augmented_readout|96.8750%|90.4514%|71.5278%|8.1597pp|98.2898%|
|星地+DURA original|57.2917%|72.9167%|53.2986%|19.4444pp|70.9262%|
|星地+DURA calibration_only|34.7222%|89.4097%|66.3194%|6.4236pp|70.9262%|
|星地+DURA augmented_readout|38.8889%|93.0556%|68.5764%|4.1667pp|86.1075%|

clean相对原状态多56/576已知接收、多51/576已知正确开放识别；未知多7/288被接受，仍单项超过95%。星地多116/576已知接收、多88/576正确开放识别，但未知多53/288被接受，拒识单项远低90%。相对同校准control，augMLP的星地AUROC增加15.1813pp，known接收+3.6458pp/open+2.2569pp/unknown拒识+4.1667pp；确有分数排序收益，尚非验收成功。

aug星地低郊区/中城市/低城市未知拒识分别47.9167%/35.4167%/33.3333%，known接收94.7917%/91.6667%/92.7083%；三景均失败。中/低城市AUROC84.3967%/82.1994%仍低于同时95%接收/90%拒识所需85.5%的必要下界，故仅挪动当前单分数阈值仍无法满足逐景保护。总体AUROC86.1075%超过必要下界也不证明存在合格阈值，不扫描开发oracle阈值。

两条件均NO_PROMOTION，不扩seed或访问正式池。仅源读出/校准不能消除域变化下的身份特征和未知分离问题；下一优先候选应在合法source训练角色上进行星地/G输入的编码器与原默认头联合增量训练，保留同预算source控制与clean身份保护，争取同时改善known正确识别和unknown排序。不回收V_dev训练、不读其真值拟合、不改变原默认F/G身份，不把更高接收率当未知拒识进展。未知发现/未来匿名注册仍未完成，完整目标ACTIVE。

9stage全部COMPLETE、实际checkpoint合成2IQ四view smoke在source IQ前通过，C0/P1也各在Q前smoke。聚焦29tests/24subtests PASS；唯一独立审查两个API P1已修复并定向确认。一次N607 preflight/单归档比较/远端compile VERIFIED。捕获实际driver2624334/export2624355的PID/CWD/命令、GPU3可见性、owned CUDA占用与日志增长；900MB为一次采样不是峰值。source导出437.2247s、两个readout拟合stage6.0401s，完整stage计时见post_state，不当作逐包部署时延。完成后本run进程退出，其他owner未停止/改动。完整报告为`docs/PAMF_PHASE3_SOURCE_PRACTICAL_READOUT_RESULTS_20261029.md`，原始评分/训练损失/后态在本run local_artifacts。

## 匹配预算的源端星地编码器联合增量结果（2026-10-29）

`20261029-phase3-source-practical-encoder-s2026092701-r01`已VERIFIED/ANALYZED_DEVELOPMENT_ONLY，实际code/config `b578c074ba68b8ff9aa539209b3410f1f103ba3d`。同一E51父状态各独立222步，累计11544，fresh AdamW CENTER_UNITS lr1e-4/wd1e-4/物理clip5；原六类默认head和Sinc都实际联合更新，未另训身份head。两arm保留native原六路监督/crossRX0.1/As和U-off，额外L128 CE权重0.68：clean_increment用clean，practical_increment用同记录随机三景固定source practical/G。OE21的64条物理记录以clean＋对应第二view共128平均uniformKL乘0.1；成员监督存在，未知TX不读取。control仍含native旧As，不称全无星地训练。

共享训练cache仅L6300/OE21共6134，12434物理/49736计算view，源U/V/novel/target0。源channel2931/receiver2912与另一个独立scene采样流2932按record选择，不按类别；primary额外L三景9568/9582/9266，共28416，额外OE4839/4663/4706，共14208。两arm各L native170496＋extra28416＝198912 packet，OE两view28416，222步全部完成。源端freeze后各用自身新clean-L几何、L/OE4四view读出各200CPU、known V四view30组校准，再novel预测/独立score；每arm40650物理/162600计算view。targetU0/正式RX24-18读取0，源/开发科学失败均不改已冻结配方。

两arm自动各完成source和C0/P1完整预测/独立truth-last。C0/P1仍同RX24-1 D2 Q864（known576/unknown288），两arm/条件不增加独立物理样本；父parent_E51精确复用上一run augmented_readout，不复用更早original读出或重新运行父模型。source与query阶段模型/聚合量冻结。以下MLP均预登记主方法，主arm一直practical_increment：

|条件/状态|未知拒识|known接收|known闭集正确|known开放正确|原F closed−open损伤|AUROC|
|---|---:|---:|---:|---:|---:|---:|
|C0 parent_E51|96.8750%|90.4514%|78.2986%|71.5278%|8.1597pp|98.2898%|
|C0 clean_increment|97.5694%|90.2778%|75.6944%|70.1389%|9.5486pp|97.5272%|
|C0 practical_increment|98.6111%|85.7639%|67.5347%|57.8125%|21.8750pp|97.1861%|
|P1 parent_E51|38.8889%|93.0556%|71.8750%|68.5764%|4.1667pp|86.1075%|
|P1 clean_increment|37.1528%|94.2708%|71.8750%|69.0972%|3.6458pp|87.2962%|
|P1 practical_increment|50.0000%|89.0625%|69.9653%|62.8472%|9.8958pp|85.2515%|

原F同输入闭集C0为79.6875%、P1为72.7431%。primary相对control，P1多拒未知37/288（+12.8472pp）却少接收known30/576、少开放正确36/576（−6.25pp），AUROC反降2.0448pp；C0闭集少47/576（−8.1597pp）、开放少71/576（−12.3264pp）。相对parent，primary的P1未知多拒32/288，但开放少33/576；不能只取拒识单项作正结果。两arm均NO_PROMOTION，不将control换成主候选、不扩seed或读正式池。

primary星地低郊区/中城市/低城市拒识60.4167%/46.8750%/42.7083%，known接收90.1042%/89.5833%/87.5000%，逐景均失败。P1整体AUROC85.2515%低于同时95%接收/90%拒识所需85.5%的必要下界；更直接地，primary C0/P1闭集本身已分别低于原F允许的77.6875%/70.7431%最低开放准确率，而开放准确率不可能高于自身闭集，所以移动阈值无法挽回身份保护。此为当前分数/模型/开发池的数学界，不推断所有方法无解，也不扫描oracle阈值。

源端known V上primary clean闭集98.3481%、lowurban闭集85.6704%，control为98.5519%/84.9519%；源域较小变化不能代替新RX的身份保持。source未知主目标仍失败，完整源/开发及逐RX/TX/scene结果见`docs/PAMF_PHASE3_SOURCE_PRACTICAL_ENCODER_RESULTS_20261029.md`。该结果不支持继续无约束地改动完整身份函数；下一优先验证保留父模型clean路径的受约束星地特征映射/身份保持，再接合法U的无标签域适应，降低附加TX监督依赖。新方案仍另行冻结目标/矩阵，不回收开发Q训练；未知发现和未来匿名注册≥80%继续保留，完整目标ACTIVE。

共享cache＋两row共15stage全部COMPLETE，native真实checkpoint梯度smoke及source/dev合成smoke通过，head/Sinc更新有实际状态比较，所有源/目标标签边界与冻结检查通过。32项唯一tests/68subtests覆盖：dev旧7/24及新增真实目录回归/邻近3/10分别保存；唯一独立review的一个parent/source/fit目录P1已修并一次定向PASS，不重复全审。一次preflight/单归档比较/compile VERIFIED；捕获driver2673776、cache2673864、两rowdrivers2675837/2675840、训练2675845/2675846和source2678276/2677698等实际PID/CWD/cmdline/CUDA及log增长/ownedGPU采样。训练分别GPU7/4、一次采样各1540MB，不是峰值；完成后owned退出、foreign0。源训练实计101.4671/86.8424s，整pipeline712.9620s含缓存/并行/加载/源开发评测，非因果加速或逐包时延证据。
