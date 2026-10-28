# PAMF Phase3：跨接收机星地无标签开放世界识别

2026-10-28。用户授权自主设计、分步分工、开启目标模式并开始实施；未知类拒识至少95%。默认N607，不停止或修改其他用户/owner任务；每张卡最多新增两个本任务训练或预测进程。本文同时承载目标prompt、设计和分步计划，不另设设计审批。

## 目标prompt

在当前三视图PAMF/Tri-SIG-TIME SSL基础上，完成跨接收机、practical residual星地条件下的开放世界RFFI。依次实现冻结模型拒识、零目标TX标签适应、未知类别发现和匿名持续注册。目标端不提供TX标签、known/unknown角色、未知类别数或类别配额；允许的无标签发现数据必须与最终query物理互斥。真实身份命名/业务授权与匿名设备注册分开。预测先固定、独立truth-last评分；不根据目标成绩调阈值、选模或选择性重跑。保留Phase1/2原产物，先做单模型seed小规模可证伪实验，源侧冻结规则且达到科学门槛后扩展3个模型seed确认。指标未达时保留负结果，不降低门槛或宣称完成。

## 验收口径

|任务|量化目标|评价限制|
|---|---|---|
|未知拒识|按真实未知TX等权的宏平均拒识率≥95%，各预登记主场景也≥95%|unknown仅在scorer连接；不能用defer冒充成功|
|旧类保护|已注册类宏平均误拒≤5%；含误拒的正确率相对相同IQ/default-head闭集基线下降≤2个百分点|同时报告逐类、RX、scene及95%区间|
|无标签适应|保持上述拒识/旧类约束，并报告同query旧类正收益|不适应、普通伪标签、可靠旧类筛选作明确控制；未证实则不得声称有效|
|匿名注册|未来独立观测新类宏平均识别≥80%；旧类相对注册前下降≤2个百分点|不在建簇样本上回测；真实TX映射只可评分端计算|
|持续开放|后续未注册设备拒识≥95%|注册后的新unknown集合独立；错误合并/拆分/重复注册分别报告|

目标针对点估计；95%统计区间另列，不将“95%拒识”写成“95%置信下界”。纯噪声/证据不足可暂缓，但全部已注册类暂缓/拒识均计错误。缺失已知或未知的组返回不适用，不能判成功。

## 数据与方法设计

默认仍为跨RX泛化：source RX与target RX不交。单星被动监听是首轮主场景，多节点协同不是前置条件。采用固定practical residual/post_sync/noEQ/25MHz单记录单接收观测；known与unknown使用同一物理规则。clean仅作匹配诊断。源训练保持当前合法旧6TX契约；严格未知TX必须未进入其监督/无标签/教师/恢复器祖先。曾无标签见过但未注册设备单列，不混称严格未见。

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

下一优先路线直接利用当前满意的完整六类PAMF，准备独立source开放开发数据；不再只依赖预算弱得多的4CE代理。元数据确认Full174类比四compact并集150多24类，其中7类在原source5RX×D1D2D3均≥70；按容量与字典序事前分为辅助暴露4类4200条、独立源新设备验证3类3150条，排除旧6和正式4unknown。科学边界已写入PROJECT_PROTOCOL，额外source已知/未知成员监督明确披露，原L/U/V和六类身份头不变；正式target仍零标签。数据run`20261029-phase3-source-oe-data-r01`的builder/6测试/一次P0/P1已就绪，待真实构建；数据与后续拒识器分开登记，后者尚未实现/训练。方案见`docs/PAMF_PHASE3_SOURCE_OPEN_DATA_20261029.md`。
