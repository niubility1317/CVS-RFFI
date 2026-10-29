# G10 共同可行点初始化的单调阈值扩展

Run `20261030-phase3-star-common-seeded-risk-s2026092701-r01`。这是新源端科学比较，保留前两轮NO_PROMOTION和全部原产物。最新用户v3指标仍为星地跨RX C_K/R_U≥90%、原F0同输入closed−A_K≤2pp，主报true-TX宏A_K/R_U/H，完整目标ACTIVE。

## 问题和固定选择

上一零初始化贪心在主joint上OE代理53.12%低于已存在common可行点56.67%，因此还不能判断独立预算是否可改善该共同阈值。这里由Agent选择以原common为初态，只允许六阈值单调提高；它隔离初始化效应，不修改NN、分数、已知保护或OE域权重。

两个冻结arm仍为detach_control/joint。三个固定运行点依次为`common_tail`、`zero_initialized_budget`、`common_initialized_budget`，唯一可晋级主为`joint::common_initialized_budget`。前两点逐值复用已冻结阈值，不能根据结果换主。新阈值逐元素≥common，故相同原始分数、默认身份和任意样本上，其接收集合是common的子集，未知拒识不会因放松其他预测类阈值而下降；这不保证新RX的已知保护或90%拒识。

模型固定来自jointOVA r02/code `5aa041122396d48c49b8cb3d5912649bdc3217e1`，两模型各累计13320 F更新，原默认六类头/966参数aux保留。OE公共分数和零初始化阈值来自 `20261030-phase3-star-oe-risk-budget-s2026092701-r01`/code `585e10b07be1f6821edea08b4853582b92512aff`；该run不作为新NN权重来源。common来自r02 ova_max六τ，zero来自OE risk conditional[1]，二者精确复现。

## 数据、成本和算法

只使用父r02 V27000的合法旧六标签/原F0同P1 baseline，以及已有OE6134公共logits/默认身份/15域。V/novel数据和模型身份完全不变，所有分数最初均由同一G10 P1生成。source五RX1/3/4/6/8与三日均已见；novel19-15/19-16/7-1各1050已有历史开发曝光。无新真实IQ前向、无G或optimizer，L/U_s/target UQ/formal读取0；42584原物理记录不增加。

每arm仅做2条合成G10 P0实际checkpoint noquery smoke，总F4；不是新真实P0/P1对照、也不是G管线验证。smoke后直接继续公共source预测校准，τ全部冻结并保存后才打开父novel公共数组；独立CPU scorer在本row全部预测完成后连接私有真值。没有novel/target参与拟合，OE只提供outside-six成员关系和域，不含TX/K。

沿用原纯NumPy greedy：每预测类下一候选是下一未拒OE完整分数层，包含跨过的全部known风险；按域宏收益/(最大新增风险÷当前剩余预算+1e-12)择优、classindex打ties。六阈值初始为common，队列从OE score<τ初始已拒层之后继续，known累计风险必须按score<τ真实初始mask计数，不能用前一OE层替代间隙中的known风险。所有42组true-known宏CK90与原F0总身份损失2pp保持，最终用原exact-Fraction核对。

共享 `star_oe_risk_budget.calibrate` 新增仅关键字参数 `initial_thresholds=None`；None保持原acceptall算法和全部旧阈值结果。提供有限FP64[6]时保留真实初始阈值／OE收益／known风险，新增步骤只增τ；初态不合格时保留原初态并标false，不静默替换。报告initialization、initial_thresholds、objective_before/after、trace和六终止候选；不宣称全局最优。sourceOE代理和未知真实TX宏拒识继续区分。

## 实现接口与责任

- 算法worker仅修改共享`star_oe_risk_budget.py`及其测试，补已有可行点起步、阈值间隙known风险、OE初始prefix、无收益保持初值、初态不合法、默认行为回归。保留旧调用接口，无新搜索超参。
- 运行worker创建`star_common_seeded_risk_run.py`及测试；exact8配置为run_id/code_commit/arm/parent_eval_config/parent_prediction_root/OE_prediction_root/output_root/batch_size128。parent_eval_config仍为r02原配置，可复用父_backend/_smoke并snapshot；OE_prediction_root为前一OE risk run对应arm的predict目录。只读取其marker/cal/thresholds/OE_scores，不得调用会提前打开novel的旧_score._prediction。新fit只新增initial_thresholds=common；common和zero报告精确复用。
- 评分／调度worker创建`star_common_seeded_risk_score.py`、`star_common_seeded_risk_launch.py`及对应测试；后者沿用每arm source→独立CPUscore两阶段、两张实际有余量卡、不影响另一健康arm。主Agent负责登记、集成验证、Git、发布、观察、报告。

新prediction schema=`pamf_phase3_star_common_seeded_risk_prediction_v1`、status=`STAR_COMMON_SEEDED_RISK_PREDICTIONS_COMPLETE`、cal status=`STAR_COMMON_SEEDED_RISK_THRESHOLDS_FROZEN`。导出RUN_ID/SCHEMA/STATUS/CALIBRATION_STATUS/ARMS/METHODS/KNOWN_FIELDS/NOVEL_FIELDS/OE_FIELDS。输出原known/novel公共字段，scores改[N,3]；conditional thresholds为FP64[3,6]，OE_scores保持8公开字段。原r02非scores字段完全相同，3margin均由同rawmax减各自τ[defaultpredclass]。

cal complete含run/code/arm、method_names顺序、conditional_thresholds、methods[三report各constraints_passed/groups42/tau]、group_count42、source_V_labels_used=true/source_OE_membership_used=true/source_OE_TX_or_K_used=false/novel_inputs_used=false/target_reads0/source_onlytrue。先保存cal/thresholds/known，再novel，最后root complete。

root complete含parent_run_id/parent_code_commit/parent_model_identity/parent_prediction_root（均r02）、OE_parent_run_id/OE_parent_code_commit/OE_prediction_root（均前OE risk）、known_count27000/novel_count3150/OE_count6134、source_only/target_reads0、source_L_reads0/source_U_reads0/neural_updates0/G_forward_rows0、F_forward_rows2/real_P1_forward_rows0/synthetic_P0_forward_rows2、model/aux_parameters_buffers_unchanged=true/all_thresholds_frozen_before_novel=true、primary_armjoint/primary_methodcommon_initialized_budget/classification_pathP1。补monotone_thresholds_from_common=true。

独立评分先核对两父run/code/arm和原rawscore身份，common与zero精确复现，newτ≥common及逐ID new接收⊆common；违反则在truth前拒绝。然后报告trueTX宏指标/42known/12unknown/配对收益、OE域宏、原cross-sourceRX结构guard；zero与control不能替主。可复用旧纯指标/结构函数，不monkeypatch全局schema或放宽旧验证。

## 执行与证伪

N607各arm一GPU、至多2arm并行、真实free≥4096MB，每卡≤2新增，不停其他owner。新code/config commit、聚焦负测＋真实checkpoint first-step smoke、一次候选P0/P1、最小prereg、新immutable run/release、一次preflight/archiveSHA/compile及实际PID/CWD/cmd/GPU/log读回；全部预测冻结后独立truth-last评分。测试不要求重复未变父suite或额外报告审查。

唯一主known42、unknown总体＋五RX＋六景≥90%、原结构guard同时过才可准备另记小目标实验；否则NO_PROMOTION，不换控制／重校准／扩seed。源端通过不代替未见目标RX、无标签适应或未来匿名注册，A_N/H_OW=N/A，完整Phase3保持ACTIVE。
