# G10 冻结 F0 的局部邻域支持度

Run `20261030-phase3-star-local-support-s2026092701-r01`。新源端候选，不追改前轮结果。主目标继续采用用户v3：跨新RX星地C_K/R_U≥90%、同输入原F0 closed−A_K≤2pp，主报trueTX宏A_K/R_U/H；完整Phase3 ACTIVE。本轮只是新评分机制的source证伪，不是目标域或注册完成。

## 固定方法

Agent选择冻结原Phase1三视图F0 E50/11100/default6head及DURA，不继承按novel较高分选择的Phase3增量。原F0的L6300/OE6134单位fused160可从jointOVA r02 teacher复用，teacher本身由原F0产生；它不是r02更新后的模型。当前已有分数以全局原型/协方差为主，本候选直接使用完整source bank。

三个固定分数顺序为`pooled_known`、`pooled_relative`、`rx_consensus_relative`，唯一主为最后一个。固定k=8、eps=1e-6，不扫k或按novel选超参。把每条FP32单位特征转换为FP64并逐条再归一化；距离为clip(2−2cos,0,4)。d_K/d_OE为分别到L/OE中最小8个距离的均值。

- pooled_known=−d_K，全部L为一个bank。
- pooled_relative=log(d_OE+eps)−log(d_K+eps)，两bank各整体取邻居。
- rx_consensus_relative：分别在每个source RX的L/OE子bank取8邻居，再分别对五个RX距离取median，最后计算同一log比。只用固定source bank分组，不读取query真实RX、日期、场景或标签路由分数。

两个对照可区分加入OE相对支持以及source-RX共识的影响，不能事后换主。默认六类身份始终由原F0 logits argmax给出，不由邻居标签替代。bank不使用每TX标签或OE真实TX/K，只使用既有known/OE成员角色及source RX；新score无NN或optimizer训练。

## 数据与顺序

同G10 MIX01/P1缓存42584物理记录：L6300/OE6134构建bank，V27000仅校准三分数各六预测类阈值，novel3150在全部阈值冻结后预测。source五RX1/3/4/6/8、三日及历史novel19-15/19-16/7-1曝光限制保持；不是新RX或新TX盲测，正式RX24-18及targetU/Q不访问。

L/OE特征复用 `20261029-phase3-star-joint-ova-s2026092701-r02/teacher/{L,OE}.npz`，核对其originalF0/DURA身份、相同role IDs和公开source-RX元数据，不能把candidate_fused_unit当原F0。所有bank在V之前固定，V/novel不追加到bank、不拟合query统计。

V缺原F0单位特征，故新F对其27000条cached P1前向；novel在freeze后也新F3150，以同时保存原F0真实默认头身份（不误用r02更新头）。额外2条actualcheckpoint synthetic G10 P0 noquery smoke，总F30152；L/OE新F0、真实G前向0、NN/G更新0、L/U_s新训练0。真实模型输入仅既有星地P1，无clean参考或新信道生成；合成P0单列，非G/P1端到端smoke。

阈值统一使用现有`star_class_risk.calibrate_known`的predicted_class六τ，true-known宏42组CK90/originalF0 loss2pp、complete ties；三分数各独立校准，无OE阈值优化。baseline使用Vcache原F0同P1 closed，fresh F0 closed单列；如有数值路径差异如实计数，不把baseline重设成新值。所有bank/模型/18τ固定后才打开novel，prediction complete后独立CPU truth-last。

## 实现接口与责任

1. 算法worker：`star_local_support_core.py`及测试。固定常量METHODS/K/EPS/RECEIVERS；`fit(L_features,L_rx,OE_features,OE_rx)->state`，state exact keys L/OE/L_rx/OE_rx（FP64归一化features与int64 RX）；`score(features,state)->FP64[N,3]`。输入feature为FP32[N,160]单位向量、各source RX至少8条。内部固定128查询分块，共用每块的两距离矩阵计算pooled及5sourceRX距离，避免Nquery×Nbank大矩阵。无queryRX/标签接口，无查询群体统计、bank更新或新分类头。完整tie只影响相同距离值，不输出邻居身份。
2. 运行worker：`star_local_support_run.py`及测试，exact7配置 run_id/code_commit/source_config/cache_root/teacher_root/output_root/batch_size128。source_config为runner.COMMON加source_cache，originalF0 6类seed2026092701，run/row/code/output匹配，row_id=`star-local-support-s2026092701`。可复用star_consensus_run._backend/_smoke/_snapshot/_check_models/_encode与既有source role loader。先实际checkpoint烟测→原F0 teacher bank+role IDs/RX→冻结bank→V P1新特征和logits→cal/known/18τ写全→novel P1新特征和logits→complete。不要加载或更新G，不消费received P0或raw。
3. 评分／调度worker：`star_local_support_score.py`、`star_local_support_launch.py`及测试。一个row=`original_F0`，source→独立CPU score；runtime沿12fields但max_concurrent_rows1/rows只original_F0。单实际free≥4096MB GPU，每卡≤2新增，不改foreign。唯一主rx_consensus_relative。

prediction schema=`pamf_phase3_star_local_support_prediction_v1`，status=`STAR_LOCAL_SUPPORT_PREDICTIONS_COMPLETE`，calstatus=`STAR_LOCAL_SUPPORT_THRESHOLDS_FROZEN`。导出RUN_ID/SCHEMA/STATUS/CALIBRATION_STATUS/METHODS/KNOWN_FIELDS/NOVEL_FIELDS/BANK_FIELDS。KNOWN_FIELDS exact ids/y/rx_i/day_i/scenario/unit_features/default_logits/closed/baseline_closed/raw_scores/scores；NOVEL_FIELDS删y/scenario/baseline_closed。unit_features FP32[N,160]、logits FP32[N,6]、raw_scores/scores FP64[N,3]。thresholds.npz exact conditional FP64[3,6]。

source_bank.npz exact `L_ids,L_unit_features,L_rx_i,OE_ids,OE_unit_features,OE_rx_i`；原features FP32，runtime由core.fit构造FP64工作状态。cal complete包含run/code、methods三组constraints_passed/groups42/conditional_thresholds、method_names顺序、conditional_thresholds、known_only=true/unknown_inputs_used=false（仅指threshold fit）、source_only/target_reads0/group_count42；bank使用OE成员监督须另如实注明，不能称整个方法known-only或零源标签。

root complete必须含run/code/model_identity(originalF0 E50/11100/232993)、teacher_run_id/code_commit/teacher_root、cache_root、known_count27000/novel_count3150/bank_L_count6300/bank_OE_count6134、methods/primary_method/classification_pathP1、all_thresholds_frozen_before_novel/source_bank_frozen_before_V/source_bank_unchanged/model_parameters_buffers_unchanged全true，source_only/target_reads0/source_novel_truth_reads0/neural_updates0/G_updates0/G_forward_rows0/F_forward_rows30152/new_V_forward_rows27000/new_novel_forward_rows3150/synthetic_P0_forward_rows2/source_L_forward_rows0/source_OE_forward_rows0，bank/role cache identities和reader原F0身份一致。

scorer在complete后验证精确字段、默认身份等于logits argmax、18τ与cal一致，bank role IDs互斥/V/novel不在bank、sourceRX五组及单位特征shape/norm，固定core重建rawscore和margin；通过后才连接private novel truth。只用纯固定bank进行复核，不重新选τ或拟合统计。报告42known/12unknown/逐TXRXscene、宏AK/RU/H/CK/AUROC/OSCR、original baseline与fresh closed、配对得失和source-only边界；A_N/H_OW=N/A。

## 成本和验收

NN部署F232993+G266951=499944，aux不使用；bank原FP32特征约7.96MB，FP64工作特征约15.92MB、RX分组及workspace另计，不能仅报NN参数掩盖bank。CPU readout耗时另列，不冒充整机B1延迟；不预先宣称加速或达标。

聚焦TDD：距离与手算一致、pooled/RXmedian差别、k/ties、分块行独立、bank不变、非法向量/缺source RX、wrong model或teacher role错配、smoke先于IQ消费、freeze先novel、原head不可替换、私有字段和truth-before-validation负测、输出不可覆盖/失败保留。一次独立P0/P1、单Git code/config commit、最小登记、新immutable root、一次preflight/archiveSHA/compile和实际PID/CWD/cmd/GPU/log读回；不重复父suite或数据重验。

只有主known42保护且unknown总体/五RX/六景宏RU≥90%，才取得另记小目标证伪资格；否则NO_PROMOTION，不换控制、调k、重校准或扩seed。source通过永远不替代完整跨新RX星地、零标签适应及未来匿名注册目标。
