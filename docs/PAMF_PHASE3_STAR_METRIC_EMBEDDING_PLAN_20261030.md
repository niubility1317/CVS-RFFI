# G10 冻结 F0 的独立拒识度量学习

Run `20261030-phase3-star-metric-embedding-s2026092701-r01`，单模型种子源端证伪。固定三臂 `identity`、`compact_oe`、`compact_oe_structure`，唯一主为 `compact_oe_structure::predicted_identity_mahalanobis`。用户v3跨未见RX星地双90%/同输入原F0闭集−A_K≤2pp、trueTX宏A_K/R_U/H保持，完整Phase3 ACTIVE。本轮源通过也不等于跨RX适应或未来匿名注册完成。

## 固定网络、训练与分数

原F0 E50/11100/default6head/232993和DURA v1.0冻结。输入仅原F0同G10 MIX01 P1的FP32单位fused160。独立网络 `MetricNet`：160→64 Linear+ReLU→160 Linear，输出 normalize(x+delta,dim=1,eps=1e-12)，20704参数；第二层weight/bias初始化为0。每臂均先torch CPU seed2026092701构建同一初始网络再移至设备，identity保持零更新，其余各1000步Adam(lr1e-3,weight_decay1e-4)，无scheduler/EMA/dropout/BN/早停。F与原默认头不参与梯度或身份替换。

固定sampling seed2026103002，每步L按6旧TX×5sourceRX各无放回抽4条，共120；OE从6134条中无放回抽128，跨步允许复用。只使用L合法旧TX/RX，OE只有outside-six成员关系，无OE真TX/K。两训练臂同初始权重/同batch轨迹，唯一区别为结构项权重。记录完整1000步三分项/total和累计曝光、实际唯一覆盖；不以loss或V选checkpoint。

设z=g(x)为单位输出，T=0.1。known对比项：对每个L anchor，正例仅同旧TX且不同sourceRX，负例为其他旧TX；同TX同RX和自身排除分母。分母包括全部正/负例，逐anchor对正例logsoftmax平均后再平均。各步每anchor有16个跨RX正例及100个其他TX负例。

prototype p_c=normalize(mean(z_L[y=c]))；OE排斥项 = mean(relu(max_c z_OE·p_c)^2)。结构项 = OE批内off-diagonal的mean((z_OE·z_OE^T − unit(x_OE)·unit(x_OE)^T)^2)。`compact_oe`总loss=known_contrast+10*OE_repulsion；唯一主`compact_oe_structure`再加10*OE_structure。控制的结构值可detach作同条件诊断，不构造额外teacher或神经前向。identity无训练，三臂均使用同架构与相同聚合器/源规则。

网络冻结后只用source L的g输出拟合四个FP64聚合数组：class_means[6,160]（均值不再归一化）、background_mean[160]、within_whitener[160,160]、background_whitener[160,160]。复用`star_relative_core._whitener`的0.2各向同性shrinkage；within残差u−classmean_y除N−6，background残差u−overallmean除N−1；u为FP32输出转FP64再逐行单位化。state无source rows/y/IDs，合52320个FP64数=418560B。

两分数顺序固定为 `minimum_relative` 和 `predicted_identity_mahalanobis`：分别为d_background−min_c d_c、−d_closed，d为对应冻结whitener的平方距离；closed始终原F0默认头argmax。只用fused160，不再五branch标准化或使用完整source邻域bank。query不接收真TX/RX/场景/K、不更新任何状态。每臂两分数各六τ均由V已知标签按原`star_class_risk.calibrate_known`校准，known42组宏CK90/原F0同P1closed−AK2pp、complete ties不变。所有网络/聚合状态在V前冻结，12τ/部署包/cal/known写全后才打开novel特征。

## 数据与来源

复用`20261030-phase3-star-local-support-s2026092701-r01`/code84cdc012f7d9641b283c407f6048bfe4e6e21cf6的original_F0/predict公有原F0特征：bankL6300/OE6134，V27000/novel3150（42584同物理记录）；合法L metadata y/rx按同IDs对齐。父bank来自jointOVA r02 teacher的originalF0而非其更新权重，cache为star-representation r01同G10 P1。可复用`star_conditional_support_run._parent/_bank/_public`的只读公有特征接口，但不调用其run/完整bankscore/真实IQ role loader或阈值控制拟合。

source RX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）、2021-03-08/15/23已见；旧六TX14-10/14-7/20-15/20-19/6-15/8-20，L各1050、V各4500；historical novel19-15/19-16/7-1各1050。原Phase1契约split392005、L6300/Us56700/V27000、10%有标签，F0 modelseed2026092701。本轮source监督含L真旧TX/RX、OE成员关系、V阈值标签，不称全系统零源监督。targetU/Q/正式RX24-18/原Us/未知真K不消费。

本轮无新真实IQ/F/G，仅每臂首步actual原F0checkpoint的2条syntheticG10P0无query smoke（总F6，非新G/P1端到端测试）。新g对L/OE训练/聚合及V/novel的前向另计，父F30152及历史P1恢复成本保留祖先；不能把features复用称总成本为零。

## 接口与文件责任

core `star_metric_embedding_core.py`：常量ARMS/METHODS/PRIMARY_ARM/PRIMARY_METHOD/TRAIN_STEPS1000/PARAMETERS20704；`MetricNet`，`losses(z_L,y_L,rx_L,z_OE,x_OE,arm)` 返回字典total/known_contrast/OE_repulsion/OE_structure，identity不调用loss；`sample_indices(L_y,L_rx,OE_count,rng)->(indexL120,indexOE128)`；`fit_metric(L_metric_units,L_y)->dict` exact四聚合keys；`metric_scores(metric_units,closed,metric)->FP64[N,2]`。source sampler固定旧6/五RX且各cell至少4，OE>=128；core无文件/真值读取。复用已稳定unit检查/whitener，数值逐行独立，无query统计。transform由runner以MetricNet分块forward实现；Torch2.1/NumPy2边界一律tolist桥，不用from_numpy/as_tensor(ndarray)/.numpy()。

runner `star_metric_embedding_run.py`：exact8配置 run_id/code_commit/arm/source_config/parent_prediction_root/cache_root/output_root/batch_size128；source_config沿runner.COMMON+source_cache，row_id=`star-metric-embedding-{arm}-s2026092701`，原F0/source契约一致。导出RUN_ID/SCHEMA=`pamf_phase3_star_metric_embedding_prediction_v1`/STATUS=`STAR_METRIC_EMBEDDING_PREDICTIONS_COMPLETE`/CALIBRATION_STATUS=`STAR_METRIC_EMBEDDING_THRESHOLDS_FROZEN`/ARMS/METHODS/PRIMARY_ARM/PRIMARY_METHOD/COUNTS/KNOWN_FIELDS/NOVEL_FIELDS/METRIC_FIELDS（四聚合名）。顺序actual F0 smoke→父公有bank/L合法元数据→固定预算g训练→冻结并保存网络+L聚合→释放训练feature bank引用→父known仅g前向/12τ→部署包冻结→父novel仅g前向→complete。F0原head/logits/closed/baseline不改，g不做身份head；网络参数/缓冲和aggregate在V/novel不变。

每臂输出`training.jsonl`（identity0行/learned1000行）、`training_complete.json`、`smoke.json`、`known_predictions.npz`、`novel_predictions.npz`、`calibration/complete.json`、root`complete.json`。部署目录 exact files `model.pt`（仅4个state_dict tensor，无optimizer/样本）、`metric.npz`（四FP64聚合）、`thresholds.npz`（conditionalFP64[2,6]）、`complete.json`（status=`STAR_METRIC_EMBEDDING_DEPLOYMENT_FROZEN`，schema=`pamf_phase3_star_metric_embedding_deployment_v1`、run/code/arm、parameters20704、source_free=true、per_record_source_rows_retained=0）。model.pt为`torch.save(model.state_dict())`，加载用同类strict state_dict；不得保存source bank/逐样本标签/ID/特征到deployment。

KNOWN_FIELDS为原local-support的ids/y/rx_i/day_i/scenario/unit_features/default_logits/closed/baseline_closed/raw_scores/scores再加`metric_unit_features`；NOVEL删y/scenario/baseline_closed。unit_features为父原F0 FP32[N160]、metric_unit_features为g FP32[N160]、logits原FP32[N6]、raw/marginFP64[N2]。阈值在deployment路径，不在root另存；所有identity字段与父逐值一致，fresh字段为父复用而非新真实F前向。

rootcomplete至少含run/code/arm、parent_run_id/code_commit/parent_prediction_root/cache_root/teacher_run_id/code_commit、model_identity/dura_identity/cache_identities、known27000/novel3150/source_L6300/source_OE6134、g_parameters20704/g_training_steps(0或1000)/g_training_L_exposures(0或120000)/g_training_OE_exposures(0或128000)/g_forward_rows（训练L+OE+L6300聚合+V27000+novel3150）、F_forward_rows2/G_forward_rows0/real_IQ_values_read0/source_L_metadata_labels_read6300/neural_F_updates0/G_updates0/source_U_reads0/target_reads0/source_novel_truth_reads0、source_only=true/classification_pathP1/default_head_replaced=false、model_parameters_buffers_unchanged=true（指F0）/metric_parameters_buffers_frozen_for_V_novel=true/aggregate_frozen_before_V=true/all_thresholds_frozen_before_novel=true/source_free_deployment_state=true/per_record_source_rows_retained0、cpu_readout_seconds（仅FP64聚合score）/g_evaluation_seconds（V/novelg前向，CUDA同步计时）/fresh_closed_vs_cached_baseline_changed_count0/A_Nnull/H_OWnull/final_phase3_completefalse。training_complete保存完整预算与实际unique覆盖及loss最终/首值，未训练identity明示0。

scorer `star_metric_embedding_score.py`：rootprediction/deployment/cal/train终态及公有来源/默认头/冻结状态/预算/model4tensor/aggregate shapes和12τ一致全部核对；独立加载部署网络，仅用原F0 units→g/aggregate重建新单位/两score和margin，确认deployment无source逐样本state；不重新训练、不重新fit聚合/τ、不重验父全部3scores或读IQ。全部公有验证后才读private novel truth，沿common._masks/_goals/metrics/CI主报每臂两法overall/42known/12unknown/逐TXRXscene；可仅scorer连接真值后比较g与原F0的source novel held-RX1NN结构，诊断不是AN且不新增结构晋级gate。每臂科学source_gate_passed=固定conditional法known42且unknown12过90%；eligible_primary仅PRIMARY_ARM科学PASS为true，nextstep主PASS=`SOURCE_METRIC_EMBEDDING_READY_FOR_TARGET_FALSIFICATION`、控制PASS=`CONTROL_ONLY_COMPLETE`、失败`NO_PROMOTION`，fullgoalACTIVE/targetaccessfalse/AN/HOWnull。

launcher `star_metric_embedding_launch.py`：沿原12fields runtime，rows exact三ARMS各source_config，max_concurrent_rows2、minfree4096MB/每卡最多2新增，保留foreign；每臂source→独立CPUscore，root全部完成后只以固定PRIMARY_ARM/PRIMARY_METHOD判科学晋级，不换控制。无query/目标访问；source完成一臂即允许该臂独立score，不因其他臂未完而阻塞。每run唯一owner/新immutable root，不热改/重跑healthy行。

## 验收与执行边界

两训练臂1000+1000=2000新optimizer步；identity0并明确非同增量预算训练控制。所有训练监督和源聚合均只L/OE，V不更新模型，novel真值仅独立scorer。部署推理只需F0给出的单位特征/默认身份、固定g权重、四聚合和六τ，不需要源样本。F/G499944+g20704=520648神经参数，g FP32权重82816B、每臂aggregate418560B、两法12τ96B，运行日志/训练feature缓存不是部署state。

一次聚焦协议负测+真实checkpoint首步smoke、一次独立P0/P1、实际code/config commit、最小登记、单N607 preflight/archiveSHA/compile与PID/CWD/cmd/GPU/log增长、每臂完整预测后独立truth-last评分。stop仅来源/角色/权限/非有限/冲突/无法合法prediction，保留partial；性能差不停止/调参/换主或选择性重跑。报告完整loss轨迹/实际曝光/源指标及成本，GPU采样与批量耗时不冒充B1。source PASS只允许另记小目标证伪；不直接扩3seed、不声称全目标已达成。
