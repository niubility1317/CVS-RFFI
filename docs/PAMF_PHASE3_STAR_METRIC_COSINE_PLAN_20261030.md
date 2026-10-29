# 冻结主度量的原型余弦读出

Run `20261030-phase3-star-metric-cosine-s2026092701-r01`。只使用前轮固定主 `compact_oe_structure` 的冻结状态，不能改用较高分的compactOE控制。固定两新分数 `predicted_cosine`、`maximum_cosine`，唯一主为maximum_cosine；父原Mahalanobis运行点作为精确复用的外部匹配参考，不重新校准。目标/阈值保护保持用户v3：跨未见RX星地CK/RU≥90%、同输入原F0closed−AK≤2pp，主报trueTX宏AK/RU/H，完整Phase3 ACTIVE。

## 方法与数据

前轮g的OE排斥最小化relu(max_c cosine(g(x),p_c))²，本轮直接检验相同变量作为拒识分数。取父部署metric.npz的L-only class_means[6,160]，FP64逐类单位化为prototypes；不重新读取L/OE逐样本特征、不拟合网络/协方差。query使用父保存的FP32 metric_unit_features，FP64逐行再单位化。完整6类余弦矩阵使用固定einsum(optimize=False)，新分数为cosine[original_closed]与max_c cosine；分数越高越known。最大余弦只作拒识，身份仍原F0 default logits argmax，不能改成argmax cosine。无query真TX/RX/scene/K或query群体统计。

父prediction_root固定 `/runs/20261030-phase3-star-metric-embedding-s2026092701-r01/rows/compact_oe_structure/predict`，父run/code为该r01/efb82a4177e9b59533b7e71408ded6b63fc8733a，g1000步/20704参数、原F0 E50/default6head/232993与DURA266951及同G10MIX01 P1祖先不变。父PRED complete/deployment/cal均只含合法公有状态；producer禁止读取任何父score文件（它含truth派生结果）。可复用`star_metric_embedding_score._deployment`恢复实际父g/四L聚合/旧τ，并用其`_reference_smoke`做2条实际checkpoint synthetic-unit noquery smoke，随后立即继续。不得调用父`_prediction`，因为它会在新τ冻结前打开父novel。

先父完整身份/来源/部署合法句柄→实际g smoke2→固定prototypes并复制同一四FP32tensor model.pt→父known27000公有特征/logits/metadata→两cos分数各六τ（`star_class_risk.calibrate_known` known42宏CK90/原F0同P1损2pp complete ties）→部署包/12τ/cal/known全部冻结→父novel3150公有特征打开→两cos预测complete。无g训练或新真实g/F/G前向；额外g2仅smoke，独立scorer新g30150 CPU FP64参考另计。父known/NV均同原g单位特征，不因新读出重新生成；baseline/fresh沿父复用，不宣称新F0实测。

同42584物理记录祖先（L6300/OE6134/V27000/novel3150），本轮只消费V/NV的既有公有特征与L聚合。source RX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）和2021-03-08/15/23已见，旧6TX14-10/14-7/20-15/20-19/6-15/8-20，V各4500，historicalnovel19-15/19-16/7-1各1050。原Phase1契约392005/L6300/Us56700/V27000/10%标签不改；g祖先含L旧TX/RX和OE成员监督，V本轮只τ。targetUQ/正式RX24-18/原Us新读取0，OE真TX/K及target标签不读取。源设备和RX历史曝光必须披露，不作新RX/TX盲测或匿名注册声明。

## 最小实现合同

仅两个模块及两测试：

- `star_metric_cosine.py`负责纯函数、producer与CPU pipeline；导出RUN_ID、SCHEMA=`pamf_phase3_star_metric_cosine_prediction_v1`、STATUS=`STAR_METRIC_COSINE_PREDICTIONS_COMPLETE`、CALIBRATION_STATUS=`STAR_METRIC_COSINE_THRESHOLDS_FROZEN`、METHODS=(predicted_cosine,maximum_cosine)、PRIMARY_METHOD=maximum_cosine、COUNTS=V27000/novel3150（可另带L/OE祖先）、KNOWN_FIELDS/NOVEL_FIELDS沿父g预测exact同keys（raw/scores仍N2）、DEPLOYMENT_SCHEMA=`pamf_phase3_star_metric_cosine_deployment_v1`/STATUS=`STAR_METRIC_COSINE_DEPLOYMENT_FROZEN`。`prototypes(class_means)->FP64[6,160]`、`cosine_scores(metric_units,closed,prototypes)->FP64[N,2]`，旧均值FP64有限非零，units FP32且norm容差2e-6，closed原F0int64[0..5]。源聚合和query输入边界必要校验，行/块/置换不变与不改state。
- producer exact4 config run_id/code_commit/parent_prediction_root/output_root，parent arm必须compact_oe_structure/原run/code及sourcefree/noFupdate/noveltruth0，output不可覆盖原输入。父KNOWN/NOVEL exact字段和原defaultargmax/baseline/metadata/单位features检查，物理ID角色互斥复用既有句柄；只有新known阈值拟合，不读sourceL/OE raw或任何score。父cal/旧12τ只用于标识旧MD参考，不能从旧novel表现选读出。
- 运行时CPU-only exact8 fields run_id/code_commit/release_root/run_root/python/launch_owner/parent_prediction_root/novel_truth；本module CLI `start --runtime`→background`pipeline --runtime`。start不可覆盖runroot，记录launch.json/pid/cwd/argv；pipeline保存config.json（producer仅4fields不含truth）、独立`predict --config` subprocess→完整成功后独立`star_metric_cosine_score` subprocess（仅此收到truth路径）→pipeline_state.json终态ANALYZED_SOURCE_ONLY或FAILED_PRESERVED。只一个source/随后一个scorer，无GPU/foreign干预；失败保留，不重启或换主。
- `star_metric_cosine_score.py`/tests负责全部公有来源/部署/12τ/score/defaulthead与父非score数组逐值复用检验后才private truth。从新部署四FP32tensor恢复g并用CPU FP64参考重建父metric_unit_features（沿既定rtol1e-5/atol2e-6），再由已核验saved production单位与prototypes重算新raw/margin（FP64atol1e-12，固定einsum），不train/refit。对比prototypes确为父L class_means单位化；不重验父全部原分数或训练全trace，不读source行/真实IQ。两新法主报overall/known42/unknown12/逐TXRXscene/AUROC/OSCR/CI，沿common._masks/_goals/metrics。

部署exact4files model.pt（仅copy父同4FP32tensor）、prototypes.npz（exact key prototypes，FP64[6,160]）、thresholds.npz（conditionalFP64[2,6]）、complete.json（schema/status/run/code、parameters20704/source_free=true/per_record_source_rows_retained0）。不导出旧Σ；原MD参考只scorer从父原raw/margin和旧τ复用，不能把该参考声称为新四文件可计算的分数。研究root还包含smoke.json/known_predictions.npz/novel_predictions.npz/calibration/complete.json/rootcomplete。

rootcomplete含run/code、parent_run_id/code_commit/parent_arm/parent_prediction_root、model_identity/dura_identity/cache_identities（父继承）、methods/primary、known_count27000/novel_count3150、g_parameters20704、inherited_g_training_steps1000/new_neural_updates0/F_forward_rows0/G_forward_rows0/g_forward_rows2/synthetic_unit_smoke_rows2/real_IQ_values_read0/source_L_forward_rows0/source_OE_forward_rows0/source_L_metadata_labels_read0/target_reads0/source_U_reads0/source_novel_truth_reads0、sourceonly/classification_pathP1/default_head_replacedfalse、prototypes_frozen_before_V/all_thresholds_frozen_before_novel/source_free_deployment_state/parent_public_features_reused/deployed_g_unchanged全true、per_record_source_rows_retained0、cpu_readout_seconds/rows30150、fresh_closed_is_parent_reuse=true/A_Nnull/H_OWnull/final_phase3_completefalse。

score STATUS=`STAR_METRIC_COSINE_SCORES_COMPLETE`、RUN_ID新run/code用新producer，primary_summary/primary_goals/source_gate_passed/eligible_primary按固定maximum_cosine known42&unknown整体/5RX/6scene12组90%判；PASS nextstep SOURCE_METRIC_COSINE_READY_FOR_TARGET_FALSIFICATION，否则NO_PROMOTION。另`reference_original_mahalanobis`从父公有saved margin列1/原六τ复用，全部公有校验后同一次truth读取算相同指标/paired newprimary−originalMD；不再次拟合或读取父score指标。pipeline只认新fixed主，不用控制较好替代。

## 执行与边界

NN部署520648参数保持（F/G499944+g20704），新g权重payload82816B、prototypes7680B、12τ96B，容器开销另计。新producer只固定读出，无训练、真实IQ/F/G0，实际gcheckpoint synthetic-unit smoke2；独立参考g30150/两cos readout30150另记为评测复核，不是B1部署时延。原2000步/605350g生产前向及F6 synthetic/父F30152和历史P1G成本保留祖先，复用不算新增独立数据或零总成本。

只本轮聚焦数值/协议负测和actualcheckpoint首步smoke、一次直接P0/P1、实际code/config commit、最小登记、单preflight/archiveSHA/compile与PID/CWD/cmd/log增长、全部prediction后的独立truth-last；不重复父fullsuite/训练/数据验证或添加批准/封印。source未知类未达标不作技术停止；技术错误/权限/非有限/输出冲突保留partial。source通过也只另记小目标证伪，目标U/Q和完整零标签DA/未来注册不在本run实现范围，完整目标不缩减。
