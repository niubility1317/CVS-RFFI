# 实验总索引

更新：2026-10-03T12:13:36+00:00

先按方法/问题检索，再用run ID读取精确配置与证据。历史组数量不等于独立实验数量。

- [管理规范与常用命令](../docs/EXPERIMENT_MANAGEMENT.md)
- [完整目录CSV](catalog.csv) · [结构化目录](catalog.jsonl) · [覆盖与缺项](coverage.json)
- 通过show的`--section artifacts/facts`读取该条目的路径或配置出处；细目按记录压缩保存于details/，不扫描全库。

## 登记规模

|记录类型|数量|
|---|---:|
|managed_run|96|
|legacy_evidence_group|724|

历史状态统一为HISTORICAL_UNVERIFIED；RUNNING等原文声明只供查证，不能证明此刻仍在运行。

## 按方法与用途查找

|路径标签（定位提示）|证据组数|
|---|---:|
|[cvs](by_method/cvs.md)|241|
|[adv3b02](by_method/adv3b02.md)|146|
|[phase2](by_method/phase2.md)|118|
|[phase1](by_method/phase1.md)|113|
|[stage2](by_method/stage2.md)|67|
|[clean_only](by_method/clean_only.md)|57|
|[ce_only](by_method/ce_only.md)|56|
|[qknn](by_method/qknn.md)|56|
|[performance_priority](by_method/performance_priority.md)|48|
|[no_augmentation](by_method/no_augmentation.md)|40|
|[source_selection](by_method/source_selection.md)|35|
|[fasttrust](by_method/fasttrust.md)|26|
|[source](by_method/source.md)|23|
|[rff_physics](by_method/rff_physics.md)|20|
|[source_selected](by_method/source_selected.md)|18|
|[frozen_prediction_reuse](by_method/frozen_prediction_reuse.md)|17|
|[source_only](by_method/source_only.md)|15|
|[d92](by_method/d92.md)|15|
|[diagnostic](by_method/diagnostic.md)|14|
|[response](by_method/response.md)|13|
|[daot](by_method/daot.md)|11|
|[drift](by_method/drift.md)|11|
|[rc4](by_method/rc4.md)|10|
|[comparison](by_method/comparison.md)|10|
|[riei](by_method/riei.md)|10|
|[no_training](by_method/no_training.md)|9|
|[practical](by_method/practical.md)|8|
|[residual_noeq](by_method/residual_noeq.md)|8|
|[mopc](by_method/mopc.md)|8|
|[csil](by_method/csil.md)|8|
|[concat](by_method/concat.md)|7|
|[residual](by_method/residual.md)|7|
|[truth_last](by_method/truth_last.md)|7|
|[lightweight](by_method/lightweight.md)|6|
|[core90](by_method/core90.md)|6|
|[full](by_method/full.md)|5|
|[zf](by_method/zf.md)|5|
|[mmse](by_method/mmse.md)|5|
|[relative_cfo](by_method/relative_cfo.md)|5|
|[final200](by_method/final200.md)|4|
|[mirror_relation](by_method/mirror_relation.md)|4|
|[receiver_agnostic](by_method/receiver_agnostic.md)|4|
|[repair](by_method/repair.md)|4|
|[shot](by_method/shot.md)|4|
|[sixscene](by_method/sixscene.md)|3|
|[identity_only](by_method/identity_only.md)|3|
|[scratch](by_method/scratch.md)|3|
|[no_target_access](by_method/no_target_access.md)|3|
|[frozen_weights](by_method/frozen_weights.md)|3|
|[packet_synchronization](by_method/packet_synchronization.md)|3|
|[fixed_parameter_budget](by_method/fixed_parameter_budget.md)|3|
|[public_only](by_method/public_only.md)|3|
|[packet_compensation](by_method/packet_compensation.md)|3|
|[smoke](by_method/smoke.md)|3|
|[evaluation](by_method/evaluation.md)|2|
|[true256](by_method/true256.md)|2|
|[final_eval](by_method/final_eval.md)|2|
|[source_scratch](by_method/source_scratch.md)|2|
|[native_methods](by_method/native_methods.md)|2|
|[architecture](by_method/architecture.md)|2|
|[cvs_identity_ce](by_method/cvs_identity_ce.md)|2|
|[cvcnn_matched](by_method/cvcnn_matched.md)|2|
|[ablation](by_method/ablation.md)|2|
|[residual_fusion](by_method/residual_fusion.md)|2|
|[coordinate_usage](by_method/coordinate_usage.md)|2|
|[phase_curvature](by_method/phase_curvature.md)|2|
|[frozen](by_method/frozen.md)|2|
|[preamble](by_method/preamble.md)|2|
|[complex_volterra_memory](by_method/complex_volterra_memory.md)|2|
|[two_extra_parameters](by_method/two_extra_parameters.md)|2|
|[phase_equivariant](by_method/phase_equivariant.md)|2|
|[complex_memory](by_method/complex_memory.md)|2|
|[benchmark_informed](by_method/benchmark_informed.md)|2|
|[valid_history](by_method/valid_history.md)|2|
|[dual_path](by_method/dual_path.md)|2|
|[fixed_comparison](by_method/fixed_comparison.md)|2|
|[three_seed](by_method/three_seed.md)|2|
|[nm fdu](by_method/nm_fdu.md)|2|
|[protonet](by_method/protonet.md)|2|
|[original_leo](by_method/original_leo.md)|1|
|[ratio](by_method/ratio.md)|1|
|[fasttrust_rc4](by_method/fasttrust_rc4.md)|1|
|[d92_parent](by_method/d92_parent.md)|1|
|[d42](by_method/d42.md)|1|
|[support_only](by_method/support_only.md)|1|
|[baseline](by_method/baseline.md)|1|
|[four_seeds](by_method/four_seeds.md)|1|
|[attentive_pooling](by_method/attentive_pooling.md)|1|
|[balanced_fusion](by_method/balanced_fusion.md)|1|
|[physics](by_method/physics.md)|1|
|[cvcnn](by_method/cvcnn.md)|1|
|[resnet1d](by_method/resnet1d.md)|1|
|[real_cnn](by_method/real_cnn.md)|1|
|[complex_coherence](by_method/complex_coherence.md)|1|
|[time_frequency_interaction](by_method/time_frequency_interaction.md)|1|
|[frozen_baseline_reuse](by_method/frozen_baseline_reuse.md)|1|
|[simplex_classifier](by_method/simplex_classifier.md)|1|
|[physical_stability](by_method/physical_stability.md)|1|
|[information_diagnostic](by_method/information_diagnostic.md)|1|
|[paired_ablation](by_method/paired_ablation.md)|1|
|[public_synthetic](by_method/public_synthetic.md)|1|
|[numerical_attribution](by_method/numerical_attribution.md)|1|
|[received_sensitivity](by_method/received_sensitivity.md)|1|
|[synthetic_only](by_method/synthetic_only.md)|1|
|[causal_envelope_memory](by_method/causal_envelope_memory.md)|1|
|[cross_channel_phase](by_method/cross_channel_phase.md)|1|
|[fixed_readout_dimensions](by_method/fixed_readout_dimensions.md)|1|
|[shared_energy_normalization](by_method/shared_energy_normalization.md)|1|
|[relative_filter_energy](by_method/relative_filter_energy.md)|1|
|[full_fp32](by_method/full_fp32.md)|1|
|[numerical_physics_consistency](by_method/numerical_physics_consistency.md)|1|
|[fractional_synchronization](by_method/fractional_synchronization.md)|1|
|[whole_identity_gauge](by_method/whole_identity_gauge.md)|1|
|[packet_moment_residual](by_method/packet_moment_residual.md)|1|
|[neural_residual](by_method/neural_residual.md)|1|
|[learned_convolutions](by_method/learned_convolutions.md)|1|
|[whole_identity_observables](by_method/whole_identity_observables.md)|1|
|[packet_orthogonal_envelope](by_method/packet_orthogonal_envelope.md)|1|
|[six_extra_parameters](by_method/six_extra_parameters.md)|1|
|[known_excitation](by_method/known_excitation.md)|1|
|[frozen_evaluation](by_method/frozen_evaluation.md)|1|
|[rf_behavior_operator](by_method/rf_behavior_operator.md)|1|
|[user_fixed_architecture](by_method/user_fixed_architecture.md)|1|
|[channel_attribution](by_method/channel_attribution.md)|1|
|[frontfilter_attribution](by_method/frontfilter_attribution.md)|1|
|[history_boundary](by_method/history_boundary.md)|1|
|[readout_attribution](by_method/readout_attribution.md)|1|
|[relation_attribution](by_method/relation_attribution.md)|1|
|[response_attribution](by_method/response_attribution.md)|1|
|[response_geometry](by_method/response_geometry.md)|1|
|[backfill](by_method/backfill.md)|1|
|[all_frozen](by_method/all_frozen.md)|1|
|[channel_order](by_method/channel_order.md)|1|
|[response_order](by_method/response_order.md)|1|
|[crosspath_relation](by_method/crosspath_relation.md)|1|
|[frontfilter](by_method/frontfilter.md)|1|
|[neural_readout](by_method/neural_readout.md)|1|
|[context_attention](by_method/context_attention.md)|1|
|[response_fusion](by_method/response_fusion.md)|1|
|[spectral_relation](by_method/spectral_relation.md)|1|
|[independent_test](by_method/independent_test.md)|1|
|[ecrs](by_method/ecrs.md)|1|
|[fedcvs](by_method/fedcvs.md)|1|

## 最近记录入口

|名称|类型|报告/原目录|
|---|---|---|
|ADV3B02＋DAOT＋FastTrust-RC4原LEO拼接增强|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-original-leo-manysig-s392005-r01/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r01/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r02/report.md)|
|DAOT＋FastTrust-RC4 practical四组实验r03|managed_run|[打开](../automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r03/report.md)|
|practical四组最新保存权重测试|managed_run|[打开](../automation_reports/CV-SincNet/20260919-phase1-daot-rc4-practical4-test-s392005-r01/report.md)|
|practical四组最新保存权重测试r02|managed_run|[打开](../automation_reports/CV-SincNet/20260919-phase1-daot-rc4-practical4-test-s392005-r02/report.md)|
|DAOT＋FastTrust-RC4 residual环境比例六组|managed_run|[打开](../automation_reports/CV-SincNet/20260920-phase1-daot-rc4-residual-ratios-manysig-s392005-r01/report.md)|
|旧Practical四组checkpoint统一六环境测试|managed_run|[打开](../automation_reports/CV-SincNet/20260920-phase1-practical4-sixscene-eval-s392005-r01/report.md)|
|CVS：DAOT＋FastTrust-RC4原始residual_noeq五种子实验|managed_run|[打开](../automation_reports/CV-SincNet/20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01/report.md)|
|CVS最终clean/星地测试与D92 E0真实256维注册|managed_run|[打开](../automation_reports/CV-SincNet/20260927-phase2-cvs-d92-practical-manytx-m5-r01/report.md)|
|D42 support-only技术诊断|managed_run|[打开](../automation_reports/CV-SincNet/20260928-diagnostic-d42-support-manytx-m2-r01/report.md)|
|D92 E0 true256 numerical recovery: all five fixed seeds|managed_run|[打开](../automation_reports/CV-SincNet/20260928-phase2-cvs-d92-practical-manytx-m5-r02/report.md)|
|Native comparison source training|managed_run|[打开](../automation_reports/CV-SincNet/20260930-phase1-native-baselines-practical-manysig-m5-r01/report.md)|
|Native comparison paired adaptation and registration|managed_run|[打开](../automation_reports/CV-SincNet/20260930-phase12-native-baselines-practical-m5-r01/report.md)|
|无信道增强四基准的 clean 测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-clean-baselines-manysig-m16-r01/report.md)|
|CVS性能优先继续优化：attentive_mean clean测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-attentive-clean-manysig-m24-r01/report.md)|
|CVS 性能优先研发：包内注意力统计池化|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-attentive-identity-manysig-m8-r01/report.md)|
|CVS性能优先继续优化：balanced_fusion clean测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-balanced-clean-manysig-m24-r01/report.md)|
|CVS继续研发：分支平衡融合与有符号PA投影|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-balanced-identity-manysig-m8-r01/report.md)|
|CVS 无增强轻量架构与常见网络基准|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-clean-architecture-manysig-m32-r01/report.md)|
|CVS性能优先继续优化：coherence_phase clean测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-coherence-clean-manysig-m24-r01/report.md)|
|CVS 连续复相关身份表征|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-coherence-identity-manysig-m8-r01/report.md)|
|CVS身份骨干交叉熵对照实验|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-identity-ce-practical-manysig-m5-r01/report.md)|
|CVS身份骨干交叉熵对照实验|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-identity-ce-practical-manysig-m5-r02/report.md)|
|CVS性能优先继续优化：tf_lowrank32 clean测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-interaction-clean-manysig-m24-r01/report.md)|
|CVS性能优先研发：时频交互|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-interaction-identity-manysig-m8-r01/report.md)|
|CVS 基础网络研发：残差物理融合|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-residual-identity-manysig-m8-r01/report.md)|
|CVS 结构优化：轻量残差融合的 clean 确认报告|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-selected-clean-manysig-m20-r01/report.md)|
|CVS性能优先继续优化：simplex_fixed clean测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-simplex-clean-manysig-m24-r01/report.md)|
|CVS 等角身份分类头|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-simplex-identity-manysig-m8-r01/report.md)|
|CVS性能优先继续优化：phase_dsq clean测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-stability-clean-manysig-m24-r01/report.md)|
|CVS 性能优先研发：物理补充表征|managed_run|[打开](../automation_reports/CV-SincNet/20261001-phase1-cvs-stability-identity-manysig-m8-r01/report.md)|
|CVS 加性坐标实际使用：完整冻结源域消融报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-diagnostic-cvs-additive-usage-source-manysig-m40-r01/report.md)|
|CVS 相对频偏坐标：完整源域身份信息诊断|managed_run|[打开](../automation_reports/CV-SincNet/20261002-diagnostic-cvs-cfo-information-source-manysig-m1-r01/report.md)|
|CVS 坐标实际使用：完整冻结源域消融报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-diagnostic-cvs-coordinate-usage-source-manysig-m40-r01/report.md)|
|CVS相位曲率：冻结源域归因结果|managed_run|[打开](../automation_reports/CV-SincNet/20261002-diagnostic-cvs-curvature-attribution-source-manysig-m8-r01/report.md)|
|冻结 CVS 源权重的公共合成相位数值归因|managed_run|[打开](../automation_reports/CV-SincNet/20261002-diagnostic-cvs-equivariant-numerics-public-m12-r01/report.md)|
|CVS 完整源 V 接收变换与身份路径诊断|managed_run|[打开](../automation_reports/CV-SincNet/20261002-diagnostic-cvs-received-sensitivity-source-manysig-m56-r01/report.md)|
|RFF物理可辨识性：合成诊断|managed_run|[打开](../automation_reports/CV-SincNet/20261002-diagnostic-rff-identifiability-synthetic-m1-r01/report.md)|
|RFF 源域已知激励诊断：预登记|managed_run|[打开](../automation_reports/CV-SincNet/20261002-diagnostic-rff-known-excitation-source-manysig-m1-r01/report.md)|
|RFF源域前导码与可观测量诊断|managed_run|[打开](../automation_reports/CV-SincNet/20261002-diagnostic-rff-preamble-source-manysig-m1-r01/report.md)|
|CVS 可学习相位记忆残差身份网络：独立 clean 测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-adaptive-volterra-clean-manysig-m32-r01/report.md)|
|CVS 可学习相位记忆残差：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-adaptive-volterra-identity-manysig-m8-r01/report.md)|
|CVS 加性坐标注入身份网络：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-additive-identity-manysig-m8-r01/report.md)|
|CVS 同步坐标保留身份网络：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-coordinate-identity-manysig-m8-r01/report.md)|
|CVS 固定参数因果包络耦合身份网络：独立 clean 测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-coupled-clean-manysig-m28-r01/report.md)|
|CVS 因果包络耦合：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-coupled-identity-manysig-m8-r01/report.md)|
|CVS 跨复数通道相干读出：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-crossphase-identity-manysig-m8-r01/report.md)|
|CVS 相对能量保留身份网络：独立 clean 测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-energy-clean-manysig-m24-r01/report.md)|
|CVS 相对滤波能量保留身份网络：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-energy-identity-manysig-m8-r01/report.md)|
|CVS 全路径复相位等变记忆网络：独立 clean 测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-equivariant-clean-manysig-m24-r01/report.md)|
|CVS 整网复相位约束：完整 FP32 独立 clean 报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-equivariant-fp32-clean-manysig-m24-r01/report.md)|
|CVS 整网复相位约束：完整 FP32 源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-equivariant-fp32-manysig-m4-r01/report.md)|
|CVS 全路径复相位等变记忆网络：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-equivariant-identity-manysig-m4-r01/report.md)|
|CVS 波形层分数频偏校正身份网络：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-fractional-identity-manysig-m8-r01/report.md)|
|完整 IQ 相位规范化 CVS：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-gauge-identity-manysig-m8-r01/report.md)|
|CVS可学习包内矩残差：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-moment-residual-identity-manysig-m8-r01/report.md)|
|CVS可学习复卷积残差：独立clean结果|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-neural-residual-clean-manysig-m36-r01/report.md)|
|CVS可学习复卷积残差：完整源训练结果|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-neural-residual-identity-manysig-m8-r01/report.md)|
|CVS性能优先继续优化：observable_phase clean测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-observable-clean-manysig-m24-r01/report.md)|
|CVS整体物理观测身份网络：源训练预登记|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-observable-identity-manysig-m8-r01/report.md)|
|CVS 包内正交包络输入：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-orthopoly-identity-manysig-m8-r01/report.md)|
|CVS六层相位曲率残差：完整源实验结果|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-phase-curvature-identity-manysig-m8-r01/report.md)|
|CVS 已知激励相对响应：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-reference-identity-manysig-m4-r01/report.md)|
|residual_fusion 完整六环境测试结果|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/report.md)|
|CVS性能优先继续优化：rf_gmp clean测试报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-rf-operator-clean-manysig-m24-r01/report.md)|
|CVS 受约束射频行为算子|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-rf-operator-identity-manysig-m8-r01/report.md)|
|CVS 固定身份网络：mid／low urban 拼接增强|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-selected-concat-manysig-m4-r01/report.md)|
|CVS 全路径逐包同步身份网络：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-synchronized-identity-manysig-m8-r01/report.md)|
|CVS 复数Volterra延迟相位记忆：完整源实验报告|managed_run|[打开](../automation_reports/CV-SincNet/20261002-phase1-cvs-volterra-identity-manysig-m8-r01/report.md)|
|CVS补偿与顺序差：完整源V冻结归因|managed_run|[打开](../automation_reports/CV-SincNet/20261003-diagnostic-cvs-channel-attribution-source-manysig-m8-r01/report.md)|
|CVS全主干前置滤波：固定权重源域反事实|managed_run|[打开](../automation_reports/CV-SincNet/20261003-diagnostic-cvs-frontfilter-attribution-source-manysig-m8-r01/report.md)|
|连续前历史与零启动：公开FIR机制验证|managed_run|[打开](../automation_reports/CV-SincNet/20261003-diagnostic-cvs-mirror-history-public-m8-r01/report.md)|
|CVS镜像关系：冻结公开信号机制分解|managed_run|[打开](../automation_reports/CV-SincNet/20261003-diagnostic-cvs-mirror-mechanism-public-m8-r01/report.md)|
|CVS镜像关系：冻结公开信号机制分解|managed_run|[打开](../automation_reports/CV-SincNet/20261003-diagnostic-cvs-mirror-mechanism-public-m8-r02/report.md)|
|CVS读出：完整源V分支归因与TX/RX/日期关联|managed_run|[打开](../automation_reports/CV-SincNet/20261003-diagnostic-cvs-readout-attribution-source-manysig-m8-r01/report.md)|
|谱关系与镜像子空间的完整源域归因|managed_run|[打开](../automation_reports/CV-SincNet/20261003-diagnostic-cvs-relation-attribution-source-manysig-m12-r01/report.md)|
|CVS显式补偿响应：完整源V冻结归因|managed_run|[打开](../automation_reports/CV-SincNet/20261003-diagnostic-cvs-response-attribution-source-manysig-m12-r01/report.md)|
|CVS补偿响应：完整源V判别几何|managed_run|[打开](../automation_reports/CV-SincNet/20261003-diagnostic-cvs-response-geometry-source-manysig-m12-r01/report.md)|
|全部既有冻结架构的clean测试结果|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-all-frozen-clean-backfill-manysig-m304-r01/report.md)|
|CVS 信道与非线性运算次序：独立 clean 结果|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-channel-order-clean-manysig-m40-r01/report.md)|
|CVS信道补偿与顺序差值：完整源训练结果|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-channel-order-identity-manysig-m16-r01/report.md)|
|CVS显式补偿响应：完整源训练结果|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-channel-response-identity-manysig-m12-r01/report.md)|
|CVS跨路径关系读出：完整源训练结果|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-crosspath-relation-identity-manysig-m8-r01/report.md)|
|CVS全主干前置滤波：完整源训练结果|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-frontfilter-identity-manysig-m8-r01/report.md)|
|CVS镜像频对关系：能量与子空间投影×四seed|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-mirror-subspace-identity-manysig-m8-r01/report.md)|
|CVS可学习注意力读出：完整源训练结果|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-neural-readout-identity-manysig-m8-r01/report.md)|
|CVS 约束响应融合：独立 clean 结果|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-response-fusion-clean-manysig-m44-r01/report.md)|
|CVS约束响应融合：完整源训练结果|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-response-fusion-identity-manysig-m8-r01/report.md)|
|CVS 频谱时序关系：独立 clean 结果|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-spectral-relation-clean-manysig-m48-r01/report.md)|
|CVS频点内可学习时间关系：整包与逐频能量归一化×四seed|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-spectral-relation-identity-manysig-m8-r01/report.md)|
|有效历史双路径：全部8份E200权重的clean测试|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-validdual-clean-manysig-m8-r01/report.md)|
|有效历史滤波与原始指纹双路径|managed_run|[打开](../automation_reports/CV-SincNet/20261003-phase1-cvs-validdual-identity-manysig-m8-r01/report.md)|
|原生DAOT＋RC4与仅动力博弈三seed对照|managed_run|[打开](../automation_reports/CV-SincNet/phase1_daot_rc4_pure_game_m3_20260917_r1/report.md)|
|原生DAOT＋RC4与仅动力博弈三seed对照|managed_run|[打开](../automation_reports/CV-SincNet/phase1_daot_rc4_pure_game_m3_20260917_r2/report.md)|
|响应博弈矩阵测试评估：VERIFIED|managed_run|[打开](../automation_reports/CV-SincNet/response_matrix_eval_20260917/report.md)|
|补偿响应融合：阻断分类无关方向的归一化捷径|legacy_evidence_group|[打开](../docs/CVS_RESPONSE_FUSION_DESIGN_20261003.md)|
|cvs_response_geometry_source_20261003_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_response_geometry_source_20261003_r01)|
|显式补偿响应：下一轮Phase1架构|legacy_evidence_group|[打开](../docs/CVS_CHANNEL_RESPONSE_DESIGN_20261003.md)|
|Phase1补偿响应架构：创新边界与证据缺口|legacy_evidence_group|[打开](../docs/CVS_CHANNEL_RESPONSE_NOVELTY_20261003.md)|
|cvs_response_attribution_source_20261003_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_response_attribution_source_20261003_r01)|
|cvs_channel_response_identity_20261003_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_channel_response_identity_20261003_r01)|
|clean_eval_protocol_fixtures|legacy_evidence_group|[打开](../local_artifacts/clean_eval_protocol_fixtures)|
|cvs_channel_response_identity_cpu_smoke|legacy_evidence_group|[打开](../local_artifacts/cvs_channel_response_identity_cpu_smoke.json)|
|cvs_channel_attribution_source_20261003_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_channel_attribution_source_20261003_r01)|
|Phase1架构研究：信道鲁棒性与射频指纹保留|legacy_evidence_group|[打开](../docs/CVS_PHASE1_CHANNEL_ROBUST_ARCHITECTURE_20261003.md)|
|CVS信道补偿与非线性交互：Phase1结构实验|legacy_evidence_group|[打开](../docs/CVS_CHANNEL_ORDER_EXPERIMENT_20261003.md)|
|cvs_channel_order_clean_eval_20261003_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_channel_order_clean_eval_20261003_r01)|
|cvs_channel_order_identity_20261003_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_channel_order_identity_20261003_r01)|
|cvs_channel_order_source_validation_20261003_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_channel_order_source_validation_20261003_r01.json)|
|CVS纯神经网络读出优化|legacy_evidence_group|[打开](../docs/CVS_NEURAL_READOUT_20261003.md)|
|cvs_neural_readout_identity_20261003_model_cpu_smoke_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_neural_readout_identity_20261003_model_cpu_smoke_r01.json)|
|CVS基础网络优化诊断|legacy_evidence_group|[打开](../docs/CVS_NETWORK_DIAGNOSIS_20261002.md)|
|CVS可学习复卷积残差结构实验|legacy_evidence_group|[打开](../docs/CVS_NEURAL_RESIDUAL_20261002.md)|
|cvs_neural_residual_clean_eval_20261002_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_neural_residual_clean_eval_20261002_r01)|
|cvs_neural_residual_identity_20261002_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_neural_residual_identity_20261002_r01)|
|cvs_curvature_attribution_source_20261002_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_curvature_attribution_source_20261002_r01)|
|cvs_phase_curvature_identity_20261002_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_phase_curvature_identity_20261002_r01)|
|CVS分层相位曲率残差|legacy_evidence_group|[打开](../docs/CVS_FEATURE_PHASE_CURVATURE_20261002.md)|
|cvs_moment_residual_identity_20261002_r01|legacy_evidence_group|[打开](../local_artifacts/cvs_moment_residual_identity_20261002_r01)|
