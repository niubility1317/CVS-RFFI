# Phase2 SA33 SCG-qKNN协同推理诊断报告

## 基本信息

- 实验ID：`phase2_sa33_collab_scg_qknn_20260704`
- 时间：2026-07-04
- 操作员/agent：Codex
- 指定权重：`SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104`
- 远程checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_sa27_optimization_central_20260527_204005/SA33_sa27_ch2_leo3_ce0p7_r010/best_primary_ood_model.pth`
- 服务器环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 服务器工作树：`/home/szu2070436088/2510044040/CV-SincNet`
- 本地Git镜像：`E:\type10-7\github_publish\CVS-RFFI-repo`
- 结论边界：本轮完成模块实现、同步和全量k=1..5诊断测试；未达到`old 99%且每类>=95%`、`seen-new 97%且每类>=93%`、`unknown reject/FAR目标`，因此只能作为diagnostic evidence，不能作为部署成功或论文成功声明。

## 目标与假设

目标是在CVS Stage2-C单星接收机部署协议下，为多台未见接收机构建可选协同数量的高效推理算法，使协同数量从1到目标接收机总数可选，并在星地信道特征上验证效率、unknown安全性和known性能。

算法假设：单接收机高unknown风险应优先拒识；当至少2个接收机对同一known候选形成支持集确认时，允许`support_confirmed_known`保护分支覆盖由星地信道位移造成的`event_unknown_risk`/`label_unknown_risk`假阳性；若known证据不足，仍由多源unknown证据拒识或defer。

## 文献/方法输入

子agent文献与方法搜索给出的可用方向包括：receiver-agnostic collaborative RFFI、RIEI/FedRIEI跨接收机DG、open-set原型学习、joint prediction+Siamese open-set、LEO联邦学习、L-TTA、HetLoRA和FedProto。落地选择为轻量`SCG-qKNN`，避免在卫星端引入重模型集成：每个接收机只上传小型证据摘要，融合端进行支持校准、风险门控和可变接收机预算决策。

## 本地改动

| 文件 | 目的 |
|---|---|
| `code/evaluation/collaborative_open_set_qknn_eval.py` | 新增`scg_qknn_cvs`融合策略；加入支持集确认、unknown多源证据、单接收机安全边界、支持保护分支和诊断字段；修复`known_guarded_rescue_cvs`在单unknown证据下可能接收的问题；暴露event alignment元数据和pairguard diagnostic-only标记。 |
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | CLI允许`--fusion_policy scg_qknn_cvs`。 |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | 增加SCG known接收、unknown拒识、支持保护、single-unknown defer和protocol元数据测试。 |

本地快照：`E:\type10-7\code\snapshots\phase2_scg_qknn_final_20260704\`

## 验证与同步

| 阶段 | 命令/证据 | 结果 |
|---|---|---|
| N607预检 | `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1` | PASS；直连N607；项目根目录可见；8张RTX 3090可见。 |
| 本地测试 | `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py -q -p no:cacheprovider` | 115 passed。 |
| Git镜像测试 | 同上，工作目录`E:\type10-7\github_publish\CVS-RFFI-repo` | 115 passed。 |
| 远程测试 | `cd /tmp && PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python .../code/tests/test_collaborative_open_set_qknn_eval.py && .../test_phase2_collaborative_open_set_qknn_eval.py` | 65 tests OK+50 tests OK。 |
| 同步方式 | `scp -F E:\type10-7\tools\n607_ssh_config ... N607:/home/szu2070436088/2510044040/CV-SincNet/...` | 三个改动文件已同步。 |

低显存GPU证据：预检和结束后`nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`均显示各GPU约`10/24576MiB`，最终评估使用GPU0/CPU型qKNN评估路径，结束后GPU0仍为`10/24576MiB`。

## 特征导出

输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_sa33_collab_scg_qknn_20260704/features_sa33_stage2c_leo.npz`

SHA256：`d17cc2a4c53c203eff65c190cfd85aa96b46582313ca539ab5075e91cb183b70`

关键命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet &&
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet \
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/export_spaceborne_features.py \
  --ckpt runs/cvs_sa27_optimization_central_20260527_204005/SA33_sa27_ch2_leo3_ce0p7_r010/best_primary_ood_model.pth \
  --wisig_pkl Dataset_WigSig/ManySig.pkl \
  --new_wisig_pkl Dataset_WigSig/ManyTx.pkl \
  --out_npz runs/phase2_sa33_collab_scg_qknn_20260704/features_sa33_stage2c_leo.npz \
  --feature_name z_id \
  --source_tx_ids 14-10,14-7,20-15,20-19,6-15,8-20 \
  --target_old_tx_ids 14-10,14-7,20-15,20-19,6-15,8-20 \
  --new_tx_ids 19-3,3-8 \
  --unknown_tx_ids 10-1,10-10 \
  --source_days 0,1 --source_rxs 1-1,1-19,14-7,18-2,19-2,2-1,2-19 \
  --target_old_days 2,3 --target_old_rxs 20-1,3-19,7-14,7-7,8-8 \
  --new_days 2,3 --new_rxs 20-1,3-19,7-14,7-7,8-8 \
  --max_samples_per_tx 400 --batch_size 512 --device cuda:0 --seed 4070401 \
  --target_new_channel_view satellite --target_new_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --target_old_channel_view satellite --target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --proxy_unknown_channel_view satellite --proxy_unknown_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --star_ground_channel_impl simplified_leo_residual
```

## 最终SCG-qKNN评估

输出JSON：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_sa33_collab_scg_qknn_20260704/collab_open_set_qknn_scg_sa33_qknn8_final_20260704.json`

输出CSV：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_sa33_collab_scg_qknn_20260704/collab_open_set_qknn_scg_sa33_qknn8_final_20260704_evidence.csv`

本地拉回目录：`E:\type10-7\remote_artifacts\phase2_sa33_collab_scg_qknn_20260704\`

JSON SHA256：`5a933181ada57188c854747ec731f9eb3214939e0213931a8ccec13afa5a1e24`

CSV SHA256：`dcb9b83e815626809260c762e8fb979bc18abce552fef82d6a31b3ca1142495e`

关键命令：

```bash
cd /tmp &&
PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code \
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_sa33_collab_scg_qknn_20260704/features_sa33_stage2c_leo.npz \
  --output_json /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_sa33_collab_scg_qknn_20260704/collab_open_set_qknn_scg_sa33_qknn8_final_20260704.json \
  --output_evidence_csv /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_sa33_collab_scg_qknn_20260704/collab_open_set_qknn_scg_sa33_qknn8_final_20260704_evidence.csv \
  --collab_counts all --event_alignment_policy receiver_domain_ranked \
  --fusion_policy scg_qknn_cvs --label_fusion_policy weighted_vote_margin \
  --receiver_class_reliability_policy support_calibrated --receiver_reliability_policy deployment_prior \
  --receiver_selection_policy fixed_receiver_order --support_selection_policy stable_first \
  --unknown_gate_mode support_envelope_evt --qknn_k 8 --k_shot 8 --query_per_class 20 \
  --prototype_score_blend 2.0 --mahalanobis_score_blend 1.0 --mahalanobis_score_temperature 0.2 \
  --class_conformal_enabled --class_conformal_min_support 2 --class_evidence_top_m 3 \
  --scenario_aware --radius_norm 0.3 --unknown_risk_threshold 0.80 \
  --accept_margin_threshold 0.10 --consensus_score_threshold 0.30 --scorer_component_vote_threshold 0.50 \
  --candidate_set_min_receivers 2 --candidate_set_min_top1_receivers 2 \
  --candidate_set_min_conformal_pvalue 0.30 --candidate_set_min_label_receiver_class_reliability 0.75 \
  --candidate_set_max_label_risk_component_agreement 0.625 \
  --candidate_set_shell_reject_risk 0.85 --candidate_set_unknown_reject_risk 0.85 \
  --candidate_set_max_receiver_pair_label_disagreement 0.80 --candidate_set_max_receiver_pair_unknown_risk_range 1.00 \
  --old_gate_min_support_density 0.50 --seen_new_gate_min_support_density 0.50 \
  --include_event_results
```

协议摘要：目标接收机5个；源接收机7个；`event_alignment_policy=receiver_domain_ranked`；`strict_same_event_collaboration=false`；`evidence_row_count=1000`；`group_count=309`；最大预算下eligible group为91。

## 逐k结果

| k | total | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | defer | scg_accept | support_protected | bytes/event | p95_ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 309 | 0.1429 | 0.0000 | 0.0333 | 0.0000 | 0.7167 | 0.1500 | 0.1133 | 67 | 0 | 40.0 | 0.1137 |
| 2 | 241 | 0.2313 | 0.0000 | 0.0400 | 0.0333 | 0.8864 | 0.0000 | 0.1618 | 36 | 36 | 80.0 | 0.1137 |
| 3 | 200 | 0.4083 | 0.1500 | 0.1750 | 0.1500 | 0.8750 | 0.0750 | 0.0550 | 60 | 60 | 120.0 | 0.1137 |
| 4 | 159 | 0.4839 | 0.0000 | 0.1667 | 0.0000 | 0.8056 | 0.0556 | 0.0629 | 56 | 56 | 160.0 | 0.1137 |
| 5 | 91 | 0.3333 | 0.0000 | 0.3000 | 0.0000 | 0.9500 | 0.0500 | 0.0000 | 29 | 29 | 200.0 | 0.1137 |

## 解释

1. 安全保守版SCG可做到unknown FAR=0，但known覆盖几乎不可用；support-protected最终版显著增加known接收，但k=1、k=3、k=4、k=5出现unknown FAR。
2. k=2是当前最稳折中：unknown FAR=0，old_acc=23.13%，seen_new_acc=4.00%，unknown_reject=88.64%，但known性能远低于可部署目标。
3. k=4给出最高old_acc=48.39%，但unknown FAR=5.56%，不可作为安全部署结果。
4. 单接收机k=1不可作为安全模式，因为部分unknown样本未触发unknown证据却被known支持路径接收，unknown FAR=15.00%。
5. 该权重在星地信道迁移下特征分离不足；仅靠后处理协同门控无法达到项目成功标准，需要训练/微调侧增强。

## 子agent审查结论

| 角色 | 结论 |
|---|---|
| 文献/方法搜索 | 支持使用轻量原型/证据协同、open-set门控、FedProto/HetLoRA/L-TTA式端侧轻量更新，不建议重型集成。 |
| 高效算法构建 | SCG-qKNN实现为证据摘要级融合，通信量为`40*k bytes/event`，p95评估延迟约0.1137ms。 |
| 合理性监督 | 指定权重、CVS-RFFI环境、星地信道、k=1..5均已覆盖；但指标不达标，必须标diagnostic-only。 |
| 查漏补缺/review | 原始pairguard作用域不能作为部署证据；single-unknown evidence不能在预算耗尽时变成known接收；已在代码中修正。 |

## 后续建议

1. 不应继续单纯调SCG阈值追求表面指标；当前数据表明known/unknown支持证据重叠，阈值会在known覆盖和unknown FAR之间直接交换。
2. 下一步应做卫星端实时微调路线：冻结SincNet主干，端侧只更新小型adapter/LoRA或receiver prototype，使用support集和高置信伪标签做约束更新；损失建议结合supervised contrastive、prototype consistency、unknown energy margin和FedProto式跨星原型聚合。
3. 若继续部署诊断，可默认`k=2`作为安全优先配置，明确标注known性能不足；`k=4`只能用于分析特征分离上限，不能用于安全部署。
