# phase2_adv3b02_virtual_negative_adapter_20260704

## 基本信息

| 字段 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_virtual_negative_adapter_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 设计并验证support-only虚拟负样本边界adapter，用于Stage2-C协同推理诊断 |
| 权重/特征来源 | ADV3B02_CORE90_SOFT_E200特征；用户指定权重名SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104作为当前协同推理目标权重族 |
| 协同范围 | `collab_counts=all`，即1到全部target receivers |
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 结论边界 | NON_DEPLOYMENT_DIAGNOSTIC；不作为部署成功或论文主结论 |

## 假设与算法

假设：现有support-calibrated qknn的风险字段不能同时保持known覆盖和unknown拒识，原因可能是support附近缺少显式open boundary。新增轻量adapter只用target-old和target-new support构造虚拟负样本，训练一个known-vs-virtual-negative边界，以期在不使用unknown query的前提下提高拒识。

核心步骤：

1. 对每个target receiver独立抽取`K=8`的target-old和target-new support。
2. 冻结特征，不训练backbone。
3. 用support拟合多类ridge head，输出old/seen-new候选标签。
4. 从support构造虚拟负样本：class shell、class midpoint和跨类mix。
5. 用support为正类、虚拟负样本为负类拟合二分类ridge边界。
6. 阈值只由support known分数分位数给出，记录`threshold_selection_label_scope=support_virtual_unknown`。
7. unknown query只参与最终评估，不参与阈值、adapter或support质量拟合。

## 本地文件变更

| 文件 | 目的 |
|---|---|
| `E:\type10-7\code\scripts\phase2_virtual_negative_adapter_eval.py` | 新增support-only虚拟负样本边界adapter诊断脚本 |
| `E:\type10-7\code\tests\test_phase2_virtual_negative_adapter_eval.py` | 覆盖协议metadata、1..N协同数量、unknown eval-only边界 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_virtual_negative_adapter_20260704\phase2_virtual_negative_adapter_eval.py` | 非Git根目录代码快照 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_virtual_negative_adapter_20260704\test_phase2_virtual_negative_adapter_eval.py` | 非Git根目录测试快照 |

Git承载目录：`E:\type10-7\github_publish\CVS-RFFI-repo`，分支`codex/cvs-rffi-release-20260626`。

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_virtual_negative_adapter_eval.py code\tests\test_phase2_virtual_negative_adapter_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_virtual_negative_adapter_eval.py -q` | PASS，3 passed；`.pytest_cache`写入警告，不影响测试 |

本地ADV3B02诊断命令：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_virtual_negative_adapter_eval.py --feature_npz remote_artifacts\phase2_adv3b02_features\features.npz --output_json local_artifacts\phase2_adv3b02_virtual_negative_adapter_20260704\virtual_negative_adapter_eval.json --output_evidence_csv local_artifacts\phase2_adv3b02_virtual_negative_adapter_20260704\virtual_negative_adapter_evidence.csv --collab_counts all --collab_group_policy available_up_to_k --partial_collab_min_receivers 1 --event_alignment_policy receiver_domain_ranked --support_selection_policy stable_first --k_shot 8 --query_per_class 20 --ridge_lambda 0.1 --boundary_ridge_lambda 0.1 --class_temperature 0.05 --boundary_temperature 0.25 --support_threshold_quantile 0.05 --virtual_negative_policy shell_mix --virtual_negative_shell_scale 1.5 --virtual_negative_mix_pairs_per_class 4 --unknown_risk_threshold 0.65 --accept_margin_threshold 0.02 --fusion_policy risk_margin --label_fusion_policy weighted_vote_margin --evidence_packet_bytes 112
```

本地输出：

| 文件 | 内容 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_virtual_negative_adapter_20260704\virtual_negative_adapter_eval.json` | 本地评估JSON |
| `E:\type10-7\local_artifacts\phase2_adv3b02_virtual_negative_adapter_20260704\virtual_negative_adapter_evidence.csv` | 本地evidence明细 |

本地结果：

| 协同接收机数 | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | known_coverage | defer_rate | p95参与接收机数 | bytes/event | latency_ms_p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6043 | 0.3000 | 0.5333 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1 | 112.0 | 0.0471 |
| 2 | 0.5989 | 0.3000 | 0.5333 | 0.4000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 2 | 203.2 | 0.0471 |
| 3 | 0.6845 | 0.5278 | 0.6000 | 0.4750 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 3 | 276.2 | 0.0471 |
| 4 | 0.7647 | 0.5500 | 0.6333 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 4 | 330.9 | 0.0471 |
| 5 | 0.7914 | 0.5500 | 0.6500 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 5 | 364.8 | 0.0471 |

本地解释：协同数量增加能提升old和seen-new，但虚拟负边界未覆盖真实unknown区域，unknown拒识为0。该路线未达到目标，不可标为Stage2-C成功；它证明当前support-only synthetic negatives过弱，下一步应使用Gaussian prototype/diag-Mahalanobis/EVT或SO-CAPR式conformal风险融合，而不是只扩大虚拟负样本阈值。

## 远端同步与复测计划

同步目标：

| 本地 | N607 |
|---|---|
| `E:\type10-7\code\scripts\phase2_virtual_negative_adapter_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_virtual_negative_adapter_eval.py` |
| `E:\type10-7\code\tests\test_phase2_virtual_negative_adapter_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_virtual_negative_adapter_eval.py` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_virtual_negative_adapter_20260704\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_virtual_negative_adapter_20260704/report.md` |

远端复测命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_virtual_negative_adapter_eval.py code/tests/test_phase2_virtual_negative_adapter_eval.py
cd /home/szu2070436088/2510044040/CV-SincNet && /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m pytest code/tests/test_phase2_virtual_negative_adapter_eval.py -q
cd /home/szu2070436088/2510044040/CV-SincNet && /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_virtual_negative_adapter_eval.py --feature_npz runs/phase2_adv3b02_features/features.npz --output_json runs/phase2_adv3b02_virtual_negative_adapter_20260704/virtual_negative_adapter_eval.json --output_evidence_csv runs/phase2_adv3b02_virtual_negative_adapter_20260704/virtual_negative_adapter_evidence.csv --collab_counts all --collab_group_policy available_up_to_k --partial_collab_min_receivers 1 --event_alignment_policy receiver_domain_ranked --support_selection_policy stable_first --k_shot 8 --query_per_class 20 --ridge_lambda 0.1 --boundary_ridge_lambda 0.1 --class_temperature 0.05 --boundary_temperature 0.25 --support_threshold_quantile 0.05 --virtual_negative_policy shell_mix --virtual_negative_shell_scale 1.5 --virtual_negative_mix_pairs_per_class 4 --unknown_risk_threshold 0.65 --accept_margin_threshold 0.02 --fusion_policy risk_margin --label_fusion_policy weighted_vote_margin --evidence_packet_bytes 112
```

## 远端结果

N607预检：

| 项 | 结果 |
|---|---|
| SSH预检 | PASS，direct `N607`可用 |
| 项目根目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19 |
| GPU选择 | GPU0；预检时8张RTX3090均为10MiB显存占用，GPU0满足低显存占用测试要求 |
| 特征同步 | 本地特征SHA256为`DB559D78DB305894307851750EF7D698DB387F0984FF13C980FEA99DB85B8532`；远端旧`ADV3B02_CORE90_SOFT_E200_PHASE1_PROXY_UNKNOWN/features.npz`哈希不同，因此本次同步本地特征到run目录 |
| SSH清理 | SCP后和拉回结果后均无N607/bridge的ESTABLISHED连接 |

远端验证：

| 命令 | 结果 |
|---|---|
| `CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_virtual_negative_adapter_eval.py code/tests/test_phase2_virtual_negative_adapter_eval.py` | PASS |
| `CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m pytest code/tests/test_phase2_virtual_negative_adapter_eval.py -q` | FAIL，远端CVS-RFFI环境无`pytest`模块 |
| `CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_virtual_negative_adapter_eval.py` | PASS，3 tests OK |

远端运行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_virtual_negative_adapter_eval.py --feature_npz runs/phase2_adv3b02_virtual_negative_adapter_20260704/features.npz --output_json runs/phase2_adv3b02_virtual_negative_adapter_20260704/virtual_negative_adapter_eval.json --output_evidence_csv runs/phase2_adv3b02_virtual_negative_adapter_20260704/virtual_negative_adapter_evidence.csv --collab_counts all --collab_group_policy available_up_to_k --partial_collab_min_receivers 1 --event_alignment_policy receiver_domain_ranked --support_selection_policy stable_first --k_shot 8 --query_per_class 20 --ridge_lambda 0.1 --boundary_ridge_lambda 0.1 --class_temperature 0.05 --boundary_temperature 0.25 --support_threshold_quantile 0.05 --virtual_negative_policy shell_mix --virtual_negative_shell_scale 1.5 --virtual_negative_mix_pairs_per_class 4 --unknown_risk_threshold 0.65 --accept_margin_threshold 0.02 --fusion_policy risk_margin --label_fusion_policy weighted_vote_margin --evidence_packet_bytes 112
```

远端输出：

| 文件 | 内容 |
|---|---|
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_virtual_negative_adapter_20260704/features.npz` | 本次同步的ADV3B02特征 |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_virtual_negative_adapter_20260704/virtual_negative_adapter_eval.json` | 远端评估JSON |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_virtual_negative_adapter_20260704/virtual_negative_adapter_evidence.csv` | 远端evidence明细 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_virtual_negative_adapter_20260704\remote_virtual_negative_adapter_eval.json` | 拉回的远端评估JSON |
| `E:\type10-7\local_artifacts\phase2_adv3b02_virtual_negative_adapter_20260704\remote_virtual_negative_adapter_evidence.csv` | 拉回的远端evidence明细 |

远端结果：

| 协同接收机数 | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | known_coverage | defer_rate | p95参与接收机数 | bytes/event | latency_ms_p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6043 | 0.3000 | 0.5333 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1 | 112.0 | 0.0828 |
| 2 | 0.5989 | 0.3000 | 0.5333 | 0.4000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 2 | 203.2 | 0.0828 |
| 3 | 0.6845 | 0.5278 | 0.6000 | 0.4750 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 3 | 276.2 | 0.0828 |
| 4 | 0.7647 | 0.5500 | 0.6333 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 4 | 330.9 | 0.0828 |
| 5 | 0.7914 | 0.5500 | 0.6500 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 5 | 364.8 | 0.0828 |

远端解释：结果与本地一致。协同数量增加提高known分类，但support-only虚拟负样本边界无法识别真实unknown。该路线没有达到`old_acc>=0.80`阶段门槛，也完全没有满足unknown拒识要求。不能作为部署成功、论文主结论或当前最优路线。

## 查漏补缺与监督

| 项 | 状态 | 说明 |
|---|---|---|
| 协同数量1..全接收机 | 本地完成 | `counts={1,2,3,4,5}` |
| 星地信道 | 本地完成 | `target_channel_view=leo_clear_weak,leo_low_elev_weak,leo_rain_weak`由特征metadata/evidence记录 |
| unknown eval-only | 本地完成 | metadata与row均记录`unknown_query_eval_only=true`和`calibration_role=query` |
| 使用CVS-RFFI远端环境 | 完成 | 使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| N607低显存GPU测试 | 完成 | GPU0显存占用10MiB；运行时设置`CUDA_VISIBLE_DEVICES=0` |
| Git版本化 | 待完成 | 需复制到Git mirror并提交 |

## 下一步算法建议

并行文献/算法review给出的更合理方向是SO-CAPR/Gaussian prototype/EVT组合，而不是继续单独加大虚拟负样本：

1. 用old source prototype做旧类shrinkage：`c_old' = normalize((beta c_old_source + sum z_support)/(beta+K))`。
2. 用seen-new support注册Gaussian prototype，低K时先用对角协方差或共享协方差。
3. 拒识用support-only conformal p-value、diag-Mahalanobis、EVT/GPD尾部阈值和class-shell risk联合，而不是单一二分类虚拟负边界。
4. 协同策略用query前固定的support quality选择receiver，输出`k=1..|R_t|`完整曲线，同时报告平均参与接收机数和defer率。
5. unknown query仍只能做最终评估；不能用于阈值、adapter、prototype或receiver quality拟合。

本次失败点的直接含义：真实unknown在当前`z_id`空间中会落入old/seen-new线性边界的高置信区域，support插值/壳层负样本不足以近似真实开放空间。下一步必须引入类条件半径/协方差/EVT尾部，而不是只调`unknown_risk_threshold`。
