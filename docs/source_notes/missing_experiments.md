# CVS-RFFI 缺失实验清单

本文件记录当前审计中无法支撑“地面训练、星上部署、小样本适应、新类识别、open-set 拒识、轻量化部署”完整主张的缺失实验。未完成项不得在论文或项目汇报中写成已验证。

## 1. True Open-set SFE: COMBINED_K20 + UNKNOWN_TX_IDS

- 缺口：Card8 的 `UNKNOWN_TX_IDS=[]`，`unknown_rejection_rate` 与 `unknown_false_accept_rate` 均为 n/a；Card9 已完成 TX 不重叠审计，但状态为 `DEFERRED_RETRY_CAPACITY`，没有运行日志和 metrics。
- 目的：补齐 OpenMax/Mahalanobis/combined gate 的核心 unknown 指标。
- 推荐优先级：P0。
- 推荐命令：

```powershell
powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1
```

容量允许后按既有报告中的 N607 命令启动：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
RUN_ID=spaceborne_fewshot_openset_card9_20260613 GPU=3 UNKNOWN_TX_IDS=6,7 \
nohup bash code/scripts/launch_spaceborne_fewshot_openset_card9_20260613.sh \
  > logs/spaceborne_fewshot_openset_card9_20260613/scheduler.out 2>&1 &
```

验收指标：`query_unknown=100`、unknown rejection、unknown false accept、known old/new accuracy、coverage、rollback reason。

## 2. SFE Support/Query 显式星地信道版本

- 缺口：Card8 SFE 是 frozen `z_id` feature-level WiSig new-TX payload；没有找到证据证明 new support/query 在 SFE payload 构造时显式经过 `H_sg/R_sat` 星地信道扰动。
- 目的：避免把普通 WiSig new-TX feature few-shot 冒充星上信道样本。
- 推荐优先级：P0。
- 推荐实验：导出 clean SFE 与 satellite SFE 两套 payload，source/new/unknown TX 一致，比较 `full_acc/new_acc/old_acc/coverage/unknown_false_accept`。
- 需要新增或确认：`export_spaceborne_features.py` 或 SFE payload 构造路径中的 `sat_channel` 参数、manifest 中记录 channel profile。

## 3. FTRC K={1,5,10,20} 完整 Adapter/LoRA 复验

- 缺口：Card8 只有 K2 adapter/LoRA 完成；两者均未通过 rollback，目标域准确率从 42.20 降至 42.11/42.16。
- 目的：判断旧类星地目标域少样本适配是否只是 K2 样本不足，还是 adapter/LoRA 方法不适配。
- 推荐优先级：P1。
- 推荐设置：`target_samples_per_rx_tx` 分别为 1/5/10/20；adapter 类型包含 `feature_residual`、`logit_lora`、`logit_calibration`；保持 rollback gate。
- 验收指标：`target_after_adapt - target_no_adapt >= 1pp`，overall drop <= 0.5pp，至少一个 checkpoint 通过 rollback。

## 4. SFE COMBINED_K20 多 Seed 复验

- 缺口：Card8 每个 candidate 只有单次结果；best K20 的 new_acc 仅 17.00%，无法判断是否稳定。
- 目的：验证 new-class prototype/gate 是否有稳定收益。
- 推荐优先级：P1。
- 推荐设置：同一 source/new/unknown TX split，至少 3 seed；报告 mean/std。
- 验收指标：new_acc、old_acc、full_acc、coverage、H-mean、unknown false accept 的均值和标准差。

## 5. Open-set Threshold Sweep

- 缺口：Mahalanobis/OpenMax/combined gate 已实现，但没有完整 threshold trade-off 曲线。
- 目的：建立 coverage-risk、unknown false accept 与 new accept 的 Pareto 曲线。
- 推荐优先级：P1/P2。
- 推荐 sweep：`unknown_threshold`、`min_margin`、`max_mahalanobis`、`openmax_quantile`。
- 验收指标：AUROC、FPR95、unknown false accept、new-to-old confusion、coverage。

## 6. highratio 36/36 最终解析

- 缺口：截至已有状态报告，36 个候选中 32 完成、4 个 R050/K500 未完成。
- 目的：确认当前 best completed `R030_K225_RATIO_STRICT` 是否仍为全矩阵最优。
- 推荐优先级：P0。
- 推荐命令：不重新训练，只在全部日志完成后重跑既有 full-log parser。
- 验收指标：best strict UDU、primary、overall、worst RX、sat floor、late drop。

## 7. 当前 Best DG Checkpoint 的复现实验

- 缺口：顶层非 git repo，缺 env lock、checkpoint hash，本地无完整 checkpoint 归档。
- 目的：确认 `R030_K225_RATIO_STRICT` 或最终 best checkpoint 可复现。
- 推荐优先级：P2。
- 验收指标：重复 strict UDU 与 original 差异 <= 0.5pp；保存 env、command、checkpoint SHA256。

## 8. 部署开销实验

- 缺口：论文草稿给出部署参数量约 0.63M，但未找到 P95 latency、峰值显存、FLOPs、功耗或星上存储开销实测。
- 目的：支撑“满足星上部署轻量化要求”的工程主张。
- 推荐优先级：P2/P3。
- 验收指标：参数量、trainable params、feature/prototype cache size、adapter state size、CPU/GPU/edge P95 latency、峰值显存。

## 9. old/new/domain/channel 分层混淆矩阵

- 缺口：SFE 已有 old/new accuracy，但缺 new-to-old confusion、per-TX/per-SNR/per-channel 分析；FTRC 缺 old-class before/after 的细粒度可解释性。
- 目的：确定新类低准确率来自 prototype 偏置、域偏移、阈值过严还是类间相似。
- 推荐优先级：P1/P2。
- 验收指标：per-class acc、confusion matrix、new-to-old misclassification rate、known-to-unknown false reject。
