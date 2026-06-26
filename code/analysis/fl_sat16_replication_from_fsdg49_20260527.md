# 从 FSDG49 复制集中式 SAT16 到联邦训练的可行性分析

生成时间：2026-05-27  
范围：CV-SincNet / CVS-RFFI / WiSig federated receiver DG  
基准联邦锚点：`FSDG49_fedprox_receiver_ra_bex02_cvs_sat`  
集中式参考：`SA16_ceonly_domain_dsq_r010`，下文按用户习惯称 `SAT16/SA16`

---

## 1. 结论

StyleBank 不应再作为主线突破口。它的问题不是实现小 bug，而是语义证据薄：本地客户端提取到的所谓 style 很难证明就是 receiver 域风格，也很容易混入 TX、day、SNR、channel、采样链路、数据量偏差等因素。即便 StyleBank 能构造 `d_style`，这个 `d_style` 也只是虚拟域，不等价真实 receiver 域。

更稳的方向是复制集中式 `SA16` 的可验证成功因素：

```text
known physical satellite view
+ clean/sat supervised CE-only view
+ domain backbone DSQ
+ receiver-client FedProx
+ rx_day domain labels inside each receiver client
```

这条路线比 StyleBank 更可靠，因为 satellite view 是已知物理扰动，不依赖从本地数据中“猜”receiver style。它仍然不是严格集中式复刻，因为普通 FedAvg/FedProx 不具备同一 step 内跨 receiver 的全局多域 batch；但它是当前 privacy-preserving FL 中最值得优先推进的工程路线。

当前代码里已经有最接近的联邦候选：

```text
FL82_16_fedprox_rx_ra_bex02_baselineview_ceonly_domain_dsq_r010
```

它比旧 `FSDG50` 更像 `SA16`，因为它显式启用了：

```text
--fl_sat_aug_mode baseline_view
--fl_baseline_view_ce_only
--fl_baseline_view_ce_weight 1.0
--domain_freq_stability_mode dsq
--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
--sat_view_prob 1.0
```

---

## 2. 证据锚点

### 2.1 FSDG49 是当前稳定联邦锚点

全量历史 metrics 扫描中，`FSDG49_fedprox_receiver_ra_bex02_cvs_sat` 是最强稳定非 r020 联邦锚点：

```text
best strict UDU = 76.2950 @ R127
latest/final strict UDU = 75.9167 @ R170
```

其配置核心是：

```text
--train_mode fedprox
--fl_client_key receiver
--fedprox_mu 0.01
--fl_local_objective receiver_agnostic_bex02
--fl_sat_aug_mode cvs_consistency
--use_sat_consistency
--sat_train_scenario mixed_orbit
--lambda_sat_cls 0.10
--lambda_sat_cons 0.00
--lambda_fishr 0.02
--fishr_min_domains 4
```

它的意义：证明 receiver-client FedProx + receiver-agnostic BEX02 框架是可用的，但它还不是 `SA16` 路线，因为它用的是 `cvs_consistency` 式 satellite loss，不是 `SA16` 的 CE-only clean/sat 监督视图。

### 2.2 旧 FSDG50 说明 baseline_view 不能粗暴搬

旧 `FSDG50_fedprox_receiver_ra_bex02_baseline_sat` 使用 baseline-view SAT，但结果弱于 FSDG49：

```text
best strict UDU = 72.8033 @ R045
latest/final strict UDU = 70.5167 @ R170
```

这说明“把 clean+sat 2B batch 直接塞进完整 DG loss”不一定好。它可能把 satellite view 也送入 receiver/domain/Fishr/MixStyle 等损失，使本来用于 TX CE 的物理视图扰动污染域解耦目标。

### 2.3 SA16 的真正成功点

集中式 `SA16_ceonly_domain_dsq_r010` 的完成结果：

```text
best primary = 84.45
strict UDU = 82.78
overall = 87.55
worst RX = 77.51
final-primary SAT avg/min = 43.66 / 39.56
unsafe skip count = 8
```

它的命令语义是：

```text
--wisig_domain rx_day
--wisig_train_ratio 0.1
--model_variant lite_d
--branch_ablation no_dac
--domain_branch_ablation no_stats
--domain_enhancer rcn_stats
--use_mixstyle
--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
--use_concat_sat_channel_aug
--concat_sat_ce_only
--concat_sat_ce_weight 1.0
--concat_sat_start_epoch 1
--sat_view_prob 1.00
--no_use_sat_consistency
--lambda_sat_cls 0.00
--lambda_sat_cons 0.00
--domain_freq_stability_mode dsq
```

关键不是“satellite loss 更强”，而是：

```text
clean 主路径仍按原来的 DG/CVS 损失训练；
satellite view 只作为同 TX 标签的 CE-only 监督样本；
domain backbone 加 DSQ 频域稳定性，但 ID backbone 不强行加同样结构。
```

---

## 3. 为什么 StyleBank 很难成为主线

StyleBank 的核心假设是：

```text
local statistics -> receiver style
remote centroids -> transferable receiver style
constructed d_style -> useful DG domain label
```

这三个箭头都不够硬。

1. 本地客户端通常只看到一个 receiver 或少量 receiver-day 组合，无法从局部统计中分离 receiver、day、TX、信道、SNR 和数据采样偏差。
2. 提取出的 style 很可能只是“容易被统计量捕获的扰动”，不一定是泛化失败真正依赖的 receiver 域因子。
3. `d_style` 是构造标签，不是真实 receiver 标签；用它驱动 GRL/Fishr 可能优化了虚拟问题，而不是目标 OOD 问题。
4. StyleBank 成熟度、centroid 数量、采样策略、物理扰动尺度都会影响上限，调参空间大，证据链长。

因此 StyleBank 可以保留为诊断或后续辅助，但不应压在最核心的“复制集中式成功案例”路径上。

---

## 4. 集中式 SA16 到 FL 会丢失什么

集中式训练最重要的优势是同一个训练 step 内有全局多 receiver/day batch：

```text
TX_i @ RX_a day_0
TX_i @ RX_b day_1
TX_j @ RX_c day_0
...
```

这使下面机制有真实语义：

```text
GRL: z_id 不能预测 receiver/day
Fishr: 对齐不同 receiver/day 的梯度统计
MixStyle: same-TX cross-domain 风格混合
consistency: 同 TX 跨域表征接近
hard-domain CE: 当前全局 batch 中难域加权
```

普通 FedAvg/FedProx 的 receiver-client 设定是：

```text
client RX0 本地训练若干 step
client RX1 本地训练若干 step
server 平均模型 delta
```

服务端平均 delta 不是全局多域 batch，也无法事后补上同一步的 GRL/Fishr/MixStyle 计算图。所以不能把普通 FedProx 叫做严格复刻集中式 BEX02。

但 satellite view 不一样。它是公共物理变换，能在每个客户端本地构造：

```text
x_clean -> y_tx
x_sat   -> y_tx
```

这部分可以比较干净地联邦化。

---

## 5. 推荐的联邦复制路线

### 5.1 主线：FSDG49 框架 + SA16 的 CE-only SAT + domain DSQ

目标不是直接复用 FSDG49 的 `cvs_consistency` SAT，而是以 FSDG49 证明过的 FedProx/receiver-client/RA-BEX02 框架为底座，替换成更接近 SA16 的训练语义：

```text
--train_mode fedprox
--fl_client_key receiver
--fl_local_objective receiver_agnostic_bex02
--fl_sat_aug_mode baseline_view
--fl_baseline_view_ce_only
--fl_baseline_view_ce_weight 1.0
--domain_freq_stability_mode dsq
--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
--sat_view_prob 1.0
--sat_cons_start_epoch 1
--lambda_sat_cls 0.00
--lambda_sat_cons 0.00
```

同时保留 FL82 公共底座：

```text
--wisig_domain rx_day
--wisig_train_ratio 0.1
--fl_rounds 200
--fl_local_epochs 2
--model_variant lite_d
--branch_ablation no_dac
--domain_branch_ablation no_stats
--domain_enhancer rcn_stats
--use_mixstyle
```

这基本就是当前 `FL82_16_fedprox_rx_ra_bex02_baselineview_ceonly_domain_dsq_r010` 的设计。

### 5.2 为什么要 `wisig_domain=rx_day`，但 client 仍用 receiver

如果 `wisig_domain=rx` 且 `fl_client_key=receiver`，每个客户端本地几乎只有一个 domain label。此时：

```text
GRL/Fishr/MixStyle/same-TX cross-domain
```

都会退化。`receiver_agnostic_bex02` 这个名字听起来强，但本地没有多域标签时，很多 DG 项只是形式上存在。

更合理的是：

```text
client = receiver
domain label = rx_day
```

这样每个 receiver client 至少能在本地看到不同 day 域，让 Fishr、MixStyle、domain branch 有一点真实多域语义；服务端再通过 FedProx 聚合不同 receiver 的模型。它仍然不是全局跨 receiver 同 step，但比 `rx` 域标签更接近集中式 `SA16`。

### 5.3 为什么要 CE-only baseline_view

集中式 `SA16` 最值得复制的点是 satellite view 不参与完整 DG 损失，而是只作为同 TX 标签监督样本：

```text
loss = clean_main_loss + weight * CE(model(x_sat), y_tx)
```

这能避免 satellite channel 被 domain branch / GRL / Fishr 当作“另一个 receiver 域”错误处理。

当前联邦代码已经支持该语义：

```text
--fl_baseline_view_ce_only
```

它会让 federated baseline-view satellite sample 走独立 TX CE，而不进入完整 2B DG batch。

---

## 6. 三档实现策略

### A. 保守且最该优先：本地 SAT16 化

只用公共物理 satellite transform，不用 StyleBank，不引入跨客户端样本共享。

预期收益：

```text
比 FSDG49 更接近 SA16；
比 StyleBank 更可解释；
隐私和通信成本不增加；
若失败，失败原因也更容易定位。
```

主要风险：

```text
domain DSQ 在每个 receiver client 内只看到 day 变化，不能看到真实跨 receiver 梯度；
CE-only satellite view 可能改善 SAT robustness，但未必能把 clean strict UDU 推到 82；
FedProx local drift 仍然存在。
```

### B. 中等增强：只共享统计，不共享 IQ

在 A 的基础上共享：

```text
per-domain loss/accuracy
class prototypes
domain prototypes
feature mean/variance
Fishr-like gradient summary
```

服务器下发：

```text
global hard-domain weights
global class/domain prototype anchors
optional BN/stat calibration
```

这比 StyleBank 更稳，因为共享的是训练诊断和 prototype 约束，而不是声称“提取 receiver style”。它不能完全复刻集中式，但能补普通 FedProx 缺失的全局域统计。

### C. 高保真但工程重：Split-BEX02 或同步 FedSGD

如果论文上必须严谨地说“最大限度复刻集中式 BEX02”，普通 FedProx 不够。更接近的做法是：

```text
Split Learning:
  client forward 前端 feature
  server 拼接多 receiver/day feature batch
  server 执行 BEX02/SA16 losses
  backward 回传 activation gradients

同步 FedSGD:
  每个 global step 采样各 client batch
  server 聚合/组合梯度
  尽量模拟 centralized optimizer step
```

代价是同步复杂、通信大、隐私边界更敏感，不适合作为下一步最快验证。

---

## 7. 推荐下一步，不启动实验

当前不需要再围绕 StyleBank 设计新主线。推荐把后续讨论收敛到这条候选：

```text
FL82_16_fedprox_rx_ra_bex02_baselineview_ceonly_domain_dsq_r010
```

它是现有代码中最像 `SA16` 的联邦实现候选。正式实验前只需要做设计审查：

1. 确认 dry-run 命令包含 `--fl_baseline_view_ce_only`、`--domain_freq_stability_mode dsq`、all5 SAT scenarios、ratio `0.1`、rounds `200`、client `receiver`。
2. 确认 federated log 会输出 `diag_baseline_sat_view_active`、`diag_sat_cls_active`、`diag_fishr_domain_count`、`diag_rx_adv_active`。
3. 确认 `wisig_domain rx_day` 没被行级参数覆盖。
4. 把 StyleBank 关闭，避免把失败归因混淆。
5. 如果未来要做 ablation，先比较：

```text
FL82_11 CE-only baseline-view anchor
FL82_16 CE-only baseline-view + domain DSQ
FSDG49 historical CVS SAT anchor
```

判断标准：

```text
clean strict UDU 是否超过 FSDG49 的 76.295；
是否接近或超过 82；
SAT avg/min 是否不低于 SA16 的 43.66 / 39.56；
worst RX 是否改善；
diag_fishr_domain_count 是否真实 >= fishr_min_domains；
diag_baseline_sat_view_active 是否稳定为 1。
```

---

## 8. 最终判断

你的判断是对的：StyleBank 不是当前最稳主线。它最大的问题是“风格是否等于 receiver 域”无法自证。

真正值得复制的是 `SA16` 里更硬的东西：

```text
可验证物理 view
+ 同 TX 监督
+ CE-only 隔离
+ domain backbone DSQ
+ receiver-client FedProx
+ rx_day local domain diversity
```

这条路线的上限仍受普通 FedProx 限制，但它比 StyleBank 更接近集中式成功原因，也比旧 FSDG50 的 baseline-view 粗搬更干净。
