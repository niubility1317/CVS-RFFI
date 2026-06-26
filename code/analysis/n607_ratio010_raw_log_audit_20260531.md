# N607 ratio 0.1 原始日志审计：联邦与集中式最强记录、已试路线、负例边界

Generated: 2026-05-31 19:25:38 +08:00

## 结论边界

- 本审计只纳入 `wisig_train_ratio/train_ratio=0.1` 或 run 名/配置明确为 `r010/ratio010` 的记录；`r020`、`0.2`、`0.2/220` 全部剔除。
- 主证据来自原始训练日志或结构化训练产物：N607 `metrics.csv`、`logs.jsonl`、`federated_config.json`、集中式 stdout `*.log/*.out`。旧报告、conversation index 和历史总结只用于定位路径与交叉检查。
- `best/peak` 与 `final/latest` 分开报告。高峰值不能覆盖最终回落；未完成 active 批次不进入最终榜。
- 硬排行只在同 family / 同 evaluator 内成立。`federated`、`VMB`、`centralized`、3-scenario SAT、5-scenario SAT、单场景 `clear_leo` 之间只作方向性解释。
- 本轮没有修改远端文件、没有停止/启动训练；SSH 后检查本地无残留 `ssh.exe`，无到 `172.31.111.215:22` 的 `ESTABLISHED` 连接。

## 审计覆盖

- 联邦主扫描：远端 `runs/**/metrics.csv`、`logs/**/metrics.csv` 以及相邻 `federated_config.json`，只保留 ratio 0.1 / r010；主解析得到 122 条 ratio 0.1 联邦记录。
- 联邦监督审计：全量读远端 `runs/` 51 个、远端 VMB `logs/` 64 个、本地 SBX02 5 个结构化结果；剔除 `r020/0.2/220` 和未能证明 ratio 0.1 的旧 run。
- 集中式主扫描：远端 `logs/**/*.out|*.log` 中完成并含 final marker 的 ratio 0.1 记录，解析 `[FINAL-PRIMARY]`、`Training finished`、`best_worst_rx`、`[SAT-TEST]`。
- 集中式监督审计：扫描 `SA* / CEN_* / BEX02*` ratio 0.1 stdout，纳入 125 条，其中完成 116 条；当前 `optimizer_20260531_175432` 批次仍在进行中，不参与最终排名。

## 联邦训练：最强记录

| 目标 | 最强记录 | best/peak | final/latest | 原始证据 |
|---|---:|---:|---:|---|
| completed r010/200 clean strict UDU peak | `VMB5_A08_pre100_satfloor_ce085_r010` | `74.9067%@R194` | `72.8367%@R200` | `N607:/home/szu2070436088/2510044040/CV-SincNet/logs/optimizer_20260530_173224_vmb_next8/VMB5_A08_pre100_satfloor_ce085_r010/metrics.csv` |
| completed r010/200 clean strict UDU final | `A15_full_warmup40_r010` | `74.8117%@R199` | `74.4633%@R200` | `N607:/home/szu2070436088/2510044040/CV-SincNet/runs/fedcvs_vmb_mechanism_20260528_000804/A15_full_warmup40_r010/metrics.csv` |
| completed r010/200 clean peak, lower rollback | `VMB5_A07_pcgrad_a07_lowproto_r010` | `74.7900%@R180` | `73.7850%@R200` | `N607:/home/szu2070436088/2510044040/CV-SincNet/logs/optimizer_20260530_173224_vmb_next8/VMB5_A07_pcgrad_a07_lowproto_r010/metrics.csv` |
| SBX02 local clean final | `SBX02_PROTO_r010` | `74.4217%@R200` | `74.4217%@R200` | `E:\type10-7\analysis_tmp\split_bex02_alternatives_20260528_122220\SBX02_PROTO_r010\metrics.csv` |
| `clear_leo` peak | `VMB5_A07_pcgrad_a07_lowproto_r010` | `43.4417%@R120` | about `41.48%@R200` | same `VMB5_A07` metrics |
| 5-scenario satellite mean peak | `VMB5_C02_a07_gate090_kd0075_r010` | mean `41.124%`, floor `37.753%@R150` | mean `39.385%@R200` | `N607:/home/szu2070436088/2510044040/CV-SincNet/logs/optimizer_20260530_173224_vmb_next8/VMB5_C02_a07_gate090_kd0075_r010/metrics.csv` |
| 5-scenario satellite mean final | `VMB6_C04_bpc2_lownoise_guard_r010` | mean `41.095%@R179` | mean `40.5767%`, floor `37.428%@R200` | `N607:/home/szu2070436088/2510044040/CV-SincNet/logs/optimizer_20260530_233142_vmb_next8/VMB6_C04_bpc2_lownoise_guard_r010/metrics.csv` |
| final rx8 in completed r010 federated | `A11_full_no_txadvr_r010` | not primary clean winner | rx8 `63.7250%`, strict final `74.1417%` | `N607:/home/szu2070436088/2510044040/CV-SincNet/runs/fedcvs_vmb_mechanism_20260528_000804/A11_full_no_txadvr_r010/metrics.csv` |

### 联邦解读

- 如果只看 completed `r010/0.1/200/receiver`，clean strict UDU 的有效上沿在 `74.8-74.9%` peak，最稳 final 是 `A15_full_warmup40_r010` 的 `74.4633%`。
- VMB 后续并非“完全没有超过旧 anchor”：`VMB5_A08` 的 clean peak 和 `VMB5_C02` 的 5-scenario satellite mean peak 都确实超过了早期 practical anchor。更准确的说法是：final 稳定 clean 没有明显超过 `A15/SBX02_PROTO`，且 satellite 提升常伴随 clean rollback。
- 单场景 `clear_leo` 可到约 `43.44%`，5-scenario satellite mean final 最强约 `40.58%`；这与 clean 74-75% 上沿同时成立时仍有明显 tradeoff。
- 旧 `FSDG49` 可作为历史线索，但不纳入本表的 completed `r010/0.1/200/receiver` 主榜，除非再按原始日志单独复核其 ratio、轮数和 evaluator。

## 集中式训练：最强记录

| 目标 | 最强记录 | clean / score | SAT / RX | 原始证据 |
|---|---:|---:|---:|---|
| clean strict UDU peak/final-primary | `CEN_A31_a22_satboost_ce1p28_stack_r010` | strict `84.88%`, score `86.06`, overall `88.83` | SAT avg `49.0767`, clear `49.68`, SAT min `47.62`, rx8 `74.83`, worstRx about `79.58` | `N607:/home/szu2070436088/2510044040/CV-SincNet/logs/optimizer_20260530_043050_centralized_next8/CEN_A31_a22_satboost_ce1p28_stack_r010.out` |
| clean runner-up | `CEN_A40_a31_mixstop150_ce128_r010` | strict `84.37%`, score `85.70`, overall `89.11` | SAT avg `47.01`, worstRx `81.62` | `N607:/home/szu2070436088/2510044040/CV-SincNet/logs/optimizer_20260530_083052_centralized_next8/CEN_A40_a31_mixstop150_ce128_r010.out` |
| clean runner-up | `CEN_C36_c26_gce008_cleanrisk_r010` | strict `84.36%`, score `85.60` | SAT avg `47.84` | `N607:/home/szu2070436088/2510044040/CV-SincNet/logs/optimizer_20260530_083052_centralized_next8/CEN_C36_c26_gce008_cleanrisk_r010.out` |
| satellite final-primary clear/mean | `CEN_A71_c58_phaseonly_joint_r010` | strict `82.74%`, score `83.94`, overall `87.01` | clear `49.95`, 3-scenario mean `49.38`, SAT min `47.76`, worstRx `72.51` | `N607:/home/szu2070436088/2510044040/CV-SincNet/logs/optimizer_20260531_003009_centralized_next8/CEN_A71_c58_phaseonly_joint_r010.out` |
| satellite high with stronger clean | `CEN_A31_a22_satboost_ce1p28_stack_r010` | strict `84.88%` | 3-scenario mean `49.0767`, clear `49.68` | same `CEN_A31` stdout |
| satellite/floor alternative | `CEN_C74_c71_satfloor_rebalance_r010` | strict `83.02%` | SAT avg `48.85`, clear `49.31` | `N607:/home/szu2070436088/2510044040/CV-SincNet/logs/optimizer_20260531_043129_centralized_next8/CEN_C74_c71_satfloor_rebalance_r010.out` |
| best worst-RX | `CEN_C26_a22_stack_fishr0_ce1p24_r010` | strict `83.75%`, score `85.33` | worstRx `82.44`, SAT avg `47.6667` | centralized raw stdout |

### 集中式解读

- ratio 0.1 原始 stdout 下，当前最强 clean/final-primary 不是后续 C73-C80，而是 `CEN_A31_a22_satboost_ce1p28_stack_r010`：strict `84.88%`，且同时保留约 `49.08%` 的三场景 SAT mean。
- 单看 SAT，`CEN_A71` 的 final-primary clear `49.95%`、三场景 mean `49.38%` 最高，但 clean strict 只有 `82.74%`，比 `CEN_A31` 低约 2.14 点。
- 集中式 clean 已明显强于联邦 current formal 记录，但 SAT floor/mean 仍停在约 `47-49%`；没有原始日志支持“当前路线能把 SAT strict/floor 推到 60%”。

## 已尝试路线与负例边界

### 联邦

| 路线 | 原始证据结论 | 判断 |
|---|---|---|
| receiver_day r010 | `FL82_01/02` clean peak 约 `66.11/66.15%`，final 约 `62.38/61.05%` | 明显低于 VMB/SBX clean anchor，不适合作为下一轮主线 |
| 早期 CVS/BEX02 + StyleBank | `FL82_04...stylebank_r010_l3` clean peak 约 `41.50%@R33`，final 约 `16.67%@R62`，`train_loss_rx_adv` 可爆到约 `233.55` | 当前 gating/adv/style 配置下崩塌；不能写成 StyleBank 机制永久失败 |
| StyleBank/GRL/Fishr earlydg/multiview | `FL82_12/14` clean peak 约 `68.3%`，final 约 `65%`，`clear_leo` 可到约 `42%` | 卫星可抬，但 clean 损伤过大，不能当强路线 |
| SBX02 Fishr/Style/LVMB | `SBX02_FISHR` clean `72.762%`，`SBX02_STYLE` clean `73.942%` 且有回落，低于 `SBX02_PROTO` `74.4217%` | 有 tradeoff，但不是 clean 最强 |
| Proto/SATCE tradeoff | `VMB5_C02` 5-scenario mean peak `41.124%`，但 final clean 约 `72.46%`；`A15` clean 更稳但 SAT 更弱 | 卫星提升常牺牲 clean/final stability |
| adv double-count / proto-only 对照 | A8/A9/A10 proto-only clean 约 `71.5%`；A11/A15 回到 `74.1-74.8%` | 支持“该配置相关失败”，但不能单独证明 double-count 是唯一因果 |
| VMB near-duplicate knob sweep | 多个 VMB5/VMB6 小改只在 peak、SAT 或 final 之一改善，未同时突破 clean final、SAT mean 和 rx8/floor | 不宜继续无新机制的小调参；应优先解决 rollback、rx8/min-RX、late/light SAT 激活 |

### 集中式

| 路线 | 原始证据结论 | 判断 |
|---|---|---|
| 强卫星/高 SATCE | `SA48_liteb_no_dac_sa34_ce1p2_r010` SAT clear 可到约 `49.34-49.65%`，但 final-primary strict 仅 `78.34%`、rx8 `63.80%` | 能抬 SAT，但 clean 代价过高 |
| SAT 高但 clean 不稳 | `SA49_id_phase_dsq_leo3_ce0p7_r010` 3-scenario SAT mean 约 `48.747%`，final-primary strict `79.04%` | 不适合作为 clean+SAT 主线 |
| late/high-CE | `CEN_A70_a62_satonly_highce_swad_r010` final-primary strict `76.65%`，SAT3 `43.57%` | 明显退化 |
| late-sat/low-rain/rain-only 高 CE | `SA65` final-primary strict `82.42%` 但 SAT3 `37.19%`；`SA66` strict `82.80%` 但 SAT3 `36.207%` | floor 崩塌，是强负例 |
| sat-cons / delayed sat | `CEN_A79_a72_delayed_satcons_floor_r010` strict `81.48%`，clear `37.78%`，SAT3 `37.067%` | 当前证据不支持继续 |
| Fishr/MixStyle/DSQ/Proto 堆叠 | `CEN_A55` strict `76.63%`、SAT3 `46.123%`；`CEN_C67` strict `77.25%`、SAT3 `46.313%` | 单纯堆机制会伤 clean，不宜继续 name-only 叠加 |
| rx8/floor 限制 | `CEN_A77` strict `80.67%`、SAT3 `47.967%`，但 rx8 `59.18%`；`CEN_A64` strict `82.24%`，rx8 `59.58%` | rx8/floor 是真实瓶颈，不是只靠提高整体 score 可解决 |
| 早期 concat-sat argparse 冲突 | 早期 SA02/SA04 类失败来自 launcher/argparse 冲突 | 这是无效科学负例，不能证明 concat-sat 思路不可行 |

## 当前 active 批次

- `optimizer_20260531_175432` 的 C81-C88 与 VMB7 当前仍是 active / startup-to-mid-run 状态。它们只能用于健康监控，不能作为 final record。
- 审查时 centralized 大约已到 E95-E97/170，VMB7 约 R30/R40；即便比更早的 E50/R16 有推进，也仍未完成。

## 当前最可信配置方向

- 联邦 clean：`A15_full_warmup40_r010` 是 final clean anchor；`VMB5_A08/A07` 是 peak/rollback 研究 anchor；`SBX02_PROTO/KDLOGIT` 是 clean 稳定替代。
- 联邦 SAT/joint：`VMB6_C04` 是 completed 5-scenario SAT final anchor；`VMB5_C02` 是 SAT mean peak anchor。下一步应围绕 final rollback、rx8/min-RX floor、late/light SAT 激活和冲突聚合，而不是继续无门控地叠 StyleBank/Fishr。
- 集中式 clean+SAT：`CEN_A31` 是当前 ratio 0.1 原始日志下最强综合 anchor；`CEN_A71/C74` 是 SAT 方向参考；`CEN_C26/A40` 是 worst-RX / clean 对照。
- 不宜继续的主方向：`r020/0.2` 口径、早期 default-on StyleBank/GRL/Fishr、late/high-CE rain-only、sat-cons delayed、无新机制的 near-duplicate knob sweep，以及把 launcher bug 当科学负例的比较。

## 最强记录的方法路线

### 联邦最强路线

所有 completed formal 联邦强记录共享同一底座：`wisig_train_ratio=0.1`、`fl_client_key=receiver`、7 个 receiver client、`fl_rounds=200`、`train_mode=fedcvs_vmb`、WiSig `rx_day` split、train days `2021_03_01/03_08`、test days `2021_03_15/03_23`、test RX `7-11`。VMB 底层机制是 transmitter identity prototype + receiver nuisance/adversarial separation，配置中启用 domain-balanced sampling/aggregation 与 transmitter-balanced batch。

| Run | 方法路线 | 关键配置 | 解释 |
|---|---|---|---|
| `A15_full_warmup40_r010` | VMB clean-first / longer warmup | `pretrain_rounds=40`, `lambda_tx_proto=0.1`, `lambda_rx_proto=0.1`, `lambda_tx_adv_r=0.1`, `freeze_rx_stage2=True`, `stage1_local_steps=1`, satellite train diagnostics inactive | 最稳 clean final。路线是先把 VMB 的 TX/RX 原型和 receiver 去捷径关系训练稳，再进入 stage2；不靠卫星训练增强，所以 clean 稳但 SAT 不强。 |
| `VMB5_A08_pre100_satfloor_ce085_r010` | VMB long-pretrain + satfloor + KD | `pretrain_rounds=100`, `stage1_local_steps=2`, `lambda_tx/rx_proto=0.16`, `prototype_clip_norm=0.45`, `prototype_ema=0.985`, `lambda_logit_kd=0.01`, `kd_reliability_gate=0.85`, `fl_sat_aug_mode=baseline_view`, `diag_sat_aug_active=1` | clean peak 最强，但 final rollback。路线是把 VMB 原型压得更紧，同时开 baseline satellite view CE 和轻 KD；可冲高峰值，但后期 satellite/KD 压力带来回落。 |
| `VMB5_A07_pcgrad_a07_lowproto_r010` | VMB low-proto / conflict-control lineage | `pretrain_rounds=90`, `stage1_local_steps=2`, `lambda_tx/rx_proto=0.12`, `prototype_clip_norm=0.45`, `prototype_ema=0.985`, `lambda_logit_kd=0.01`, satellite view active | `clear_leo` peak 最强。比 A08 的 proto 权重更低，倾向减少原型约束冲突；对单场景 SAT 有利，但 final clean 仍低于 A15。 |
| `VMB5_C02_a07_gate090_kd0075_r010` | VMB satellite-gated KD | `pretrain_rounds=90`, `lambda_tx/rx_proto=0.16`, `lambda_logit_kd=0.0075`, `kd_reliability_gate=0.90`, satellite view active | 5-scenario SAT mean peak 最强。路线是保留较强原型和卫星视图，同时用更高 reliability gate 限制 KD；SAT mean 能上去，但 final clean 只有约 `72.46%`。 |
| `VMB6_C04_bpc2_lownoise_guard_r010` | VMB BPC2 / low-noise guard / satellite final | `batches_per_client=2`, `pretrain_rounds=105`, `lambda_tx/rx_proto=0.14`, `prototype_clip_norm=0.35`, `prototype_ema=0.99`, `server_lr=0.006`, `server_momentum=0.75`, `lambda_logit_kd=0.006`, `kd_reliability_gate=0.93` | 5-scenario SAT final 最强。路线是用更多 client batch、更低 server update、更强 EMA/clip 和更严 KD gate 降低噪声，换来 SAT final 稳定，但 clean final 低于 A15。 |
| `SBX02_PROTO_r010` | split-BEX02 ProtoEvidenceBank / FedProto stats | `lambda_fed_proto=0.05`, `use_proto_evidence_bank=True`, `use_fed_proto_stats=True`, `proto_fusion_eval=True`, `lambda_tx/rx_proto=0.15`, `pretrain_rounds=20` | 本地 SBX02 clean 稳定替代。路线是把跨 receiver 的原型证据作为辅助，而不是强推 SAT；clean final 稳，卫星不是最强。 |

StyleBank 注意：这些强记录的配置里保留了 StyleBank 字段，但 top VMB metrics 中 `diag_style_batch_active=0`，所以不能把最强成绩归因于 StyleBank 生效。当前 strongest evidence 更支持 VMB/proto/KD/satellite-view/guard，而不是 StyleBank 主导。

### 集中式最强路线

集中式强记录共享底座：`train_mode=centralized`、WiSig `rx_day`、`train_ratio=0.100`、`epochs=170`、AdamW `lr=2e-4`、`lite_d`、`slim_group=none`、`id_branch=no_dac`、`domain_branch=no_stats`、`domain_enhancer=rcn_stats`、`exp_group=s3_rxrobust_no_dac`、eval on `test_unseen_day_unseen_rx`。它们的核心不是完整 satellite consistency loss，而是 `use_concat_sat_channel_aug + concat_sat_ce_only`：把卫星信道视图当监督 CE 训练样本，`lambda_sat_cls=lambda_sat_cons=0`，同时在 clean/SAT 之间用 MixStyle、GroupCE、Proto/SupCon、PA auxiliary 和 checkpoint selection 平衡。

| Run | 方法路线 | 关键配置 | 解释 |
|---|---|---|---|
| `CEN_A31_a22_satboost_ce1p28_stack_r010` | clean+SAT stack anchor | concat-sat CE-only, 5-scenario train cycle, `ce_weight=1.28`, `view_prob=1.0`, DSQ domain-freq stability, MixStyle `p=0.18/strength=0.70/late_start=110`, `lambda_group_ce=0.10`, `lambda_proto=0.015`, `lambda_supcon_id=0.020`, `lambda_fishr=0.005` | 当前 ratio 0.1 最强综合记录。路线是 A22/C26 stack 的增强版：强 clean identity backbone + moderate SAT CE boost + DSQ/MixStyle/GroupCE/proto/SupCon/Fishr 小权重叠加。clean strict `84.88%`，SAT mean 约 `49.08%`。 |
| `CEN_A40_a31_mixstop150_ce128_r010` | A31 clean runner-up / late MixStyle stop | 与 A31 同底座，lineage 是 A31 + `mixstyle_stop_epoch=150` | clean runner-up，worstRx 更强。路线是减少后期 MixStyle 干扰，牺牲部分 SAT mean，换更好的 worst-RX。 |
| `CEN_A71_c58_phaseonly_joint_r010` | phase-only joint / SAT mean anchor | concat-sat CE-only 3-scenario, `ce_weight=1.24`, `view_prob=0.95`, `id_time=phase_delta`, domain freq off, lighter MixStyle `p=0.10/strength=0.42/late_start=95`, stronger proto/SupCon `0.025/0.030`, Fishr off | SAT clear/mean 最强，但 clean 低。路线是从 clean-heavy DSQ stack 转向 phase-only joint 表示和更强 proto/SupCon，减少 MixStyle 强度，提高 SAT 泛化，代价是 strict clean 降到 `82.74%`。 |
| `CEN_C74_c71_satfloor_rebalance_r010` | satfloor rebalance / scheduled SAT view | concat-sat CE-only, `ce_weight=1.30`, `view_prob=1.0`, schedule `1@0.95` then `120@0.85`, phase_delta, light MixStyle `p=0.08/strength=0.38`, `lambda_proto=0.025`, `lambda_supcon_id=0.030`, Fishr off | SAT floor 候选。路线是继承 A71 的 phase-only SAT 方向，再用 schedule 降低后期 SAT view 压力，改善 floor，但 clean 仍低于 A31/A40。 |
| `CEN_C26_a22_stack_fishr0_ce1p24_r010` | worst-RX / group-risk anchor | concat-sat CE-only, 5-scenario train cycle, `ce_weight=1.24`, DSQ domain-freq, MixStyle `p=0.18/strength=0.70`, `lambda_group_ce=0.10`, `lambda_proto=0.015`, `lambda_supcon_id=0.020`, Fishr off | worst-RX 最强。路线偏向 group-risk 和 receiver floor，clean/SAT 不如 A31，但 rx8/worst-RX 更稳。 |

集中式总路线可以概括为：`A22/C26 DSQ stack` 给 clean 与 rx floor，`A31` 在这个 stack 上加更强 SAT CE 和轻 Fishr 得到最强综合；`A71/C74` 则切向 phase-only + light MixStyle + stronger proto/SupCon，把 SAT mean/floor 推高，但 clean 回落。

## Supervision notes

- 联邦子审计修正：不能写“VMB 后续未超过 anchor”；应写“VMB 的 peak 和 5-scenario mean 已超过部分旧 anchor，但 final stable clean 未显著超越 A15/SBX02，且 rollback/tradeoff 仍在”。
- 集中式子审计修正：原始 stdout 支持 `CEN_A31` 是 ratio 0.1 clean/final-primary 最强；`CEN_A71` 是 SAT clear/mean 最强但 clean 低。
- 方法审查约束：任何“不可行”都限定为当前配置/launcher/gating/数据口径下不宜继续，不等于机制本身被数学证明失败。
