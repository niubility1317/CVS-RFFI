# N607 联邦与集中式训练全历史记录/失败路线审计

Generated: 2026-05-31 19:35 Asia/Hong_Kong

Scope: 本轮做了 direct `N607` 只读 preflight/状态检查，并重读本地历史报告、结构化 JSON、conversation index 和三个子 agent 审计意见。没有改远端文件、没有 kill 进程、没有启动新实验。

## 审计边界

- SSH 边界：direct `N607` preflight 通过，检查了服务器时间、项目根目录和 GPU；每次 SSH 后本地确认无残留 `ssh.exe`、无到 `172.31.111.215:22` 的 ESTABLISHED 连接。
- 当前 N607 状态：每张 GPU 正好 2 个训练进程，1 个 centralized + 1 个 federated/VMB；因此本轮严格 monitor-only。
- 当前 active centralized：`CEN_C81_c76_satfloor_bridge_r010` 到 `CEN_A88_c76_satmax_swad_r010`，检查时约 E50-E54/170，test eval 尚未进入主窗口，不能算 final record。
- 当前 active federated/VMB：`VMB7_C01_a05_stylebank_finalguard_r010` 到 `VMB7_A08_a08_allsat_styleproto_r010`，检查时约 R16-R19/200，strict/test 仍为 heavy-eval 前状态，不能算 final record。
- 口径原则：只在同协议/同 evaluator 内做硬排行；跨 federated、VMB、centralized、3-scenario SAT、5-scenario SAT、旧 `0.2/220` 之间只做 directional 解释。

## 指标定义

- `clean strict UDU`：通常对应 `test_unseen_day_unseen_rx`，是主 clean cross-domain 排名指标。
- `final` 与 `best` 分开：best 是训练过程中最高点，final/latest 是最终或最后可用 checkpoint；历史上多条路线 best 高但 final rollback。
- `satellite clear_leo`：单场景卫星指标，不能和 3/5-scenario satellite mean 直接相减。
- Centralized 近批 `sat avg/min`：多数是 `clear_leo, low_elev_leo, rain_leo` 三场景 strict UDU 的均值/最小值。
- VMB/SBX/B60 多数 satellite 表是 5-scenario：`clear_leo, low_elev_leo, rain_leo, storm_mp, mixed_orbit`；有些 parser 输出 aggregate mean 与 strict mean，两者也不能混写。
- `r020/0.2/220`、`ratio=0.2` centralized sat-aug、target adaptation、Stage2-only bug run 都是历史参考或无效证据，不进入当前 formal 主榜。

## 子 Agent 审计结论

- 覆盖面审计：原报告像“结论摘要”，缺 SBX02、StyleBank/GRL/Fishr 证据卫生、VMB6 前 lineage、SA11-SA66 桥接、CEN batch ledger 和 concat-sat validity ledger。
- 数值核验：`FSDG49` 的 `75.9167%` 是 final/latest，不是 best；best 是 `76.2950%@R127`。SBX02 有 5/8 本地结构化核验与后续 report snapshot 8/8 两种边界。A40 的 caveat 不是“缺 satellite 字段”，而是“早期批次/选择口径不同”。
- 方法学审查：不要把“当前配置失败”写成“机制本身不可行”。StyleBank/Fishr/VMB DG 的结论应限定为当前配置、gating、batch-domain、metric/launcher 证据；cross-lane 数值不能硬相减。

## Evidence Map

| Family | Source | Status | Hard-comparable scope |
|---|---|---|---|
| Early FSDG / FedAvg / FedProx | `20260521_205515`, `20260522_195855`, `20260523_*`, `20260527_1549_federated_best_scan` | completed historical | FL internal only |
| FL82 / FLCL formal ladder | `20260526_004220_fl82_fed_validation`, `20260526_215453_classifiability_ladder`, `20260527_1549_federated_best_scan` | completed/part invalid | split by `r020` vs `r010` |
| StyleBank/GRL/Fishr FL82 | `fed_pvs_r010_validation_20260526_121958`, `20260527_104701_*`, `20260527_111318_*`, `stylebank_hetero_grl_collab_traceability.md` | diagnostic + partial | mechanism evidence, not final main榜 |
| SBX02 split-BEX02 | `20260528_split_bex02_alternatives`, `20260528_162922_federated_log_analysis`, `split_bex02_alternatives_traceability.md` | completed, 8/8 snapshot, 5/8 earlier structured cache | SBX02/VMB-like only |
| VMB A1/A5/A8-A15 | `20260527_192102_*`, `20260528_000804_*`, `20260528_162922_*` | completed/diagnostic | VMB same launcher family |
| VPT/B60/VMB2-VMB6 | `20260529_015529_*`, `optimizer_20260529_233451`, `optimizer_20260530_053154`, `optimizer_20260530_173224`, `optimizer_20260530_233142`, `optimizer_20260531_175432` | completed | VMB optimizer family |
| Central concat-sat/BOSV/CE-only | `20260525_223950`, `20260525_233854_*`, `20260526_094539_*`, `cvs_sat_ceonly_central_20260527_113422` | mixed valid + launcher-failure ledger | centralized sat route only |
| Central SA11-SA42 | `sat_log_organization_20260527_191519`, `cvs_backbone_stability_*`, `cvs_sa27_optimization_*`, `cvs_sat_leo_multival_*` | completed historical | SA internal; fair LEO subset separate |
| Central SA43-SA66 | `cvs_sa43_stable_aggressive_*`, `cvs_sa51_sa58_*`, `20260529_024719_centralized_sa59_sa66_optimizer` | completed/full-log parsed | SA full-log family |
| CEN optimizer batches | `optimizer_20260530_123129`, `optimizer_20260530_163217`, `optimizer_20260530_203118`, `optimizer_20260531_003009`, `optimizer_20260531_043129`, `optimizer_20260531_175432` | completed through C80; C81-C88 active | current centralized optimizer family |

## 最强记录总览

### Federated / FL Historical Tiers

| Tier | Strongest row | Metric | Interpretation |
|---|---:|---:|---|
| All historical peak, invalid for current target | `FL82_04_fedprox_rx_ra_bex02_cvs_stylebank_r020_l3` | best clean strict UDU `79.0400%@R105`; final collapsed to `16.6667%` | 旧 `r020/0.2/220`，只能说明历史峰值存在，不能作为当前 `r010/0.1/200/receiver` 记录 |
| Stable non-r020 historical anchor | `FSDG49_fedprox_receiver_ra_bex02_cvs_sat` | best `76.2950%@R127`; final/latest `75.9167%@R170` | 仍是稳定 FL clean 历史锚点；支持 receiver-client + receiver-agnostic BEX02/CVS 比 receiver-day 更稳 |
| Current formal pre-VMB anchor | `FLCL_07_monotonic_clean60_sat_style_r010` | best/final `73.3983%@R200`; clear_leo `32.2967` | `r010/200/receiver` formal ladder内 clean 最强 |
| Current formal satellite reference | `FL82_09_fedprox_rx_ra_bex02_baselineview_clearleo_l3_r010` | clean best `71.1650`; clear_leo best `43.0333` | clear_leo 单场景最强历史参考，不是 clean winner |

### Federated / SBX02 and VMB

| Family | Strongest clean | Satellite/joint note | Evidence boundary |
|---|---:|---:|---|
| VMB A8-A15 | `A15_full_warmup40_r010`, best strict UDU `74.81%@R199`, last `74.46` | clear_leo about `34.33` | completed current-contract VMB; clean-first |
| SBX02 split-BEX02 | `SBX02_PROTO_r010`, best/final `74.4217%@R200`; `KDLOGIT` `74.2700` | `COMBO` clear `38.52`, `SATCE` clear `37.98` but lower clean | 8/8 final snapshot available; earlier local structured cache was 5/8 |
| VPT/B60 | `VPT_B60_mixstyle_satce_r010`, best clean `74.298%@R194` but final `72.493` | `VPT_B60_proto015_satce_r010` final clean `73.790`, best sat avg `39.639`, final sat avg `38.967`, floor `36.147` | report-level parsed; selected as practical anchor but no-launch for near-duplicate followups |
| VMB5 | `VMB5_A07_pcgrad_a07_lowproto_r010`, best `74.790%@R180`, final `73.785` | satellite/joint fields null in `vmb5_analysis.json` | clean verified; not satellite record |
| VMB6 clean peak | `VMB6_C02_minrx_groupce_uniform_nokd_r010`, best `73.8517%@R177`, final `72.4717` | final sat avg `39.414` | completed same VMB6 batch |
| VMB6 joint/risk | `VMB6_C04_bpc2_lownoise_guard_r010`, best `73.725%@R179`, final `72.855` | best/final sat avg `41.095/40.5767`, risk `72.681` | best completed VMB6 satellite/joint/risk anchor |
| VMB6 final-clean stability | `VMB6_A05_stylebank_receiver_balanced_r010`, best `73.6633`, final `72.955` | final sat avg `39.1123` | style receiver balancing reduces rollback but not enough |

Federated bottom line: 如果只看当前 formal/current completed family，clean 上限大约是 `74.8%` strict UDU；如果允许早期稳定非 r020 历史，`FSDG49` 到 `76.295/75.917`；如果允许旧 invalid r020，`FL82_04` 峰值到 `79.04` 但 final 崩塌且不合当前合同。卫星方面最多是 clear_leo 单场景 `43.03` 或 VMB/SBX 5-scenario 低 40 档，离 `>=60` floor 很远。

### Centralized / Historical and Current Tiers

| Tier | Strongest row | Metric | Interpretation |
|---|---:|---:|---|
| Ratio 0.2 historical sat-aug reference | `SA01_cvs_loss_mixed`, `SA02_concat_sat_mixed` | `SA01` best primary `87.75`, strict `86.42`; `SA02` clean strict `84.18`, five-sat strict clear/low/rain/storm/mixed `50.45/49.78/48.34/43.73/48.51` | not current r010 protocol; useful only as design reference |
| Early r010 CE/backbone clean anchor | `SA18_domain_dsq_ch2_r010` | primary `85.69`, strict `84.18`, LEO `44.77/43.66` | clean capacity anchor before later satellite balancing |
| Early fair LEO route | `SA34_sa27_ch2_leo3_ce1p2_r010` | primary `83.91`, strict `82.41`, LEO `47.49/46.22` | balanced LEO route before SA47/SA60+ |
| SA59-SA66 clean anchor | `SA60_sa55_ce1p32_eval81_r010` | primary `84.42`, overall `87.12`, strict `83.26`, SAT mean/min `46.22/44.91` | clean strong but satellite floor short |
| SA59-SA66 satellite/joint anchor | `SA64_floor_lowrain_ce1p45_eval81_r010` | primary `84.19`, strict `82.87`, SAT mean/min `47.94/46.41` | best SA59-SA66 joint/satellite route |
| CEN clean-only optimizer peak | `CEN_A40_a31_mixstop150_ce128_r010` | strict `84.37`, score `85.70`, overall `89.11` | strongest clean row in local optimizer-summary family; earlier/selection口径不同 |
| CEN early satellite row | `CEN_C35_c26_clean_satce128_nofishr_r010` | strict `84.04`, sat mean/floor `48.843/47.34` | clean still high with strong satellite floor in that batch |
| CEN C49-C56 risk/sat rows | `CEN_A56_a48_proto_supcon_fishr_swad_r010`, `CEN_C51_c36_dualworst_cleanrisk_r010` | A56 strict `84.02`, sat mean/min `47.633/46.40`; C51 strict `84.22`, sat mean/min `48.133/46.85` | C51 best clean/sat/joint; A56 best risk by that summary |
| CEN C57-C64 clean/risk | `CEN_C58_a56_jointfeat_lowfishr_r010` | strict `83.75`, sat mean/min `47.757/46.35`, best strict `83.83` | clean/risk anchor in C57-C64 |
| CEN C57-C64 satellite/floor | `CEN_A62_a56_jointfeat_proto035_latestable_r010` | strict `83.00`, sat mean/min `48.887/47.35`, joint `65.944` | explicit satellite mean/min best in that parser |
| CEN C65-C72 | `CEN_A71_c58_phaseonly_joint_r010` | strict `82.74`, score `83.94`, sat avg about `49.38` | middle batch, later beaten on clean/risk by C76 but strong sat reference |
| CEN C73-C80 balanced/risk | `CEN_C76_c68_ema_phase_bridge_r010` | score `84.43`, strict `83.65`, overall `87.22`, sat avg `48.20`, risk `84.275` | current best balanced/risk completed centralized record |
| CEN C73-C80 satellite | `CEN_C74_c71_satfloor_rebalance_r010` | score `84.18`, strict `83.02`, sat avg `48.85`, sat floor `47.36` | current best recent satellite-balanced completed CEN row |

Centralized bottom line: centralized clean identity 已经稳定超过 `82%`，多个 current r010 centralized 配置在 `83-84%` strict UDU；最强 clean-only optimizer row 是 A40/C36/C35 一带，当前最实用三角是 `C76` clean/risk、`C74` satellite-balanced、`A62/A71` satellite-floor reference。卫星 strict floor 仍在 47-49 左右，未达到 60。

## 已尝试路线总账

### Federated

- FedAvg/FedProx receiver-day CE：作为早期 baseline；receiver-day 客户端太单域，后续被 receiver-client + receiver-agnostic BEX02/CVS 路线压过。
- Receiver-client FedProx + receiver-agnostic BEX02 + CVS satellite consistency：`FSDG49` 是稳定历史 anchor，best `76.295`，final `75.917`。
- Direct BEX02 DG inside receiver-day FL：作为“blind DG”方向证据不足，核心问题不是代码一定错，而是 client/local batch 没有足够域结构，Fishr/GRL/MixStyle 条件弱。
- FL82 StyleBank/ProtoBank/GRL/Fishr：发现过 default-on StyleBank 污染 control、Fishr domain count 不满足、adv head duplicate pressure、sat loss 与 style batch gating 语义问题；后续本地修了 opt-in、double count、sat loss gating、conservative gates。
- FLCL classifiability ladder：证明 cleaner monotonic schedule 比激进 StyleBank 更稳，`FLCL_07` 是 formal pre-VMB clean anchor。
- StyleBank targetrx/zdom and methods ablation：target-domain label、zdom probe、target-balanced、soft style、real-mix 上界都做过；`real_mix32` 是 diagnostic/privacy-violating，不可作为 FL 方法。
- SBX02 split-BEX02 alternatives：`LVMB/PROTO/FISHR/STYLE/KDLOGIT/QTOKEN/SATCE/COMBO` 八路完成；PROTO/KDLOGIT clean 最强，COMBO/SATCE satellite 更强但 clean lower；StyleBank batch active 证据不足，explicit FedProto loss 也显示 inactive。
- VMB A1/A5/A8-A15：验证 full VMB/更长 warmup 比 CE-only stage2 更强；A5 曾有 Stage2-only launch bug，不能作科学负例。
- VPT/B60 pretrain SDG：B60_proto015 是 practical sat/joint anchor，B60_mixstyle 有 clean peak 但 rollback；no-launch 审查拒绝了近重复 knob sweep。
- VMB2-VMB6 optimizer：从 proto/clip/RFDR/KD/PCGrad/GroupCE/BPC2/StyleBank receiver balancing 逐步走向 rollback、rx8/min-RX、sat floor、risk-adjusted selection。
- VMB7 current active：正在测试 finalguard、BPC2 satfloor、asymproto、minrx no-KD、stylebank BPC2、unfreeze style lowLR、PCGrad groupCE、allsat styleproto；尚非 final。

### Centralized

- Early CVS/satellite augmentation：`SA01/SA03` late weak satellite CE 完成，`SA02/SA04` 初始 concat-sat 因 mutually exclusive flags 失败，后续 recovery 才是有效证据。
- Baseline-origin sat view / BOSV：确认 baseline-style supervised satellite view 是重要参考；但 early `ratio=0.2` 与 current r010 不能硬比。
- Concat satellite CE-only：`SA09/SA10` 路线把 satellite sample 只作为 supervised TX-CE，避免进入完整 CVS loss stack；这是后续 CEN satellite CE-only 的语义基础。
- SA11-SA18 backbone/DSQ/phase：建立 clean identity capacity，`SA16/SA18` 是早期 clean anchors。
- SA26-SA30 fair LEO set：`SA27` 是 fair LEO optimization winner，SA34/SA36 继续探索 satellite tradeoff。
- SA43-SA50：围绕 SA34 做 stable/aggressive central exploration，`SA47` 成为 SA51-SA58 的中心 anchor。
- SA51-SA58：探索 seed stability、CE weights、late SAT start、rain-only；证明 late/high satellite emphasis 不自动解决 floor。
- SA59-SA66：`SA60` clean，`SA64` joint/sat，`SA65/SA66` 是 late high-CE low/rain 或 rain-only collapse 的强负例。
- CEN optimizer C33-C80：系统扫 MixStyle stop/anneal、sat CE weight/schedule、dual-worst GroupCE、DSQ/phase、PA/DAC/no-DAC、proto/SupCon、EMA/SWAD/SWA、rx8 floor guards；C76/C74/A62/A71 是当前强三角。
- CEN C81-C88 current active：在 C76/C74/A78/A79 上继续桥接 satfloor、clean recovery、Fishr floor、joint sat recovery、rxchain、nosat control、satmax SWAD；尚不能用于结论。

## 当前证据下不宜继续的路线

这里的“不宜继续”不是数学证明机制永远不可行，而是“按当前日志、配置与 evaluator 证据，不应作为下一轮主路线”。

- 旧 `FL82_04 r020/0.2/220`：峰值高但 final 崩塌，且不合 current formal `0.1/200/receiver` 合同。
- Receiver-day + direct BEX02 DG + full Fishr/GRL：当前证据显示 client/local batch 域不足，强 DG loss 没有可用域结构支撑；不宜盲目加损失。
- 早期开闸 StyleBank：default-on contamination、`fishr_min_domains` 不匹配、adv loss fallback double count、sat loss gating、`replay_start=2/dg_start=3/replay_prob=1` 的组合已被证明不稳；这不是 StyleBank/Fishr 机制本身失败的证明。
- StyleBank/Fishr name-only knob：如果 `diag_style_batch_active=0`、`style_dg_ready=0`、`fishr_active=0` 或 FedProto loss 为 0，不能把结果解释为机制有效/无效。
- SATCE/COMBO 直接强推 satellite：SBX02 和 FLCL 证据都显示 satellite 可以上升，但 clean strict UDU 常掉 1-3 点；要有 clean guard 和 late/light schedule。
- COMBO all-at-once：StyleBank 与 baseline satellite CE 在旧代码路径里 per local step 互斥，所谓组合实际部分交替；不宜再堆全部机制，除非先修同 step mixed-batch 语义。
- Prototype/fusion 替代主分类器：当前较适合作为 gated auxiliary evidence/KD/proto regularization，直接替代 CE head 风险高。
- VMB near-duplicate knob sweep：B60 no-launch 已拒绝 lower proto、微调 stage length、重复 StyleBank 等近重复路线；没有新机制/本地测试前不宜 launch。
- VMB A5 Stage2-only：这是 launch config bug，无效证据，不能参与方法优劣比较。
- Central SA65/SA66 式 late high-CE low/rain 或 rain-only：clean 可保持但 satellite mean/min 大崩，当前是明确负例。
- Uniform all5 strong satellite supervision：SA04/相关 all5 参考显示太钝，容易牺牲 clean 或不如 mixed/low-rain targeted route。
- Concat-sat 初始失败日志：SA02/SA04 早期失败是 argparse/launcher 冲突，不是 concat-sat 科学负例。
- CEN A79/A72 式只看 joint 高分：joint/score 可能被 clean 或选择口径抬高，但 satellite collapse/failed stale 必须作为硬风险。
- Centralized target adaptation：改变协议和 checkpoint dependency，应继续独立成 family，不进入 current centralized 主榜。

## 当前最可信配置

Federated:

- Clean anchor：`A15_full_warmup40_r010`、`VMB5_A07`、`SBX02_PROTO/KDLOGIT`、`FSDG49`。
- Satellite/joint anchor：`VMB6_C04`、`VPT_B60_proto015`、`SBX02_COMBO/SATCE`。
- 下一步不该是“更多无门控 DG loss”，而应是稳定 VMB/full-objective 域信号、warmup、conflict aggregation、final rollback control、rx8/min-RX floor、late/light satellite activation，以及明确 StyleBank/SATCE 同步或互斥语义。

Centralized:

- Clean/risk anchor：`CEN_C76`，同时保留 `CEN_A40/C36/C35` 作为 clean-only 历史参考。
- Satellite/floor anchor：`CEN_C74`、`CEN_A62`、`CEN_A71`。
- 下一步瓶颈不是 clean identity capacity，而是 clean-satellite tradeoff、rx8/min-RX floor、satellite floor 和 checkpoint selection。

## Source Appendix

- `E:\type10-7\conversation_index\`：本轮已 rebuild，182 entries。
- `E:\type10-7\automation_reports\CV-SincNet\20260527_1549_federated_best_scan\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\fed_pvs_r010_validation_20260526_121958\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\20260527_104701_fl82_stylebank_targetrx_zdom\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\20260527_111318_fl82_stylebank_methods_ablation\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\20260528_split_bex02_alternatives\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\20260528_162922_federated_log_analysis\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\20260528_000804_fedcvs_vmb_mechanism_matrix\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\20260529_015529_vmb_optimizer_no_launch\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\optimizer_20260530_173224\artifacts\vmb4_evidence_summary.json`
- `E:\type10-7\automation_reports\CV-SincNet\optimizer_20260530_233142\vmb5_analysis.json`
- `E:\type10-7\automation_reports\CV-SincNet\optimizer_20260531_175432\artifacts\evidence_summary.json`
- `E:\type10-7\automation_reports\CV-SincNet\20260525_223950\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\20260525_233854_baseline_origin_sat_view_design\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\20260526_094539_concat_sat_recovery\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\cvs_sat_ceonly_central_20260527_113422\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\sat_log_organization_20260527_191519\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\cvs_sa27_optimization_central_20260527_204005\related_central_full_log_analysis.md`
- `E:\type10-7\automation_reports\CV-SincNet\cvs_sa43_stable_aggressive_central_20260528_120112\sa43_sa50_full_log_analysis.json`
- `E:\type10-7\automation_reports\CV-SincNet\cvs_sa51_sa58_sa47_eval91_central_20260528_174616\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\20260529_024719_centralized_sa59_sa66_optimizer\report.md`
- `E:\type10-7\automation_reports\CV-SincNet\optimizer_20260530_123129\artifacts\centralized_evidence_summary.json`
- `E:\type10-7\automation_reports\CV-SincNet\optimizer_20260530_203118\artifacts\centralized_c49_c56_full_log_summary.json`
- `E:\type10-7\automation_reports\CV-SincNet\optimizer_20260531_003009\artifacts\centralized_c57_c64_full_log_summary.json`
- `E:\type10-7\automation_reports\CV-SincNet\optimizer_20260531_043129\artifacts\centralized_c65_c72_full_log_summary.json`
