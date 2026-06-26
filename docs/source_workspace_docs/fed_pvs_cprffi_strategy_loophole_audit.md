# Fed-PVS-CPRFFI 策略漏洞审计与修正版

更新时间：2026-05-25  
对象：`C:/Users/lh594/Downloads/fed_pvs_cprffi_final_design.md` 与现有 `E:/type10-7/code` CVS-RFFI 主体  
结论类型：策略审计，不是新实验结果

## 0. 置信边界

我不能对“该策略最终一定提升指标”给出事实意义上的 100% 信心。只要还没有跑完受控消融，经验结果就不可能 100% 确定。

我可以给出 100% 确定的边界是：

1. 原始策略不能直接宣称已经闭环，因为当前代码还缺少 StyleBank 多视图本地目标、构造域标签 `d_style`、多原型 ProtoBank、可靠性融合评估等关键部件。
2. 直接把集中式 DG loss 放进单域化 federated client 是高风险路径，N607 的 receiver-day BEX02 FL 结果已经显示它弱于 CE-only FL 对照。
3. 修正版策略必须先建立诊断与门槛，再逐层打开 StyleBank、DG loss、ProtoBank fusion、few-shot adaptation。
4. 任何“继续推进下一阶段”的判断都必须由本文件中的 stop/gate 条件触发，而不是由架构直觉触发。

因此，修正版不是“我 100% 确信一定赢”，而是“我 100% 确信这是当前证据下更严格、更少漏洞、可证伪的推进策略”。

## 1. 审计循环

| 轮次 | 审计问题 | 发现 | 策略变化 |
|---|---|---|---|
| Loop 0 | 原始融合是否结构上成立？ | 成立：CVS-RFFI backbone/base classifier 可以保留，StyleBank/ProtoBank/FSL 可以作为外层机制。 | 不重写主干，围绕 federated trainer 增加统计、采样、融合模块。 |
| Loop 1 | 是否已经可安全实施？ | 不安全：当前 federated local objective 仍是 single-main-view first，DG loss 还没有构造域 `d_style`。 | StyleBank 先做 diagnostics/no-op，再做 style-conditioned augmentation，最后才打开 DG loss。 |
| Loop 2 | 原型头是否能直接复用？ | 不能：当前 FedProto 是 class sum/count 单均值，FJMP/SGC 是 post-stage 思路。 | ProtoBank 改成 inference-only conservative evidence，先离线评估 harm/rescue，再进入训练闭环。 |
| Loop 3 | N607 证据是否支持关键假设？ | 支持“盲目 DG 有害”和“卫星强监督有 clean-vs-sat tradeoff”；但 strict concat-sat 的 SA02/SA04 失败，不能作为负证据。 | 增加 Phase -1 实验卫生：先修正 launcher 冲突和证据标签，再引用结果。 |
| Loop 4 | 是否存在可观察退出条件？ | 原始策略没有足够硬门槛。 | 加入 style coverage、random-vs-conditioned、clean drop、harm/rescue、privacy probe、FedProx ratio 等 gates。 |

## 2. 已核验证据

### 2.1 本地代码证据

- `code/train.py` 已有 `fedavg/fedprox`、`receiver/receiver_day/...` client key、FedProto stats、BEX02/receiver-agnostic objective、Fishr、GRL、MixStyle、sat consistency、concat-sat 等入口。
- `code/federated/fed_trainer.py` 的 `global_proto_stats` 当前是 `class_sum/class_count -> class_proto`，不是多 prototype mixture。
- `FederatedTrainer._compute_local_objective` 当前围绕 `x_main` 组织训练，可选 sat consistency 或 baseline sat view，但没有构造 `x_local + x_remote_style + x_style_phys + x_sat_style` 和独立 `d_style`。
- `fed_aggregate.py` 当前聚合默认没有排除本地 adapter/domain-private 参数；如果新增 client-local heads，必须显式处理。
- `model_dual_cvsincnet.py` 已有 `z_id`、`z_dom`、RCNStatEncoder、GRL heads、MixStyle hooks，适合承接风格/域诊断，但不能自动解决 federated single-domain 条件。
- `DataAugmentation.py` 和 `sat_channel.py` 提供物理扰动和卫星信道基底；缺的是 StyleBank 条件化采样层。

### 2.2 N607 结果证据

- CE-only receiver-day FL 在 `FSDG12/12A/13/14/14A/14B/15/16/17` 一带 strict UDU 约 70.7-71.7。
- 直接把 BEX02 DG 放进 receiver-day FL 的 `FSDG18/FSDG19/FSDG1A/FSDG1B` strict UDU 约 69.6-69.9，弱于 CE-only FL 对照。
- `FSDG49_fedprox_receiver_ra_bex02_cvs_sat` 达到 overall 80.30、strict UDU 75.92，是当前 inspected FL 家族里最强证据。
- `FSDG50_fedprox_receiver_ra_bex02_baseline_sat` strict UDU 70.52，说明强卫星监督/基线式扩张在 FL 下可能压坏身份学习。
- centralized satellite logs 显示强 sat-view supervision 能提高 satellite strict UDU 到约 48-50，但伴随 clean strict UDU 降低，存在 clean-vs-sat tradeoff。
- strict concat-sat 的 `SA02/SA04` launcher 触发 mutually exclusive args 错误：`--use_sat_consistency` 与 `--no_use_sat_consistency` 同时出现。因此这些失败不能被解释成 concat-sat 方法无效。
- `fed_proto_smoke` 只证明 FedProto plumbing 能跑，不能证明 multi-prototype head 有性能收益。

## 3. 漏洞清单与修复

| 漏洞 | 为什么会破坏原策略 | 证据状态 | 正确修复 | 通过门槛 |
|---|---|---|---|---|
| 1. DG loss 没有真实多域局部 batch | GRL/Fishr/SupCon 依赖跨域统计；receiver-day client 内部很容易退化成单域或弱多域。 | 本地代码 + N607 BEX02 FL 结果已验证。 | 先构造 `d_style` 多视图 batch，再启用 DG。 | 每个 batch 至少 3 个 style domain；Fishr pair 非零；domain labels 不是 raw receiver/day。 |
| 2. MixStyle 仍可能用 raw domain | 用 raw domain 会把真实 receiver/day 当成 style source，无法表达远端 StyleBank 构造域。 | 本地代码路径显示需要改造。 | MixStyle、Fishr、GRL、same-TX consistency 全部接收 `d_style`；`d_raw` 只用于日志/评估。 | 单测断言 `d_style` 被传入 loss hooks；raw-only batch 不启用跨域 loss。 |
| 3. FedProto 被误认为 ProtoBank | 单 class mean 会抹掉 client/style/mode mixture，且没有 reliability。 | 本地代码已验证。 | 新建 ProtoEvidenceBank，保留多原型与 metadata；FedProto 只做 baseline。 | 每类保留多个 prototype，含 count/margin/entropy/intra_var/drift/clean_sat_kl/reliability。 |
| 4. FJMP/SGC 不能直接移植为训练头 | 现有 FJMP/SGC 是 post-stage 或 adapter/fusion 思路，强替换 classifier 可能伤害 base classifier。 | 本地模块语义已验证。 | 只复用 safe fusion、harm/rescue、prototype diagnostics 思想。 | `p_final=(1-rho)p_base+rho p_proto`，初始 `rho<=0.05`，必须报告 harm/rescue。 |
| 5. strict concat-sat 证据污染 | SA02/SA04 失败来自参数冲突，不是方法失败。 | N607 日志已验证。 | 修 launcher，重跑 strict concat-sat 前不得引用其性能结论。 | dry-run 无互斥参数；日志中确认 concat branch 生效。 |
| 6. 强卫星视图可能牺牲 clean identity | sat strict 提升伴随 clean strict 下降，FL 中 `FSDG50` 更明显。 | N607 日志已验证。 | 使用 clean-anchored objective：sat view 只能作为一致性/受限辅助，不得压过 clean CE。 | clean strict drop 不超过 1-2 pp；sat strict 有统计显著提升；否则回滚。 |
| 7. FedProx 可能只是名义组件 | 多个 FedProx run 的 prox/cls ratio 很小，准确率近似 FedAvg。 | N607 metrics 已验证。 | FedProx 保留为 optimizer baseline，不作为核心贡献。 | 只有当 prox_ratio 非零且改善 drift/strict UDU 时才写入贡献。 |
| 8. StyleBank 统计可能泄露设备指纹 | RFFI 的风格统计本身可能携带 TX/receiver 可识别信息。 | 风险未实验验证。 | class-marginalized、coarse-bin、cluster centroid、EMA、可选 DP noise；做 membership/client probe。 | probe AUC 接近随机或低于预设阈值；否则降低粒度。 |
| 9. Style-conditioned physical sampler 可能退化成随机增强 | 如果 StyleBank stats 不能映射到 CFO/SRO/IQ/channel 参数，方法只是随机 DR 的复杂版本。 | 风险未实验验证。 | 设置 random physical、style-conditioned、oracle/local-stat 三个对照。 | style-conditioned 明显优于 random，且 style coverage/MMD 指标改善。 |
| 10. client 粒度还未定论 | receiver 与 receiver_day 的结论不同，最终 FL 设置不能凭直觉选。 | N607 显示 receiver-client + RA + CVS sat 当前更强。 | receiver、receiver_day 作为独立假设运行；不要混写结论。 | 至少报告 strict UDU、overall、client drift、communication cost。 |
| 11. 一次加入 StyleBank+DG+Proto 会不可归因 | 指标变化无法定位来自风格增强、DG loss、sat view 还是 fusion。 | 策略风险。 | 分阶段消融：diagnostic -> no-DG style aug -> DG -> Proto eval -> few-shot。 | 每阶段只增加一个主变量，且保留上一阶段 best checkpoint。 |
| 12. 聚合边界可能污染本地参数 | 如果新增 adapter/domain heads 后仍全部 FedAvg，会破坏本地特化假设。 | 本地 aggregation 入口需改造。 | 显式 `exclude_keys` 或 local-only module registry；报告哪些参数聚合。 | 单测检查 excluded keys 不进入 global state。 |
| 13. 缺少通信/存储成本约束 | StyleBank/ProtoBank 可能带来不可接受的上行成本。 | 风险未实验验证。 | 每轮记录 packet bytes、prototype count、server bank size。 | 成本相对 FedAvg 参数上传可解释，且随轮数有上限。 |
| 14. few-shot 过早介入会掩盖主问题 | FSL 可以在目标域校准，但不能修复 FL-DG 主训练失败。 | 策略风险。 | few-shot 放到 Stage 5，只对已冻结或稳定 base 做 calibration/adaptation。 | 先证明 no-fewshot model 稳定，再评估 new domain/new class few-shot。 |
| 15. 单 seed/单 run 会制造假确定性 | strict UDU 小幅波动可能来自随机种子、GPU nondeterminism、early stop。 | 风险未实验验证。 | 关键结论至少 3 seeds，报告 mean/std 和 best/last。 | 新策略必须在均值上赢，且方差不能覆盖全部收益。 |
| 16. split 或 eval 口径可能泄漏 | StyleBank/ProtoBank 如果看见 test-unseen receiver/day/sat 信息，会把泛化问题变成 transductive tuning。 | 风险未实验验证。 | 明确 train clients、server bank 来源、val/test split；禁止从 test eval 更新 bank。 | run report 中列出 bank update 数据源，测试集只读。 |
| 17. 本地与 N607 代码版本可能漂移 | local-first 设计如果 sync 不完整，会导致本地审计和远端运行不是同一个策略。 | 项目流程风险。 | 每次 N607 实验前写 report、git/diff/hash、SYNC_MANIFEST/local snapshot，并用 `scp` 同步。 | report 中有文件映射、hash/diff 摘要、远端命令和验证输出。 |

## 4. 修正版策略 V2

### Phase -1：证据卫生与最小修复

目标：清理会污染判断的实验和代码入口。

- 修正 strict concat-sat launcher 的互斥参数冲突。
- 给 sat/concat branches 增加 dry-run 或 unit test，确认实际走的是预期 branch。
- 把 `FSDG50`、SA01-SA04、concat-sat 等证据明确标注为“有效结果”或“失败/不可解释结果”。
- 记录 FedProx ratio；FedProx 在当前阶段只作为 baseline optimizer。
- 每个 N607 新实验都按项目规则写本地 report，记录 local diff/hash、sync mapping、远端命令、日志、PID、GPU、conda env。
- 对任何会形成论文结论的比较预先设定 seed 列表和统计口径。

停止条件：如果 launcher 仍然不能无歧义地证明使用了目标增强路径，不进入新的卫星增强结论。

### Phase 0：冻结当前可信 baseline

目标：建立后续比较锚点。

可信锚点建议：

- CE-only receiver-day FL：约 71 strict UDU。
- receiver-client + receiver-agnostic BEX02 + CVS sat consistency：`FSDG49`，75.92 strict UDU。
- direct receiver-day BEX02 FL：约 69.6-69.9 strict UDU，作为“盲目 DG 失败”反例。
- centralized clean/sat tradeoff：clean strict 高，sat strict 低；强 sat view 提升 sat 但伤 clean。

停止条件：如果复查发现这些 run 配置不一致或日志缺失，先补 report/summary，再进入新实验。

统计要求：单 run 只作为方向性证据；进入论文级结论前至少补齐多 seed 或说明为什么某些历史 run 只能作为 pilot。

### Phase 1：StyleBank V0，只诊断不改变训练

目标：验证 StyleBank 是否有信息量、是否安全、是否可通信。

实现最小模块：

- `code/federated/style_packet.py`
- `code/federated/rf_style_extractor.py`
- `code/federated/style_bank.py`

第一版 packet 只收集：

- class-marginalized RCN stats
- spectrum coarse stats
- shallow feature mean/std
- sample count、client key、round、EMA age

不做：

- 不做 DG loss
- 不做 Proto fusion
- 不做 few-shot
- 不改变训练样本

通过门槛：

- style clusters 与 receiver/day/sat 场景有可解释相关性。
- TX/class leakage probe 不显著。
- packet bytes 和 bank size 可控。
- no-op 模式与 baseline 指标一致。

### Phase 2：StyleBank V1，style-conditioned augmentation 但不启用 DG loss

目标：先证明远端风格能比随机物理增强更有效。

实现最小模块：

- `code/federated/virtual_domain_sampler.py`
- `code/federated/conditioned_receiver_dg.py`

训练只加入：

```text
x_remote_style
+x_style_anchored_phys
CE / clean-anchored consistency
no GRL/Fishr/SupCon yet
```

关键对照：

- baseline CE/FSDG49-style control
- random physical DR
- style-conditioned physical DR
- local-stat/oracle-ish diagnostic upper bound

通过门槛：

- style-conditioned > random physical。
- clean strict drop 不超过 1-2 pp。
- strict UDU 或 satellite strict 至少一个核心指标提升。
- style coverage/MMD 指标改善。

### Phase 3：打开 DG loss，但只对 `d_style`

目标：让 DG loss 回到它成立的统计前提。

实现要求：

- batch 同时保留 `d_raw` 与 `d_style`。
- `d_raw` 用于日志、split/eval、client analysis。
- `d_style` 用于 GRL、Fishr、MixStyle、same-TX consistency、GroupDRO。
- 如果 `num_style_domains < min_domains`，自动禁用 DG loss。

通过门槛：

- loss logs 显示 Fishr pair 数量非零且稳定。
- GRL/domain accuracy 不坍缩成单类。
- 开启 DG 后优于 Phase 2，而不是只增加训练复杂度。

### Phase 4：ProtoEvidenceBank 与 conservative fusion

目标：把多 prototype 作为跨 client 身份证据，不替代 base classifier。

实现最小模块：

- `code/federated/proto_evidence_bank.py`
- `code/federated/reliability_fusion.py`

先做 eval-only：

```text
p_final = (1 - rho) * p_base + rho * p_proto
rho <= 0.05 at start
```

必须记录：

- net gain
- rescue count
- harm count
- ECE/NLL
- per-class/per-style harm
- prototype reliability distribution

通过门槛：

- rescue 明显大于 harm。
- no-regression class 数量可接受。
- clean/sat/new-style split 上没有系统性伤害。

### Phase 5：Few-shot adaptation

目标：处理新接收机、新卫星链路、新 TX 类别，不提前替代主训练。

实现方式：

- frozen base + calibrator/prototype adapter
- new-domain calibration 和 new-class support set 分开评估
- 继承 Phase 4 的 harm/rescue 约束

通过门槛：

- few-shot 提升来自 support set 适配，而不是泄漏测试域。
- `K=1/5/10` 分别报告。
- 不损害 no-support baseline。

## 5. 最先应实施的代码任务

优先级从高到低：

1. 修复 strict concat-sat launcher 的互斥参数问题，并补一个 dry-run/test 验证 branch。
2. 在 federated trainer 中引入 `d_style` 数据结构和 no-op plumbing，先不改变训练。
3. 实现 StylePacket/StyleBank V0 diagnostics，记录通信成本和 privacy probe 输入。
4. 给 MixStyle/Fishr/GRL/SupCon 增加只接收 `d_style` 的单测。
5. 实现 style-conditioned sampler 的 random-vs-conditioned 对照。
6. 给每个 N607 策略实验建立本地 report 和 sync/hash 记录。
7. 最后再实现 ProtoEvidenceBank eval-only fusion。

## 6. 事实置信结论

我现在不接受原始策略的“直接实施版”。它有太多会把失败归因混在一起的漏洞。

我接受修正版 V2，原因是它把所有当前能识别的漏洞都变成了可测试的门槛：

- 先修实验卫生，再引用结果。
- 先诊断 StyleBank，再让它改变训练。
- 先证明 style-conditioned augmentation 胜过 random，再启用 DG。
- DG 只吃 `d_style`，不吃 raw receiver/day。
- ProtoBank 先做保守 eval-only fusion，不替代 base classifier。
- few-shot 放在最后，避免掩盖主训练问题。

这不是保证一定涨点的策略，而是当前最不容易自欺、最容易定位失败原因、也最符合现有 CVS-RFFI 主体的策略。

## 7. 2026-05-26 实现后再审计

Phase -1/Phase 1 代码落地后，又做了一轮 loophole loop。新增发现与修复如下：

| 新漏洞 | 风险 | 修复 | 验证 |
|---|---|---|---|
| `d_style` 直接使用远端 `style_id` | StyleBank 的 `style_id` 可能大于模型 domain head 输出维度，导致 CE 越界或错误监督。 | `VirtualDomainSampler` 改为构造紧凑域标签：clean=0，每个虚拟 view 从 1 开始；原始 style id 放入 `metadata.raw_style_ids`。 | 本地与 N607 `test_virtual_domain_sampler_builds_d_style_without_mutating_raw_domain` 通过。 |
| domain head 维度小于构造域数 | 即使 `d_style` 紧凑，某些模型配置仍可能无法表示所有构造域。 | `FederatedTrainer._domain_logits_and_targets` 在 domain CE 前检查标签范围；超出 head 维度时跳过 CE 类 domain loss，保留 Fishr/consistency 等不依赖 head 维度的项。 | 本地与 N607 `test_domain_losses_skip_when_style_domains_exceed_head_dimension` 通过。 |
| StyleBank packet 统计键漂移 | 有些 packet 有 feature stats，有些没有，向量长度/语义可能不一致。 | `FederatedStyleBank` 维护稳定 numeric stat key schema；新 key 出现时重编码已有 centroid。 | 本地与 N607 `test_style_bank_handles_schema_drift_and_keeps_recent_centroids` 通过。 |
| StyleBank trim 可能偏向旧 centroid | 原排序存在保留旧 centroid 的风险。 | trim 改为优先保留高 count、低 age、较新 round 的 centroid。 | 同上测试覆盖 old centroid 被淘汰。 |
| 本地 launcher 测试可能卡住 | Windows 本地 bash/WSL 行为异常时，测试可能超时而不是报告可解释状态。 | launcher subprocess 测试增加 30s timeout；本地可 skip，N607 必须通过。 | 本地 16 项中 15 passed/1 skipped；N607 16/16 passed。 |

实现后当前事实边界：

- 对“代码策略是否仍有已知逻辑漏洞”：在本轮静态审计与单测覆盖范围内，未发现剩余未修复漏洞。
- 对“实验指标是否一定提升”：仍不能 100% 保证，必须进入 Phase 1 no-op run、Phase 2 random-vs-style-conditioned、Phase 3 `d_style` DG、Phase 4 Proto fusion 消融后才能判断。
- 对“是否可以进入下一步实验设计”：可以，但每个 N607 实验仍必须按项目规则写本地 report、记录 local-to-remote mapping/hash、命令、PID、GPU、日志、预期文件和 stop criteria。
