# 面向天基 RFFI 的 FL-DG-FSL-Prototype 统筹方案

更新时间：2026-05-25
操作者：Codex
工作区：E:/type10-7

## 1. 一句话结论

这四个方向不要并列堆叠，而要分层：

```text
Federated learning = 系统与隐私框架
StyleBank/物理虚拟域 = 让联邦本地重新具备多域条件
Domain generalization = 在多风格本地 batch 上学习跨 rx/day/sat 不变身份特征
Few-shot adaptation = 新地面站/新接收机/新星地链路上线后的轻量校准
Multi-prototype bank = 跨客户端身份证据与保守推理校正器
```

当前最值得推进的主线是：

```text
Fed-PVS-RFFI:
  Federated Physics-guided Virtual Style-bank RFFI

核心机制：
  客户端上传低维 RF style/prototype 统计，而不是 IQ；
  服务端维护 StyleBank + ProtoBank；
  客户端用远端真实风格和物理扰动重构多域 batch；
  DG 损失只在多风格 batch 条件成立后启用；
  原型只做可靠性加权的证据融合和 hard-sample rescue。
```

## 2. 为什么不能简单相加

联邦学习、域泛化、小样本、多原型头各自都合理，但直接叠加会互相伤害。

1. receiver/receiver-day 作为客户端时，本地 batch 往往只有一个真实域。GRL、Fishr、same-TX consistency、MixStyle cross-domain、GroupDRO 都依赖多域比较条件，直接搬进单域客户端会退化。
2. Few-shot 和 prototype 很容易变成源域过拟合器。过去 FJMP 的经验显示，prototype 分支短期能 rescue，长期容易 harm，尤其是 CE 直接打在 fused logits 上时。
3. 星地增强存在 clean-vs-sat tradeoff。强监督 satellite view 能提高 satellite strict UDU，但会牺牲 clean strict UDU，需要门控、阶段调度和评估指标共同约束。
4. FedProx 只能约束 client drift，不会凭空创造跨域信息。N607 结果中 FedProx 在 local-epoch-2 下 proximal 项很小，和 FedAvg 几乎没有实质差异。

所以统筹原则是：先修复“联邦本地缺多域”的结构问题，再使用 DG；先有稳定 base，再引入 prototype/few-shot 校正。

## 3. 已有证据

### 3.1 本地材料

- `C:/Users/lh594/Downloads/fed_pvs_rffi_research_plan.md` 已提出 RF StyleBank、style-anchored physical virtual domains、Identity ProtoBank、fingerprint semantic preservation。
- `E:/type10-7/docs/federated_collaborative_prototype_fusion_report.md` 已明确：prototype 不应作为替代 base classifier 的强分类头，而应作为多客户端证据、校正器、可拒绝的辅助来源。
- 当前代码具备落地基础：`code/train.py` 支持 FedAvg/FedProx、receiver/receiver_day 客户端、BEX02 local objective、receiver-agnostic BEX02、satellite eval/train；`code/federated/fed_trainer.py` 已有 FedProto stats；`code/sat_channel.py` 支持 LEO/MEO/GEO、天气、多径、Doppler/CFO/IQ imbalance；`code/FJMP` 已有多原型、rho/gate、harm/rescue 诊断。

### 3.2 N607 实验结果

远端路径：`/home/szu2070436088/2510044040/CV-SincNet`

关键结果：

| 组别 | 配置 | 结论 |
|---|---|---|
| `FSDG12/13/14` | 10% labeled, receiver-day FedAvg/FedProx, CE | strict UDU 约 71%；FedProx 与 FedAvg 差异很小 |
| `FSDG18/19/1A/1B` | receiver-day FL 中直接放 BEX02 DG | strict UDU 约 69.6-69.9%，低于 CE-only FL |
| `FSDG49` | receiver-client FedProx + receiver-agnostic BEX02 + CVS sat consistency | overall 80.30%，strict UDU 75.92%，是已查到的最佳 FL 结果 |
| `FSDG50` | 同上，但 baseline-style supervised sat view | strict UDU 70.52%，说明强 satellite view 在 FL 中可能压坏 identity |
| B3b/CVS-RFFI central | 强 DG 中心化训练 | clean strict UDU 约 86-87%，但 satellite strict UDU 多在 40-46% |
| `SA02/SA04` approximate strong sat view | 强 satellite supervision | satellite strict UDU 可到约 48-50%，但 clean strict UDU 降到约 83-84% |
| `fed_proto_smoke` | FedProto stats smoke | 只证明 prototype 统计链路可跑，不能说明性能 |

最重要的实验含义：`FSDG49` 说明“receiver 粒度 + receiver adversarial + 稳健 sat consistency”比“receiver-day 单域里强塞集中式 DG”更靠谱。

## 4. 外部文献支撑

- FedAvg：McMahan 等提出在不集中原始数据的情况下做 iterative model averaging，是 FL 基线。
- FedProx：Li 等针对统计/系统异质性加入 proximal 项，但它解决的是 drift 稳定性，不是跨域信息缺失。
- Fishr：Rame 等通过匹配域级梯度方差做 DG，前提是训练中存在多个源域。
- MixStyle：Zhou 等通过混合 instance-level feature statistics 隐式合成新域，也依赖跨域 style 的可用性。
- Receiver-agnostic RFFI / RIEI / FedRIEI：RFFI 的 receiver effect 是真实问题，adversarial/disentangled receiver suppression 是合理方向。
- CCST / FedGCA / FedKA：FedDG 文献已经承认 FL 中传统 DG 因无法集中多域数据而失效，并转向 cross-client style transfer、single-source style augmentation、feature distribution matching。
- FedProto：客户端原型通信可以缓解异质性，但裸平均 prototype 容易抹平 domain modes；本课题应保留 mixture 和 reliability。
- ProtoNet / MAML / cross-domain few-shot：少样本适配适合用 metric/prototype/adapter 思路，但 domain shift 下必须引入特征扰动或风格扰动，否则会过拟合 support 域。

参考链接：

- FedAvg: https://arxiv.org/abs/1602.05629
- FedProx: https://arxiv.org/abs/1812.06127
- Fishr: https://arxiv.org/abs/2109.02934
- MixStyle: https://arxiv.org/abs/2104.02008
- RA-Collab RFFI: https://arxiv.org/abs/2207.02999
- RIEI/FedRIEI: https://arxiv.org/abs/2411.03636
- Receiver-agnostic RFFI via FL: https://openresearch-repository.anu.edu.au/entities/publication/53156d23-889a-41cf-a830-97ec565a32e9
- Channel-robust receiver-independent RFFI: https://arxiv.org/abs/2512.12070
- FedProto: https://ojs.aaai.org/index.php/AAAI/article/view/20819
- CCST FedDG: https://arxiv.org/abs/2210.00912
- FedGCA single-source FedDG: https://arxiv.org/abs/2409.14671
- FedKA: https://arxiv.org/abs/2203.11635
- Prototypical Networks: https://arxiv.org/abs/1703.05175
- MAML: https://arxiv.org/abs/1703.03400
- Cross-domain few-shot feature-wise transformation: https://arxiv.org/abs/2001.08735

## 5. 推荐总体架构

### 5.1 联邦层：先选对客户端粒度

优先主线：

```text
client = receiver
local objective = receiver_agnostic_bex02
optimizer = FedProx only when local epochs/client drift 足够大
eval = day/rx/sat strict OOD
```

原因：当前 `receiver_day` 太碎，10% few-shot 下每个客户端数据少、本地单域严重，BEX02 DG 直接放进去反而变弱。`receiver` 客户端至少能保留同一接收机下跨日期变化，更适合作为第一阶段。

第二阶段再引入：

```text
client = receiver_day 或 receiver_channel
但必须配套 StyleBank，让每个本地客户端看到远端风格。
```

### 5.2 StyleBank 层：让 DG 条件重新成立

客户端上传的不是样本，而是风格统计包：

```text
S_c = {
  rcn_stats,
  spectral mean/std,
  feature mean/std from z_dom or early features,
  estimated CFO/SRO/IQ imbalance/AGC/noise/multipath ranges,
  satellite meta distribution if available,
  reliability: n, val margin, entropy, domain drift
}
```

服务端维护：

```text
StyleBank = clustered mixture of client/domain styles
```

客户端本地训练时构造：

```text
x_clean
x_remote_style = T_style(x_clean; sampled remote style)
x_phys = T_phy(x_clean; remote style + small physical jitter)
x_sat = T_sat(x_clean; sampled satellite scenario/style)
```

然后再启用：

```text
GRL / Fishr / same-TX SupCon / MixStyle / group CE
```

关键点：DG 损失的输入不再是单域本地 batch，而是 StyleBank 重建后的多风格 batch。

### 5.3 DG 层：分阶段，而不是一开始拉满

推荐训练日程：

1. Round 0-30：CE + receiver-agnostic adversarial + weak sat consistency，先稳定 TX identity。
2. Round 31-90：引入 StyleBank virtual domains，开启 same-TX consistency 和轻量 MixStyle。
3. Round 91-150：开启 Fishr/group CE，但要求 `min_domains >= 3/4` 且每个虚拟域有足够样本。
4. Round 151+：降低 DG 权重，转向 calibration、worst-domain selection、SWA/SWAD 或 best-primary checkpoint。

不建议：

```text
receiver_day client + local single-domain + BEX02 DG 全权重
```

N607 已经显示这个方向不优。

### 5.4 ProtoBank 层：从“分类头”改成“协作证据库”

客户端上传：

```text
P_i,c,k
n_i,c,k
margin_i,c,k
entropy_i,c,k
client_val_acc
drift_i
sat/clean reliability
```

服务端不要粗暴平均成一个 `P_global,c`，而是：

```text
Bank_c = {P_i,c,k, reliability_i,c,k}
```

推理：

```text
p_base = softmax(global_logits)
p_proto = reliability_weighted_vote(z, Bank)
p_final = (1-rho) * p_base + rho * p_proto
```

约束：

- `rho` 初期 0.02/0.05/0.10 小步扫描；
- easy/high-confidence base 样本要强保护；
- boundary/low-margin 样本才允许 prototype 介入；
- 每次报告 harm/rescue/net_gain；
- 不把 CE 直接压在 fused logits 上作为主目标。

### 5.5 Few-shot 层：只做上线后轻量适配

few-shot 不应和全局训练早期搅在一起。推荐部署阶段：

```text
新 receiver / 新地面站 / 新 satellite pass
K-shot support per TX
冻结 backbone 主体
只更新：
  prototype reliability
  small adapter / BN affine
  gate/rho calibration
  optional last-layer temperature
```

任务构造：

```text
episode = leave-one-receiver/day/sat-scenario-out
support = K packets per TX on new domain
query = remaining packets + satellite perturbed variants
```

目标：

```text
min CE(query) + prototype pull + clean/sat assignment consistency
max preserve base margin on easy samples
```

## 6. 与现有 CVS-RFFI 的有机结合

最自然的落点不是重写模型，而是在现有模块外加三层统计机制：

| 现有能力 | 建议扩展 |
|---|---|
| `train.py` FedAvg/FedProx/receiver_agnostic_bex02 | 增加 StyleBank-fed local objective |
| `federated/fed_trainer.py` FedProto stats | 从单均值 proto stats 扩展为 client/class/K mixture + reliability |
| `sat_channel.py` | 把 scenario sampler 改成 StyleBank/meta-conditioned sampler |
| `DataAugmentation.py` / RCN stats | 提取 RF style packet，驱动物理虚拟域 |
| `FJMP` safe residual prototype | 借鉴 harm/rescue/rho/gate，但在 FL 中转为 inference evidence fusion |
| `training_test_eval.py` / satellite eval | 固定 clean strict UDU + sat mean/worst + worst rx/day/sat 指标 |

建议新增概念模块：

```text
code/federated/style_packet.py
code/federated/style_bank.py
code/federated/virtual_domain_sampler.py
code/federated/proto_evidence_bank.py
code/federated/reliability_fusion.py
```

先实现最小闭环，不急于做复杂生成器。

## 7. 推荐实验路线

### Phase A：把已有实验补齐和清洁化

1. 修复 strict concat-sat launcher 的互斥参数问题。
2. 运行 `FSDG51_fedprox_receiver_proto_stats`，因为脚本里已经有 prototype-stat 组，但 N607 目前只有 smoke。
3. 补 central 10% refs：`FSDG02-FSDG07` 若未完整跑完，用来判断 FL gap 的真实上界。

### Phase B：验证“客户端粒度”命题

矩阵：

```text
receiver vs receiver_day
FedAvg vs FedProx
CE vs receiver_agnostic_bex02
with/without CVS sat consistency
```

主指标：

```text
strict_udu
seen_day_unseen_rx
unseen_day_seen_rx
sat_mixed_orbit strict UDU
worst receiver strict UDU
```

预期：receiver-client RA-BEX02 继续优于 receiver-day BEX02。

### Phase C：最小 StyleBank

先不做神经 style transfer，做统计/物理参数级 StyleBank：

```text
server stores client style packets
client samples remote packet
local applies parameterized RF augment + sat augment
labels unchanged
domain labels become virtual domain ids
```

对比：

```text
FSDG49 baseline
FSDG49 + random physical aug
FSDG49 + StyleBank-conditioned physical aug
FSDG49 + StyleBank + Fishr/same-TX consistency
```

成功条件：strict UDU 上升，同时 satellite worst 不下降超过 1-2 个点。

### Phase D：ProtoBank 作为推理证据

对比：

```text
base only
global averaged prototype
client mixture prototype
reliability-weighted client mixture
reliability + rho/gate
```

必须报告：

```text
base_acc
proto_acc
fused_acc
harm_rate
rescue_rate
net_gain
rho_mean/p95
per-rx/per-day/per-sat worst
```

成功条件：net_gain > 0，harm_rate 可控，worst-domain 不恶化。

### Phase E：Few-shot new-domain adaptation

任务：

```text
holdout rx 7/8/9/10/11
holdout day 2/3
holdout satellite scenarios: low_elev/rain/storm/mixed
K = 1, 5, 10, 20 packets per TX
```

对比：

```text
no adaptation
prototype-only update
temperature/gate calibration
adapter/BN-only update
full fine-tune small LR
```

成功条件：K-shot 后 strict UDU 和 satellite worst 同时提升，且不破坏 old-domain clean accuracy。

## 8. 风险控制

- Style packets 可能泄露客户端统计特征：需要类别边缘化、聚合、噪声或 top-level clustering，避免上传 per-sample/per-class 细粒度敏感信息。
- Virtual domain 可能偏离真实物理：必须用真实远端 style anchor，不能只扩大随机增强范围。
- Prototype 可能产生 harm：必须默认 base-anchor、小 rho、reliability gate、harm/rescue 诊断。
- Satellite supervision 可能牺牲 clean：每个实验都要同时报告 clean strict UDU 和 sat mean/worst，不能只看 sat。
- FedProx 可能名义存在但实际无效：每轮记录 `prox/cls` 或 fedprox_ratio，确认 proximal 项是否真的起作用。

## 9. 推荐论文叙事

题目可以围绕：

```text
Federated Physics-Guided Virtual Style Banks for Spaceborne Radio Frequency Fingerprint Identification
```

核心创新点：

1. 指出 FL-RFFI 中“客户端单域”会破坏传统 DG 损失前提。
2. 提出不上传 IQ 的 RF StyleBank，使客户端本地重构跨 receiver/day/satellite 的多域训练条件。
3. 提出 StyleBank + ProtoBank 双统计机制：前者修复 DG 条件，后者稳定 TX identity。
4. 将 few-shot 新域适配限制在 prototype/gate/adapter 层，避免大模型过拟合新站点。
5. 用 clean strict UDU、satellite worst、harm/rescue、通信/隐私开销构成严谨评估。

## 10. 当前最优下一步

最推荐的下一个工程/实验动作：

```text
1. 修复 concat-sat launcher 互斥参数。
2. 跑 FSDG51：receiver-client RA-BEX02 + FedProto stats。
3. 实现最小 RF StylePacket/StyleBank，不做生成器，只做统计驱动物理增强。
4. 在 FSDG49 基础上加 StyleBank-conditioned virtual domains。
5. 再逐步打开 Fishr/same-TX consistency/MixStyle，并用 harm/strict/sat worst 控制。
```

这条路线最贴近已有 CVS-RFFI 工作，也最能形成论文里清楚的因果链：不是“把很多方法拼起来”，而是“发现 FL 破坏 DG 前提 -> 用物理风格库修复前提 -> 用原型证据稳定身份 -> 用 few-shot 完成部署适配”。
