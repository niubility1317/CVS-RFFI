# Fed-PVS-CPRFFI 与 CVS-RFFI 主体融合分析

更新时间：2026-05-25  
输入设计：`C:/Users/lh594/Downloads/fed_pvs_cprffi_final_design.md`  
本地代码：`E:/type10-7/code`

## 1. 总判断

最新版设计报告的主线是成立的，而且和当前 CVS-RFFI 代码基础是匹配的。关键不是重写 CVS-RFFI 主干，而是在现有主干外增加一层联邦统计与证据机制：

```text
CVS-RFFI backbone / base classifier
  + Federated StyleBank
  + Style-conditioned multi-view local objective
  + DG losses on constructed d_style
  + reliability-aware ProtoBank
  + inference-only conservative prototype fusion
  + few-shot adapter/calibration
```

最重要的融合原则是：`base classifier` 仍然是主判别器，CVS-RFFI 仍然承担 TX 身份识别的核心表达；StyleBank 修复联邦本地单域问题；DG loss 只在本地 batch 被重构成多风格 batch 后启用；ProtoBank 只作为跨客户端身份支持证据，不直接替代主分类头。

## 2. 现有 CVS-RFFI 能承接什么

| 设计需求 | 当前代码基础 | 结论 |
|---|---|---|
| 联邦训练 | `code/train.py` 已支持 `fedavg/fedprox`；`code/federated/fed_trainer.py` 负责单进程 FL 模拟 | 可直接作为主入口 |
| 客户端粒度 | `client_split.py` 支持 `receiver`、`receiver_day`、`receiver_channel`、`receiver_day_channel` | 满足报告形式化设定 |
| FedProx | `fedprox.py` 与 `FederatedTrainer` 已实现 proximal loss 与日志 | 可保留为 drift 稳定项 |
| FedProto baseline | `fed_trainer.py` 已收集 class sum/count 并生成 class prototype | 可作为 ProtoBank 的起点，但目前只是单均值 |
| DG loss | `train.py` 和 `fed_trainer.py` 已有 GRL、Fishr、same-TX consistency、group CE、MixStyle | 需要把输入域标签从 raw receiver/day 改成构造出的 `d_style` |
| 模型分支 | `model_dual_cvsincnet.py` 已有 `z_id`、`z_dom`、RCNStatEncoder、GRL head、MixStyle hooks | 与报告的 `z_id/z_dom/z_hw` 语义一致 |
| 物理增强 | `DataAugmentation.py` 已有 CFO/SRO/IQ imbalance/multipath/receiver-chain DR | 缺少从 StyleBank 条件化采样 |
| 星地链路 | `sat_channel.py` 和 `concat_sat_channel_aug.py` 已有 sat eval、sat consistency、strict concat-sat | 可扩展成 Satellite StyleBank |
| 安全原型融合 | `code/FJMP` 和 `code/SGC/v3` 有多原型、gate、harm/rescue 诊断 | 可复用诊断思想，但不要照搬成强训练头 |
| 评估 | `training_test_eval.py`、`FederatedTrainer._evaluate`、sat eval 已覆盖 named split 和 strict UDU | 需要补 harm/rescue、style coverage、通信/隐私指标 |

当前主体不是缺模型，而是缺“跨客户端统计状态”：StyleBank、ProtoBank mixture、style-conditioned sampler、fusion evaluation。

## 3. 当前最大断点

### 3.1 联邦本地目标仍是 single-view first

`FederatedTrainer._compute_local_objective` 目前先构造一个 `x_main`，再可选加 satellite consistency 或 baseline satellite view。它没有报告要求的：

```text
x_local
x_remote_style
x_style_anchored_phys
x_sat_style
d_style = local / remote_style / remote_phys / sat_style
```

因此当前 `bex02_dg` 或 `receiver_agnostic_bex02` 在 `receiver_day` 客户端下仍容易遇到报告指出的结构性问题：本地真实域太单一，Fishr/GRL/SupCon 的跨域统计条件不足。

### 3.2 FedProto 仍是单原型平均

当前 `global_proto_stats` 的核心是：

```text
class_sum / class_count -> class_proto
```

这适合作为 FedProto baseline，但和报告中的 ProtoBank 不同。报告要求每类保留多客户端、多风格、多模式原型，并附带：

```text
count, margin, entropy, intra_var, client_drift, clean_sat_kl, style_id, age, reliability
```

所以不能直接把现有 FedProto 当作最终协作多原型分类头，只能作为 F2 baseline 或第一版 prototype pull。

### 3.3 现有 DG 标签是 raw domain，不是 constructed style domain

当前 `d_raw` 来自 WiSig 的 day/rx/rx_day，集中式训练时合理；但在 FL 客户端内，当 client = receiver 或 receiver_day 时，`d_raw` 往往不足以形成多域比较。报告要求把真实域和构造域分开：

```text
d_raw   -> 日志、真实 split、隐私诊断、client identity
d_style -> GRL/Fishr/SupCon/group CE
```

这应成为第一轮实现的硬边界。

### 3.4 FJMP/SGC 不能原样成为 Fed-PVS-CPRFFI 的原型头

FJMP 已经有 `rho/gate/safe_logits/harm/rescue`，SGC v3 也有 safe adapter 与 prototype distance，但它们更像 post-stage 或 frozen-base 模块。Fed-PVS-CPRFFI 要的是跨客户端 ProtoBank evidence：

```text
p_final = (1-rho) * p_base + rho * p_proto
```

并且 `rho_max` 从 0.02/0.05 小步扫。FJMP 里的可学习 residual/logit fusion 只能借鉴诊断与安全门控，不能直接改成主训练 CE 目标。

## 4. 有机融合后的架构

### 4.1 保留 CVS-RFFI 主干作为 anchor

第一原则：不动或少动 `model_dual_cvsincnet.py` 的主结构。它已经有：

```text
id_backbone -> z_id -> tx_logits
dom_backbone / RCNStatEncoder -> z_dom
adv_head / dom_head
feat_dac / feat_pa / feat_joint
MixStyle hooks: time_down, t1
```

Fed-PVS-CPRFFI 应该把它作为稳定主干，新增模块围绕 `FederatedTrainer` 和 `DataAugmentation/sat_channel` 工作。这样最小化风险，也能复用已验证的 BEX02/receiver-agnostic BEX02 配置。

### 4.2 StyleBank 作为联邦服务器状态

建议新增：

```text
code/federated/style_packet.py
code/federated/rf_style_extractor.py
code/federated/style_bank.py
code/federated/virtual_domain_sampler.py
code/federated/conditioned_receiver_dg.py
```

第一版 Style packet 不要过大，只做：

```text
RCN stats
spectrum stats
shallow feature mean/std
count/reliability/privacy metadata
```

落点：

```text
train_one_client()
  -> batch forward or no-grad hook collects style stats
  -> return "fed_style_stats"

FederatedTrainer.train()
  -> update self.global_style_bank after model aggregation
  -> next round broadcast/sample styles
```

### 4.3 Multi-style local objective 替换 single-view objective

在 `FederatedTrainer._compute_local_objective` 内新增受开关控制的路径：

```text
if use_fed_style_bank and round >= style_replay_start_round:
    build x_all, y_all, d_style
else:
    keep current x_main path
```

构造顺序建议：

```text
Stage 0: x_local only
Stage 1: x_local + x_remote_feature_style
Stage 2: + x_style_anchored_phys
Stage 3: + DG losses on d_style
Stage 4: proto fusion only in eval
```

重要：`d_raw` 保留给 split/eval/client，`d_style` 只给 DG losses 和 MixStyle pairing。

### 4.4 DG recovery 复用现有损失，但改变启用条件

现有 GRL/Fishr/SupCon 不需要重写。需要做的是 gate：

```text
num_style_domains_per_batch >= 3
same_tx_cross_style_pair_count > 0
fishr_nonzero_ratio > 0
per-style-domain min samples satisfied
```

报告里的“直接 DG 在 receiver_day FL 中退化”已经和现有 N607 结果一致。因此不要把 `receiver_day + bex02_dg + full Fishr` 作为最终路线；它应作为反例/诊断组。

### 4.5 ProtoBank 从统计链路升级，不先做强融合

建议新增：

```text
code/federated/proto_evidence_bank.py
code/federated/reliability_fusion.py
code/eval_proto_fusion.py
```

升级路径：

```text
F2: 当前 FedProto 单均值 pull
F8: reliability-aware multi-prototype regularization
F9: frozen/base-only checkpoint 上做 inference-only fusion sweep
```

第一版不要训练 learnable fusion，不要对 fused logits 做主 CE。只报告：

```text
base_acc
proto_acc
fused_acc
changed_pred_rate
rescue_rate
harm_rate
net_gain
rho_mean / rho_p95
worst-domain change
```

如果 `net_gain <= 0`，prototype 只保留为 representation regularization，不进入最终推理。

### 4.6 Few-shot 放在最后，不要早期混入主训练

现有 `train_target_adapt.py`、SGC target adaptation、FJMP frozen-base 训练经验可以服务 few-shot，但应作为 Stage 5：

```text
freeze backbone
support set extracts new style packet
update adapter / BN affine / temperature / gate / prototype reliability
query evaluates strict UDU and old-domain clean drop
```

新接收机/新星地链路和新 TX 是两类问题：

```text
new domain, old classes -> adapter/calibration/prototype reliability
new TX, new classes     -> prototype-only new-class evidence + old/new calibration
```

## 5. 第一轮落地建议

### 5.1 不建议第一轮做的事

不要第一轮就做：

```text
完整 phy parameter estimator
复杂生成器
learnable prototype fusion
new TX few-shot
强 satellite CE
receiver_day + full DG 主实验
```

这些都会把风险叠加，难以解释失败原因。

### 5.2 第一轮应做的最小闭环

建议第一轮只做 “StyleBank V1 + style-conditioned physical 对照”：

```text
1. RFStyleExtractor: RCN/spec/shallow feature stats
2. FederatedStyleBank: merge/sample/broadcast
3. virtual_domain_sampler: local/style/phys view construction
4. conditioned_receiver_dg: 从 style bucket 映射到 apply_receiver_dg 参数
5. FederatedTrainer: 收集 fed_style_stats，下一轮采样远端 style
6. logging: style_bank_size, accept_rate, num_style_domains_per_batch
7. experiments: random physical vs style-conditioned physical
```

第一轮验收只看：

```text
训练不崩
style_bank_size > 0
客户端能采样非本地 style
num_style_domains_per_batch >= 2
clean strict UDU 不明显下降
style-conditioned physical >= random physical
```

### 5.3 第一轮代码文件映射

| 文件 | 动作 |
|---|---|
| `code/federated/style_packet.py` | 定义 style packet dataclass / tensor pack/unpack |
| `code/federated/rf_style_extractor.py` | 提取 RCN/spec/shallow feature stats |
| `code/federated/style_bank.py` | merge、EMA、reliability filter、sample |
| `code/federated/virtual_domain_sampler.py` | 构造 `x_all/y_all/d_style` |
| `code/federated/conditioned_receiver_dg.py` | style -> receiver-chain augmentation 参数 |
| `code/federated/fed_trainer.py` | 新增 `global_style_bank`、收集/更新/采样、multi-view objective |
| `code/train.py` | 新增 CLI flags，并向 `FederatedTrainer` 传参 |
| `code/tests/` | 加 style bank merge/sample、multi-view labels、no-op fallback、dry-run integration tests |

### 5.4 建议新增 CLI

```bash
--use_fed_style_bank
--style_bank_type rcn_spec
--fed_style_momentum 0.8
--fed_style_min_count 32
--style_bank_k 8
--style_num_remote 2
--style_replay_start_round 20
--style_phys_start_round 40
--style_dg_loss_start_round 80
--style_jitter_scale 0.10
--lambda_style_cls 0.3
--lambda_style_cons 0.03
```

DG 相关现有参数可以继续使用：

```bash
--lambda_adv
--lambda_fishr
--lambda_supcon_id
--fishr_min_domains
--min_batch_domains_for_domain_loss
```

但它们应绑定 `d_style`，不是绑定本地单域 `d_raw`。

## 6. 实验路线

优先把现有 FSDG 结果组织成因果链，而不是再铺大矩阵。

### 6.1 现有实验应作为背景证据

当前已有证据支持：

```text
receiver-client + receiver-agnostic BEX02 + mild sat consistency
>
receiver_day + direct DG
```

这正好支撑论文叙事：联邦本地单域会让直接 DG 退化，必须用 StyleBank 修复多域条件。

### 6.2 第一组新实验

建议新建 `FED_PVS_STYLEBANK_V1` 队列：

```text
PVS00: FedProx receiver CE + FedProto single-proto
PVS01: PVS00 + random receiver-chain physical aug
PVS02: PVS00 + StyleBank feature replay
PVS03: PVS00 + RCN/spec StyleBank replay
PVS04: PVS03 + style-conditioned physical aug
PVS05: PVS04 + style identity consistency
PVS06: PVS04 + DG losses after warmup
PVS07: PVS06 + FedProto pull
```

核心结论目标：

```text
PVS04 > PVS01
PVS06 > PVS04 only if DG diagnostics are valid
```

如果 `PVS06 < PVS04`，不要硬说 DG 有效；回到 `d_style`、pair count、Fishr nonzero ratio、style strength 检查。

### 6.3 Prototype 单独做第二组

在第一组有稳定 checkpoint 后，再做：

```text
PROTO00: base only
PROTO01: averaged class prototype
PROTO02: client mixture prototype
PROTO03: reliability-weighted mixture
PROTO04: reliability + rho gate
```

这组不重新训练主干，只做 inference/eval sweep，能最快判断 ProtoBank 是 rescue 还是 harm。

## 7. 风险与防线

### 风险 1：StyleBank 不如随机增强

防线：

```text
style coverage MMD/Wasserstein
jitter_scale 0.05/0.10/0.20
style packet receiver/day separability probe
random physical 对照必须保留
```

### 风险 2：强去域导致去指纹化

防线：

```text
降低 lambda_adv/lambda_fishr
保留 feat_dac/feat_pa 不做强 invariance
TX probe 高、RX/SAT probe 低才算成功
clean strict UDU 不可明显下降
```

### 风险 3：Prototype fusion harm

防线：

```text
rho_max 从 0.02 开始
概率级融合优先
base confident 时 rho=0
proto unreliable 时拒绝介入
必须报告 harm/rescue/net_gain
```

### 风险 4：Satellite tradeoff

防线：

```text
先 weak sat consistency
再 satellite style replay
不要早期强 sat CE
同时报告 clean strict UDU 和 sat worst
```

### 风险 5：FedProx 没有效果

防线：

```text
记录 prox/cls ratio
扫 mu
local_epochs=1 或 prox_loss 很小时，不把 FedProx 当贡献点
```

## 8. 结论

Fed-PVS-CPRFFI 与现有 CVS-RFFI 的最佳结合方式是“保守增量式”：

```text
CVS-RFFI 主干不重写；
FederatedTrainer 成为 StyleBank/ProtoBank 的服务器状态协调器；
DataAugmentation/sat_channel 成为 style-conditioned virtual domain 的执行器；
已有 DG losses 只在 d_style 成立后启用；
FJMP/SGC 的 prototype/fusion 经验只转化为 inference-only evidence fusion 和诊断；
few-shot 放到最后作为部署适配层。
```

下一步如果进入实现，第一轮应只做 StyleBank V1 和 random-vs-style-conditioned physical 对照。只要这条链成立，后续 DG recovery、ProtoBank fusion、few-shot adaptation 才有扎实基础。
