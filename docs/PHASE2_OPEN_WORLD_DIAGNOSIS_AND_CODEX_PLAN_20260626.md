# Phase 2 Open-World RFFI 诊断与 Codex 改进计划

日期：2026-06-26  
范围：当前 `main` 分支根目录代码，不修改 `type10-*`、`5.7/`、`unkown/` 等历史快照目录。  
用途：把 ChatGPT 对 Phase 2 在轨部署、少样本域适应、新类识别、未知类拒识、TX/RX 解耦和 SGC 使用方式的诊断转成 Codex 可直接读取的工程任务清单。

---

## CODEX-READ-ME

本文件是给本地 Codex 的入口文档。优先读取以下文件和函数：

| 优先级 | 文件 | 重点位置 / 符号 | 需要理解的问题 |
|---|---|---|---|
| P0 | `train.py` | `main()`、`configure_sgc_trainable_params()`、`compute_core_losses()`、`PrototypeMemoryBank`、`domain_aware_supcon_loss()`、`fishr_logit_gradient_variance_loss()`、SAT/SGC CLI 参数 | 当前只有 `source/sgc_augment/sgc_adapt`，不是完整 open-world Phase 2；`sgc_adapt` 冻结非 adapter 参数；pseudo-label/entropy 参数多数未真正接入主损失。 |
| P0 | `model_dual_cvsincnet.py` | `DualCVSincNetDisentangle.forward()`、`z_id/z_dom`、`fast_infer_when_no_aux`、SGC adapter 接入点 | SGC 先处理 IQ 后同时送入 ID backbone 和 domain backbone；常规 `return_aux=False` 推理会跳过 domain backbone。 |
| P0 | `model.py` | `CosFaceHead`、`PhysicalAwareClassifier`、`feat_joint/feat_imp/feat_pa/feat_dac` | 当前闭集分类用 angular margin，但没有显式 open-set unknown margin；`feat_imp` 不等于纯 RX 特征。 |
| P0 | `dataset_wisig.py` | `_rms_normalize_iq()`、`WiSigCompactDataset.__getitem__()`、`WiSigIndex`、domain lut | 训练输入已经做 joint IQ RMS normalization；domain label 可为 day/rx/rx_day。 |
| P0 | `sgc_adapter.py` | `AmplitudeNormalizer`、`FrequencyOffsetCompensator`、`SpectralInterferenceSuppressor`、`ResidualChannelCompensator` | full SGC 的 per-channel RMS normalization 可能抹掉 RFFI 有用的幅度/IQ imbalance 信息；residual comp 内部用 BatchNorm，小样本适配有风险。 |
| P1 | `eval_feature_diagnosis.py` | feature extraction、Fisher ratio、NCM、domain centroid metrics | 扩展为角空间 open-world 诊断工具。 |
| P1 | `docs/SGC_5_7_analysis_and_merged_plan_20260507.md` | SGC 5.7 实验结论 | full SGC from source 不是主线；no_amp 是较值得继续看的 SGC 方向；当前 `sgc_adapt` 不应当作最终适配路线。 |
| P1 | `findings.md` | SGC/5.7 evidence notes | 已有日志证据摘要，可作为改动前的事实基线。 |

Codex 搜索关键词：

```text
CODEX-PHASE2-OPENWORLD
CODEX-TXRX-GEOMETRY
CODEX-OPENSET-REJECTION
CODEX-DOMAIN-BACKBONE
CODEX-SGC-NOAMP
```

---

## 一句话结论

当前失败更像是三件事叠加：

```text
1. 地面基模只优化闭集 TX 分类，角空间没有为 few-shot 新类与 unknown 留出足够可校准边界；
2. 当前 TX/RX 解耦是单向的：只让 z_id 去 RX/day，但没有让 z_dom 去 TX，也没有显式建模 TX×RX 交互；
3. 当前 sgc_adapt 只训练很小的 adapter，未真正利用域骨干做目标域识别/校准，也没有完整的新类注册与未知拒识路径。
```

---

## 当前代码诊断

### D1. 当前不是完整 Phase 2 open-world pipeline

当前 `train.py` 的训练阶段只有：

```text
source
sgc_augment
sgc_adapt
```

这三者服务的是 source 训练、卫星信道增强和 SGC adapter 适配；还没有实现真正的：

```text
旧类保持 + 新类 few-shot 注册 + 目标 RX 域校准 + 未知类拒识 + 无标签流自适应
```

现有风险：

- `pseudo_label_threshold`、`lambda_ent` 等参数存在，但多数属于 reserved/未完整接入主损失。
- `sgc_adapt` 会冻结除 SGC adapter 之外的参数；这不是域骨干参与的目标域学习。
- 常规推理 `return_aux=False` 会跳过 domain backbone，因此域骨干默认不参与最终判决。
- 没有 episode-style open-set validation：没有 held-out TX 作为 pseudo-unknown 校准 FPR95/AUROC/OSCR。

### D2. 地面基模问题应按角空间诊断，不要只看 t-SNE/欧氏范围

当前 `CosFaceHead` 已经把分类放到归一化角空间：

```text
logit_k = s * cos(z, w_k)
true-class logit = s * (cos(z, w_y) - m)
```

因此二维可视化中“聚成小空间”不一定代表闭集不可分。真正要诊断：

```text
class angular radius
minimum inter-class angular distance
class safety margin = inter-class angle - radius_i - radius_j
feature effective rank
same-TX cross-RX centroid drift
held-out unknown TX nearest-prototype distance
```

如果类内 95% 角半径之和大于类间角距，Phase 2 新类和 unknown 必然容易被最近旧类吸收。

### D3. 当前 prototype push 可能没有有效梯度

`PrototypeMemoryBank` 中 class/domain prototype 由 EMA memory 更新。当前实现中：

- 当前样本向本类 prototype 的 pull 有梯度；
- prototype-to-prototype push 多数情况下只作用在 memory tensor 上，不能有效反向推开当前 batch 特征；
- domain prototype 与 class prototype 对齐同样主要发生在 memory 内部。

建议新增对当前样本有梯度的 prototype margin loss：

```text
L_proto_margin = mean_i [ margin + max_{j != y_i} sim(u_i, p_j) - sim(u_i, p_yi) ]_+
```

这里 `p_j` 可以 stop-gradient，但 `u_i` 必须参与梯度。

### D4. 当前 TX/RX 解耦是单向的

当前核心逻辑：

```text
z_dom -> classify RX/day domain
GRL(z_id) -> remove RX/day from identity
cov(z_id, z_dom) -> global linear orthogonality
same_tx_cross_domain_consistency(z_id) -> same TX across domains pulled together
```

它主要回答：

```text
同一个 TX 在不同 RX/day 下，z_id 能否靠近？
```

没有显式回答：

```text
同一个 RX 接收不同 TX 时，z_dom 是否靠近？
z_dom 中是否泄漏 TX 身份？
TX 间相对方向是否随 RX 改变？
RX 间相对方向是否随 TX 改变？
TX×RX 交互项被放在哪里？
```

尤其是当前 `z_dom` 默认来自第二骨干的 `feat_imp`，它是 PA/DAC impairment embedding 的融合，不是纯 RX 特征。对于 `no_dac` 主线，`feat_imp` 基本更接近 PA/impairment cue，可能包含 TX 身份。

### D5. 当前 domain loss 对 unseen target RX 不够自然

`wisig_domain=rx_day` 把 domain 设成 RX×day 联合类别。训练外的 target RX/day 会被 remap 成 `-1`，不参与 domain CE。此设计适合 source domain regularization，但不适合直接外推新 RX。

建议把 domain 分解为：

```text
z_rx  : receiver hardware / IQ imbalance / AGC / ADC / LNA/Mixer/filter cues
z_env : day/channel/SNR/Doppler/weather/environment cues
z_int : TX×RX or RX×channel residual interaction
```

然后分别使用 RX head、environment head 和 interaction residual，而不是只用一个 `rx_day` softmax。

### D6. SGC full amp normalization 可能破坏 RFFI 信息

数据集已有 joint complex RMS normalization：

```text
x <- x / sqrt(mean(I^2 + Q^2))
```

但 `SGCAdapter.AmplitudeNormalizer` 又对 I/Q 两个通道分别按时间 RMS 归一化：

```text
I <- I / RMS(I)
Q <- Q / RMS(Q)
```

这会削弱 IQ gain imbalance 等特征。另一方面，domain enhancer 的 RCN stats 正在显式使用 `log(std_I/std_Q)` 等统计量。full SGC 在 domain backbone 前先做 per-channel normalization，可能直接抹掉域分支想使用的关键统计。

这和 5.7 结论一致：后续优先看 `no_amp` SGC，不建议把 full SGC from source 作为主线。

### D7. 小样本适配时 SGC residual 中 BatchNorm 有风险

`ResidualChannelCompensator` 内部使用 `BatchNorm1d`。在轨 few-shot 或小 batch 适配时，BN running statistics 可能不稳定。

建议二选一：

```text
A. 将 SGC residual 内 BatchNorm1d 改成 GroupNorm/LayerNorm；
B. 适配阶段冻结 BN running statistics，仅训练 affine/residual gamma 或 LoRA/adapter 参数。
```

---

## 必须新增的诊断指标

### CODEX-TODO P0.1：扩展 `eval_feature_diagnosis.py`

新增函数：

```python
angular_class_geometry(X, y)
effective_rank(X)
tx_rx_two_way_anova(X, y_tx, y_rx)
rx_probe_on_id_feature(X_id, y_rx)
tx_probe_on_domain_feature(X_dom, y_tx)
open_set_distance_scores(X_known, y_known, X_unknown)
```

输出 JSON 至：

```text
diagnostics/phase2_geometry_<checkpoint>_<split>.json
```

必须包含：

```text
class_radius_mean
class_radius_p95_mean
min_interclass_angle
margin_violation_pairs
feature_effective_rank
same_tx_cross_rx_centroid_cos
train_test_class_centroid_cos
var_tx_ratio_on_z_id
var_rx_ratio_on_z_id
var_interaction_ratio_on_z_id
var_tx_ratio_on_z_dom
var_rx_ratio_on_z_dom
rx_probe_acc_on_z_id
tx_probe_acc_on_z_dom
known_unknown_auroc
known_unknown_fpr95
oscr_auc
```

验收标准：

```text
python eval_feature_diagnosis.py --help
```

能看到新增 phase2/open-set 诊断参数；离线运行后能生成 JSON 与 Markdown 摘要。

### CODEX-TODO P0.2：实现 TX×RX two-way ANOVA

对每个特征空间计算：

```text
mu_tr = mean(feature | TX=t, RX=r)
mu = global mean
a_t = mean_r(mu_tr) - mu
b_r = mean_t(mu_tr) - mu
e_tr = mu_tr - mu - a_t - b_r
```

报告方差占比：

```text
Var_TX / Var_total
Var_RX / Var_total
Var_interaction / Var_total
```

目标预期：

```text
z_id  : Var_TX 高，Var_RX 与 Var_interaction 低
z_rx  : Var_RX 高，Var_TX 低
z_int : 可以保留部分 interaction，但不能直接进入 TX classifier
```

---

## 数据采样与训练结构改进

### CODEX-TODO P1.1：新增 balanced TX×RX sampler

新增文件建议：

```text
balanced_tx_rx_sampler.py
```

目标：每个 batch 构造矩形结构：

```text
N_TX × N_DOMAIN × N_PER_CELL
例如 8 TX × 4 RX/day × 8 packets = batch 256
```

要求：

- 同一 batch 内同一 TX 至少覆盖 2 个 RX/day；
- 同一 RX/day 至少覆盖 2 个 TX；
- 尽量形成 2×2 rectangle：`(tx1,rx1),(tx1,rx2),(tx2,rx1),(tx2,rx2)`；
- 打印并记录有效 pair/rectangle 数量。

训练日志新增：

```text
[BATCH-GEOM] same_tx_cross_domain_pairs=...
[BATCH-GEOM] same_rx_cross_tx_pairs=...
[BATCH-GEOM] tx_rx_rectangles=...
```

### CODEX-TODO P1.2：新增 TX×RX 四角一致性损失

新增函数建议放到 `train.py` 或独立 `tx_rx_geometry.py`：

```python
tx_rx_rectangle_identity_loss(z_id, tx_label, rx_label)
tx_rx_rectangle_receiver_loss(z_rx, tx_label, rx_label)
```

身份差分一致性：

```text
1 - cos(
  z_id(tx1,rx1) - z_id(tx2,rx1),
  z_id(tx1,rx2) - z_id(tx2,rx2)
)
```

接收机差分一致性：

```text
1 - cos(
  z_rx(tx1,rx1) - z_rx(tx1,rx2),
  z_rx(tx2,rx1) - z_rx(tx2,rx2)
)
```

初始权重：

```text
lambda_rectangle_id = 0.02
lambda_rectangle_rx = 0.02
```

如果有效 rectangle 太少，loss 应自动返回 graph-preserving zero，并在日志中记录 coverage，不要 silently fail。

### CODEX-TODO P1.3：新增反向 TX 去泄漏约束

当前只有：

```text
GRL(z_id) -> domain classifier
```

新增：

```text
GRL(z_rx or z_dom) -> TX classifier
```

目标：让 receiver/domain feature 不携带 TX 身份。

建议命名：

```text
lambda_tx_adv_on_dom
```

初始搜索：

```text
0.02, 0.05, 0.10
```

注意：不要一开始大权重，否则可能损坏 domain feature 本身。

---

## Domain backbone 正确使用方式

### CODEX-TODO P2.1：不要把 z_dom 直接拼进 TX classifier

当前建议：

```text
z_dom 不能作为 TX identity evidence 直接参与分类；
它只能生成接收机上下文，用于小幅修正 ID feature 或调节 open-set 阈值。
```

推荐结构：

```text
raw IQ statistics / RCN stats -> domain context encoder -> q_rx
normalized IQ -> no_amp SGC -> ID backbone -> z_id
q_rx -> residual FiLM / adapter -> corrected z_id
```

其中 `q_rx` 应按 target RX 的一段 packet window 做 EMA：

```text
q_rx_t = beta * q_rx_{t-1} + (1-beta) * mean(z_rx_window)
```

`q_rx` 默认 stop-gradient 进入 ID correction adapter，防止 TX loss 反向把 TX 信息写入 receiver context。

### CODEX-TODO P2.2：拆分 domain 表征

在 `model_dual_cvsincnet.py` 中逐步从：

```text
z_dom
```

扩展为：

```text
z_rx
z_env
z_int
```

最小实现可以先不改整体模型，只新增 projection heads：

```python
self.rx_proj = MLPHead(self.emb_dim, rx_emb_dim)
self.env_proj = MLPHead(self.emb_dim, env_emb_dim)
self.int_proj = MLPHead(self.emb_dim, int_emb_dim)
```

训练时：

```text
z_rx  -> RX classifier, TX adversary
z_env -> day/channel classifier, TX adversary
z_int -> weakly regularized interaction buffer，不直接进 TX classifier
```

---

## Phase 2 open-world 训练路线

### Stage A：目标 RX 校准

输入应包含：

```text
目标 RX 上若干旧已知 TX 的少量锚点样本
目标 RX 上新类 few-shot support
目标 RX 上无标签混合流
source prototypes 或少量 replay anchors
```

先冻结：

```text
ID backbone
旧类 CosFace/prototype weights
旧类 source prototypes
```

只训练：

```text
no_amp SGC / residual adapter
domain context encoder
small FiLM / domain-conditioned adapter
```

损失：

```text
old_known_target_proto_align
multi_view_consistency
source_distillation
adapter_delta_regularization
domain_context_loss
```

### Stage B：新类注册

不要立即全网络 fine-tune。优先用 prototype/weight imprinting：

```text
new_class_proto = normalize(weighted_mean(z_id_support_augmented))
classifier_weight_new <- new_class_proto
```

少样本类半径用 shrinkage：

```text
r_new = alpha(K) * r_source_prior + (1-alpha(K)) * r_empirical_new
```

K 越少，越依赖 source prior。

### Stage C：open-world 无标签流适配

无标签样本先分成：

```text
known-old high confidence
known-new high confidence
uncertain / unknown
```

只对 high confidence 样本做 pseudo-label。必须同时满足：

```text
nearest-prototype angular distance under class radius
energy score under known threshold
multi-view prediction agreement
temporal streak agreement
```

unknown / uncertain 样本：

```text
不要做普通 entropy minimization
不要强行分配 max-softmax pseudo label
放入 unknown buffer
用 energy margin / prototype repulsion 维持 unknown space
等待后续聚类或人工注册
```

---

## 未知拒识评分设计

不要只用 max softmax 阈值。

建议组合：

```text
1. prototype angular distance score
2. energy score
3. multi-view inconsistency score
4. domain correction reliability score
```

示意：

```text
S_unknown = a * normalized_proto_distance + b * energy + c * multiview_variance
```

其中 domain reliability 只调节阈值，不单独判断是否 unknown：

```text
threshold = threshold_base + gamma * domain_uncertainty
```

必须用 held-out TX 做 pseudo-unknown 校准：

```text
train known classes = subset of source TX
validation unknown = held-out source TX
report AUROC, FPR95, OSCR, known acc, new-class acc
```

---

## SGC 后续路线

### 保留方向

```text
sgc_lite_b_no_dac_no_amp
residual-only / gated residual as diagnostic
mild SAT consistency
```

### 暂停方向

```text
full sgc_lite_b_no_dac from source
默认 1.0/1.0 SAT consistency from epoch 1
当前 sgc_adapt 作为最终 Phase 2 适配
no_spec 作为主线
no_amp_freq 作为主线
```

### 建议改动

- `ResidualChannelCompensator` 中 BatchNorm -> GroupNorm 或适配阶段冻结 BN；
- adapter 输出 residual stats，供 domain reliability 评分使用；
- SGC 不再对 domain backbone 输入强制使用同一 processed IQ，至少保留 raw IQ stats 给 domain context；
- 默认 Phase 2 使用 no_amp，避免 per-channel normalization 抹掉 IQ imbalance/PA 幅度 cue。

---

## 推荐文件新增/修改顺序

### Step 1：纯诊断，不改训练行为

```text
eval_feature_diagnosis.py
  + angular geometry
  + effective rank
  + TX/RX ANOVA
  + RX probe on z_id
  + TX probe on z_dom
  + open-set AUROC/FPR95/OSCR
```

### Step 2：采样与 loss

```text
balanced_tx_rx_sampler.py        # 新增
tx_rx_geometry.py                # 新增
train.py                         # 接入 sampler/loss CLI
```

新增 CLI：

```text
--use_tx_rx_balanced_sampler
--tx_per_batch
--domain_per_tx
--samples_per_tx_domain
--lambda_rectangle_id
--lambda_rectangle_rx
--lambda_tx_adv_on_dom
--lambda_proto_margin
```

### Step 3：域上下文结构

```text
model_dual_cvsincnet.py
  + z_rx/z_env/z_int projection
  + tx adversary on z_rx/z_env
  + optional domain-conditioned adapter
```

### Step 4：Phase 2 独立入口

```text
phase2_adapt.py
open_world_head.py
open_world_memory.py
eval_open_world.py
```

不要继续把所有 open-world 逻辑塞进当前 `train.py`，否则很难隔离 source training 与 deployment adaptation。

---

## 建议的第一批实验

### EXP-0：只诊断当前最佳基模

输入 checkpoint：优先使用当前已有最强 R19 Lite-B no-DAC + Fishr / GroupDRO / mild SAT 候选。

输出：

```text
diagnostics/phase2_geometry_r19_fishr.json
diagnostics/phase2_open_set_r19_fishr.json
```

看三件事：

```text
1. class angular safety margin 是否大量为负；
2. z_id 中 RX 方差占比是否过高；
3. z_dom 中 TX probe acc 是否显著高于随机。
```

### EXP-1：balanced sampler + SupCon/proto-margin

基线：`rxrobust_lite_b_no_dac_mix015 + Fishr 0.02`。  
新增：balanced sampler、`lambda_supcon_id=0.03`、`lambda_proto_margin=0.05`。

通过线：

```text
Primary OOD 不低于 baseline -0.3
strict UDU 不低于 baseline -0.3
open-set AUROC 提升
margin_violation_pairs 下降
```

### EXP-2：加入 rectangle loss

在 EXP-1 上加入：

```text
lambda_rectangle_id = 0.02
lambda_rectangle_rx = 0.02
```

观察：

```text
TX/RX ANOVA 中 z_id 的 Var_RX_ratio 是否下降
same-TX cross-RX centroid cosine 是否提升
known/new/unknown 三类是否更可分
```

### EXP-3：no_amp SGC 目标域小步适配

从强基模 checkpoint 加载：

```text
preset = sgc_lite_b_no_dac_no_amp
stage  = sgc_augment or new phase2 adapter stage
```

禁止使用默认 1.0/1.0 SAT 权重。建议：

```text
lambda_sat_cls  = 0.05 or 0.08
lambda_sat_cons = 0.02 or 0.04
sat_cons_start_epoch = 20 or later
```

---

## 明确不要做的事情

```text
DO NOT treat current sgc_adapt as full Phase 2.
DO NOT directly concatenate z_dom into TX classifier.
DO NOT rely on max softmax threshold for unknown rejection.
DO NOT continue full SGC from source as default route.
DO NOT use strong SAT consistency as a universal fix.
DO NOT ignore target RX old-known anchors; without them TX and RX effects are confounded.
DO NOT optimize unknown samples with ordinary entropy minimization.
```

---

## 最重要的结构性前提

在轨 Phase 2 若要分开 TX 与 RX，目标 RX 必须看到一些旧已知 TX：

```text
old_known_tx_i on target_rx
new_tx_j       on target_rx
unlabeled_mix  on target_rx
```

如果只有：

```text
tx1 only from rx1
tx2 only from rx2
```

那么 TX 差异与 RX 差异在统计上不可分。任何模型都只能猜测差异来源。Codex 在实现 Phase 2 数据接口时必须检查并记录 target RX anchor coverage。

---

## Codex 输出要求

每次实现一个步骤后，请同时更新：

```text
progress.md
findings.md
```

并生成或追加：

```text
diagnostics/*.json
docs/*phase2*.md
```

日志中必须能看到：

```text
[PHASE2-GEOM]
[OPENSET]
[TXRX-ANOVA]
[BATCH-GEOM]
[DOMAIN-LEAK]
[UNKNOWN-REJECT]
```

这样后续可以用纯日志判断改动是否真的改善了 Phase 2，而不是只提升闭集 accuracy。
