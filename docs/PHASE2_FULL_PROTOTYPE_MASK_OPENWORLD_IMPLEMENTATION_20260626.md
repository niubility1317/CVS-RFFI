# Phase 2 原型维护、TX/RX 掩码解耦与多原型开放集分类头完整实现方案

日期：2026-06-26  
面向仓库：`niubility1317/CVS-RFFI` 当前 `main` 根目录代码  
建议文档位置：`docs/PHASE2_FULL_PROTOTYPE_MASK_OPENWORLD_IMPLEMENTATION_20260626.md`  
关键词：`CODEX-PHASE2-PROTOTYPE-MASK-OPENWORLD`、`CODEX-TXRX-PROTOTYPES`、`CODEX-MULTI-PROTO-HEAD`、`CODEX-FEATURE-MASKS`

---

## 0. Codex 读取指令

本文件是完整实现蓝图，不只是实验建议。Codex 应按以下顺序读取：

```text
1. docs/PHASE2_OPEN_WORLD_DIAGNOSIS_AND_CODEX_PLAN_20260626.md
2. docs/PHASE2_FULL_PROTOTYPE_MASK_OPENWORLD_IMPLEMENTATION_20260626.md   <- 本文件
3. train.py
4. model_dual_cvsincnet.py
5. model.py
6. dataset_wisig.py
7. eval_feature_diagnosis.py
8. sgc_adapter.py
9. docs/SGC_5_7_analysis_and_merged_plan_20260507.md
10. findings.md
```

本文件要求完整实现以下能力：

```text
A. 地面训练维护 TX 原型、RX/域原型、TX×域局部原型；
B. 建立 TX 原型和域原型之间的“域漂移/条件校正”关系；
C. 在地面训练中使用特征维度掩码和关系掩码区分 TX 身份空间、RX/域空间和交互空间；
D. 阶段二使用多原型 open-world 分类头支持旧类保持、新类注册、目标 RX 校准和未知拒识；
E. 给出可落地的模块、函数、CLI 参数、损失项、训练流程、日志、测试与验收指标。
```

---

## 1. 总体目标

当前模型的主干思路是：

```text
IQ -> DualCVSincNetDisentangle
      ├── ID backbone     -> z_id / tx_logits
      └── domain backbone -> z_dom / dom_logits / adv_dom_logits
```

已有代码能做 source 训练、SGC augment、SGC adapter 适配，但还不具备完整 open-world Phase 2：

```text
旧类保持 + 新类 few-shot 注册 + 目标 RX 域校准 + 未知类拒识 + 无标签流自适应
```

本方案将其扩展为：

```text
IQ input
  │
  ├── raw/RCN/domain statistics
  │       └── RX/domain encoder
  │              ├── z_rx
  │              ├── q_d = domain prototype
  │              └── predicted domain shift s_d
  │
  └── normalized IQ / no_amp SGC / ID backbone
          └── shared feature h
                 ├── z_tx  = M_tx  ⊙ h
                 ├── z_rx' = M_rx  ⊙ h 或 domain branch projection
                 ├── z_int = M_int ⊙ h
                 ├── TX prototype bank P_tx[t]
                 ├── TX-domain local bank P_tx_dom[t,d]
                 └── OpenWorldMultiPrototypeHead
```

核心原则：

```text
TX 原型定义发射机身份中心；
RX/域原型定义接收机/环境上下文；
TX×域局部原型定义该域对该 TX 的实际观测中心；
域原型不直接参与 TX 分类，而是预测/索引身份空间中的域漂移；
阶段二多原型头利用旧类原型、目标域校正原型、新类 support 原型和半径进行分类与拒识。
```

---

## 2. 符号定义

| 符号 | 含义 |
|---|---|
| `x` | IQ packet，形状 `[B, 2, T]` |
| `y` | TX / 发射机标签 |
| `d` | domain 标签，可为 RX、day、RX×day，地面阶段优先 RX 或 RX×day |
| `h` | backbone 输出的共享或投影后特征 |
| `z_tx` | 发射机身份特征，归一化后在单位超球面上 |
| `z_rx` | 接收机/域特征，归一化后在单位超球面上 |
| `z_int` | TX×RX 交互残差特征，小容量、受约束 |
| `P_tx[t]` | 第 `t` 个发射机的全局身份原型 |
| `P_dom[d]` | 第 `d` 个接收机/域的全局域原型 |
| `P_tx_dom[t,d]` | 第 `t` 个发射机在第 `d` 个域下的局部身份原型 |
| `r_tx[t]` | 第 `t` 个 TX 的类内角半径，建议维护 p50/p90/p95 |
| `s_d` | 第 `d` 个域在 TX 身份空间中的公共漂移 |
| `g(q_d)` | 由域原型/域上下文预测身份空间漂移的函数 |
| `M_tx, M_rx, M_int` | 特征维度软掩码，区分身份、接收机、交互子空间 |

所有用于原型比较的特征都必须 L2 normalize：

```text
u = z / ||z||
```

---

## 3. 原型设计

### 3.1 为什么不能只维护一个 class prototype

单一 class prototype：

```text
P_tx[t] = mean(z_tx | TX=t)
```

会把不同接收机/日期造成的系统性偏移混在一起。阶段二遇到 target RX 时，会出现：

```text
旧类在 target RX 下整体漂移；
新类 support 因为 target RX 偏移靠近某个旧类；
未知类被旧类 prototype 吸收。
```

因此需要三层原型：

```text
P_tx[t]       : canonical TX identity center
P_dom[d]      : receiver/domain context center
P_tx_dom[t,d] : local observation center for TX t in domain d
```

### 3.2 发射机全局原型 `P_tx[t]`

定义：

```text
P_tx[t] = normalized balanced mean over domains of mean(z_tx | TX=t, domain=d)
```

不要直接按所有样本平均。正确方式：

```text
for each tx t:
    local_centers = []
    for each domain d that has samples of t:
        local_centers.append(mean(z_tx[y==t and domain==d]))
    P_tx[t] = normalize(mean(local_centers))
```

原因：避免样本量最多的 RX/day 主导发射机原型。

EMA 更新：

```text
P_tx[t] <- normalize(m * P_tx[t] + (1-m) * batch_balanced_center[t])
```

推荐：

```text
proto_momentum = 0.95 or 0.98
min_count_per_update = 2
```

用途：

```text
1. 地面 TX prototype margin loss；
2. 地面类半径估计；
3. 阶段二旧类 source prototype；
4. 阶段二 unknown rejection 的已知类边界。
```

### 3.3 域全局原型 `P_dom[d]`

定义：

```text
P_dom[d] = normalized balanced mean over TX of mean(z_rx | domain=d, TX=t)
```

同样不要按所有样本直接平均。正确方式：

```text
for each domain d:
    local_centers = []
    for each tx t that has samples in d:
        local_centers.append(mean(z_rx[y==t and domain==d]))
    P_dom[d] = normalize(mean(local_centers))
```

目标：

```text
同一 RX/domain 下，不同 TX 的 z_rx 应靠近；
不同 RX/domain 的 P_dom 应可分；
z_rx 不应包含 TX 身份。
```

用途：

```text
1. 地面 domain prototype compactness；
2. TX leakage probe / TX adversary on z_rx；
3. 阶段二 target RX context estimation；
4. 预测或索引 identity-space domain shift s_d；
5. 调节 unknown rejection threshold。
```

禁止：

```text
P_dom[d] 不直接作为 TX 分类证据；
不要把 z_dom/P_dom 直接 concatenate 到 TX classifier。
```

### 3.4 发射机-域局部原型 `P_tx_dom[t,d]`

定义：

```text
P_tx_dom[t,d] = normalize(mean(z_tx | TX=t, domain=d))
```

用途：

```text
1. 估计同一个 TX 在不同 RX/domain 下的身份漂移；
2. 建立 domain prototype 与 TX prototype 的联系；
3. 训练 domain shift predictor；
4. 阶段二为旧类生成 target-domain prototypes。
```

局部原型不作为阶段二唯一分类中心，因为它可能带有历史 RX/domain 偏置。它主要用于估计漂移。

---

## 4. TX 原型与域原型的联系

### 4.1 正确关系：加性漂移 + 受限交互

不要要求：

```text
P_tx[t] 和 P_dom[d] 在同一空间远离或靠近
```

它们语义不同，直接比较没有物理意义。

建议建模：

```text
P_tx_dom[t,d] ≈ normalize(P_tx[t] + s_d + r_{t,d})
```

其中：

| 项 | 含义 | 约束 |
|---|---|---|
| `P_tx[t]` | 发射机 canonical 身份 | 类间角间隔大，类内半径小 |
| `s_d` | domain d 对身份空间的公共漂移 | 可由 `P_dom[d]` 或 target RX context 预测 |
| `r_{t,d}` | TX×domain 交互残差 | 小容量、能量受限，不直接作为 TX 证据 |

### 4.2 域漂移计算

欧氏近似版，容易实现：

```text
delta[t,d] = P_tx_dom[t,d] - P_tx[t]
s_d = mean_t(delta[t,d])
r[t,d] = delta[t,d] - s_d
```

由于原型都在单位球面上，更严谨可以用 tangent-space log map，但第一版不必过度复杂。第一版建议：

```text
使用 delta = normalize(P_tx_dom) - normalize(P_tx)
使用 cosine loss / L2 loss
最后所有用于分类的 prototype 再 normalize
```

### 4.3 域漂移一致性损失

同一域对不同 TX 的公共漂移应相似：

```text
L_shift_cons = mean_d mean_{t1,t2 in active(d)} [1 - cos(delta[t1,d], delta[t2,d])]
```

交互残差应受限：

```text
L_interaction_small = mean_{t,d} ||delta[t,d] - s_d||_2^2
```

### 4.4 域漂移预测损失

新增 `DomainShiftPredictor`：

```text
g(P_dom[d]) -> predicted_s_d
```

损失：

```text
L_shift_pred = 1 - cos(g(P_dom[d]), stopgrad(s_d))
```

阶段二没有地面 domain id 时，可以从 target RX anchors 得到 target context：

```text
q_target = EMA(mean(z_rx on target RX old-known anchors / unlabeled window))
s_target = g(q_target)
```

然后：

```text
correct feature:       z_tx_corr = normalize(z_tx - s_target)
generate old proto:    P_tx_target[t] = normalize(P_tx[t] + s_target)
```

---

## 5. 特征空间目标几何

### 5.1 TX 身份空间 `Z_tx`

目标：

```text
同一 TX 跨 RX/domain 紧凑；
不同 TX 球冠之间留出角间隔；
未知 TX 落在所有已知球冠外；
新类 support 能在空余角空间中注册。
```

对每个 TX：

```text
radius_p95[t] = quantile_0.95(arccos(z_tx_i · P_tx[t]))
```

对每对 TX：

```text
inter_angle[i,j] = arccos(P_tx[i] · P_tx[j])
safety_margin[i,j] = inter_angle[i,j] - radius_p95[i] - radius_p95[j]
```

期望：

```text
safety_margin[i,j] > gamma_open
```

建议初始：

```text
gamma_open = 5° ~ 10°  # 或用弧度 0.087 ~ 0.175
```

如果大量 pair 的 safety margin 为负，阶段二新类与 unknown 拒识一定困难。

### 5.2 RX/域空间 `Z_rx`

目标：

```text
同一 RX/domain 跨 TX 紧凑；
不同 RX/domain 分开；
TX probe 在 z_rx 上接近随机；
RX/domain probe 在 z_tx 上尽量低。
```

指标：

```text
rx_radius_p95[d]
rx_inter_angle[d1,d2]
tx_probe_acc_on_z_rx
rx_probe_acc_on_z_tx
```

### 5.3 交互空间 `Z_int`

目标：

```text
保留小容量 TX×RX residual；
不直接进入 TX 分类头；
用于适配可靠度、unknown threshold 调节或 residual uncertainty。
```

约束：

```text
||z_int|| 不应过大；
TX/RX probe 不能同时很高；
高 z_int energy 表示当前样本存在强交互/不可靠，阶段二阈值应更保守。
```

---

## 6. 掩码设计

### 6.1 为什么需要 mask

当前双骨干已经有 `z_id` 和 `z_dom`，但：

```text
z_id 可能仍含 RX/domain 信息；
z_dom 可能含 TX 身份，特别是 feat_imp/PA 相关特征；
只靠 covariance orthogonal 不能消除条件依赖和非线性泄漏。
```

因此需要结构化 mask 来实现：

```text
哪些特征维度服务 TX 身份；
哪些特征维度服务 RX/domain；
哪些特征维度仅保存小交互 residual。
```

### 6.2 特征维度 mask

给共享或投影后的特征 `h` 学三个 soft masks：

```text
M_tx, M_rx, M_int ∈ [0,1]^D
```

生成：

```text
z_tx  = normalize(M_tx  ⊙ h)
z_rx  = normalize(M_rx  ⊙ h)
z_int = normalize(M_int ⊙ h)
```

如果当前不想重构模型，可做最小侵入实现：

```text
h_tx_base = project(out_main["z_id"])
h_rx_base = project(out_main["z_dom"])
h = LayerNorm(h_tx_base + h_rx_base)   # 或 concat 后 linear
z_tx  = normalize(M_tx  ⊙ h)
z_rx  = normalize(M_rx  ⊙ h)
z_int = normalize(M_int ⊙ h)
```

更稳的第一版：

```text
不要替换原有 tx_logits；
mask features 先只用于 prototype/geometry/open-set auxiliary losses；
验证稳定后再让 multi-prototype head 使用 z_tx。
```

### 6.3 mask 正则

防重叠：

```text
L_overlap = ||M_tx*M_rx||_1 + ||M_tx*M_int||_1 + ||M_rx*M_int||_1
```

覆盖：

```text
L_cover = ||M_tx + M_rx + M_int - 1||_1
```

近二值：

```text
L_binary = mean(M_tx*(1-M_tx)) + mean(M_rx*(1-M_rx)) + mean(M_int*(1-M_int))
```

容量比例：

```text
L_balance = (mean(M_tx)-rho_tx)^2 + (mean(M_rx)-rho_rx)^2 + (mean(M_int)-rho_int)^2
```

建议：

```text
rho_tx  = 0.60 ~ 0.70
rho_rx  = 0.20 ~ 0.30
rho_int = 0.05 ~ 0.15
```

总 mask loss：

```text
L_mask = λ_overlap*L_overlap + λ_cover*L_cover + λ_binary*L_binary + λ_balance*L_balance
```

初始：

```text
lambda_mask = 0.001 ~ 0.005
lambda_binary 前 30% epoch 不开或很小，后期逐步增大
```

### 6.4 关系 mask

除了维度 mask，还要构造 pair/relationship masks。

TX 空间正样本：

```text
same TX, different RX/domain
A_tx[i,j] = 1[y_i == y_j and d_i != d_j]
```

TX 空间负样本：

```text
different TX
N_tx[i,j] = 1[y_i != y_j]
```

RX 空间正样本：

```text
same RX/domain, different TX
A_rx[i,j] = 1[d_i == d_j and y_i != y_j]
```

RX 空间负样本：

```text
different RX/domain
N_rx[i,j] = 1[d_i != d_j]
```

这四个 mask 用于 masked SupCon / prototype contrast，不要再只依赖 batch-global covariance。

---

## 7. 地面训练损失设计

### 7.1 TX prototype margin loss

```text
u_i = normalize(z_tx_i)
pos = cos(u_i, P_tx[y_i])
neg = max_{j != y_i} cos(u_i, P_tx[j])
L_tx_proto_margin = mean([m + neg - pos]_+)
```

推荐：

```text
m = 0.15 ~ 0.30 cosine margin
lambda_tx_proto = 0.03 ~ 0.08
```

### 7.2 TX prototype pull loss

```text
L_tx_pull = mean(1 - cos(u_i, P_tx[y_i]))
```

可与 margin 合并：

```text
L_tx_proto = L_tx_pull + alpha_margin * L_tx_proto_margin
```

### 7.3 Domain prototype loss

```text
v_i = normalize(z_rx_i)
L_rx_pull = mean(1 - cos(v_i, P_dom[d_i]))
```

不同 domain 分开：

```text
neg_dom = max_{k != d_i} cos(v_i, P_dom[k])
L_rx_margin = mean([m_dom + neg_dom - cos(v_i, P_dom[d_i])]_+)
```

### 7.4 TX leakage adversary on RX space

新增：

```text
GRL(z_rx) -> TX classifier
```

损失：

```text
L_tx_adv_on_rx = CE(tx_head_on_rx(GRL(z_rx)), y_tx)
```

训练时正常加到总 loss，GRL 会让 `z_rx` 去 TX 信息，同时让 adversarial head 学会识别 TX。

初始：

```text
lambda_tx_adv_on_rx = 0.02
max 0.10
```

### 7.5 Domain leakage adversary on TX space

当前已有类似：

```text
GRL(z_id) -> domain classifier
```

后续应切换或增加到 masked `z_tx`：

```text
GRL(z_tx) -> domain classifier
```

### 7.6 Masked SupCon

TX SupCon：正样本 `same TX different domain`，负样本 `different TX`。

```text
L_tx_supcon = SupCon(z_tx, positive_mask=A_tx, valid_negative_mask=N_tx)
```

RX SupCon：正样本 `same domain different TX`，负样本 `different domain`。

```text
L_rx_supcon = SupCon(z_rx, positive_mask=A_rx, valid_negative_mask=N_rx)
```

### 7.7 Domain shift losses

```text
L_shift_cons
L_interaction_small
L_shift_pred
```

组合：

```text
L_shift = L_shift_cons + beta_int*L_interaction_small + beta_pred*L_shift_pred
```

推荐：

```text
lambda_shift = 0.02
beta_int = 0.5
beta_pred = 1.0
```

### 7.8 TX×RX rectangle loss

需要 batch 中有四角结构：

```text
(tx1, rx1)  (tx1, rx2)
(tx2, rx1)  (tx2, rx2)
```

TX 身份差分一致性：

```text
L_rect_tx = mean(1 - cos(
    z_tx(tx1,rx1) - z_tx(tx2,rx1),
    z_tx(tx1,rx2) - z_tx(tx2,rx2)
))
```

RX 差分一致性：

```text
L_rect_rx = mean(1 - cos(
    z_rx(tx1,rx1) - z_rx(tx1,rx2),
    z_rx(tx2,rx1) - z_rx(tx2,rx2)
))
```

如果 batch 没有有效 rectangle，返回 graph-preserving zero，并记录：

```text
[BATCH-GEOM] tx_rx_rectangles=0
```

### 7.9 Open-set pseudo-unknown loss

地面阶段必须模拟未知类。

方法：在每个 episode 中把一部分 TX 暂时作为 pseudo-unknown：

```text
episode_known_tx = subset
pseudo_unknown_tx = held-out subset
```

对 known 样本：

```text
energy known should be low
prototype radius score should be inside class boundary
```

对 pseudo-unknown 样本：

```text
energy should be high
nearest prototype distance should be outside boundary
```

简单 loss：

```text
L_energy_known   = mean(ReLU(E_known - margin_known))
L_energy_unknown = mean(ReLU(margin_unknown - E_unknown))
L_open = L_energy_known + L_energy_unknown
```

或 radius margin：

```text
L_unknown_radius = mean(ReLU(radius_margin - nearest_distance_unknown))
```

### 7.10 总 loss

```text
L_total =
    L_ce_tx
  + lambda_dom_ce          * L_domain_ce
  + lambda_domain_adv      * L_GRL_domain_on_tx
  + lambda_tx_proto        * L_tx_proto
  + lambda_rx_proto        * L_rx_proto
  + lambda_tx_supcon       * L_tx_supcon
  + lambda_rx_supcon       * L_rx_supcon
  + lambda_tx_adv_on_rx    * L_tx_adv_on_rx
  + lambda_shift           * L_shift
  + lambda_rect_tx         * L_rect_tx
  + lambda_rect_rx         * L_rect_rx
  + lambda_mask            * L_mask
  + lambda_open            * L_open
  + existing SAT/Fishr/SGC losses as configured
```

建议初始权重：

```text
lambda_tx_proto       = 0.05
lambda_rx_proto       = 0.03
lambda_tx_supcon      = 0.03
lambda_rx_supcon      = 0.02
lambda_tx_adv_on_rx   = 0.02
lambda_shift          = 0.02
lambda_rect_tx        = 0.02
lambda_rect_rx        = 0.02
lambda_mask           = 0.001
lambda_open           = 0.02
```

---

## 8. Batch sampler 设计

### 8.1 为什么必须换 sampler

如果 batch 是普通 shuffle，很多 loss 会缺有效 pair：

```text
same TX cross domain pair
same RX cross TX pair
TX×RX rectangle
```

这会导致解耦 loss 间歇性空转。

### 8.2 Balanced TX×RX sampler

新增文件：

```text
balanced_tx_rx_sampler.py
```

目标 batch：

```text
N_TX × N_DOMAIN × N_PER_CELL
```

推荐：

```text
8 TX × 4 RX/day × 8 packets = batch 256
```

如果样本不足，策略：

```text
1. 优先保证 N_TX 和 N_DOMAIN；
2. per-cell 不足时 replacement sampling；
3. 记录 replacement ratio；
4. 若 rectangle coverage 低于阈值，打印 warning。
```

日志：

```text
[BATCH-GEOM] tx_per_batch=8 domain_per_batch=4 per_cell=8
[BATCH-GEOM] same_tx_cross_domain_pairs=...
[BATCH-GEOM] same_domain_cross_tx_pairs=...
[BATCH-GEOM] tx_rx_rectangles=...
[BATCH-GEOM] replacement_ratio=...
```

### 8.3 CLI 参数

```text
--use_tx_rx_balanced_sampler
--tx_per_batch 8
--domain_per_batch 4
--samples_per_tx_domain 8
--sampler_domain_mode rx_day
--sampler_replacement
--min_rectangles_per_batch 16
```

---

## 9. 阶段二多原型开放集分类头

### 9.1 目标

新增文件：

```text
open_world_head.py
```

核心类：

```python
class OpenWorldMultiPrototypeHead(nn.Module):
    """Multi-prototype cosine head with class radii and unknown rejection scores."""
```

每个类可以有多个 prototype：

```text
old class:
  source global prototype
  target-calibrated prototype
  historical domain prototypes, optional

new class:
  support prototype
  augmented support prototype
  target RX prototype
```

### 9.2 数据结构

```python
prototypes: Tensor[num_classes, max_proto, feat_dim]
radii: Tensor[num_classes, max_proto]
sigmas: Tensor[num_classes, max_proto]
valid_mask: BoolTensor[num_classes, max_proto]
proto_type: List[List[str]]  # source / target / support / augmented / historical-domain
class_type: List[str]        # old / new / reserved
class_ids: List[int]
```

### 9.3 分类分数

输入：

```text
z_tx normalized
optional domain_shift s_target
optional mode: feature_corrected or proto_shifted
```

特征校正：

```text
z_corr = normalize(z_tx - s_target)
```

原型校正：

```text
P_target[t] = normalize(P_tx[t] + s_target)
```

class score：

```text
sim[c,m] = cos(z_corr, prototypes[c,m])
S_c = tau * logsumexp(sim[c,:] / tau)
```

也支持保守版：

```text
S_c = max_m sim[c,m]
```

建议先实现两种：

```text
--multi_proto_pool logsumexp|max
```

### 9.4 半径分数

最近 prototype：

```text
best_c, best_m = argmax sim
angle = arccos(sim[best_c,best_m])
radius_score = (angle - radius[best_c,best_m]) / (sigma[best_c,best_m] + eps)
```

### 9.5 Energy 分数

```text
E = -T * logsumexp(S_c / T)
```

### 9.6 多视图一致性

对同一输入构造多视图：

```text
clean
mild CFO
mild phase noise
mild AWGN
mild channel/SAT view
```

输出：

```text
view_pred_agreement
view_score_variance
view_feature_variance
```

### 9.7 Unknown decision

```python
unknown = (
    radius_score > theta_radius
    or energy > theta_energy
    or view_disagreement > theta_view
)
```

更细化：

```text
known if:
  radius_score <= theta_radius[class]
  energy <= theta_energy
  multiview_agreement >= theta_agree
  temporal_streak >= min_streak
else unknown/uncertain
```

### 9.8 新类注册

```python
def register_new_class(self, class_id, support_features, support_aug_features=None, source_radius_prior=None, k_shot=None):
    p_target = normalize(mean(support_features))
    p_aug = normalize(mean(concat(support_features, support_aug_features)))
    r_new = shrinkage_radius(empirical_radius, source_radius_prior, k_shot)
    add prototypes [p_target, p_aug]
```

Shrinkage：

```text
alpha(K) = source_prior_weight = 1 / sqrt(K)
alpha clipped to [0.2, 1.0]
r_new = alpha * r_source_prior + (1-alpha) * r_empirical
```

### 9.9 旧类目标域校准

目标 RX 上旧类 anchors：

```text
old known TX on target RX
```

更新：

```text
q_target = EMA(mean(z_rx))
s_target = g(q_target)
P_old_target[t] = normalize(P_tx[t] + s_target)
```

如果某个旧类在 target RX 有 anchor，也可直接生成：

```text
P_old_target_observed[t] = normalize(mean(z_tx | old TX=t, target RX))
```

head 中保留：

```text
P_tx_source[t]
P_tx_target_pred[t]
P_tx_target_observed[t] if available
```

---

## 10. Phase 2 适配入口

新增文件：

```text
phase2_adapt.py
```

### 10.1 输入数据

至少需要：

```text
--source_ckpt
--source_proto_path
--target_old_known_dir / pkl split
--target_new_support_dir / pkl split
--target_unlabeled_dir / pkl split
--output_dir
```

阶段二数据必须检查：

```text
target RX 是否有 old-known TX anchors；
new support 是否来自 target RX；
unlabeled 是否混合 old/new/unknown；
每个 new class support 数 K；
每个 old known anchor 数；
```

如果 target RX old-known anchors 不足，必须打印：

```text
[PHASE2-WARN] insufficient old-known target RX anchors; TX/RX effects are confounded.
```

### 10.2 Stage A：目标 RX 校准

冻结：

```text
ID backbone
old class prototypes
old CosFace/classifier weights
```

训练：

```text
no_amp SGC / residual adapter
domain context encoder
domain shift predictor
optional small FiLM adapter
```

损失：

```text
old_known_proto_align
source_distillation
multi_view_consistency
adapter_delta_regularization
domain_context_stability
```

### 10.3 Stage B：新类注册

```text
extract z_tx for support
estimate target RX context q_target
correct support feature if needed
create p_new_target and p_new_src_like
estimate radius using shrinkage
add to OpenWorldMultiPrototypeHead
```

### 10.4 Stage C：无标签 open-world 适配

对每个 unlabeled 样本：

```text
scores = multi_proto_head(x)
if high_conf_old_or_new:
    use pseudo-label CE/prototype alignment/consistency
elif uncertain_or_unknown:
    put into unknown buffer
    do not entropy-minimize as known
```

高可信条件：

```text
radius_score < threshold_radius
energy < threshold_energy
multi_view_agreement > threshold_agree
temporal_streak >= min_streak
```

unknown buffer：

```text
store feature, timestamp, nearest class, scores
cluster periodically
if cluster stable and far from all known/new prototypes -> candidate unknown class
```

---

## 11. 新增/修改文件清单

### 11.1 新增文件

```text
phase2_prototypes.py
feature_masks.py
tx_rx_geometry.py
balanced_tx_rx_sampler.py
open_world_head.py
phase2_adapt.py
eval_open_world.py
```

### 11.2 修改文件

```text
model_dual_cvsincnet.py
train.py
eval_feature_diagnosis.py
findings.md
progress.md
docs/*.md
```

### 11.3 测试文件

```text
tests/test_phase2_prototypes.py
tests/test_feature_masks.py
tests/test_tx_rx_geometry.py
tests/test_open_world_head.py
tests/test_balanced_tx_rx_sampler.py
tests/test_phase2_smoke.py
```

---

## 12. 代码结构草案

### 12.1 `phase2_prototypes.py`

```python
class BalancedPrototypeBank:
    def __init__(self, num_items, feat_dim, momentum=0.95, device=None): ...
    def update_from_features(self, z, labels, group_labels=None): ...
    def get(self, labels=None): ...
    def initialized_mask(self): ...

class TxDomainPrototypeBank:
    def __init__(self, num_tx, num_domains, feat_dim, momentum=0.95, device=None): ...
    def update(self, z_tx, y_tx, d): ...
    def compute_domain_shifts(self, tx_bank): ...
    def local_proto(self, tx, domain): ...

class PrototypeRadiusTracker:
    def update(self, z_tx, y_tx, tx_proto): ...
    def radius(self, class_id, quantile='p95'): ...
    def sigma(self, class_id): ...
```

### 12.2 `feature_masks.py`

```python
class FeatureMaskRouter(nn.Module):
    def __init__(self, feat_dim, tx_ratio=0.65, rx_ratio=0.25, int_ratio=0.10, temperature=1.0): ...
    def forward(self, h):
        masks = self.current_masks()
        z_tx = normalize(masks['tx'] * h)
        z_rx = normalize(masks['rx'] * h)
        z_int = normalize(masks['int'] * h)
        return z_tx, z_rx, z_int, masks
    def mask_regularization(self): ...
```

### 12.3 `tx_rx_geometry.py`

```python
def pair_masks(y_tx, d): ...
def masked_supcon_loss(z, positive_mask, valid_mask=None, temperature=0.12): ...
def tx_rx_rectangle_identity_loss(z_tx, y_tx, d): ...
def tx_rx_rectangle_receiver_loss(z_rx, y_tx, d): ...
def tx_rx_anova_metrics(z, y_tx, d): ...
def domain_shift_losses(tx_domain_bank, tx_bank, domain_bank, shift_predictor=None): ...
```

### 12.4 `open_world_head.py`

```python
class OpenWorldMultiPrototypeHead(nn.Module):
    def add_old_classes(self, prototypes, radii, sigmas=None): ...
    def add_target_prototypes(self, class_ids, prototypes, radii=None, proto_type='target'): ...
    def register_new_class(self, class_id, support_features, support_aug_features=None, radius_prior=None): ...
    def forward(self, z, domain_shift=None, return_details=True): ...
    def unknown_scores(self, class_scores, best_angles, view_stats=None): ...
    def decide(self, scores, thresholds): ...
```

### 12.5 `balanced_tx_rx_sampler.py`

```python
class BalancedTxDomainBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, tx_per_batch, domain_per_batch, samples_per_tx_domain, replacement=True, seed=1337): ...
    def __iter__(self): ...
    def batch_geometry_stats(self, batch_indices): ...
```

---

## 13. `train.py` CLI 新增参数

```text
# prototype banks
--use_phase2_ground_prototypes
--proto_momentum 0.95
--proto_min_count 2
--lambda_tx_proto 0.05
--lambda_rx_proto 0.03
--lambda_proto_margin 0.05
--proto_margin 0.20

# feature masks
--use_feature_masks
--mask_feat_dim 256
--mask_tx_ratio 0.65
--mask_rx_ratio 0.25
--mask_int_ratio 0.10
--lambda_mask 0.001
--mask_binary_start_epoch 80

# TX/RX geometry
--use_txrx_geometry_losses
--lambda_tx_supcon_masked 0.03
--lambda_rx_supcon_masked 0.02
--lambda_tx_adv_on_rx 0.02
--lambda_domain_adv_on_tx 0.0   # optional extra beyond existing adv
--lambda_shift 0.02
--lambda_rect_tx 0.02
--lambda_rect_rx 0.02

# sampler
--use_tx_rx_balanced_sampler
--tx_per_batch 8
--domain_per_batch 4
--samples_per_tx_domain 8
--sampler_replacement

# open-set episodic training
--use_open_set_ground
--episode_known_tx_ratio 0.75
--lambda_open_energy 0.02
--open_energy_margin_known -5.0
--open_energy_margin_unknown -2.0

# logging/checkpoint
--phase2_proto_save_path checkpoints/phase2_proto_bank.pt
--phase2_diag_json diagnostics/phase2_train_diag.json
```

---

## 14. 日志要求

训练日志必须新增：

```text
[PROTO-TX] initialized=... pull=... margin=... radius_p95_mean=... min_inter_angle=... violation_pairs=...
[PROTO-RX] initialized=... pull=... margin=... tx_leak_probe=... rx_radius_p95_mean=...
[SHIFT] cons=... pred=... int=... shift_norm=...
[MASK] tx_mean=... rx_mean=... int_mean=... overlap=... binary=...
[BATCH-GEOM] same_tx_cross_domain_pairs=... same_rx_cross_tx_pairs=... rectangles=...
[OPENSET] auroc=... fpr95=... oscr=... known_energy=... unknown_energy=...
[DOMAIN-LEAK] rx_probe_on_z_tx=... tx_probe_on_z_rx=...
[TXRX-ANOVA] z_tx_var_tx=... z_tx_var_rx=... z_tx_var_int=... z_rx_var_tx=... z_rx_var_rx=...
```

阶段二日志：

```text
[PHASE2-CALIB] target_rx=... old_anchor_tx=... old_anchor_samples=... q_target_norm=... shift_norm=...
[PHASE2-NEW] class=... k=... radius=... proto_count=...
[UNKNOWN-REJECT] known_acc=... new_acc=... unknown_auroc=... fpr95=... reject_rate=...
[UNKNOWN-BUFFER] size=... clusters=... candidate_new_unknown=...
```

---

## 15. 诊断指标扩展

修改 `eval_feature_diagnosis.py`，新增：

```text
angular class geometry
feature effective rank
TX/RX ANOVA
domain leakage probes
open-set pseudo-unknown AUROC/FPR95/OSCR
multi-prototype evaluation
```

输出 JSON 示例：

```json
{
  "checkpoint": "...",
  "feature": "z_tx_masked",
  "class_radius_p95_mean_deg": 12.4,
  "min_interclass_angle_deg": 28.7,
  "margin_violation_pairs": 3,
  "effective_rank": 47.2,
  "txrx_anova": {
    "var_tx_ratio": 0.72,
    "var_rx_ratio": 0.08,
    "var_interaction_ratio": 0.05
  },
  "domain_leakage": {
    "rx_probe_acc_on_z_tx": 18.3,
    "tx_probe_acc_on_z_rx": 9.1
  },
  "open_set": {
    "auroc": 0.91,
    "fpr95": 0.28,
    "oscr_auc": 0.84
  }
}
```

---

## 16. 地面训练推荐实验

### 16.1 Baseline 诊断

```bash
python eval_feature_diagnosis.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --ckpt finalist_runs/final_lite_b_fishr_sat_mild_v1/best_model_primary_ood.pth \
  --features z_id,z_dom,id_feat_joint,dom_feat_imp \
  --phase2_geometry \
  --open_set_eval \
  --heldout_tx_ratio 0.25 \
  --output_json diagnostics/baseline_phase2_geometry.json
```

### 16.2 原型统计 dry run

```bash
python train.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --slim_group rxrobust_lite_b_no_dac_mix015 \
  --epochs 5 \
  --batch_size 256 \
  --use_phase2_ground_prototypes \
  --use_tx_rx_balanced_sampler \
  --tx_per_batch 8 \
  --domain_per_batch 4 \
  --samples_per_tx_domain 8 \
  --lambda_tx_proto 0.0 \
  --lambda_rx_proto 0.0 \
  --lambda_shift 0.0 \
  --latest_save_path checkpoints/proto_dryrun_latest.pth
```

目标：确认 prototype bank 更新、日志和 sampler coverage 正常。

### 16.3 TX prototype margin 实验

```bash
python train.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --slim_group rxrobust_lite_b_no_dac_mix015 \
  --epochs 200 \
  --batch_size 256 \
  --primary_udu_weight 0.65 \
  --lambda_fishr 0.02 \
  --use_phase2_ground_prototypes \
  --use_tx_rx_balanced_sampler \
  --lambda_tx_proto 0.05 \
  --lambda_proto_margin 0.05 \
  --phase2_proto_save_path checkpoints/tx_proto_bank.pt \
  --best_primary_save_path runs/phase2_proto_v1/best_primary.pth
```

### 16.4 原型 + mask + TX/RX geometry 实验

```bash
python train.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --slim_group rxrobust_lite_b_no_dac_mix015 \
  --epochs 200 \
  --batch_size 256 \
  --primary_udu_weight 0.65 \
  --lambda_fishr 0.02 \
  --use_phase2_ground_prototypes \
  --use_feature_masks \
  --use_txrx_geometry_losses \
  --use_tx_rx_balanced_sampler \
  --lambda_tx_proto 0.05 \
  --lambda_rx_proto 0.03 \
  --lambda_tx_supcon_masked 0.03 \
  --lambda_rx_supcon_masked 0.02 \
  --lambda_tx_adv_on_rx 0.02 \
  --lambda_shift 0.02 \
  --lambda_rect_tx 0.02 \
  --lambda_rect_rx 0.02 \
  --lambda_mask 0.001 \
  --best_primary_save_path runs/phase2_mask_proto_v1/best_primary.pth
```

---

## 17. 阶段二推荐命令

### 17.1 从地面 checkpoint 构建原型头

```bash
python eval_open_world.py \
  --source_ckpt runs/phase2_mask_proto_v1/best_primary.pth \
  --proto_bank checkpoints/phase2_proto_bank.pt \
  --build_multi_proto_head \
  --output_head checkpoints/open_world_head_source.pt \
  --output_report diagnostics/open_world_source_report.json
```

### 17.2 目标 RX 校准 + 新类注册

```bash
python phase2_adapt.py \
  --source_ckpt runs/phase2_mask_proto_v1/best_primary.pth \
  --source_proto_bank checkpoints/phase2_proto_bank.pt \
  --open_world_head checkpoints/open_world_head_source.pt \
  --target_old_known_split data/phase2/target_rx_old_known.json \
  --target_new_support_split data/phase2/target_rx_new_support.json \
  --target_unlabeled_split data/phase2/target_rx_unlabeled.json \
  --stage all \
  --adapter_mode no_amp_sgc \
  --calib_epochs 20 \
  --lambda_old_proto_align 1.0 \
  --lambda_source_distill 0.5 \
  --lambda_multiview_cons 0.2 \
  --unknown_buffer \
  --output_dir phase2_runs/target_rx_v1
```

---

## 18. 验收标准

### 18.1 代码级验收

```bash
python -m py_compile phase2_prototypes.py feature_masks.py tx_rx_geometry.py balanced_tx_rx_sampler.py open_world_head.py phase2_adapt.py eval_open_world.py
python -m py_compile train.py model_dual_cvsincnet.py eval_feature_diagnosis.py
```

测试：

```bash
python -m pytest tests/test_phase2_prototypes.py
python -m pytest tests/test_feature_masks.py
python -m pytest tests/test_tx_rx_geometry.py
python -m pytest tests/test_open_world_head.py
python -m pytest tests/test_balanced_tx_rx_sampler.py
```

如果环境没有 pytest/torch，至少保留 `py_compile` 和 mock shape tests。

### 18.2 训练级验收

相对 baseline：

```text
Primary OOD 不低于 baseline -0.30
strict UDU 不低于 baseline -0.30
overall 不低于 baseline -0.30
worst-RX 不低于 baseline -0.50
```

几何指标：

```text
margin_violation_pairs 下降
class_radius_p95_mean 不增加或下降
min_interclass_angle 不下降
effective_rank 不塌缩
rx_probe_acc_on_z_tx 下降
tx_probe_acc_on_z_rx 下降
same_tx_cross_rx_centroid_cos 上升
```

开放集指标：

```text
held-out TX unknown AUROC 上升
FPR95 下降
OSCR AUC 上升
known accuracy 与 new-class accuracy 平衡
```

阶段二：

```text
target old-known accuracy 高
registered new-class accuracy 高
unknown reject AUROC 高
false unknown reject rate 可控
unknown buffer 不被强制 pseudo-label 到旧类
```

---

## 19. 实现顺序

### Phase I：只做统计，不影响训练

```text
1. 新增 phase2_prototypes.py 的 prototype banks；
2. 修改 train.py，在 forward 后更新 banks，但所有 lambda=0；
3. 扩展 eval_feature_diagnosis.py 输出角空间和 open-set 诊断；
4. 确认日志和 JSON 正常。
```

### Phase II：接入 TX prototype margin

```text
1. 接入 L_tx_proto；
2. 确认闭集性能不明显下降；
3. 观察 class radius 和 margin violation 是否改善。
```

### Phase III：接入 domain prototype + TX adversary

```text
1. 接入 L_rx_proto；
2. 接入 GRL(z_rx)->TX classifier；
3. 观察 tx_probe_acc_on_z_rx 是否下降。
```

### Phase IV：接入 mask 和 relationship losses

```text
1. 新增 FeatureMaskRouter；
2. prototype/contrastive 先使用 masked z_tx/z_rx；
3. 接入 masked SupCon；
4. 接入 rectangle loss；
5. 观察 TX/RX ANOVA 和 leakage probe。
```

### Phase V：多原型 head 离线评估

```text
1. 从训练集抽旧类原型和半径；
2. 用 held-out TX 模拟 unknown；
3. 用 pseudo-new episode 模拟 few-shot 注册；
4. 输出 AUROC/FPR95/OSCR。
```

### Phase VI：真实 Phase 2

```text
1. target RX old-known anchors 校准 q_target 和 s_target；
2. 注册新类；
3. 在 target unlabeled stream 上做 open-world 判别；
4. 只对高可信旧类/新类做 pseudo-label；
5. unknown buffer 只聚类和暂存，不强行并入 known。
```

---

## 20. 风险与回滚

| 风险 | 表现 | 回滚/缓解 |
|---|---|---|
| mask 过强导致闭集掉点 | Primary/strict UDU 明显下降 | 降低 `lambda_mask`，延后 binary regularization，只把 mask 用于辅助 loss |
| TX adversary on z_rx 过强 | domain acc 下降，RX 原型不稳 | `lambda_tx_adv_on_rx` 从 0.02 降到 0.005，或 warmup 50 epoch 后再开 |
| prototype margin 过强 | 类内过度压缩，new/unknown 分界反而差 | 降低 margin 或只用 hard negatives top-k |
| rectangle loss 覆盖不足 | loss 长期为 0 | 使用 balanced sampler，增加 `domain_per_batch` |
| domain shift 估计不稳 | target old classes 校正后性能下降 | 使用 EMA q_target，限制 shift norm，保留 source prototype fallback |
| multi-proto 过多导致误吸 unknown | unknown FPR95 变差 | 限制每类 prototype 数，半径 shrink 更保守 |
| SGC no_amp 仍伤害指纹 | clean/OOD 掉点 | 冻结 SGC，仅用 domain-conditioned prototype correction |

---

## 21. 禁止事项

```text
DO NOT 直接把 z_dom 拼到 TX classifier。
DO NOT 让 domain prototype 直接作为 TX 判别证据。
DO NOT 把 uncertain/unknown 样本做普通 entropy minimization。
DO NOT 用 max-softmax 阈值作为唯一 unknown reject。
DO NOT 在没有 target RX old-known anchors 的情况下宣称完成 TX/RX 解耦适配。
DO NOT 默认使用 full SGC per-channel amplitude normalization。
DO NOT 一次性开启所有强 loss；必须按 Phase I -> VI 递进。
```

---

## 22. 最终目标形态

完成后，仓库应支持：

```text
地面训练：
  - 维护 TX/RX/TX-domain prototypes；
  - 约束 TX 空间紧凑且类间有 open margin；
  - 约束 RX 空间不泄漏 TX；
  - 使用 mask 和 relationship masks 区分 TX/RX/interaction；
  - 输出可用于阶段二的 prototype bank、radius、domain shift predictor。

阶段二：
  - 用 target RX old-known anchors 估计 q_target/s_target；
  - 用多原型头保持旧类、注册新类；
  - 用原型半径、energy、多视图一致性进行 unknown rejection；
  - 只对高可信 known/new 样本自训练；
  - unknown buffer 聚类而不强制归入旧类。
```

如果只记一句话：

```text
TX 原型负责身份，RX 原型负责上下文，TX×域原型负责估计身份漂移，多原型头负责阶段二扩展与拒识，mask 负责防止两类信息在特征维度和训练关系上互相污染。
```
