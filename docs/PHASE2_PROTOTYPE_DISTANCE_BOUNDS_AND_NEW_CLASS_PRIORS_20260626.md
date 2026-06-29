# Phase 2 原型距离边界导出与新类少样本原型先验设计

日期：2026-06-26  
关联文档：

```text
docs/PHASE2_OPEN_WORLD_DIAGNOSIS_AND_CODEX_PLAN_20260626.md
docs/PHASE2_FULL_PROTOTYPE_MASK_OPENWORLD_IMPLEMENTATION_20260626.md
```

Codex 搜索关键词：

```text
CODEX-PROTOTYPE-DISTANCE-BOUNDS
CODEX-CLASS-RADIUS-MAX
CODEX-NEW-CLASS-PRIOR
CODEX-FEW-SHOT-PROTO-SHRINKAGE
CODEX-UNKNOWN-REJECTION-BOUNDS
```

---

## 1. 本补充解决的问题

地面训练不仅要维护原型，还必须导出每个训练样本到所属原型的距离分布，尤其是：

```text
每类最大距离
每类 p95/p99 距离
每类 top-k hardest distances
每类按域分解的最大距离 / p95 / p99
全局 open-set radius prior
```

这些统计用于阶段二：

```text
1. 旧类 unknown rejection 边界；
2. 新类少样本原型半径初始化；
3. target RX 域校正后的阈值调整；
4. 多原型分类头中的 class radius / sigma / tail threshold；
5. 判断样本是否落在所有已知类边界之外。
```

此外，面对新类 support 很少的情况，不能只用 K 个 support 样本建立原型和半径。应借助地面已有原型库提供：

```text
1. 半径先验；
2. 方向先验 / 最近旧类邻域先验；
3. 域漂移先验；
4. 类内形状先验；
5. 多视图增强下的稳定性先验。
```

---

## 2. 地面训练必须导出的距离统计

### 2.1 距离定义

所有 TX 身份特征和原型先 L2 normalize：

```text
u_i = normalize(z_tx_i)
p_y = normalize(P_tx[y_i])
```

角距离：

```text
d_i = arccos(clamp(u_i · p_y, -1, 1))
```

建议用角距离作为主指标，单位可同时保存：

```text
radian
degree
cosine_distance = 1 - u_i · p_y
```

阶段二分类和拒识建议统一使用角距离或 cosine similarity，不要混用未归一化欧氏距离。

### 2.2 每类距离边界

对每个 TX 类 `c`，地面训练/诊断阶段导出：

```text
count[c]
mean[c]
std[c]
median[c]
p90[c]
p95[c]
p99[c]
max[c]
topk_max[c]              # top-k hardest sample distances
robust_max[c]            # 推荐用于拒识的稳健最大距离
mad[c]                   # median absolute deviation
tail_mean_top5pct[c]
tail_std_top5pct[c]
```

其中：

```text
max[c]
```

必须保存，但不建议直接作为唯一拒识阈值，因为单个异常样本、标注噪声、低 SNR 样本或强 RX 偏移会让 max 过大，导致阶段二 unknown 被吸收。

推荐同时保存：

```text
robust_max[c] = min(max[c], p99[c] + alpha * tail_std_top5pct[c])
```

初始：

```text
alpha = 1.0 或 1.5
```

阶段二默认使用：

```text
radius_known[c] = robust_max[c] + safety_pad[c]
```

而不是直接 `max[c]`。

### 2.3 按域分解的距离边界

对每个 TX 类 `c` 和 domain `d`，保存：

```text
count[c,d]
mean[c,d]
p95[c,d]
p99[c,d]
max[c,d]
robust_max[c,d]
center_shift[c,d] = angular_distance(P_tx_dom[c,d], P_tx[c])
```

用途：

```text
1. 判断哪个 RX/domain 让某个 TX 类半径变大；
2. 阶段二 target RX 如果与某个 source domain 相似，可借用相似 domain 的半径；
3. 估计 domain-conditioned radius expansion；
4. 发现 TX×RX interaction 异常大的类。
```

### 2.4 全局先验统计

保存全局分布：

```text
global_radius_p50
global_radius_p90
global_radius_p95
global_radius_p99
global_radius_robust_max
class_radius_mean
class_radius_std
class_radius_iqr
class_radius_by_nearest_neighbor_density
```

还要保存类间几何：

```text
interclass_angle_min[c]
interclass_angle_mean[c]
nearest_class[c]
safety_margin_min[c] = min_j angle(P_tx[c], P_tx[j]) - radius[c] - radius[j]
```

如果某类 `safety_margin_min[c] < 0`，说明该类和最近邻类的边界已经重叠。阶段二对这类的拒识阈值必须更保守。

---

## 3. 原型距离统计文件格式

建议新增导出文件：

```text
checkpoints/phase2_proto_bank.pt
checkpoints/phase2_proto_bounds.json
diagnostics/phase2_proto_bounds_report.md
```

### 3.1 JSON 示例

```json
{
  "version": "phase2_proto_bounds_v1",
  "feature": "z_tx",
  "distance": "angular_radian",
  "num_classes": 16,
  "num_domains": 28,
  "global": {
    "radius_p95": 0.213,
    "radius_p99": 0.288,
    "radius_robust_max": 0.331,
    "class_radius_mean": 0.226,
    "class_radius_std": 0.041
  },
  "classes": {
    "0": {
      "count": 10240,
      "mean": 0.094,
      "std": 0.031,
      "p90": 0.151,
      "p95": 0.174,
      "p99": 0.231,
      "max": 0.402,
      "robust_max": 0.283,
      "tail_mean_top5pct": 0.209,
      "tail_std_top5pct": 0.036,
      "nearest_class": 7,
      "nearest_angle": 0.512,
      "safety_margin_p95": 0.164,
      "safety_margin_robust": -0.014
    }
  },
  "class_domain": {
    "0": {
      "3": {
        "count": 320,
        "p95": 0.188,
        "p99": 0.254,
        "max": 0.331,
        "robust_max": 0.289,
        "center_shift": 0.071
      }
    }
  }
}
```

### 3.2 `.pt` 文件内容

```python
{
    "P_tx": Tensor[num_tx, feat_dim],
    "P_dom": Tensor[num_domains, feat_dim],
    "P_tx_dom": Tensor[num_tx, num_domains, feat_dim],
    "P_tx_dom_valid": BoolTensor[num_tx, num_domains],
    "radius_p95": Tensor[num_tx],
    "radius_p99": Tensor[num_tx],
    "radius_max": Tensor[num_tx],
    "radius_robust_max": Tensor[num_tx],
    "radius_sigma": Tensor[num_tx],
    "domain_radius_p95": Tensor[num_tx, num_domains],
    "domain_radius_robust_max": Tensor[num_tx, num_domains],
    "nearest_class": LongTensor[num_tx],
    "nearest_angle": Tensor[num_tx],
    "safety_margin_p95": Tensor[num_tx],
    "safety_margin_robust": Tensor[num_tx],
    "metadata": {...}
}
```

---

## 4. 为什么 max distance 要保存，但不能单独使用

最大距离的优点：

```text
保守覆盖所有已知训练样本；
阶段二可直接判断“是否超出训练已知类支撑集”；
对安全拒识有解释性。
```

最大距离的风险：

```text
1. 一个噪声样本会把半径撑得很大；
2. 少量错误标签会严重污染阈值；
3. 某个弱 RX/domain 会让某类半径过宽；
4. 使用 max 会提高 unknown false accept。
```

因此建议同时提供三套边界：

```text
strict_radius[c]  = p95[c]
balanced_radius[c] = p99[c]
loose_radius[c]   = robust_max[c]
raw_max[c]        = max[c]  # 只作审计，不默认作为阈值
```

阶段二根据任务选择：

```text
高安全拒识：strict_radius
平衡识别/拒识：balanced_radius
最大限度保旧类召回：loose_radius
人工审计/异常分析：raw_max
```

---

## 5. 阶段二如何使用距离边界进行未知拒识

对样本 `x`：

```text
u = normalize(z_tx(x))
score_c = max_m cos(u, proto[c,m])
angle_c = arccos(score_c)
best_c = argmin angle_c
```

半径分数：

```text
radius_score = (angle_best - radius[best_c]) / (sigma[best_c] + eps)
```

拒识规则初版：

```text
if angle_best > radius[best_c] + margin_unknown:
    unknown
else:
    known_or_new
```

多信号融合版：

```text
unknown if any/weighted:
    angle_best > radius[best_c]
    energy > theta_energy
    multiview_agreement < theta_view
    temporal_streak < min_streak
```

建议保存不同半径模式：

```text
--unknown_radius_mode strict|balanced|loose|raw_max
```

对应：

```text
strict   -> p95
balanced -> p99
loose    -> robust_max
raw_max  -> max
```

默认：

```text
balanced
```

---

## 6. 新类样本稀少时，如何借助已有地面原型

答案：可以，而且应该这样做。但不能把新类直接拉向某个旧类原型；应借助旧类原型提供“半径、形状、域漂移、局部密度”的先验。

### 6.1 新类原型的基本建立

新类 support：

```text
S_n = {x_1, ..., x_K}
```

提取：

```text
u_i = normalize(z_tx(x_i))
```

目标 RX 已估计 domain shift `s_target` 时：

```text
u_i_corr = normalize(u_i - s_target)
```

新类 canonical prototype：

```text
P_new_src_like = normalize(mean_i u_i_corr)
```

新类 target prototype：

```text
P_new_target = normalize(mean_i u_i)
```

阶段二多原型头同时保存：

```text
class new_n:
    prototype[0] = P_new_src_like
    prototype[1] = P_new_target
```

### 6.2 借用已有原型的半径先验

由于 K 很小，新类经验半径不可靠：

```text
r_empirical_new = quantile(angle(u_i, P_new), q)
```

K=1 时甚至为 0。

因此使用 shrinkage：

```text
r_new = alpha(K) * r_prior + (1 - alpha(K)) * r_empirical_new
```

推荐：

```text
alpha(K) = clamp(1 / sqrt(K), min=0.2, max=1.0)
```

例子：

```text
K=1  -> alpha=1.00，几乎完全用地面先验
K=2  -> alpha=0.71
K=5  -> alpha=0.45
K=10 -> alpha=0.32
K>=25 -> alpha≈0.20
```

`r_prior` 如何选：

```text
方案 A：global class radius prior，例如 median(radius_p95 或 p99)
方案 B：nearest-neighbor prior，从和新类 support 最近的若干旧类半径加权平均
方案 C：domain-conditioned prior，使用 target RX 相似域下的旧类半径
方案 D：hybrid prior，A/B/C 加权
```

建议默认 hybrid：

```text
r_prior = 0.4 * global_prior + 0.4 * nearest_old_class_prior + 0.2 * domain_prior
```

### 6.3 最近旧类邻域先验

计算新类 support prototype 与所有旧类 prototype 的角距离：

```text
angle_to_old[c] = arccos(P_new · P_tx[c])
```

选择 top-M 最近旧类：

```text
N_M = top M nearest old classes
M = 3 or 5
```

用 soft weight：

```text
w_c = softmax(-angle_to_old[c] / tau_prior)
```

半径先验：

```text
r_prior_neighbor = sum_c w_c * radius[c]
```

形状/不确定性先验：

```text
sigma_prior_neighbor = sum_c w_c * sigma[c]
```

注意：不要让新类原型向旧类原型方向移动太多，否则会吞掉新类差异。旧类只提供半径/不确定性/尾部分布先验，不直接替代新类 prototype。

### 6.4 旧类原型能否帮助新类 prototype 方向？

可以，但只能作为弱正则和去偏，不应强拉。

两种安全方式：

#### 方式 1：域漂移去偏

如果 support 来自 target RX，则新类方向受 target RX 偏移影响。使用旧类 target anchors 估计：

```text
s_target = g(q_target)
```

然后：

```text
P_new_src_like = normalize(mean(u_i - s_target))
```

这是最推荐的“用旧类帮助新类方向”的方式。

#### 方式 2：局部 tangent shrinkage

如果新类 support 很少且噪声大，可用最近旧类子空间做轻微去噪。

构建最近旧类局部子空间：

```text
C = covariance of features from nearest old classes or their class-domain centers
```

把新类 prototype 的高噪声分量做轻微收缩：

```text
P_new_denoised = normalize(P_new - beta * projection_to_noise_subspace(P_new))
```

第一版可以不实现，先只做域漂移去偏和半径 shrinkage。

### 6.5 新类多原型增强

少样本下一个 prototype 太脆弱。建议至少建立：

```text
P_new_target_clean
P_new_target_aug_mean
P_new_src_like_corrected
```

增强视图：

```text
mild CFO
mild phase noise
mild AWGN
mild RX/channel perturbation
no_amp SGC view
```

保留增强均值：

```text
P_new_aug = normalize(mean(z_tx(aug(x_i))))
```

如果增强后 prototype 和 clean prototype 夹角过大：

```text
angle(P_new_clean, P_new_aug) > threshold
```

说明新类 support 不稳定，应增大 `r_new` 或标记低置信注册。

### 6.6 新类半径的域条件扩张

如果 target RX 的 domain uncertainty 高，或旧类 anchors 显示 target drift 很大：

```text
domain_uncertainty = norm(s_target) or variance(z_rx window)
```

则：

```text
r_new_final = r_new + gamma_domain * domain_uncertainty
```

但要上限裁剪：

```text
r_new_final <= r_prior_loose
```

防止新类半径过大吞 unknown。

---

## 7. 新类原型建立算法

### 7.1 输入

```text
support_features U = [K, D]
support_aug_features optional
source P_tx, radius stats, sigmas
optional target q_target / s_target
optional target old-known anchors
```

### 7.2 输出

```text
P_new_src_like
P_new_target
P_new_aug
r_new_strict
r_new_balanced
r_new_loose
sigma_new
registration_confidence
```

### 7.3 算法

```text
1. normalize all support features u_i.
2. if s_target exists:
       u_i_corr = normalize(u_i - s_target)
   else:
       u_i_corr = u_i
3. P_new_target = normalize(mean(u_i))
4. P_new_src_like = normalize(mean(u_i_corr))
5. compute empirical distances to P_new_target and P_new_src_like.
6. compute nearest old classes to P_new_src_like.
7. compute global radius prior.
8. compute nearest-neighbor radius prior.
9. compute domain-conditioned radius prior if target domain known.
10. r_prior = weighted sum of global/neighbor/domain priors.
11. alpha = clamp(1/sqrt(K), 0.2, 1.0)
12. r_new = alpha*r_prior + (1-alpha)*r_empirical
13. if augmentation exists:
        compute P_new_aug and view dispersion
        r_new += gamma_view * view_dispersion
14. if domain uncertainty exists:
        r_new += gamma_domain * domain_uncertainty
15. clamp r_new to [r_min, r_max] from source prior.
16. add prototypes to multi-prototype head.
```

### 7.4 注册置信度

```text
conf = weighted combination of:
    K support count
    support compactness
    augmentation consistency
    distance to nearest old class
    target domain reliability
```

如果新类 prototype 距离最近旧类太近：

```text
angle(P_new, P_nearest_old) < radius_new + radius_old + gamma_open
```

则标记：

```text
[PHASE2-NEW-WARN] new class overlaps old prototype boundary
```

策略：

```text
1. 要求更多 support；
2. 使用 stricter radius；
3. 暂时不自动注册，进入 pending class；
4. 降低 pseudo-label 扩展速度。
```

---

## 8. 多原型头中如何保存新类先验

每个新类保存：

```python
{
    "class_id": new_id,
    "class_type": "new",
    "prototypes": {
        "target_clean": P_new_target,
        "src_like_corrected": P_new_src_like,
        "augmented_mean": P_new_aug
    },
    "radii": {
        "strict": r_new_strict,
        "balanced": r_new_balanced,
        "loose": r_new_loose
    },
    "sigma": sigma_new,
    "support_count": K,
    "nearest_old_classes": [...],
    "registration_confidence": conf,
    "domain_context": {...}
}
```

分类时：

```text
score_new = max or logsumexp similarity over new class prototypes
radius_new = selected radius mode for winning prototype
```

拒识时：

```text
如果最佳类是 new class，但 registration_confidence 低，则阈值更保守。
```

---

## 9. 地面训练代码需要新增的统计类

建议在 `phase2_prototypes.py` 增加：

```python
class PrototypeDistanceTracker:
    def __init__(self, num_classes, num_domains, topk=32): ...
    def update(self, z_tx, y, d, P_tx, P_tx_dom=None): ...
    def finalize(self): ...
    def to_json_dict(self): ...
    def state_dict(self): ...
```

功能：

```text
1. 在线累计每类距离的 count/mean/std；
2. 保存 top-k 最大距离；
3. 支持 epoch 末计算 p90/p95/p99；
4. 支持 class-domain 距离统计；
5. 支持 nearest class 和 safety margin 计算；
6. 导出 JSON 和 PT。
```

实现注意：

```text
如果完整保存所有训练样本距离内存太大，可按 epoch 保存 histogram + top-k；
如果样本量可接受，第一版可保存 CPU list，epoch end 计算 quantile；
final training 后建议用一次 full train loader eval pass 重新精确统计。
```

---

## 10. 训练流程中的边界统计时机

不要只用训练中 EMA 过程的 batch 估计。建议：

```text
1. 训练过程中每 epoch 记录粗略距离；
2. best checkpoint 保存后，加载 best checkpoint；
3. 对 full train set / calibration set 跑一遍 eval；
4. 用 eval 模式重新提取 z_tx；
5. 用最终 P_tx/P_tx_dom 计算距离边界；
6. 导出 phase2_proto_bounds.json / pt。
```

原因：

```text
训练中 dropout/mixstyle/augment 会改变距离；
阶段二需要的是 eval-mode 稳定边界；
最好用与部署一致的 feature extraction path。
```

---

## 11. CLI 新增建议

`train.py`：

```text
--track_proto_distances
--proto_distance_topk 32
--proto_bounds_save_path checkpoints/phase2_proto_bounds.json
--proto_bounds_pt_save_path checkpoints/phase2_proto_bounds.pt
--proto_radius_modes p95,p99,robust_max,max
--proto_robust_max_alpha 1.5
--proto_bounds_eval_after_train
```

`phase2_adapt.py`：

```text
--proto_bounds_path checkpoints/phase2_proto_bounds.json
--unknown_radius_mode balanced
--new_class_radius_prior hybrid
--new_class_prior_neighbor_k 5
--new_class_prior_tau 0.1
--new_class_radius_shrinkage sqrt
--new_class_domain_radius_weight 0.2
--new_class_global_radius_weight 0.4
--new_class_neighbor_radius_weight 0.4
--new_class_min_support_warn 5
```

---

## 12. 日志新增要求

地面训练日志：

```text
[PROTO-BOUNDS] class=0 count=... p95=... p99=... max=... robust_max=... nearest=... safety=...
[PROTO-BOUNDS-SUMMARY] p95_mean=... p99_mean=... robust_mean=... max_mean=... violation_pairs=...
[PROTO-BOUNDS-DOMAIN] class=0 domain=3 p95=... max=... shift=...
```

阶段二新类注册日志：

```text
[NEW-PROTO] class=17 K=3 r_emp=... r_prior_global=... r_prior_neighbor=... r_prior_domain=... alpha=... r_final=...
[NEW-PROTO] nearest_old=7 angle=... old_radius=... overlap_margin=...
[NEW-PROTO-WARN] low support or overlap with old boundary; use strict pseudo-label admission.
```

未知拒识日志：

```text
[UNKNOWN-REJECT] radius_mode=balanced best_class=... angle=... radius=... radius_score=... energy=... view_agree=...
```

---

## 13. 验收标准

### 13.1 地面导出

必须生成：

```text
checkpoints/phase2_proto_bounds.json
checkpoints/phase2_proto_bounds.pt
diagnostics/phase2_proto_bounds_report.md
```

检查项：

```text
每个类 count > 0
每个类 p95/p99/max/robust_max 有限
max >= p99 >= p95 >= mean
robust_max <= max
nearest_class 不等于自身
safety_margin 可计算
class-domain 统计有 valid mask
```

### 13.2 阶段二新类

模拟 K-shot：

```text
K = 1, 2, 5, 10
```

要求：

```text
K=1 时 radius 不为 0，而来自 source prior；
K 增大时 empirical radius 权重上升；
新类与最近旧类 overlap 时能报警；
unknown AUROC/FPR95 相比无 prior 不下降，理想应提升。
```

### 13.3 unknown rejection

对 held-out TX：

```text
AUROC 提升或持平
FPR95 下降
known accuracy 不大幅下降
```

如果直接使用 raw max 导致 FPR95 变差，默认切回 balanced/p99 或 robust_max。

---

## 14. 最终建议

地面训练必须导出训练样本到原型的最大距离，但应同时导出 p95/p99/top-k/robust max。阶段二默认不要直接用 raw max，而使用：

```text
balanced: p99
loose: robust_max
strict: p95
```

新类样本稀少时，可以借助已有地面原型，但借用方式应是：

```text
1. 借旧类半径分布作为新类半径先验；
2. 借最近旧类邻域作为不确定性/形状先验；
3. 借目标 RX old-known anchors 估计域漂移，修正新类 support 特征；
4. 借多视图增强估计新类稳定性；
5. 不把新类原型强行拉向旧类原型。
```

一句话：

```text
旧类原型库帮助新类“定半径、去域偏、估不确定性”，但不能替代新类 support 的身份方向。
```
