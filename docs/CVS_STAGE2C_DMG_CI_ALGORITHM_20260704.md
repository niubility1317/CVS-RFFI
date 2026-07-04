# CVS Stage2-C DMG-CI算法设计

## 目标

`DMG-CI`全称为`Dual-Metric Guarded Collaborative Inference`。它用于验证一个保守假设：保旧类分类仍使用原始`ADV3B02_CORE90_SOFT_E200+qknn8`证据，source-heldout hard negative训练得到的metric证据只作为seen-new救援和未知类二级拒识门。

## 协议边界

| 项目 | 约束 |
|---|---|
| 底座模型 | 不改ADV3B02 backbone |
| 在轨方法 | qknn8证据和轻量risk sketch |
| hard negative来源 | source/proxy_unknown证据 |
| target_unknown | 仅最终评估，不参与阈值、profile、adapter或metric选择 |
| 成功口径 | 必须同一row满足old、seen-new、unknown和资源约束 |

## 融合逻辑

输入两条已封闭evidence流：

1. `base_evidence`：原始ADV3B02/qknn8，用于旧类核心保护。
2. `metric_evidence`：source-heldout hard-negative metric后的qknn8证据，用于seen-new救援和拒识风险。

每个receiver-row按同一个`event_id,receiver_id,role,true_label`对齐。DMG-CI不重新拟合阈值，只按预设门控合成：

```text
if base old core and metric agrees:
    keep base old label and cap risk
elif metric predicts seen-new with enough score and margin:
    use metric seen-new label
elif metric unknown risk is high and no old core:
    keep base label but raise reject risk
else:
    use base guarded prediction
```

## 当前本地诊断结论

在`features_proxy_mined.npz`上，DMG-CI能降低unknown FAR，但会严重破坏旧类和seen-new覆盖。本地ENPC/SLEV结果均为`target_pass=false`，最强unknown行约为：

| backend | M | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| ENPC | 1 | 0.1042 | 0 | 0 | 0 | 0.7917 | 0.0417 |
| SLEV | 1 | 0.1042 | 0 | 0 | 0 | 0.7917 | 0.0417 |

该结果说明当前hard-negative metric风险与旧类风险高度重叠，不能作为部署成功。下一步不应继续堆叠拒识门控，而应回到特征训练目标：提升旧类核心区与source-heldout/真实target_unknown之间的几何间隔，同时加入逐类old floor保护。
