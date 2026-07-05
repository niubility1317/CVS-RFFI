# CVS Stage2-C OSPR-CI Algorithm

日期：2026-07-05

## 定义

`OSPR-CI`表示`Open-set Source-heldout Prototype Repair Collaborative Inference`。它面向天基RFFI卫星群Stage2-C：冻结`ADV3B02_CORE90_SOFT_E200`特征，使用`qknn8`少样本support记忆，在特征层训练小型source-heldout开集修复adapter，并把协同接收机数量从`M=1`报告到目标接收机总数。

## 协议边界

`target_unknown`只能作为最终评估集。OSPR-CI禁止使用`target_unknown`进行adapter训练、support建库、阈值拟合、profile选择或receiver权重学习。

训练允许数据：

| 数据 | 用途 |
|---|---|
| `source_fit` | 旧类原型和旧类保持 |
| `source_holdout` | source侧校准与旧类margin保持 |
| `proxy_unknown` | source-side hard negative开集压力 |
| `target_old_support` | target旧类K-shot support |
| `target_new_support` | seen-new K-shot enrollment |

禁止数据：

| 数据 | 禁止用途 |
|---|---|
| `target_unknown` | 任何训练、阈值、profile、权重或回滚门设置 |
| `target_old/query` | 阈值/profile选择 |
| `target_new/query` | 阈值/profile选择 |

## 机制

1. 从`source`旧类中按类切分`source_fit/source_holdout`。
2. 使用`source_fit+target support`构建旧类和seen-new原型。
3. 训练低秩残差adapter，目标是保护旧类、提升support几何、压低proxy/virtual open-space已知类logit。
4. 保存adapter后的特征包，复用现有ENPC/SLEV qknn8协同后端。
5. 报告`M=1..R`、同一行`old_acc/min_old/seen_new_acc/min_seen/unknown_reject/unknown_FAR`与资源代理字段。

损失结构：

```text
L = L_source_fit_ce
  + L_source_holdout_ce
  + L_support_ce
  + lambda_preserve * KL(base_logits || adapted_logits)
  + lambda_old * old_floor_margin
  + lambda_seen * seen_floor_margin
  + lambda_proxy * proxy_open_space
  + lambda_virtual * virtual_boundary_open_space
  + lambda_compact * support/source_holdout_compact
  + lambda_residual * ||z_adapted - z||^2
```

## 资源字段

OSPR-CI只声明代理资源字段：

| 字段 | 含义 |
|---|---|
| `bytes_per_event` | 后端证据包估算，非真实链路实测 |
| `latency_ms` | 本地融合代理估算，非星间端到端p95 |
| `qknn8_support_int8_bytes` | qknn8 support int8状态估算 |
| `total_fp16_state_bytes` | adapter/prototype/support合计代理状态 |

在真实端侧序列化、batch=1 profiler、网络排队p95未完成前，不写`resource_real_pass=true`。

## 验收

正式成功必须来自同一实验行：

```text
old_acc >= 0.99
min_old >= 0.95
seen_new_acc >= 0.97
min_seen >= 0.93
unknown_reject >= 0.99
unknown_FAR <= 0.01
```

若通过defer降低FAR，必须单独报告coverage/defer，不能把accepted-only准确率当full-denominator准确率。

## 当前状态

本地真实smoke已证明实现可运行，并输出`M=1..5`，但仍为负诊断。当前`features_proxy_mined.npz`上，OSPR-CI未达到正式成功门槛；后续N607全量复验也应按`NON_DEPLOYMENT_DIAGNOSTIC`边界解释，除非同一行全部门槛达标。
