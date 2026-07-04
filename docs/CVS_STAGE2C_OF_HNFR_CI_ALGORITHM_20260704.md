# OF-HNFR-CI:旧类 floor 约束的硬负样本协同推理诊断

## 场景边界

OF-HNFR-CI 面向 Stage2-C 天基 RFFI 卫星群协同推理诊断。输入为 ADV3B02/Proxy-mined 特征包，协同后端仍使用 qknn8 的 ENPC/SLEV 证据栈，`target_unknown`不进入训练、适配器更新或阈值拟合。摘要中的`best_joint_row`只允许作为事后诊断排序，不允许作为部署 profile 选择依据。

## 算法机制

该方法训练低秩残差特征适配器，训练数据只包含：

| 数据角色 | 用途 |
|---|---|
| `source`旧类 | 旧类 CE、旧类 KL 保真、旧类 margin floor |
| `proxy_unknown` | source-side 硬负样本开放集排斥 |
| `target_old` K-shot support | target 旧类保真与 floor |
| `target_new` K-shot support | seen-new CE、support KL 保真、seen margin floor |
| `target_unknown` | 禁止训练，仅评估 |

核心损失为：

```text
L = L_source_ce
  + L_support_ce
  + w_old_kl L_source_old_kl
  + w_support_kl L_support_kl
  + w_target_old_kl L_target_old_kl
  + w_old_floor (L_source_old_floor + L_target_old_floor)
  + w_seen_floor L_seen_floor
  + w_proxy L_proxy_open
  + w_virtual L_virtual_boundary_open
  + w_compact L_support_compact
  + w_residual L_residual
  + w_old_core L_old_core_residual
  + w_support_core L_support_core_residual
```

其中 floor loss 使用真类 logit 与最高非真类 logit 的 margin：

```text
L_floor = mean(max(0, margin - (s_y - max_{c!=y}s_c)))
```

## 部署资源

适配器状态为低秩残差参数、原型和轻量 verifier 状态，摘要字段写入`state_bytes.total_fp16_state_bytes`。协同通信仍由后端 M 接收机 qknn8 事件摘要决定，当前全量评估使用`collab_count=1..5`、`query_per_class=12`。

## 当前验证状态

本地单元测试覆盖：

| 测试项 | 结果 |
|---|---|
| `old_floor_margin_loss`低 margin 惩罚与达标置零 | PASS |
| 训练指标中`target_unknown_training_count=0` | PASS |
| OPR 原有 target_unknown 排除与 manifest 测试 | PASS |

本地诊断显示 OF-HNFR-CI 能同时提高 target-old support 与 seen-new support 的 prototype accuracy，并显著降低 proxy_unknown 最大已知 logit；但真实`target_unknown`仍未在不伤害 old/seen 的情况下达到拒识目标。该结论仅适用于当前 ADV3B02/Proxy-mined 特征、OF-HNFR-CI 训练配置和 ENPC/SLEV 后端；当前状态为`NON_DEPLOYMENT_DIAGNOSTIC`，不能作为部署成功候选。
