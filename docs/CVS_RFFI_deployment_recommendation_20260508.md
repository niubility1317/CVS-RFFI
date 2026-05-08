# CVS-RFFI 星上部署推荐

日期: 2026-05-08

## 当前最优模型 (开发集)

**E2_residual_only_std_res001**

| 指标 | 数值 |
|---|---:|
| Primary score | 88.24 |
| Overall TX accuracy | 90.70% |
| Strict unseen-day unseen-RX | 86.92% |
| Worst unseen-RX | 86.99% |
| SAT Avg (5 scenarios) | 41.58 |
| Parameters | 1,672,844 |
| Training route | D1 source (200ep) -> E2 residual SGC (60ep) |

**注意:** 这是开发集最优, checkpoint 选择使用了 test-derived 指标。论文最终表格应使用 validation-only 选择或 fresh holdout。

## 星上部署推理模型

### 最小推理架构 (lite_b)

```
IQ [B, 2, 256]
  -> SincConv1d (24 filters, learnable bandpass)
  -> HighFreqEmphasis (fixed 1st/2nd diff)
  -> Time: fuse -> DSConv(96->144->144) -> t_emb(192)
  -> Freq: FFT -> FreqBandGate(36) -> DSConv(24->48->48) -> f_emb(192)
  -> PA: MemPolyLift(1,3,5) -> DilatedConv(72->96->96) -> pa_local(192)
  -> PhysicalAwareClassifier -> CosFace logits [B, 6]
```

| 属性 | 数值 |
|---|---|
| 参数量 | ~750K |
| FLOPs | ~150M |
| 内存 (FP32) | ~3MB |
| 内存 (INT8) | ~750KB |
| 输入 | 256 点 I/Q |
| 输出 | 6 类 TX |
| 延迟 (估算) | <1ms @ 1GHz ARM |

### 极限压缩架构 (lite_d)

| 属性 | 数值 |
|---|---|
| 参数量 | ~500K |
| 内存 (INT8) | ~500KB |
| Primary (参考) | ~87.85 |

## 可移除的训练组件

推理时移除以下组件不影响分类精度:

| 组件 | 参数量 | 用途 |
|---|---|---|
| dom_backbone | ~630K | 域解耦训练 |
| dom_head | ~10K | 域分类 |
| adv_head | ~10K | GRL 对抗 |
| DomainFeatureEnhancer | ~23K | RCN 域增强 |
| RCNStatEncoder | ~8K | 统计编码 |
| MixStyle1D | ~0 | 训练风格混合 |
| SGC Adapter | ~50K | 信道补偿 (可选) |

## 可选星上组件

| 组件 | 参数 | 用途 | 何时启用 |
|---|---|---|---|
| Residual SGC Adapter | ~50K | 卫星信道补偿 | 需要信道鲁棒性时 |

E2 模型已包含 residual SGC adapter (32ch, 2 blocks, gamma=0 init)。星上部署时:
- 如果卫星信道已知/稳定: 移除 adapter, 使用纯 id_backbone
- 如果卫星信道未知/多变: 保留 adapter, 用 source checkpoint 初始化

## 量化路径

| 阶段 | 方法 | 参数大小 | 精度损失 |
|---|---|---|---|
| FP32 | 基线 | 3MB (lite_b) | 0% |
| FP16 | 半精度 | 1.5MB | <0.1% |
| INT8 | 量化感知训练 | 750KB | <1% |
| INT4 | 极限量化 | 375KB | 需要验证 |

## 知识蒸馏路径

如果 lite_d (500K) 精度不够, 可以从 lite_b 蒸馏:

```
Teacher: lite_b no_dac (~750K params, Primary ~88.24)
Student: lite_e no_dac (~300K params)
Loss: KL(teacher_logits, student_logits) + CE(student_logits, y)
```

## 部署前检查清单

- [ ] 验证推理模型输出与训练模型一致
- [ ] 在 holdout 测试集上确认 Primary > 87.5
- [ ] INT8 量化后精度损失 <1%
- [ ] 单帧推理延迟 <2ms
- [ ] 内存占用 <1MB
- [ ] 如果使用 SGC adapter: 在目标卫星场景上验证 SAT Avg > 40
