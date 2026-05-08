# CVS-RFFI 结构裁剪清单

日期: 2026-05-08

## 已验证可裁剪 (立即执行)

| # | 结构 | 裁剪方式 | 证据 | 涉及实验数 | 性能影响 | 风险 |
|---|---|---|---|---|---|---|
| 1 | DAC Branch | `branch_ablation=no_dac` | type10-5 C01, R19, R25, 5.8 全部 | 100+ | 无害甚至有益 | 无 |
| 2 | DomainFeatureEnhancer | `domain_enhancer=off` | 5.8 D1 最强 source (87.87) | 6 | Primary +0.63 | 低 |
| 3 | Full SGC: AmplitudeNormalizer | 不启用 | 5.7 no_amp 对照 | 11 | 净负 | 无 |
| 4 | Full SGC: FrequencyOffsetCompensator | 不启用 | 5.7 no_freq 对照 | 11 | 净负 | 无 |
| 5 | Full SGC: SpectralInterferenceSuppressor | 不启用 | 5.7 no_spec 崩溃 | 11 | 净负 | 无 |
| 6 | ECC loss | `ecc_w=0` | 5.8 C4 SAT Avg -3.56 | 4 | 不稳定 | 无 |
| 7 | DAC auxiliary losses | `aug_p_dac=0, lambda_cls_dac=0` | no_dac 时自动禁用 | -- | 无 | 无 |
| 8 | DAC-only training view | `enable_dac_aux=False` | no_dac 时自动禁用 | -- | 无 | 无 |

## 推理时可裁剪 (部署优化)

| # | 结构 | 参数量 | 推理时需要 | 裁剪后总参数 |
|---|---|---|---|---|
| 1 | dom_backbone 独立部分 | ~630K | 否 | |
| 2 | dom_head | ~10K | 否 | |
| 3 | adv_head | ~10K | 否 | |
| 4 | DomainFeatureEnhancer | ~23K | 否 | |
| 5 | RCNStatEncoder | ~8K | 否 | |
| 6 | MixStyle1D | ~0 | 否 | |
| 7 | SGC Adapter (可选) | ~50K | 可选 | |
| | **推理最小 (lite_b)** | | | **~750K** |
| | **推理最小 (lite_d)** | | | **~500K** |

## 待验证可裁剪 (需要消融实验)

| # | 结构 | 当前证据 | 消融建议 | 预期参数节省 | 预期性能影响 |
|---|---|---|---|---|---|
| 1 | PA Branch | D4(no_pa) Primary -0.22, Overall +0.23 | 专项消融 | -200K | 可能 -0.2 Primary |
| 2 | dom_backbone 部分共享 | D3(same) Primary -1.16 | 测试部分共享 | -300K | 需要验证 |
| 3 | Orth loss 权重 | D6 Primary +0.25 但 UDU -0.53 | 测试降低至 0.02 | 0 | UDU 可能下降 |
| 4 | Prototype loss | 无 5.8 数据 | 保持关闭 | 0 | 待验证 |
| 5 | SupCon loss | 无 5.8 数据 | 保持关闭 | 0 | 待验证 |

## 必须保留 (有强证据)

| # | 结构 | 参数量 | 证据 | 移除后果 |
|---|---|---|---|---|
| 1 | SincConv1d | ~2.4K | type1-type9 全部使用 | 基础架构 |
| 2 | Time Branch | ~350K | C00 last 16.67% 崩溃 | 训练崩溃 |
| 3 | Freq Branch | ~80K | C03 last 16.67% 崩溃 | 训练崩溃 |
| 4 | PhysicalAwareClassifier | ~120K | type10-2 消融有效 | 分类头 |
| 5 | PA Branch | ~200K | D4 Primary 仅 -0.22 | 稳定作用 |
| 6 | GRL | 0 | D5 UDU -1.05 | 域解耦失效 |
| 7 | Hard-domain CE | 0 | R05 Worst-RX +7.75 | 最大单项提升 |
| 8 | Conservative MixStyle | ~0 | D02 +3.16% last | 显著提升 |
| 9 | Fishr | 0 | SAT37 87.95 Primary | 可靠 OOD 提升 |
| 10 | Residual SGC | ~50K | E2 +0.40 Primary | 保守有效 |

## 裁剪后模型对比

| 配置 | 参数量 | Primary | Strict UDU | Worst-RX | SAT Avg | 推荐场景 |
|---|---|---|---|---|---|---|
| 当前最优 E2 | 1,673K | 88.24 | 86.92 | 86.99 | 41.58 | 主论文 |
| 推理最小 lite_b | ~750K | 88.24* | 86.92* | 86.99* | N/A | 星上部署 |
| 推理最小 lite_d | ~500K | ~87.85* | ~86.27* | ~84.67* | N/A | 极限压缩 |
| INT8 lite_b | ~375KB | <1% loss | <1% loss | <1% loss | N/A | 最终部署 |
| INT8 lite_d | ~250KB | <1% loss | <1% loss | <1% loss | N/A | 极限部署 |

*推理时移除训练专用组件后，分类精度不变
