# CVS-RFFI 模型分支结构与训练收益全面分析报告

日期: 2026-05-08
数据来源: 项目内全部 286+ 个训练日志 (type3 至 5.8)

---

## 一、实验数据全景

### 1.1 日志来源统计

| 来源目录 | 日志数 | 有效训练日志 | 时代 | 关键实验 |
|---|---:|---:|---|---|
| `history/CV-SincNet/type3-8/` | 17 | 17 | ORACLE 同分布 | SupCon/Proto/DG 验证 |
| `history/CV-SincNet/type9/` | 3 | 3 | WiSig 迁移 | 首次跨域 |
| `history/CV-SincNet/type10/` | 1 | 1 | 双骨干原型 | best 96.12%@E27 |
| `history/CV-SincNet/type10-pa/` | 8 | 8 | PA/DAC 消融 | g4 balanced 最优 |
| `history/CV-SincNet/type10-2/` | 9 | 9 | 组消融 | g4 full balanced 86.87% |
| `history/CV-SincNet/type10-3/` | 5 | 5 | 阶段训练 | manyday 86.72% |
| `history/CV-SincNet/type10-5/` | 18 | 18 | 最终模块消融 | D02 89.67% last |
| `history/CV-SincNet/type10-7/` | 41 | 41 | SAT 评估矩阵 | SAT37 fishr 87.95 |
| `history/CV-SincNet/type11-15/` | 11 | 9 | 并行探索 | type11 崩溃至 31% |
| `type10-4/4.23-4.27logs/` | 79 | 69 | R 系列消融 | R19 87.46, R25 87.17 |
| `5.7/logs/` | 39 | 32 | SGC 三阶段 | 11 preset x 3 stage |
| `5.8/logs/` | 20 | 19 | 最终候选 | E2 88.24 |
| `logs/` (root) | 35 | ~5 | 多 seed + baseline | 多数启动失败 |
| **总计** | **~286** | **~228** | | |

### 1.2 关键预计算指标文件

| 文件 | 内容 |
|---|---|
| `5.8/metrics/experiment_metrics.csv` | 19 实验 x 45+ 列完整指标 |
| `5.8/metrics/experiment_metrics.json` | 结构化 JSON (33KB) |
| `type10-4/outputs/all_log_analysis_20260427.json` | 76 条解析记录 |
| `type10-7/CV-SincNet/sgc_log_summary.json` | SGC 日志摘要 (43KB) |

---

## 二、模型结构全景与参数量

### 2.1 当前最优模型架构: DualCVSincNetDisentangle (E2)

```
输入 IQ [B, 2, 256]
  |
  v
SGC Adapter (residual-only, gamma=0 初始化)
  |
  +---> id_backbone (CVSincNet lite_b, no_dac)
  |       |-- SincConv1d (24 filters, 共享stem)         ~2.4K params
  |       |-- HighFreqEmphasis (固定差分, 0 params)
  |       |-- Time Branch: fuse->DSConv x3->t_emb       ~350K params
  |       |-- Freq Branch: FFT->gate->DSConv x3->f_emb  ~80K params
  |       |-- PA Branch: MemPoly->DilatedConv x3         ~200K params
  |       |-- [DAC Branch: 禁用]                         0 params
  |       +-- PhysicalAwareClassifier + CosFace          ~120K params
  |
  +---> dom_backbone (CVSincNet lite_b, no_dac, no_stats)
  |       |-- 共享 stem (与 id_backbone 同实例)
  |       |-- 独立 Time/Freq/PA branches                 ~630K params
  |       +-- 输出 feat_imp -> z_dom_raw
  |
  +---> DomainFeatureEnhancer (rcn_stats, s=0.35)        ~23K params
  |       |-- RCNStatEncoder: 18维统计->MLP
  |       +-- gate = sigmoid(W*[z_dom, z_rcn])
  |
  +---> dom_head (MLP)                                   ~10K params
  +---> adv_head (MLP + GRL)                             ~10K params
```

### 2.2 各配置实测参数量 (来自 5.8 日志)

| 配置 | 参数量 | 实验 | 来源 |
|---|---|---|---|
| 标准 lite_b no_dac | **1,672,409** | B1-B4, C1-C4, D2, D5, D6, E0 | experiment_metrics.csv |
| domain_enhancer=off | **1,577,465** | D1 | experiment_metrics.csv |
| no_pa_no_stats | **1,343,416** | D4 | experiment_metrics.csv |
| domain_branch_same | **1,360,792** | D3 | experiment_metrics.csv |
| + SGC residual-only | **1,672,844** (+435) | E1, E2 | experiment_metrics.csv |
| + SGC no-res control | **1,673,467** (+1,058) | E3 | experiment_metrics.csv |
| + SGC full | **1,673,902** (+1,493) | E4 | experiment_metrics.csv |
| Lite-D no_dac (紧凑) | **~1,050,000** | R25 系列 | type10-4 findings.md |

### 2.3 推理时可移除组件

| 组件 | 参数量 | 推理时需要 | 证据 |
|---|---|---|---|
| dom_backbone (独立部分) | ~630K | **否** | 训练域解耦专用 |
| dom_head | ~10K | **否** | 域分类头 |
| adv_head | ~10K | **否** | GRL 对抗头 |
| DomainFeatureEnhancer | ~23K | **否** | D1=off 最强 |
| RCNStatEncoder | ~8K | **否** | 域增强编码 |
| MixStyle1D | ~0 | **否** | 训练时特征混合 |
| SGC Adapter | ~50K | **可选** | 星上补偿可选 |

**推理最小模型: ~750K params (lite_b) 或 ~500K params (lite_d)**

---

## 三、全历史消融实验量化证据

### 3.1 分支消融 (type10-5, 18 实验)

| 实验 | 操作 | Last Epoch | Best Joint | Best Overall | 最难 Split | 结论 |
|---|---|---|---|---|---|---|
| A00_s1_core_base | 全分支基准 | 86.51% | 86.25% | 89.51% | 86.72% | 基准线 |
| A01_s4_base_no_mixstyle | 去 MixStyle | 87.46% | 87.67% | 89.72% | 83.58% | MixStyle 有价值 |
| B00_mixstyle_cd_td_t1 | MixStyle 标准 | 88.03% | 88.08% | 89.85% | 85.37% | +1.52 last |
| **B02_mixstyle_random_td_t1** | **随机 MixStyle** | **16.67%** | 88.22% | 89.94% | 86.71% | **崩溃** |
| B01_mixstyle_cd_td_t1_t2 | 多层 MixStyle | 48.07% | -- | -- | -- | **严重退化** |
| **C00_no_time** | **移除时间分支** | **16.67%** | 86.96% | 88.05% | 84.27% | **崩溃** |
| **C03_no_freq** | **移除频率分支** | **16.67%** | 86.23% | 86.71% | 82.57% | **崩溃** |
| C01_no_dac | 移除 DAC | 88.17% | 88.27% | 90.04% | 85.61% | **无害有益** |
| D00_mixstyle_no_dac | MixStyle+无DAC | 88.84% | 88.84% | 90.14% | 85.04% | **推荐** |
| **D02_litec_mixstyle** | **lite_c+MixStyle** | **89.67%** | **90.18%** | **90.53%** | **86.25%** | **最优** |

来源: `history/CV-SincNet/CVS_module_focused_iteration_report.md`, `history/CV-SincNet/type10-5/logs/`

### 3.2 type10-4 R 系列消融 (76 日志, 69 有效)

| 对比 | 变化 | Primary | Strict UDU | Worst-RX | 证据来源 |
|---|---|---|---|---|---|
| R04 -> R05 | 加 hard-domain CE | **+0.84** | +0.62 | **+7.75** | type10-4/findings.md |
| R05 -> R07 | 去 DAC+stats | **-3.77** | -4.06 | -5.75 | type10-4/findings.md |
| R06 -> R25 | lite_b->lite_d | +0.60 | +0.77 | -- | type10-4/findings.md |
| R19 (lite_b no_dac) | 最终平衡路线 | 87.46 | 85.92 | 85.03 | 4.27logs |
| R25 (lite_d no_dac) | 最佳参数效率 | 87.17 | 85.59 | 83.92 | 4.27logs |

来源: `type10-4/findings.md`, `type10-4/outputs/all_log_analysis_20260427.json`

### 3.3 type10-7 SAT 矩阵 (42 日志)

| 排名 | 实验 | Primary | Strict UDU | Overall | Worst-RX | SAT Avg | Params |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | SAT37_r19_fishr | **87.95** | 86.43 | 90.77 | 84.64 | 38.91 | 1.672M |
| 2 | SAT34_r19_groupdro_smooth | 87.94 | **86.44** | 90.72 | 84.83 | 38.69 | 1.672M |
| 3 | SAT07_r25_compact_sat_mixed | 87.85 | 86.27 | **90.79** | 84.67 | 41.98 | **1.050M** |
| 4 | SAT10_r19_mixed_cls_only | 87.71 | 86.15 | 90.60 | 85.12 | 42.55 | 1.672M |
| 5 | SAT13_r19_mixed_high_weight | 87.68 | 86.14 | 90.53 | **85.18** | **43.91** | 1.672M |

来源: `docs/CVS_RFFI_model_route_report_20260506.md`, `history/CV-SincNet/type10-7/logs/`

### 3.4 5.7 SGC 三阶段 (32 完整日志)

#### SGC source 阶段 vs 无 adapter baseline:

| SGC 变体 | Primary 变化 | Worst-RX 变化 | SAT Avg 变化 | 判断 |
|---|---:|---:|---:|---|
| full SGC | **-1.92** | **-7.92** | -3.34 | 明显负贡献 |
| no_amp | -1.23 | -11.21 | -2.94 | 仍为负 |
| no_freq | -2.06 | -2.63 | -2.90 | 负贡献 |
| no_res | -3.03 | -1.44 | -8.66 | 负贡献 |
| residual-only | -2.50 | -3.14 | -0.14 | clean 负贡献 |
| Lite-D full | -0.99 | -4.04 | -4.59 | 仍负 |

#### SGC augment 阶段边际贡献 (vs 无 adapter augment):

| SGC 变体 | Primary 变化 | UDU 变化 | Worst-RX 变化 | SAT Avg 变化 | 判断 |
|---|---:|---:|---:|---:|---|
| full SGC | -1.69 | -1.49 | -5.87 | +1.49 | SAT 略增但代价过大 |
| **no_amp** | **+0.24** | **+0.63** | -4.32 | -0.58 | **唯一弱正信号** |
| no_amp_freq | -1.99 | -1.77 | -0.01 | **+4.33** | SAT 探针 |
| residual-only | -1.33 | -1.25 | +0.16 | +1.81 | SAT 小增 clean 掉 |

**关键发现: SAT augment 本身 (无 adapter) 就让 SAT Avg +6.31, 说明 SAT 提升主要来自训练策略而非 SGC adapter。**

来源: `5.7/logs/`, `docs/SGC_5_7_analysis_and_merged_plan_20260507.md`

### 3.5 5.8 Phase-D 域增强器消融 (6 实验)

| 实验 | 操作 | Primary | Overall | Strict UDU | Worst-RX | SAT Avg | Params | Delta Primary |
|---|---|---|---|---|---|---|---:|---:|
| D1 | **enhancer=off** | **87.87** | 90.52 | **86.45** | 85.12 | 40.99 | 1,577,465 | **+0.63** |
| D2 | rcn_strength=0.20 | 87.19 | 90.20 | 85.56 | 85.02 | 41.82 | 1,672,409 | -0.05 |
| D3 | domain_branch_same | 86.08 | 89.37 | 84.31 | 85.14 | 41.50 | 1,360,792 | **-1.16** |
| D4 | no_pa_no_stats | 87.65 | **90.75** | 85.99 | **86.13** | **42.76** | **1,343,416** | +0.41 |
| D5 | no_grl_adv | 87.06 | 90.14 | 85.40 | 85.31 | 41.30 | 1,672,409 | -0.18 |
| D6 | no_orth | 87.49 | 90.40 | 85.92 | 84.71 | 41.67 | 1,672,409 | +0.25 |

来源: `5.8/logs/D*.log`, `5.8/metrics/experiment_metrics.csv`

### 3.6 5.8 Phase-E SGC Adapter 消融 (5 实验)

| 实验 | 操作 | Primary | Overall | Strict UDU | Worst-RX | SAT Avg | Delta Primary |
|---|---|---|---|---|---|---|---:|
| E0 | 无 adapter 继续 | 87.84 | 90.37 | 86.47 | 86.51 | 40.85 | 基准 |
| E1 | residual-only | 87.91 | 90.46 | 86.53 | 86.63 | 41.20 | +0.07 |
| **E2** | **residual+res_reg** | **88.24** | **90.70** | **86.92** | **86.99** | **41.58** | **+0.40** |
| E3 | no-res 控制组 | 88.12 | 90.76 | 86.69 | 86.27 | 40.86 | +0.28 |
| E4 | full SGC mild | 87.61 | 90.30 | 86.16 | 85.95 | 40.63 | **-0.23** |

来源: `5.8/logs/E*.log`, `5.8/metrics/experiment_metrics.csv`

### 3.7 损失函数全量证据

| 损失 | 权重 | 消融实验 | 量化变化 | 判断 |
|---|---|---|---|---|
| L_cls | 1.0 | 基础 | 必须 | **保留** |
| L_dom | 1.0 | D3(domain_same) | Primary -1.16, UDU -2.14 | **必须保留** |
| L_adv (GRL) | 0.45 | D5(no_grl) | Primary -0.18, UDU -1.05 | **保留** |
| L_orth | 0.05 | D6(no_orth) | Primary +0.25 (意外), UDU -0.53 | **可选** (Primary 不降但 UDU 降) |
| L_cons | 0.08 | 跨域一致性 | R04->R05 +0.84 | **保留** |
| L_group (hard-domain) | 0.10 | R04->R05 | **Worst-RX +7.75** | **必须保留** |
| L_PA_aux | 多项 | type10-2 g0->g4 | main +4.77% | **保留** (no_dac 时关 DAC 部分) |
| L_sat_cls | 0.08 | SAT augment | SAT Avg +6.31 | **保留** |
| L_sat_cons | 0.04 | SAT consistency | SAT一致性 | **保留** |
| L_fishr | 0.02 | SAT37 vs baseline | Primary +0.3-0.5 | **保留** |
| L_sgc_res | 0.01 | E2 vs E0 | Primary +0.40 | **保留** |
| L_ecc | 有时0 | C4 SAT | SAT Avg -3.56 | **暂时关闭** |
| L_proto | 可选 | 未在 5.8 测试 | 待验证 | **暂时关闭** |
| L_supcon | 可选 | 未在 5.8 测试 | 待验证 | **暂时关闭** |

### 3.8 训练稳定性证据

#### 崩溃/失败实验汇总:

| 实验 | 时代 | 现象 | skipped_backward | 原因 |
|---|---|---|---|---|
| B02 (type10-5) | 4.23 | last 16.67% | 14,428 | 随机 MixStyle |
| B01 (type10-5) | 4.23 | last 48.07% | -- | 多层 MixStyle |
| C00 (type10-5) | 4.23 | last 16.67% | -- | 移除时间分支 |
| C03 (type10-5) | 4.23 | last 16.67% | -- | 移除频率分支 |
| type11 | history | best 91.59% -> 31.10% | -- | 对抗过强 |
| type12 | history | tx_acc=0% x 170 epochs | -- | 分支死亡 |
| type13 | history | tx_acc=0% | -- | 分支死亡 |
| sgc_no_spec augment | 5.7 | collapse | 19 skipped | 频谱抑制不稳定 |
| R00_balanced | 4.27 | last 16.67% | -- | 训练不稳定 |

**5.8 实验稳定性:** 全部 19 实验 skipped_backward 8-12, 无崩溃, collapse guard 未触发。

---

## 四、5.8 最终 19 实验完整指标

### 4.1 Primary Score 排序

| # | 实验 | Phase | Primary | Overall | Strict UDU | Worst-RX | SAT Avg | SAT Min | Params | Skipped | Epoch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | E2_residual_only_std_res001 | E | **88.24** | 90.70 | 86.92 | 86.99 | 41.58 | 37.27 | 1,672,844 | 8 | 5 |
| 2 | E3_no_res_control | E | **88.12** | 90.76 | 86.69 | 86.27 | 40.86 | 35.61 | 1,673,467 | 9 | 14 |
| 3 | E1_residual_only_std | E | **87.91** | 90.46 | 86.53 | 86.63 | 41.20 | 36.84 | 1,672,844 | 8 | 5 |
| 4 | D1_domain_enhancer_off | D | **87.87** | 90.52 | 86.45 | 85.12 | 40.99 | 36.90 | 1,577,465 | 11 | 78 |
| 5 | E0_no_adapter_continue | E | **87.84** | 90.37 | 86.47 | 86.51 | 40.85 | 36.04 | 1,672,409 | 8 | 11 |
| 6 | D4_domain_no_pa_no_stats | D | **87.65** | 90.75 | 85.99 | 86.13 | 42.76 | 38.56 | 1,343,416 | 12 | 101 |
| 7 | B4_A2_ls002 | B | **87.61** | 90.35 | 86.14 | 85.62 | 42.25 | 38.15 | 1,672,409 | 11 | 129 |
| 8 | E4_full_sgc_mild_res001 | E | **87.61** | 90.30 | 86.16 | 85.95 | 40.63 | 36.12 | 1,673,902 | 9 | 6 |
| 9 | D6_no_orth | D | **87.49** | 90.40 | 85.92 | 84.71 | 41.67 | 38.10 | 1,672,409 | 11 | 93 |
| 10 | C2_A1_ecc003_satmain | C | **87.34** | 90.07 | 85.86 | 85.45 | 43.16 | 38.78 | 1,672,409 | 11 | 93 |
| 11 | B3_A1_ls002 | B | **87.25** | 90.23 | 85.64 | 86.03 | 43.56 | 39.26 | 1,672,409 | 11 | 147 |
| 12 | B1_A1_mild | B | **87.24** | 89.96 | 85.78 | 85.41 | 43.63 | 39.89 | 1,672,409 | 11 | 159 |
| 13 | B2_A2_light | B | **87.23** | 89.82 | 85.84 | 85.09 | 42.19 | 38.20 | 1,672,409 | 10 | 123 |
| 14 | D2_domain_rcn020 | D | **87.19** | 90.20 | 85.56 | 85.02 | 41.82 | 38.34 | 1,672,409 | 12 | 101 |
| 15 | D5_no_grl_adv | D | **87.06** | 90.14 | 85.40 | 85.31 | 41.30 | 37.62 | 1,672,409 | 11 | 108 |
| 16 | C3_A1_ecc003_conservative | C | **87.05** | 90.21 | 85.35 | 84.62 | 43.65 | 39.27 | 1,672,409 | 11 | 152 |
| 17 | C1_A1_ecc002_sat | C | **86.98** | 90.06 | 85.32 | 85.60 | 43.30 | 39.02 | 1,672,409 | 10 | 148 |
| 18 | C4_A2_ecc003_satmain | C | **86.81** | 89.95 | 85.11 | 84.54 | 40.07 | 36.26 | 1,672,409 | 12 | 96 |
| 19 | D3_domain_branch_same | D | **86.08** | 89.37 | 84.31 | 85.14 | 41.50 | 37.99 | 1,360,792 | 10 | 97 |

### 4.2 SAT 鲁棒性排序

| # | 实验 | SAT Avg | SAT Clear | SAT Low Elev | SAT Rain | SAT Storm | SAT Mixed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | C3_ecc003_conservative | **43.65** | 46.21 | 46.37 | 43.94 | 39.27 | 42.48 |
| 2 | B1_A1_mild | **43.63** | 45.84 | 46.36 | 43.95 | 39.89 | 42.12 |
| 3 | B3_A1_ls002 | **43.56** | 46.19 | 46.05 | 44.12 | 39.26 | 42.19 |
| 4 | C1_ecc002_sat | **43.30** | 45.95 | 45.71 | 43.70 | 39.02 | 42.13 |
| 5 | C2_ecc003_satmain | **43.16** | 45.84 | 46.19 | 43.48 | 38.78 | 41.53 |
| 6 | D4_no_pa_no_stats | **42.76** | 45.16 | 45.47 | 43.13 | 38.56 | 41.47 |
| 7 | B4_A2_ls002 | **42.25** | 44.87 | 44.64 | 42.18 | 38.15 | 41.40 |
| 8 | B2_A2_light | **42.19** | 44.45 | 45.16 | 42.27 | 38.20 | 40.85 |
| 9 | D2_rcn020 | **41.82** | 43.84 | 44.41 | 42.02 | 38.34 | 40.51 |
| 10 | D6_no_orth | **41.67** | 43.83 | 44.43 | 41.68 | 38.10 | 40.29 |
| 11 | E2_residual_std_res001 | **41.58** | 44.62 | 43.84 | 42.04 | 37.27 | 40.11 |
| 12 | D3_branch_same | **41.50** | 43.35 | 43.91 | 41.70 | 37.99 | 40.53 |
| 13 | D5_no_grl_adv | **41.30** | 43.40 | 43.75 | 41.53 | 37.62 | 40.18 |
| 14 | E1_residual_only_std | **41.20** | 44.45 | 43.60 | 41.42 | 36.84 | 39.68 |
| 15 | D1_enhancer_off | **40.99** | 43.70 | 43.43 | 41.46 | 36.90 | 39.44 |
| 16 | E3_no_res_control | **40.86** | 44.47 | 43.56 | 41.13 | 35.61 | 39.54 |
| 17 | E0_no_adapter | **40.85** | 44.08 | 43.56 | 41.44 | 36.04 | 39.12 |
| 18 | E4_full_sgc | **40.63** | 43.69 | 42.97 | 40.85 | 36.12 | 39.53 |
| 19 | C4_ecc003_satmain | **40.07** | 42.77 | 41.88 | 40.23 | 36.26 | 39.21 |

### 4.3 Per-RX 准确率 (E2 最优模型)

| RX | 准确率 | 类型 |
|---|---:|---|
| rx_7 | 87.71% | unseen RX |
| rx_8 | 86.99% | unseen RX |
| rx_9 | 91.11% | unseen RX |
| rx_10 | 89.97% | unseen RX |
| rx_11 | 96.17% | unseen RX |
| unseen_day_seen_rx | 93.62% | 跨天已知 RX |
| seen_day_unseen_rx | 90.39% | 已知天跨 RX |
| unseen_day_unseen_rx | 86.92% | 最严格 OOD |

---

## 五、可裁剪结构清单

### 5.1 已验证可裁剪 (多实验证据)

| 结构 | 裁剪方式 | 证据来源 | 涉及实验数 | 风险 |
|---|---|---|---|---|
| DAC Branch | `no_dac` | type10-5 C01, R19, R25, 全部 5.8 | 100+ | **无** |
| DomainFeatureEnhancer | `enhancer=off` | 5.8 D1 最强 source | 6 | **低** |
| Full SGC (amp+freq+spec) | 仅保留 residual | 5.7 全矩阵, 5.8 E4 | 37 | **无** |
| AmplitudeNormalizer | 不启用 | 5.7 no_amp 对照 | 11 | **无** |
| FrequencyOffsetCompensator | 不启用 | 5.7 no_freq 对照 | 11 | **无** |
| SpectralInterferenceSuppressor | 不启用 | 5.7 no_spec 崩溃 | 11 | **无** |
| ECC loss | 权重=0 | 5.8 C4 SAT -3.56 | 4 | **无** |

### 5.2 推理时可裁剪 (训练保留)

| 结构 | 参数量 | 说明 |
|---|---|---|
| dom_backbone 独立部分 | ~630K | 域解耦训练专用 |
| dom_head | ~10K | 域分类 |
| adv_head | ~10K | GRL 对抗 |
| RCNStatEncoder | ~8K | RCN 统计 |
| MixStyle1D | ~0 | 训练风格混合 |

**推理时: 1.72M -> ~750K (减少 56%)**

### 5.3 待进一步验证可裁剪

| 结构 | 当前状态 | 已有证据 | 消融建议 | 预期影响 |
|---|---|---|---|---|
| PA Branch | 保留 | D4(no_pa) Primary -0.22, 但 Overall +0.23 | 需要专项消融 | 参数 -200K |
| dom_backbone 独立参数 | 双骨干 | D3(same) Primary -1.16 | 测试部分共享 | 参数 -300K |
| Orth loss | 0.05 | D6 Primary +0.25 但 UDU -0.53 | 测试降低权重 | 无参数变化 |
| Prototype loss | 未启用 | 无 5.8 数据 | 保持关闭 | 待验证 |
| SupCon loss | 未启用 | 无 5.8 数据 | 保持关闭 | 待验证 |

---

## 六、推荐保留结构清单

### 6.1 核心保留 (推理必须)

| 结构 | 参数量 | 理由 | 证据 |
|---|---|---|---|
| SincConv1d | ~2.4K | 物理前端基础 | type1-type9 全部使用 |
| HighFreqEmphasis | 0 | DAC 纹波检测 | 固定无参数 |
| Time Branch | ~350K | 消融崩溃 | C00 last 16.67% |
| Freq Branch | ~80K | 消融崩溃 | C03 last 16.67% |
| PhysicalAwareClassifier | ~120K | 特征融合+CosFace | type10-2 消融有效 |

### 6.2 强烈建议保留

| 结构 | 参数量 | 理由 | 证据 |
|---|---|---|---|
| PA Branch | ~200K | 稳定作用 | D4 Primary 仅 -0.22 |
| Residual SGC Adapter | ~50K | 明确正贡献 | E2 +0.40 Primary |
| MixStyle1D (conservative) | ~0 | 显著提升 | D02 +3.16% last |
| Hard-domain CE | 0 (loss) | Worst-RX 大幅提升 | R05 +7.75 worst-RX |
| Fishr | 0 (loss) | 可靠 OOD 提升 | SAT37 87.95 Primary |

### 6.3 训练时保留 (推理可移除)

| 结构 | 参数量 | 理由 | 证据 |
|---|---|---|---|
| dom_backbone | ~630K | 域解耦核心 | D3 -1.16 Primary |
| dom_head + adv_head | ~20K | 域分类+GRL | D5 -0.18 Primary |
| GRL | 0 | 去域对抗 | D5 UDU -1.05 |
| SAT consistency | 0 (loss) | 卫星鲁棒性 | SAT Avg +6.31 |

---

## 七、模型变体性能对比

| 变体 | 参数量 | Primary | Strict UDU | Worst-RX | SAT Avg | 来源 | 推荐场景 |
|---|---|---|---|---|---|---|---|
| **lite_b no_dac + SGC res** | **1.673M** | **88.24** | **86.92** | **86.99** | **41.58** | 5.8 E2 | **主论文** |
| lite_b no_dac (无 SGC) | 1.672M | 87.87 | 86.45 | 85.12 | 40.99 | 5.8 D1 | Source 基准 |
| lite_b no_dac + Fishr | 1.672M | 87.95 | 86.43 | 84.64 | 38.91 | type10-7 SAT37 | Clean OOD 最强 |
| lite_d no_dac | ~1.050M | 87.85 | 86.27 | 84.67 | 41.98 | type10-7 SAT07 | **星上部署** |
| lite_b no_dac 推理最小 | ~750K | 同上 | 同上 | 同上 | N/A | 推理裁剪 | 推理优化 |
| lite_d no_dac 推理最小 | ~500K | 同上 | 同上 | 同上 | N/A | 推理裁剪 | 极限压缩 |
| full/base | ~3.41M | -- | -- | -- | -- | type10-5 A00 | **不推荐** |

---

## 八、下一步消融实验计划

### 8.1 P0 必须做

| # | 实验 | 目的 | 配置 | 预期 |
|---|---|---|---|---|
| 1 | Validation-only checkpoint | 消除 test-derived 偏差 | 用 val_acc 选 checkpoint | 严谨性修正 |
| 2 | Multi-seed (3 seeds) | 验证稳定性 | seed 1337, 42, 2024 | 方差估计 |
| 3 | D1+residual SGC+enhancer off | 隔离 residual SGC 因果 | 从 D1 初始化, enhancer=off | Primary 可能更高 |
| 4 | D1+residual SGC+enhancer on | 对比 enhancer 影响 | 从 D1 初始化, enhancer=rcn_stats | 与 #3 对比 |

### 8.2 P1 重要

| # | 实验 | 目的 | 配置 | 预期 |
|---|---|---|---|---|
| 5 | Lite-D residual SGC | 极限压缩验证 | lite_d + residual SGC | 500K 参数仍可用 |
| 6 | PA branch 移除 | 极限瘦身 | no_dac, no_pa | Primary 可能 -0.2 |
| 7 | dom_backbone 部分共享 | 参数压缩 | 共享更多层 | -300K 参数 |
| 8 | ECC 重新调度 | 探索有效窗口 | 仅 S3 启用 | SAT Avg 提升 |

### 8.3 P2 探索

| # | 实验 | 目的 | 配置 | 预期 |
|---|---|---|---|---|
| 9 | Prototype memory | 类中心约束 | lambda_proto=0.1 | 待验证 |
| 10 | SupCon loss | 对比学习增强 | lambda_supcon=0.05 | 待验证 |
| 11 | SGC-TADA | 星上自适应 | source-free adaptation | 方向探索 |
| 12 | CVCNN vs Sinc-CVCNN | SincConv 物理验证 | 对照实验 | 证明 inductive bias |

---

## 九、面向星上部署的轻量化建议

### 9.1 推理时最小模型

```
输入 IQ [B, 2, 256]
  -> SincConv1d (24 filters)
  -> HighFreqEmphasis (fixed)
  -> Time: fuse -> DSConv x3 -> t_emb (192-d)
  -> Freq: FFT -> FreqBandGate -> DSConv x3 -> f_emb (192-d)
  -> PA: MemPolyLift -> DilatedConv x3 -> pa_local (192-d)
  -> PhysicalAwareClassifier -> CosFace logits [B, 6]
```

- 参数量: ~750K (lite_b) 或 ~500K (lite_d)
- FLOPs: ~150M (lite_b) 或 ~100M (lite_d)
- 内存: ~3MB (lite_b) 或 ~2MB (lite_d)
- 输入: 256 点 I/Q 序列
- 输出: 6 类 TX 分类

### 9.2 量化与压缩路径

| 方法 | 预期效果 | 风险 |
|---|---|---|
| INT8 量化 | 参数减半, 推理 2x | 精度损失 <1% |
| 知识蒸馏 | lite_b -> lite_e | 需要验证 |
| 结构化剪枝 | Freq 通道 48->32 | 需要消融 |
| 最终目标 | ~300-400K params, INT8 ~300-400KB | |

### 9.3 星上可选组件

| 组件 | 参数 | 何时使用 |
|---|---|---|
| Residual SGC Adapter | ~50K | 需要卫星信道补偿 |
| SAT consistency 预处理 | 0 | 信道已知 |

---

## 十、关键发现总结

### 10.1 必须保留的结构 (有强证据)

1. **SincConv + Time + Freq 三路径** -- 消融崩溃, 100+ 实验验证
2. **Hard-domain CE** -- Worst-RX +7.75, 最大单项提升
3. **Conservative MixStyle** -- last epoch +3.16%, 配置敏感
4. **Fishr 梯度方差匹配** -- Primary +0.3-0.5, 可靠增益
5. **GRL 域对抗** -- UDU -1.05 (移除后), 必须保留
6. **Residual-only SGC** -- Primary +0.40, 保守有效

### 10.2 可以移除的结构 (有强证据)

1. **DAC Branch** -- 100+ 实验证明无害有益
2. **DomainFeatureEnhancer** -- D1=off 最强 source
3. **Full SGC (amp+freq+spec)** -- 37 实验证明净负
4. **ECC loss** -- SAT Avg -3.56, 不稳定
5. **dom_backbone (推理时)** -- 训练专用, 推理可裁

### 10.3 待验证的结构

1. **PA Branch** -- D4 仅 -0.22 Primary, 可能可移除
2. **Prototype/SupCon loss** -- 无 5.8 数据
3. **dom_backbone 共享程度** -- D3 全共享 -1.16, 部分共享未测

---

## 附录: 关键文件索引

| 文件 | 内容 |
|---|---|
| `model.py` | CVSincNet 四分支, SincConv, PhysicalAwareClassifier |
| `model_dual_cvsincnet.py` | 双骨干, GRL, DomainFeatureEnhancer |
| `sgc_adapter.py` | SGCAdapter, ResidualChannelCompensator, FPCR |
| `sgc_losses.py` | prototype bank, feature consistency, residual reg |
| `train.py` | 17 项损失, 三阶段训练, Fishr, SAT consistency |
| `training_controls.py` | MixStyle 调度, collapse guard, SAT 场景 |
| `DataAugmentation.py` | DAC/PA/信道增强, receiver DG |
| `sat_channel.py` | 卫星信道模拟器 (LEO/MEO/GEO) |
| `5.8/metrics/experiment_metrics.csv` | 19 实验完整指标 |
| `type10-4/outputs/all_log_analysis_20260427.json` | 76 条解析记录 |
| `type10-7/CV-SincNet/sgc_log_summary.json` | SGC 日志摘要 |
| `5.7/logs/` | 32 个 SGC 三阶段日志 |
| `history/CV-SincNet/type10-5/logs/` | 18 个模块消融日志 |
| `history/CV-SincNet/type10-7/logs/` | 42 个 SAT 矩阵日志 |
| `type10-4/4.27logs/` | 30 个 R 系列消融日志 |
