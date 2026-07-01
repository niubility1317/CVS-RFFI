# Stage2 OPGAC-Net替换未知类评估报告

- 实验ID：`stage2_opgac_jrefc9_unknown_alt_20260701_1126`
- 时间：2026-07-01
- 操作者：Codex
- 目标：在保持地面模型、旧类、新类、target receiver、K-shot和OPGAC评估脚本不变的前提下，将unknown TX从`10-1,10-10`替换为`1-10,1-12`，测试未知类选择是否显著影响Unknown FAR。

## 协议边界

- 地面模型：`JREF_C9_MULTICOMP_M2_E220`
- checkpoint：`runs/phase1_jointmain_refine_20260630/JREF_C9_MULTICOMP_M2_E220/best_joint_safe_ssdg.pth`
- source receiver：`1-1,1-19,14-7,18-2,19-2,2-1,2-19`
- target receiver：`3-19,7-14,7-7,8-8`
- 旧类`Y_old`：`14-10,14-7,20-15,20-19,6-15,8-20`
- seen-new`Y_new`：`1-16,1-18`
- 替换后的unknown`Y_unknown`：`1-10,1-12`
- K-shot：target-old每TX`10`个support；target-new每TX`10`个support。
- query：target-old每TX`50`个；target-new每TX`30`个；unknown每TX`30`个。
- 权限边界：target query和unknown query只用于最终评估，不参与原型、阈值、能量或半径校准。

## 本地状态与验证

| 项目 | 结果 |
|---|---|
| N607预检 | direct `N607`通过；项目根和GPU可见 |
| 主工作区Git | `E:\type10-7`不是Git仓 |
| 发布仓Git | `github_publish/CVS-RFFI-repo`干净，当前分支ahead 1 |
| OPGAC评估脚本 | `tools/evaluate_opgac_stage2.py`存在，SHA256=`A0D4C0BA3843B7C545037347C7277A958301C9A04A56D4615CAB32F3A90591F4` |
| 特征导出脚本 | `code/export_spaceborne_features.py`存在，SHA256=`40BE1197980CDFAE131252C9764DE8A51830CF533AD5A54B48511E0B9DD57A6E` |

## 计划命令

远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`

远端输出根：`runs/stage2_opgac_jrefc9_unknown_alt_20260701_1126`

核心参数：

```bash
SOURCE_TX="14-10,14-7,20-15,20-19,6-15,8-20"
NEW_TX="1-16,1-18"
UNKNOWN_TX="1-10,1-12"
TARGET_DOMAINS="3-19 7-14 7-7 8-8"
SCENARIOS="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
```

导出脚本会显式传入`--sample_rate_hz 25000000`，因为该checkpoint内记录的`sample_rate_hz=0.0`不能直接用于SincConv重建。

## 结果

### 样本审计

| 域 | unknown TX | 可用样本 | query-per-TX=30 |
|---|---|---:|---|
| `3-19` | `1-10,1-12` | `1-10:200,1-12:200` | PASS |
| `7-14` | `1-10,1-12` | `1-10:200,1-12:200` | PASS |
| `7-7` | `1-10,1-12` | `1-10:200,1-12:200` | PASS |
| `8-8` | `1-10,1-12` | `1-10:200,1-12:200` | PASS |

互斥审计：`source_new=[]`、`source_unknown=[]`、`new_unknown=[]`。四个域的`split_overlap_audit`均为空。

### 远端输出

- `runs/stage2_opgac_jrefc9_unknown_alt_20260701_1126/opgac_summary.csv`
- `runs/stage2_opgac_jrefc9_unknown_alt_20260701_1126/opgac_eval.json`
- `runs/stage2_opgac_jrefc9_unknown_alt_20260701_1126/opgac_summary_q95.csv`
- `runs/stage2_opgac_jrefc9_unknown_alt_20260701_1126/opgac_eval_q95.json`

本地回收：

- `automation_reports/CV-SincNet/stage2_opgac_jrefc9_unknown_alt_20260701_1126/artifacts/opgac_summary.csv`
- `automation_reports/CV-SincNet/stage2_opgac_jrefc9_unknown_alt_20260701_1126/artifacts/opgac_eval.json`
- `automation_reports/CV-SincNet/stage2_opgac_jrefc9_unknown_alt_20260701_1126/artifacts/opgac_summary_q95.csv`
- `automation_reports/CV-SincNet/stage2_opgac_jrefc9_unknown_alt_20260701_1126/artifacts/opgac_eval_q95.json`

## 结果表：q99默认门控

| 域 | 变体 | Old acc | Seen-new acc | Known acc | Coverage | Unknown FAR | Full acc | 无拒识Known acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `3-19` | strict | 53.67% | 0.00% | 44.72% | 93.10% | 95.00% | 39.05% | 36.94% |
| `3-19` | confirm-new | 44.33% | 51.67% | 45.56% | 92.62% | 90.00% | 40.48% | 36.94% |
| `7-14` | strict | 74.00% | 0.00% | 61.67% | 93.57% | 98.33% | 53.10% | 57.78% |
| `7-14` | confirm-new | 71.33% | 33.33% | 65.00% | 92.62% | 93.33% | 56.67% | 57.78% |
| `7-7` | strict | 72.67% | 0.00% | 60.56% | 89.76% | 86.67% | 53.81% | 54.72% |
| `7-7` | confirm-new | 68.67% | 35.00% | 63.06% | 92.38% | 90.00% | 55.48% | 54.72% |
| `8-8` | strict | 74.00% | 0.00% | 61.67% | 96.43% | 100.00% | 52.86% | 63.61% |
| `8-8` | confirm-new | 72.67% | 8.33% | 61.94% | 95.71% | 95.00% | 53.81% | 63.61% |

均值：

| 变体 | Old acc | Seen-new acc | Known acc | Coverage | Unknown FAR | Full acc | 无拒识Known acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict | 68.58% | 0.00% | 57.15% | 93.21% | 95.00% | 49.70% | 53.26% |
| confirm-new | 64.25% | 32.08% | 58.89% | 93.33% | 92.08% | 51.61% | 53.26% |

## 结果表：q95保守门控

| 域 | 变体 | Old acc | Seen-new acc | Known acc | Coverage | Unknown FAR | Full acc | 无拒识Known acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `3-19` | strict | 40.67% | 0.00% | 33.89% | 52.62% | 43.33% | 37.14% | 36.94% |
| `3-19` | confirm-new | 34.67% | 50.00% | 37.22% | 63.33% | 50.00% | 39.05% | 36.94% |
| `7-14` | strict | 65.33% | 0.00% | 54.44% | 73.57% | 75.00% | 50.24% | 57.78% |
| `7-14` | confirm-new | 64.67% | 31.67% | 59.17% | 80.00% | 81.67% | 53.33% | 57.78% |
| `7-7` | strict | 64.33% | 0.00% | 53.61% | 74.76% | 71.67% | 50.00% | 54.72% |
| `7-7` | confirm-new | 62.33% | 33.33% | 57.50% | 82.38% | 78.33% | 52.38% | 54.72% |
| `8-8` | strict | 67.33% | 0.00% | 56.11% | 81.67% | 93.33% | 49.05% | 63.61% |
| `8-8` | confirm-new | 66.33% | 8.33% | 56.67% | 82.14% | 88.33% | 50.24% | 63.61% |

均值：

| 变体 | Old acc | Seen-new acc | Known acc | Coverage | Unknown FAR | Full acc | 无拒识Known acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict | 59.42% | 0.00% | 49.51% | 70.65% | 70.83% | 46.61% | 53.26% |
| confirm-new | 57.00% | 30.83% | 52.64% | 76.96% | 74.58% | 48.75% | 53.26% |

## 与上一组unknown对比

上一组unknown为`10-1,10-10`。本组unknown为`1-10,1-12`。

| 阈值 | 变体 | 旧unknown FAR | 新unknown FAR | 变化 |
|---|---|---:|---:|---:|
| q99 | strict | 80.42% | 95.00% | +14.58 pp |
| q99 | confirm-new | 83.33% | 92.08% | +8.75 pp |
| q95 | strict | 41.25% | 70.83% | +29.58 pp |
| q95 | confirm-new | 53.33% | 74.58% | +21.25 pp |

旧类和seen-new指标基本不变，因为source、target-old、target-new划分没有变化；变化集中在unknown FAR。这说明`1-10,1-12`比`10-1,10-10`更靠近当前旧类/seen-new接受区，是更难的未知类组合。

## 结论

替换unknown后性能更差，且差异主要出现在Unknown FAR，而不是Old acc或Seen-new acc。该结果进一步支持前一轮判断：当前失败不是因为unknown里混入旧类，而是开放集边界本身不稳定；不同非旧TX与旧类/seen-new的几何距离差异很大，`1-10,1-12`明显落在OPGAC表的接受区内。

下一步若继续验证，应做unknown TX sweep：固定旧类和seen-new，遍历多组`Y_unknown`，输出每个unknown TX的最近旧类、接受率、能量分布和距离分位数。这样可以区分“某些unknown特别难”与“整体拒识机制失效”。
