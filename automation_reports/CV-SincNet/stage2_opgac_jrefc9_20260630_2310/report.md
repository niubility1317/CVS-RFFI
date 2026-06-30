# Stage2 OPGAC-Net评估报告

- 实验ID：`stage2_opgac_jrefc9_20260630_2310`
- 时间：2026-06-30
- 操作者：Codex
- 目标：以地面训练模型`JREF_C9_MULTICOMP_M2_E220`为冻结主干，测试阶段二`OPGAC-Net`在target receiver域的旧类域适应、新类注册识别、未知类拒识效果。

## 协议边界

- 阶段：Stage2-C。
- 地面模型：`runs/phase1_jointmain_refine_20260630/JREF_C9_MULTICOMP_M2_E220/best_joint_safe_ssdg.pth`。
- 地面旧类：`14-10,14-7,20-15,20-19,6-15,8-20`。
- source receiver：`1-1,1-19,14-7,18-2,19-2,2-1,2-19`。
- target receiver：本次计划优先跑`3-19,7-14,7-7,8-8`，延续前序“除20-1外的剩余域”比较口径；脚本支持后续补跑`20-1`。
- target-new TX：`1-16,1-18`。
- unknown TX：`10-1,10-10`。
- K-shot：旧类每TX`10`个target-old support；新类每TX`10`个target-new support。
- query：target-old每TX`50`个；target-new每TX`30`个；unknown每TX`30`个。原因是`10-1`在部分target receiver下过滤后只满足30个unknown query。
- 严格边界：support只用于旧类OPGAC校准和新类注册；target query与unknown query只用于最终测试，不参与原型、半径、能量阈值或密度门校准。

## OPGAC-Net评估实现

- 新增本地脚本：`tools/evaluate_opgac_stage2.py`。
- 输入：`export_spaceborne_features.py`导出的`z_id`特征NPZ。
- 旧类表：从source特征建立多组件对角高斯，默认按source receiver分组件并限制每类最多4个组件，随后用target-old support做support-only校准。
- 新类表：用target-new support追加注册为seen-new高斯状态。
- 拒识头：使用OPGAC能量、类内半径、old-new margin、top2 margin共同决定`old_class/new_class/unknown/ambiguous`。
- 输出：JSON全量诊断和CSV汇总，包括带拒识`old_acc/seen_new_acc/coverage/unknown_far/full_acc`，以及关闭拒识的`no_reject_*`指标。
- 评估变体：
  - `opgac_strict`：保留新类overlap生命周期判断。
  - `opgac_confirm_new`：诊断变体，强制已给support的新类进入confirmed，检查新类注册能力是否被overlap保护过度压制。

## 本地变更与验证

| 文件 | 目的 | 本地验证 |
|---|---|---|
| `code/export_spaceborne_features.py` | 修复直接执行时`code/cvsrffi`被根目录旧包遮蔽的问题 | `python -m py_compile code/export_spaceborne_features.py`通过 |
| `tools/evaluate_opgac_stage2.py` | 新增OPGAC Stage2-C特征级评估入口 | `python -m py_compile tools/evaluate_opgac_stage2.py`通过；合成NPZ smoke test通过 |

## N607同步计划

| 本地文件 | N607目标 |
|---|---|
| `code/export_spaceborne_features.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/export_spaceborne_features.py` |
| `code/cvsrffi/opgac_net.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/opgac_net.py` |
| `tools/evaluate_opgac_stage2.py` | `/home/szu2070436088/2510044040/CV-SincNet/tools/evaluate_opgac_stage2.py` |

## 远端命令计划

工作目录：`/home/szu2070436088/2510044040/CV-SincNet`。

Conda/Python：`~/.conda/envs/CVS-RFFI/bin/python`。

输出根目录：`runs/stage2_opgac_jrefc9_20260630_2310`。

每个target receiver先导出特征：

```bash
CUDA_VISIBLE_DEVICES=4 ~/.conda/envs/CVS-RFFI/bin/python code/export_spaceborne_features.py \
  --ckpt runs/phase1_jointmain_refine_20260630/JREF_C9_MULTICOMP_M2_E220/best_joint_safe_ssdg.pth \
  --wisig_pkl Dataset_WigSig/ManySig.pkl \
  --new_wisig_pkl Dataset_WigSig/ManyTx.pkl \
  --out_npz runs/stage2_opgac_jrefc9_20260630_2310/<domain>/features.npz \
  --feature_name z_id \
  --source_tx_ids 14-10,14-7,20-15,20-19,6-15,8-20 \
  --target_old_tx_ids 14-10,14-7,20-15,20-19,6-15,8-20 \
  --new_tx_ids 1-16,1-18 \
  --unknown_tx_ids 10-1,10-10 \
  --source_rxs 1-1,1-19,14-7,18-2,19-2,2-1,2-19 \
  --target_old_rxs <domain> \
  --new_rxs <domain> \
  --max_samples_per_tx 400 \
  --batch_size 512 \
  --device cuda:0 \
  --sample_rate_hz 25000000 \
  --target_old_channel_view satellite \
  --target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --target_new_channel_view satellite \
  --target_new_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --star_ground_channel_impl simplified_leo_residual \
  --seed 362017
```

然后合并评估：

```bash
~/.conda/envs/CVS-RFFI/bin/python tools/evaluate_opgac_stage2.py \
  --feature-npz 'runs/stage2_opgac_jrefc9_20260630_2310/*/features.npz' \
  --source-tx-ids 14-10,14-7,20-15,20-19,6-15,8-20 \
  --target-old-tx-ids 14-10,14-7,20-15,20-19,6-15,8-20 \
  --new-tx-ids 1-16,1-18 \
  --unknown-tx-ids 10-1,10-10 \
  --shots 10 \
  --target-old-support-per-tx 10 \
  --target-old-query-per-tx 50 \
  --query-per-tx 30 \
  --source-proto-per-tx 240 \
  --component-mode rx \
  --max-components-per-class 4 \
  --output-json runs/stage2_opgac_jrefc9_20260630_2310/opgac_eval.json \
  --summary-csv runs/stage2_opgac_jrefc9_20260630_2310/opgac_summary.csv
```

## 执行记录

- N607预检：直接`N607`可访问，项目根目录与GPU可见。
- 文件同步：`export_spaceborne_features.py`、`opgac_net.py`、`evaluate_opgac_stage2.py`均已同步到N607并通过`py_compile`。
- 远端导出：4个target receiver均导出成功，`z_id`维度为160。
- 导出异常修复：`JREF_C9_MULTICOMP_M2_E220`checkpoint中`sample_rate_hz=0.0`，直接重建会触发`sample_rate too low or min_band_hz too large`，最终导出命令显式加入`--sample_rate_hz 25000000`。
- 远端评估异常修复：N607当前NumPy/MKL组合在`np.quantile`、`np.isfinite`、布尔向量路径上出现segfault或dtype冲突，评估脚本已改为纯Python分位数和逐样本计数。
- 远端输出：
  - `runs/stage2_opgac_jrefc9_20260630_2310/opgac_summary.csv`
  - `runs/stage2_opgac_jrefc9_20260630_2310/opgac_eval.json`
  - `runs/stage2_opgac_jrefc9_20260630_2310/opgac_summary_q95.csv`
  - `runs/stage2_opgac_jrefc9_20260630_2310/opgac_eval_q95.json`
- 本地回收：
  - `automation_reports/CV-SincNet/stage2_opgac_jrefc9_20260630_2310/artifacts/opgac_summary.csv`
  - `automation_reports/CV-SincNet/stage2_opgac_jrefc9_20260630_2310/artifacts/opgac_eval.json`
  - `automation_reports/CV-SincNet/stage2_opgac_jrefc9_20260630_2310/artifacts/opgac_summary_q95.csv`
  - `automation_reports/CV-SincNet/stage2_opgac_jrefc9_20260630_2310/artifacts/opgac_eval_q95.json`

## 结果表：q99默认门控

| 域 | 变体 | Old acc | Seen-new acc | Known acc | Coverage | Unknown FAR | Full acc | 无拒识Known acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `3-19` | strict | 53.67% | 0.00% | 44.72% | 89.05% | 66.67% | 43.10% | 36.94% |
| `3-19` | confirm-new | 44.33% | 51.67% | 45.56% | 89.52% | 68.33% | 43.57% | 36.94% |
| `7-14` | strict | 74.00% | 0.00% | 61.67% | 91.43% | 83.33% | 55.24% | 57.78% |
| `7-14` | confirm-new | 71.33% | 33.33% | 65.00% | 92.38% | 91.67% | 56.90% | 57.78% |
| `7-7` | strict | 72.67% | 0.00% | 60.56% | 88.57% | 78.33% | 55.00% | 54.72% |
| `7-7` | confirm-new | 68.67% | 35.00% | 63.06% | 90.95% | 80.00% | 56.90% | 54.72% |
| `8-8` | strict | 74.00% | 0.00% | 61.67% | 95.48% | 93.33% | 53.81% | 63.61% |
| `8-8` | confirm-new | 72.67% | 8.33% | 61.94% | 95.48% | 93.33% | 54.05% | 63.61% |

均值：

| 变体 | Old acc | Seen-new acc | Known acc | Unknown FAR | Full acc | 无拒识Known acc |
|---|---:|---:|---:|---:|---:|---:|
| strict | 68.58% | 0.00% | 57.15% | 80.42% | 51.79% | 53.26% |
| confirm-new | 64.25% | 32.08% | 58.89% | 83.33% | 52.86% | 53.26% |

## 结果表：q95保守门控

| 域 | 变体 | Old acc | Seen-new acc | Known acc | Coverage | Unknown FAR | Full acc | 无拒识Known acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `3-19` | strict | 40.67% | 0.00% | 33.89% | 48.33% | 13.33% | 41.43% | 36.94% |
| `3-19` | confirm-new | 34.67% | 50.00% | 37.22% | 60.48% | 30.00% | 41.90% | 36.94% |
| `7-14` | strict | 65.33% | 0.00% | 54.44% | 69.52% | 46.67% | 54.29% | 57.78% |
| `7-14` | confirm-new | 64.67% | 31.67% | 59.17% | 78.33% | 70.00% | 55.00% | 57.78% |
| `7-7` | strict | 64.33% | 0.00% | 53.61% | 72.14% | 53.33% | 52.62% | 54.72% |
| `7-7` | confirm-new | 62.33% | 33.33% | 57.50% | 79.52% | 58.33% | 55.24% | 54.72% |
| `8-8` | strict | 67.33% | 0.00% | 56.11% | 75.71% | 51.67% | 55.00% | 63.61% |
| `8-8` | confirm-new | 66.33% | 8.33% | 56.67% | 77.38% | 55.00% | 55.00% | 63.61% |

均值：

| 变体 | Old acc | Seen-new acc | Known acc | Unknown FAR | Full acc | 无拒识Known acc |
|---|---:|---:|---:|---:|---:|---:|
| strict | 59.42% | 0.00% | 49.51% | 41.25% | 50.83% | 53.26% |
| confirm-new | 57.00% | 30.83% | 52.64% | 53.33% | 51.79% | 53.26% |

## 解释

- q99默认门控能显著提高旧类带拒识Old acc，尤其`7-14/7-7/8-8`达到约72%-74%，但Unknown FAR为66.67%-93.33%，说明接受区仍过大，不能作为未知拒识可部署结果。
- q95保守门控把strict平均Unknown FAR从80.42%压到41.25%，但平均Old acc也从68.58%降到59.42%，seen-new在strict下仍为0%。这说明单纯收紧半径/能量阈值只能做风险交换，不能从机制上解决开放集分离。
- `opgac_strict`下所有新类状态均为`provisional`，所以Seen-new acc为0；`opgac_confirm_new`强制把有support的新类确认后，Seen-new acc恢复到8.33%-51.67%，但Unknown FAR同步升高，说明新类注册球与unknown/old尾部仍有明显重叠。
- 无拒识Known acc均值为53.26%，而OPGAC q99 confirm-new Known acc均值为58.89%，说明OPGAC对旧类/新类已带来一定域适应收益；失败点主要不是closed-set识别完全无效，而是开放集拒识边界不可靠。
- 未知类最近旧类分布集中落到少数旧类，尤其`3-19/7-14/7-7`大量靠近旧类label 1，对应旧类尾部/半径逃逸仍是主要风险。

## 结论

以`JREF_C9_MULTICOMP_M2_E220`为冻结主干，当前OPGAC-Net可以验证“原型高斯表+support-only校准+新类扩表”的执行链路，但不能作为可部署Stage2-C方案：旧类域适应有收益，新类注册需要解除过强provisional保护才有识别率，未知拒识仍明显不足。

下一步不应继续只调q值，而应加入显式未知边界建模：source leave-one-old-out伪未知、support局部密度门、old/new重叠惩罚、以及新类confirmed条件的几何检验，否则扩大表会继续扩大unknown接受区。
