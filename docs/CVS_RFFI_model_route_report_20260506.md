# CVS-RFFI 版本代码与训练日志综合分析报告

日期：2026-05-06

## 1. 结论摘要

综合当前工作区内的代码版本、实验文档和训练日志，最适合作为主线继续推进的模型路线是：

**主推荐：R19 Lite-B no-DAC + conservative MixStyle + Fishr**

对应已验证日志：`type10-7/CV-SincNet/logs/SAT37_r19_fishr_20260428_203517.log`

理由：
- 在所有已解析训练日志中，`SAT37_r19_fishr` 的 Primary OOD 最高：87.95。
- 严格 unseen-day unseen-RX 为 86.43%，整体测试为 90.77%，都在第一梯队。
- 参数量约 1.67M，明显小于 full/base 路线，同时优于旧 R19 baseline。
- 训练稳定，`skipped_backward_batches=13`，没有 NaN collapse。
- 结构上延续已经反复验证的 `Lite-B + no_dac + RCN/stat domain cues + PA aux` 主干，不是大幅冒险重构。

**紧随备选：R19 Lite-B no-DAC + GroupDRO smooth**

对应日志：`SAT34_r19_groupdro_smooth`

它与 Fishr 几乎并列，Primary OOD 为 87.94，strict UDU 为 86.44。如果部署目标更偏向弱接收机分组或 worst-RX，可把 GroupDRO smooth 作为第一备选。

**部署压缩候选：R25 Lite-D no-DAC + SAT mixed consistency**

对应日志：`SAT07_r25_compact_sat_mixed`

它只有约 1.05M 参数，但 Primary OOD 达 87.85，strict UDU 86.27，整体测试 90.79，是当前最强的参数效率路线。

**不建议现在直接选 SGC-Adapter 作为最终最佳模型。**

SGC-Adapter 在当前根目录代码中已经集成，并且设计方向合理，但根目录 `logs/` 中 2026-05-06 的 SGC/slim/SSDG 日志基本是 dry-run、空日志或 `python: command not found` 启动失败，没有完整训练证据。因此 SGC 应作为下一阶段实验路线，而不是本轮报告里的已验证冠军。

## 2. 分析范围与证据可信度

本次共解析 `.log` 文件 183 个，其中 110 个包含训练 epoch，6 个为空日志。

| 来源 | 日志数 | 有训练 epoch | 说明 |
|---|---:|---:|---|
| `type10-4/4.23logs` | 18 | 18 | 早期主干、MixStyle、分支消融日志 |
| `type10-4/4.24logs` | 17 | 15 | 早期稳定性与分支路线补充 |
| `type10-4/4.26logs` | 11 | 8 | 瘦身、domain enhancer、R 路线前身 |
| `type10-4/4.27logs` | 30 | 28 | R19/R25 no-DAC 主线成熟日志 |
| `type10-7/CV-SincNet/logs` | 42 | 41 | SAT 评估、SAT consistency、Fishr、GroupDRO 等完整矩阵 |
| `logs/` | 65 | 0 | 根目录新启动日志，多数为 dry-run 或启动失败 |

证据分级：

| 等级 | 含义 | 本报告采用方式 |
|---|---|---|
| A | 完整训练日志，含 best/final、split、SAT 评估 | 可用于排名和路线选择 |
| B | 代码和测试存在，但训练日志未完成 | 可列为下一步实验 |
| C | 文档或脚本计划，无有效训练日志 | 不作为最佳模型依据 |

`type10-7` 和 `type10-4/4.27logs` 是本次最可靠的实证来源。根目录 SGC/SSDG/slim 日志不能证明模型优劣，因为多条日志内容是 `python: command not found` 或 dry-run 命令。

## 3. 各版本代码路线演进

### 3.1 `type10-4`：主干能力、消融与瘦身路线

`type10-4` 是当前项目最重要的历史证据库。它包含从 full/base 到 Lite-B/Lite-D 的结构对比，并形成了后续主线判断：

- full/base 模型可以取得很高 strict UDU，但参数大、训练比设置不同，部署性较弱。
- DAC 分支去除后不但没有明显伤害，反而在多条路线中提升或保持 OOD 表现。
- PA 分支、时间分支、频率分支、RCN/stat domain cues 仍然有价值。
- `no_dac,no_stats`、`time_only`、过度瘦身路线风险较高。
- conservative MixStyle 比强 MixStyle 更稳，尤其在 no-DAC 主线上。

代表日志：
- `R19_r05_mix_p015`：Primary OOD 87.46，strict UDU 85.92，参数 1.67M。
- `R25_r06_refined_default`：Primary OOD 87.17，strict UDU 85.59，参数 1.05M。

这说明 R19/R25 no-DAC 已经是成熟主线。

### 3.2 `type10-7`：SAT 评估与鲁棒训练增强

`type10-7` 基于 R19/R25 no-DAC 主线继续加入：

- SAT channel evaluation。
- SAT consistency training。
- Fishr。
- GroupDRO smooth/capped。
- prototype memory、SupCon、SWA/SWAD 等增强项。

这里产生了本次最强的已验证候选：

- `SAT37_r19_fishr`
- `SAT34_r19_groupdro_smooth`
- `SAT07_r25_compact_sat_mixed`
- `SAT13_r19_mixed_high_weight`

结论是：**干净 OOD 仍由 R19/R25 no-DAC 主线支撑，Fishr/GroupDRO 是更可靠的增益项；SAT consistency 对卫星扰动有帮助，但不能单独解决卫星场景准确率偏低的问题。**

### 3.3 当前根目录：SGC-Adapter、SSDG、统一 preset

当前根目录代码比 `type10-7` 多了约 262 行训练集成，主要新增：

- `sgc_adapter.py`：RMS 幅度归一化、频偏/Doppler 补偿、频域干扰抑制、残差通道补偿。
- `sgc_losses.py`：source prototype bank、feature consistency、pseudo label、entropy、residual regularization。
- `train.py` 的 `sgc_*` preset、`source -> sgc_augment -> sgc_adapt` 三阶段控制。
- `SSDG-SSL` preset 文档和测试。
- `run_all_preset_experiments.sh`、`run_sgc_experiments.sh` 启动矩阵。

SGC-Adapter 的设计很贴合卫星-地面信道问题，但目前只具备 B 级证据：代码和单元测试存在，完整训练日志不存在。

### 3.4 `unkown`：更激进的 satellite hybrid 实验分支

`unkown` 目录包含一个更实验性的升级版本：

- `sat_hybrid_lite_b_no_dac`
- `sat_hybrid_lite_d_no_dac`
- `sat_hybrid_lite_f`
- `sat_adapter_lite_f`
- classic DG losses：MMD、CORAL、RIEI
- `baseline_cvcnn`、`baseline_resnet1d`

这条路线的想法很有价值，但它没有与当前工作区日志对应的完整训练结果，并且当前根目录主线实际保留的是 SGC-Adapter 而非 `sat_hybrid_*`。因此不建议把 `unkown` 作为现在的最佳路线，只建议吸收其中的 classic DG 对照思想。

## 4. 核心候选模型对比

指标说明：

- Primary OOD：日志中的综合 OOD 选择指标，当前排序最重要。
- strict UDU：unseen-day unseen-RX，是最硬的泛化切分。
- SAT Avg：final-primary checkpoint 下 5 个 SAT scenario 的 strict UDU 平均值。
- Worst-RX：最弱接收机分组最佳值。

| 排名 | 候选 | 路线 | 参数量 | Primary OOD | strict UDU | overall | Worst-RX | SAT Avg | 结论 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `SAT37_r19_fishr` | Lite-B no-DAC + Fishr | 1.672M | 87.95 | 86.43 | 90.77 | 84.64 | 38.91 | 主推荐，干净 OOD 最强 |
| 2 | `SAT34_r19_groupdro_smooth` | Lite-B no-DAC + smooth GroupDRO | 1.672M | 87.94 | 86.44 | 90.72 | 84.83 | 38.69 | 几乎并列，偏弱域稳健 |
| 3 | `SAT07_r25_compact_sat_mixed` | Lite-D no-DAC + SAT mixed | 1.050M | 87.85 | 86.27 | 90.79 | 84.67 | 41.98 | 最佳紧凑部署候选 |
| 4 | `SAT10_r19_mixed_cls_only` | Lite-B no-DAC + SAT cls only | 1.672M | 87.71 | 86.15 | 90.60 | 85.12 | 42.55 | Clean/SAT 折中较好 |
| 5 | `SAT13_r19_mixed_high_weight` | Lite-B no-DAC + 高权重 SAT consistency | 1.672M | 87.68 | 86.14 | 90.53 | 85.18 | 43.91 | SAT Avg 最高，但 clean OOD 略低 |
| 6 | `R19_r05_mix_p015` | Lite-B no-DAC baseline | 1.672M | 87.46 | 85.92 | 90.32 | 85.03 | 无 | 4.27 成熟 baseline |
| 7 | `R25_r06_refined_default` | Lite-D no-DAC baseline | 1.050M | 87.17 | 85.59 | 90.10 | 83.92 | 无 | 旧 compact baseline |

一个容易误判的点：`A00_s1_core_base` 曾有 strict UDU 86.72，但它是更早 full/base 路线，参数约 3.41M，训练比例与后续主线不完全一致，也没有 Primary OOD 和 SAT 评估。它证明 full/base 容量有效，但不适合作为现在的推荐路线。

## 5. 不推荐路线与原因

| 路线 | 不推荐原因 |
|---|---|
| full/base 主干直接部署 | 参数量大，旧日志条件不同，后续 Lite-B/Lite-D no-DAC 已经达到接近或更好 OOD |
| `no_dac,no_stats` | 日志和先前 findings 显示 strict UDU、Primary、Worst-RX 均明显下滑 |
| 只保留 time 或去掉 time/freq 主路径 | 属于高风险瘦身边界，容易容量不足或泛化坍缩 |
| `type10-4` 中不稳定强辅助路线 | 部分日志有 NaN/collapse 或大量 skipped backward |
| 当前根目录 SGC/slim/SSDG 2026-05-06 日志 | 没有完成训练，多数是 `python` 缺失、dry-run 或空日志 |
| `unkown/sat_hybrid_*` | 代码想法可参考，但没有当前工作区内完整训练日志验证 |

## 6. 推荐的最终模型与路线

### 6.1 主线模型

选择：

```text
DualCVSincNetDisentangle
model_variant = lite_b
branch_ablation = no_dac
domain_branch_ablation = no_stats
domain_enhancer = rcn_stats
MixStyle = conservative same_tx_crossdomain
Fishr = 0.02
```

推荐原因：

- 已验证分数最高。
- 比 full/base 更轻。
- 保留 PA、time、freq、RCN/stat cues，避免过度瘦身。
- 去掉 DAC 分支，减少冗余复杂度。
- Fishr 对 Primary OOD 有最直接的证据增益。

建议命令骨架：

```bash
python -u train.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --batch_size 256 \
  --slim_group rxrobust_lite_b_no_dac_mix015 \
  --epochs 200 \
  --wisig_train_ratio 0.2 \
  --primary_udu_weight 0.65 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --eval_sat_channel \
  --eval_sat_on test_unseen_day_unseen_rx \
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_eval_max_batches -1
```

### 6.2 弱域优先备选

如果更关心 worst-RX 或弱域风险，选择：

```text
R19 Lite-B no-DAC + smooth GroupDRO
```

对应 `SAT34_r19_groupdro_smooth`，Primary OOD 与 Fishr 只差 0.01，strict UDU 还略高 0.01。

### 6.3 参数效率部署候选

如果部署端参数预算更紧，选择：

```text
R25 Lite-D no-DAC + SAT mixed consistency
```

对应 `SAT07_r25_compact_sat_mixed`，参数约 1.05M，Primary OOD 87.85，几乎追平 R19 路线。

建议把它作为第二条长期保留路线，而不是简单降级版，因为它的参数效率非常好。

## 7. SGC-Adapter 后续路线

SGC-Adapter 不应被本轮直接选为最佳模型，但它应该成为下一阶段最重要的实验路线。

原因：

- 当前 satellite scenario 的绝对准确率仍偏低。最好的 SAT Avg 也只有 43.91，说明仅靠 SAT consistency 还没有真正学到强补偿。
- SGC-Adapter 的模块设计正好覆盖幅度归一化、频偏/Doppler、频域干扰和残差信道补偿。
- 当前代码已有单元测试，参数预算小于 50k，适合部署侧补偿。

建议下一阶段做严格三阶段实验：

```bash
# 1. source baseline
python -u train.py --preset sgc_baseline_no_adapter --stage source --epochs 200

# 2. source with SGC adapter
python -u train.py --preset sgc_lite_b_no_dac --stage source --epochs 200

# 3. satellite-channel augmentation
python -u train.py \
  --preset sgc_lite_b_no_dac \
  --stage sgc_augment \
  --source_ckpt sgc_runs/sgc_lite_b_no_dac/source/best_model.pth \
  --epochs 100 \
  --lambda_feat 1.0 \
  --lambda_res 0.01

# 4. adapter-only adaptation
python -u train.py \
  --preset sgc_lite_b_no_dac \
  --stage sgc_adapt \
  --source_ckpt sgc_runs/sgc_lite_b_no_dac/augment/best_model.pth \
  --adapt_epochs 50 \
  --adapt_lr 1e-4 \
  --pseudo_label_threshold 0.85 \
  --lambda_proto 1.0 \
  --lambda_cons 0.5 \
  --lambda_ent 0.01 \
  --lambda_res 0.01
```

启动器层面还要先修正环境：根目录日志显示 shell 脚本里调用 `python` 失败。需要确认训练机器上 `python` 可用，或把脚本改成可配置 `PYTHON_BIN=${PYTHON_BIN:-python3}`。

## 8. 最终建议

建议采用“双路线”策略：

| 路线 | 用途 | 当前状态 |
|---|---|---|
| R19 Lite-B no-DAC + Fishr | 主论文/主结果/默认推荐 | 已验证，立即可用 |
| R19 Lite-B no-DAC + GroupDRO smooth | 弱域/worst-RX 备选 | 已验证，几乎并列 |
| R25 Lite-D no-DAC + SAT mixed | 轻量部署候选 | 已验证，参数效率最高 |
| SGC-Adapter 三阶段链路 | 卫星信道补偿下一阶段 | 代码已集成，需完整训练验证 |

当前最适合写入主线结论的模型是：

```text
SAT37 路线：
R19 Lite-B no-DAC + conservative MixStyle + Fishr(0.02)
```

当前最适合工程部署压缩评估的模型是：

```text
SAT07 路线：
R25 Lite-D no-DAC + SAT mixed consistency
```

当前最值得继续投入的新路线是：

```text
SGC-Adapter：
先复现 no-adapter baseline，再跑 source -> augment -> adapt，最后与 SAT37/SAT07 对照。
```

