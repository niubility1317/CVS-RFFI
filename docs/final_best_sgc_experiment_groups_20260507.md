# 最佳模型 + SGC Residual 完整预设实验组

日期: 2026-05-07

## 总体原则

本轮实验不再把 full SGC from source 当作主线。5.7 日志表明，SGC 模块有可观测影响，但 full SGC 当前净收益不足；残差分支有信号，值得保留并单独验证。因此实验分两阶段:

1. **Phase A: 第一最佳模型候选。** 先用无 SGC 的 `Lite-B no-DAC + conservative MixStyle + Fishr + mild SAT mixed consistency` 跑出强 backbone。
2. **Phase B: SGC/residual 机制验证。** 从 Phase A 选中的 best checkpoint 出发，比较无 adapter continuation、residual-only、no-res、no-amp residual、full SGC 等路线。

所有候选统一使用:

```text
dataset = wisig
wisig_domain = rx_day
wisig_train_ratio = 0.2
primary_udu_weight = 0.65
eval_sat_scenarios = clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
sat_eval_on = test_unseen_day_unseen_rx
sat_eval_max_batches = -1
```

## 通过门槛

第一最佳模型门槛:

| 指标 | 门槛 |
|---|---:|
| Primary OOD | >= 87.80 |
| strict UDU | >= 86.20 |
| overall | >= 90.50 |
| worst-RX | >= 84.50 |
| SAT Avg | >= 41.50 |
| skipped_backward_batches | <= 50 |

SGC/residual 进入主线的门槛:

| 指标 | 门槛 |
|---|---:|
| SAT Avg 相对 no-adapter continuation | >= +1.00 |
| Primary 相对 no-adapter continuation | >= -0.30 |
| strict UDU 相对 no-adapter continuation | >= -0.30 |
| worst-RX 相对 no-adapter continuation | >= -0.50 |

如果 SGC/residual 不能满足这些条件，只作为机制分析结果，不进入最终模型。

## Phase A: 第一最佳模型候选组

| 实验名 | SAT 设置 | Fishr | seed | 目的 |
|---|---|---:|---:|---|
| `A0_fishr_only_ref` | 无 SAT train，仅 SAT eval | 0.02 | 1337 | 当前代码下复现 Fishr clean/OOD 参照 |
| `A1_fishr_sat_mild_v1` | cls 0.08, cons 0.04, start 20 | 0.02 | 1337 | 主候选 |
| `A2_fishr_sat_light_v2` | cls 0.05, cons 0.02, start 20 | 0.02 | 1337 | clean/OOD 掉时的保守版本 |
| `A3_fishr_sat_mid_v3` | cls 0.12, cons 0.06, start 20 | 0.02 | 1337 | SAT Avg 不足时的增强版本 |
| `A4_fishr_sat_delayed_v4` | cls 0.08, cons 0.04, start 60 | 0.02 | 1337 | SAT 过早干扰时的延迟版本 |
| `A5_sat_mild_no_fishr_ablation` | cls 0.08, cons 0.04, start 20 | 0.00 | 1337 | 验证 Fishr 是否必要 |
| `A6_fishr_sat_mild_seed2026` | cls 0.08, cons 0.04, start 20 | 0.02 | 2026 | 主候选稳定性复跑 |
| `A7_fishr_sat_mild_seed3407` | cls 0.08, cons 0.04, start 20 | 0.02 | 3407 | 主候选稳定性复跑 |

Phase A 结束后，脚本会从日志中解析 `[FINAL-PRIMARY]`，优先选择过门槛且 Primary OOD 最高的 checkpoint。若没有候选过门槛，则选择可解析候选中 Primary OOD 最高的 checkpoint，并在输出里标记为 fallback。

## Phase B: SGC / residual 机制验证组

Phase B 全部从 Phase A 选中的 checkpoint 加载，训练 60 epoch，SAT 权重保持温和:

```text
lambda_sat_cls = 0.08
lambda_sat_cons = 0.04
sat_cons_start_epoch = 20
fine-tune lr = 5e-5
lambda_res = 0.02
```

| 实验名 | adapter 设置 | 目的 |
|---|---|---|
| `B0_no_adapter_sat_continue` | 不启用 SGC adapter | Phase B 的继续训练对照，防止把 SAT continuation 的收益误算给 SGC |
| `B1_residual_only_std` | 只启用 residual compensator, channels 32, blocks 2 | 验证标准 residual-only 是否有效 |
| `B2_residual_only_small` | 只启用 residual compensator, channels 16, blocks 1 | 验证更小 residual 是否更不伤指纹 |
| `B3_residual_only_wide` | 只启用 residual compensator, channels 48, blocks 2 | 验证 residual 容量上限 |
| `B4_no_res_control` | amp/freq/spec 开，residual 关 | residual 的直接反证组 |
| `B5_no_amp_residual_full` | amp 关，freq/spec/residual 开 | 继承 5.7 里最有希望的 no-amp 信号 |
| `B6_no_amp_no_res_control` | amp 关，freq/spec 开，residual 关 | 检查 B5 的收益是否真来自 residual |
| `B7_full_sgc_mild` | full SGC，但 mild SAT | 验证 full SGC 在强 backbone + 小权重下是否仍有负作用 |
| `B8_no_amp_freq_sat_probe` | amp/freq 关，spec/residual 开 | SAT 上限探针，不作为主模型 |

## 建议解读顺序

1. 先看 Phase A 是否有模型过第一最佳门槛。
2. Phase B 先比较 `B0_no_adapter_sat_continue` 与 `B1/B2/B3`，判断 residual-only 是否带来净增益。
3. 再比较 `B5_no_amp_residual_full` 与 `B6_no_amp_no_res_control`，判断 no-amp 路线里的 residual 是否真的有效。
4. `B7_full_sgc_mild` 只用于确认 full SGC 是否仍然伤指纹。
5. `B8_no_amp_freq_sat_probe` 只看 SAT 上限，不参与最佳模型选择。

## 启动方式

当前建议先只跑最佳模型探索组，也就是 Phase A。SGC/residual 放到最后，等 Phase A 产出最佳 checkpoint 后再启动。

### 现在先跑 Phase A

Linux/Git Bash/WSL:

```bash
bash run_final_best_sgc_queue.sh
```

等价显式写法:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 PHASES=A bash run_final_best_sgc_queue.sh
```

PowerShell:

```powershell
$env:GPU_IDS="0,1,2,3,4,5,6,7"; $env:PHASES="A"; bash run_final_best_sgc_queue.sh
```

### Phase A 跑完后再跑 SGC/residual

脚本会自动从 Phase A 日志中选择 checkpoint；如果你想让它自动选择:

```bash
PHASES=B bash run_final_best_sgc_queue.sh
```

如果你想手动指定 Phase A 里某个 checkpoint:

```bash
PHASES=B PHASE_B_SOURCE_CKPT=finalist_runs/A1_fishr_sat_mild_v1/best_model_primary_ood.pth bash run_final_best_sgc_queue.sh
```

### 若要全自动先 A 后 SGC

只有确认不需要中间人工复核时才用:

```bash
PHASES=A,B bash run_final_best_sgc_queue.sh
```

常用控制:

```bash
PHASES=A bash run_final_best_sgc_queue.sh
PHASES=B PHASE_B_SOURCE_CKPT=finalist_runs/A1_fishr_sat_mild_v1/best_model_primary_ood.pth bash run_final_best_sgc_queue.sh
DRY_RUN=1 bash run_final_best_sgc_queue.sh
```
