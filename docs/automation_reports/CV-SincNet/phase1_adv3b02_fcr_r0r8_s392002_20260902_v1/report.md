# ADV3B02-FCR单seed最小证伪矩阵预登记

## 状态

- run_id：`phase1_adv3b02_fcr_r0r8_s392002_20260902_v1`
- 当前状态：`LOCAL_VERIFIED`
- protocol_scope：Phase1 source-only；不访问Phase2 support/query/truth
- implementation_commit：`7fbde1e8547d90c821438b8401f754383d493bc3`
- branch：`codex/adv3b02-fcr-20260901`

## 候选与冻结矩阵

| Row | 冻结语义 | Decoder模式 |
|---|---|---|
| R0 | FCR身份CE基线，无FCR辅助损失 | `control` |
| R1 | R0+`L_self+L_eta` | `control` |
| R2 | R1+`L_swap` | `control` |
| R3 | R2+`L_shared` | `control` |
| R4 | R3+`L_latent_cycle` | `control` |
| R5 | R4+同样本basic`L_drop_f`必要性 | `control` |
| R6 | R5+严格Fingerprint Pair定向移植 | `control` |
| R7 | R6+完整物理顺序Decoder及Fisher门控`L_phys` | `full_physics` |
| R8 | R7+严格Nuisance/Content/Fingerprint三轴干预 | `full_physics` |

所有row固定`seed=392002`、`epochs=200`、Meta-SSL`L_s/U_s/V=0.07/0.63/0.30`、E80卫星辅助CE、三段LEO_WEAK日程以及同一source split。真实严格Fingerprint Pair能力当前仍为`blocked`；缺失轴必须为连接零并报告`N/A`，不得随机回退，也不得因此停止其他合法row。

## 环境、路径与命令

- 远端环境：N607普通账户；Python`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 远端CWD/代码根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r0r8_s392002_20260902_v1`
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r0r8_s392002_20260902_v1`
- launcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r0r8_s392002_20260902_v1.launcher.out`
- GPU：`0`；row顺序执行，不超过每GPU两个训练任务

```text
nohup env RUN_ID=phase1_adv3b02_fcr_r0r8_s392002_20260902_v1 OUTPUT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r0r8_s392002_20260902_v1 ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r0r8_s392002_20260902_v1 GPU=0 SEED=392002 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r0r8_s392002_20260902_v1/code/scripts/launch_phase1_adv3b02_fcr_20260901.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r0r8_s392002_20260902_v1.launcher.out 2>&1 &
```

## 直接技术停止规则

只在数据/query权限越界、错误row/seed/split、输出覆盖、错误checkout、命令不能运行、无prediction闭合、进程归属不明，或至少两个row出现同一确定性pre-prediction异常时，停止该run拥有的精确进程树并保留全部产物。低性能、负收益、严格Fingerprint Pair不可用或缺少非必要字段不得作为技术停止理由。

## 预期artifact

每个`R0`至`R8`row独立保存`best_joint.pth`、`fcr_diagnostics.json`、`fcr_predictions.json`、`train.log`和`status.txt`。完成训练的row必须用选定checkpoint产生clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`四场景完整prediction。launcher最多标记`PREDICTIONS_READY`；只有独立truth-last scorer完成同row评分后才可进入`ARTIFACTS_COMPLETE/ANALYZED`。
