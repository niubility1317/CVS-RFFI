# ADV3B02-FCR R1-R8八卡并行实验v3预登记

## 状态

- run_id：`phase1_adv3b02_fcr_r1r8_s392002_20260902_v3`
- 当前状态：`LOCAL_VERIFIED`
- protocol_scope：Phase1 source-only；不访问Phase2 support/query/truth
- implementation_commit：`684ec110ffd7306ef836d82cf0cc5967ebc3c596`
- branch：`codex/adv3b02-fcr-20260901`
- predecessor：v1修复空nuisance掩码；v2确认默认`lite_c`的192维身份嵌入与报告固定160维schema冲突。两次失败产物均保留。

## 冻结矩阵与GPU映射

| Row | 语义 | GPU | Decoder |
|---|---|---:|---|
| R1 | FCR身份CE+`L_self+L_eta` | 0 | `control` |
| R2 | R1+`L_swap` | 1 | `control` |
| R3 | R2+`L_shared` | 2 | `control` |
| R4 | R3+`L_latent_cycle` | 3 | `control` |
| R5 | R4+同样本basic`L_drop_f`必要性 | 4 | `control` |
| R6 | R5+严格Fingerprint Pair定向移植 | 5 | `control` |
| R7 | R6+完整物理顺序Decoder和Fisher门控`L_phys` | 6 | `full_physics` |
| R8 | R7+严格三轴干预 | 7 | `full_physics` |

R0及旧ADV3B02基线均不启动。所有row固定`seed=392002`、`epochs=200`、`model_variant=lite_d`、Meta-SSL`L_s/U_s/V=0.07/0.63/0.30`、E80卫星辅助CE、三段LEO_WEAK日程和同一source split。`lite_d`是报告固定`id_feature_raw/z_f_id=160`的现有兼容主干，不新增投影器，不改变FCR因子、损失或消融递进。严格Fingerprint Pair仍为`blocked`，缺失轴连接零并报告`N/A`。

## 修复验证

- v1空掩码回归与v2 160维launcher回归均完成RED→GREEN。
- 完整FCR聚焦组95项通过。
- 真实模型前向：`emb_dim=160`、`z_id_raw=(2,160)`、`z_f_id=(2,160)`、`fcr_tx_logits=(2,6)`。
- 定点P0/P1复审：无阻断问题。
- 既有真实ADV3B02 checkpoint+Phase1 source无query smoke有效；不连接Phase2 query或truth。

## 用户资源授权

用户明确授权本次05:00启动忽略既有显卡进程数量限制，每张GPU新增一个本实验row。该授权不允许停止、重启、修改或影响任何既有进程。

## 环境、路径与命令

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r1r8_s392002_20260902_v3`
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v3`
- row输出：`<run root>/jobs/Rk/Rk`
- launcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v3.Rk.launcher.out`
- 启动入口：`docs/automation_reports/CV-SincNet/phase1_adv3b02_fcr_r1r8_s392002_20260902_v3/launch_r1r8_remote.sh`

## Release归档映射

- archive_source_commit：`e3d4cf5889d8d7e674fd4d329cc199838d5d3078`
- 本地归档：`E:\type10-7\release_archives\phase1_adv3b02_fcr_r1r8_s392002_20260902_v3_e3d4cf58.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/archives/phase1_adv3b02_fcr_r1r8_s392002_20260902_v3_e3d4cf58.tar.gz`
- SHA256：`d311f4f6104ff399357c0290261434c45e16dcc56b8675155efc9c2052e35798`

## 直接技术停止规则

只在数据/query权限越界、错误row/seed/split、输出覆盖、错误checkout、命令不能运行、无prediction闭合、进程归属不明，或至少两个row出现同一确定性pre-prediction异常时，停止本run拥有的精确进程树并保留全部产物。低性能、负收益、严格Fingerprint Pair不可用或既有GPU任务数量不得作为停止理由。

## 预期artifact

每个R1-R8独立保存`best_joint.pth`、`fcr_diagnostics.json`、`fcr_predictions.json`、`train.log`和`status.txt`。完成训练的row必须产生clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`四场景prediction。启动闭合只证明`RUNNING`；独立truth-last评分后才能进入`ARTIFACTS_COMPLETE/ANALYZED`。
