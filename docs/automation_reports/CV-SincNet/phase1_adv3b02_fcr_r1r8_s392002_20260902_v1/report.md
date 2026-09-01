# ADV3B02-FCR R1-R8八卡并行实验预登记

## 状态

- run_id：`phase1_adv3b02_fcr_r1r8_s392002_20260902_v1`
- 当前状态：`LOCAL_VERIFIED`
- protocol_scope：Phase1 source-only；不访问Phase2 support/query/truth
- implementation_commit：`7fbde1e8547d90c821438b8401f754383d493bc3`
- prereg_base_commit：`9fc27f311bd5fb5e7d278bd5f07ca19e9f507509`
- branch：`codex/adv3b02-fcr-20260901`

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

R0及任何旧ADV3B02对比基线均不启动。所有row固定`seed=392002`、`epochs=200`、Meta-SSL`L_s/U_s/V=0.07/0.63/0.30`、E80卫星辅助CE、三段LEO_WEAK日程和同一source split。真实严格Fingerprint Pair能力仍为`blocked`；缺失轴必须为连接零并报告`N/A`，不得随机回退。

## 用户资源授权

用户明确授权本次05:00启动忽略既有显卡进程数量限制，每张GPU新增一个本实验row。该授权不允许停止、重启、修改或影响任何既有进程，也不允许管理员账户或破坏性操作。

## 环境、路径与命令

- N607普通账户Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r1r8_s392002_20260902_v1`
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v1`
- row输出：`<run root>/jobs/Rk/Rk`
- launcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v1.Rk.launcher.out`
- 启动入口：`docs/automation_reports/CV-SincNet/phase1_adv3b02_fcr_r1r8_s392002_20260902_v1/launch_r1r8_remote.sh`

## Release归档映射

- archive_source_commit：`73458d90324b9f0e4bbf1706c79743a1522ae855`
- 本地归档：`E:\type10-7\release_archives\phase1_adv3b02_fcr_r1r8_s392002_20260902_v1_73458d90.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/archives/phase1_adv3b02_fcr_r1r8_s392002_20260902_v1_73458d90.tar.gz`
- SHA256：`eae6c8bba62311a9ee0ccef5b59791dc7c2e117f472868e03dfa4ec9fb45f43c`
- 解压策略：在不可覆盖release root内使用`--strip-components=1`。

## 直接技术停止规则

只在数据/query权限越界、错误row/seed/split、输出覆盖、错误checkout、命令不能运行、无prediction闭合、进程归属不明，或至少两个row出现同一确定性pre-prediction异常时，停止本run拥有的精确进程树并保留全部产物。低性能、负收益、严格Fingerprint Pair不可用或既有GPU任务数量不得作为停止理由。

## 预期artifact

每个R1-R8独立保存`best_joint.pth`、`fcr_diagnostics.json`、`fcr_predictions.json`、`train.log`和`status.txt`。完成训练的row必须产生clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`四场景完整prediction。启动闭合仅证明`RUNNING`；只有prediction完整并经独立truth-last scorer评分后才能进入`ARTIFACTS_COMPLETE/ANALYZED`。
