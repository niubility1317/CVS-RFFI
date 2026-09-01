# ADV3B02-FCR R1-R8八卡并行实验v2预登记

## 状态

- run_id：`phase1_adv3b02_fcr_r1r8_s392002_20260902_v2`
- 当前状态：`LOCAL_VERIFIED`
- protocol_scope：Phase1 source-only；不访问Phase2 support/query/truth
- implementation_commit：`df19a485347ac18e350cddfd533ebc9894762e79`
- prereg_base_commit：`df19a485347ac18e350cddfd533ebc9894762e79`
- branch：`codex/adv3b02-fcr-20260901`
- predecessor：v1因统一的pre-prediction `nuisance_valid=None` TypeError技术失败，全部产物保留；v2仅包含该定点修复，矩阵不变。

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

## 修复与本地验证

- 原因：合法的未应用卫星view显式携带`nuisance_valid=None`，构造器错误地只处理属性缺失。
- 修复：显式None归一化为每样本全False；非None路径不变。
- TDD：回归测试修复前复现N607同一TypeError，修复后通过。
- 完整FCR聚焦验证：95项通过。
- 定点P0/P1复审：无阻断问题。
- 既有真实ADV3B02 checkpoint+Phase1 source无query smoke仍有效；本次修复另由真实故障路径和回归测试覆盖，不连接Phase2 query或truth。

## 用户资源授权

用户明确授权本次05:00启动忽略既有显卡进程数量限制，每张GPU新增一个本实验row。该授权不允许停止、重启、修改或影响任何既有进程，也不允许管理员账户或破坏性操作。

## 环境、路径与命令

- N607普通账户Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r1r8_s392002_20260902_v2`
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v2`
- row输出：`<run root>/jobs/Rk/Rk`
- launcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v2.Rk.launcher.out`
- 启动入口：`docs/automation_reports/CV-SincNet/phase1_adv3b02_fcr_r1r8_s392002_20260902_v2/launch_r1r8_remote.sh`

## Release归档映射

- archive_source_commit：`c008b8481bbe9009c3b72628233b78450528469d`
- 本地归档：`E:\type10-7\release_archives\phase1_adv3b02_fcr_r1r8_s392002_20260902_v2_c008b848.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/archives/phase1_adv3b02_fcr_r1r8_s392002_20260902_v2_c008b848.tar.gz`
- SHA256：`61ab6252238f3428b53e2dae799ec4001e6246288f69b01cff66f7f6421864d1`
- 解压策略：在不可覆盖release root内使用`--strip-components=1`。

## 直接技术停止规则

只在数据/query权限越界、错误row/seed/split、输出覆盖、错误checkout、命令不能运行、无prediction闭合、进程归属不明，或至少两个row出现同一确定性pre-prediction异常时，停止本run拥有的精确进程树并保留全部产物。低性能、负收益、严格Fingerprint Pair不可用或既有GPU任务数量不得作为停止理由。

## 预期artifact

每个R1-R8独立保存`best_joint.pth`、`fcr_diagnostics.json`、`fcr_predictions.json`、`train.log`和`status.txt`。完成训练的row必须产生clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`四场景完整prediction。启动闭合仅证明`RUNNING`；只有prediction完整并经独立truth-last scorer评分后才能进入`ARTIFACTS_COMPLETE/ANALYZED`。
