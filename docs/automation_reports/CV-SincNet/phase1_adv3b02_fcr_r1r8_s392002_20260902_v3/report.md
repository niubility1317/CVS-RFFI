# ADV3B02-FCR R1-R8八卡并行实验v3报告

## 状态

- run_id：`phase1_adv3b02_fcr_r1r8_s392002_20260902_v3`
- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
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

## 启动与技术失败闭合

- 05:22完成v3落地并启动R1-R8；未启动R0或旧ADV3B02基线。
- launcher PID：R1=`4029158`、R2=`4029159`、R3=`4029160`、R4=`4029161`、R5=`4029162`、R6=`4029163`、R7=`4029164`、R8=`4029165`。
- 首次进程核验确认训练PID为R1=`4029176`、R2=`4029189`、R3=`4029188`、R4=`4029178`、R5=`4029186`、R6=`4029185`、R7=`4029184`、R8=`4029190`，工作目录均为本v3 release，命令均含`--model_variant lite_d`，GPU映射严格为R1→GPU0至R8→GPU7。
- release归档本地/远端SHA256一致；远端`bash -n`、Python编译和launcher dry-run均通过。
- 05:23，8个row在首次真实FCR前向中以同一确定性异常退出：`RuntimeError: "cuda_scatter_gather_base_kernel_func" not implemented for 'ComplexHalf'`。异常位置为`phase1_fcr_fingerprint.py:116`的`s.gather(1,index)`；上游在自动混合精度下构造了CUDA `ComplexHalf`张量。
- R1-R8的`status.txt`均为`TRAIN_FAILED`；未生成checkpoint、diagnostics或prediction，未连接truth，因此没有性能结果。
- 所有v3进程自然退出，没有停止、修改或重启任何既有GPU进程；8个row的`train.log`和`status.txt`均保留在不可覆盖run root。
- 这是v1空nuisance掩码、v2身份嵌入维度之后的第三个独立预训练启动故障。为避免盲目修复/重启循环，本次不自动创建v4；后续应先在与N607一致的CUDA自动混合精度环境复现并解决复数半精度算子兼容性，再以新run ID重新发布。
