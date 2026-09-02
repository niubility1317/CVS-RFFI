# ADV3B02-FCR R1-R8八卡并行实验v4报告

## 状态

- run_id：`phase1_adv3b02_fcr_r1r8_s392002_20260902_v4`
- 当前状态：`RUNNING`
- protocol_scope：Phase1 source-only；不访问Phase2 support/query/truth
- implementation_commit：`9e2b75a4ed70a4d206cbd87a4c9750773d8299c8`
- branch：`codex/adv3b02-fcr-20260901`
- predecessor：v3因CUDA AMP将FCR复数链路降为`ComplexHalf`，在`gather`处确定性失败；v1-v3产物全部保留，本v4使用新的不可覆盖run root。

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

R0及旧ADV3B02基线均不启动。所有row固定`seed=392002`、`epochs=200`、`model_variant=lite_d`、Meta-SSL`L_s/U_s/V=0.07/0.63/0.30`、E80卫星辅助CE、三段LEO_WEAK日程和同一source split。严格Fingerprint Pair仍为`blocked`，缺失轴连接零并报告`N/A`。

## 本次修复与本地验证

- 在FCR聚合前向、`ContentGenerator`、Fingerprint响应算子和物理顺序Decoder建立模块级FP32/`complex64`精度边界；未关闭ADV3B02主干AMP，未改变因子、Decoder顺序、损失、数据或矩阵。
- CPU autocast RED复现后修复；本地真实CUDA FP16 AMP下`U_s`和`L_s`完整objective+backward通过，主干输出保持FP16、FCR目标为FP32、33组FCR梯度有限。
- 全量FCR聚焦组90项通过；Python语法和差异检查通过。
- 原P1定点复审：PASS，0项P0/P1。

## 用户资源授权

用户明确授权本次05:00启动忽略既有显卡进程数量限制，每张GPU新增一个本实验row。该授权不允许停止、重启、修改或影响任何既有进程。

## 环境、路径与命令

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4`
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4`
- row输出：`<run root>/jobs/Rk/Rk`
- launcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4.Rk.launcher.out`
- 启动入口：`docs/automation_reports/CV-SincNet/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4/launch_r1r8_remote.sh`
- 启动命令：`bash <release root>/docs/automation_reports/CV-SincNet/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4/launch_r1r8_remote.sh`

## Release归档映射

- archive_source_commit：`d55daeb206f358e9d20d7c19561ac6329383eccb`
- 本地归档：`E:\type10-7\release_archives\phase1_adv3b02_fcr_r1r8_s392002_20260902_v4_d55daeb2.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/archives/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4_d55daeb2.tar.gz`
- SHA256：`45828f1c017cb10d71229f2ac16705a75381ae0ef67e76a1e8734410713df53e`

## 直接技术停止规则

只在数据/query权限越界、错误row/seed/split、输出覆盖、错误checkout、命令不能运行、无prediction闭合、进程归属不明，或至少两个row出现同一确定性pre-prediction异常时，停止本run拥有的精确进程树并保留全部产物。低性能、负收益、严格Fingerprint Pair不可用或既有GPU任务数量不得作为停止理由。

## 预期artifact

每个R1-R8独立保存`best_joint.pth`、`fcr_diagnostics.json`、`fcr_predictions.json`、`train.log`和`status.txt`。完成训练的row必须产生clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`四场景prediction。启动闭合只证明`RUNNING`；独立truth-last评分后才能进入`ARTIFACTS_COMPLETE/ANALYZED`。

## N607发布与启动闭合

- 2026-09-02 10:32 CST完成直接SSH只读preflight；普通账户、项目路径、8张GPU和7.2TB可用磁盘均可见。v4 release、run和archive目标在发布前均不存在。
- release归档本地/远端SHA256一致；远端Python编译和`bash -n`通过。N607 Torch2.1/CUDA12.1下直接运行完整CUDA FP16 AMP`compute_fcr_pair_objective+backward`，结果为`N607_CUDA_AMP_PAIR_OBJECTIVE_PASS`。
- 远端训练环境未安装`pytest`，因此未以pytest runner执行；同一回归函数通过标准库直接加载并执行成功，不安装服务器软件。
- 2026-09-02 10:35 CST启动R1-R8；未启动R0或旧ADV3B02基线。launcher PID：R1=`4166651`、R2=`4166652`、R3=`4166653`、R4=`4166654`、R5=`4166655`、R6=`4166656`、R7=`4166657`、R8=`4166658`。
- 主训练PID：R1=`4166678`、R2=`4166679`、R3=`4166682`、R4=`4166680`、R5=`4166674`、R6=`4166684`、R7=`4166681`、R8=`4166683`。CWD全部绑定v4 release，GPU映射逐一为R1→GPU0至R8→GPU7。
- 10:36 CST，8个`train.log`均增长至约16.5KB并写出E-01 checkpoint事件；`ComplexHalf|Traceback|RuntimeError`扫描为零，原v3确定性故障未复现。当前最高可证明状态仅为`RUNNING`，尚无完整训练、prediction或性能结果。
