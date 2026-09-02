# ADV3B02-FCR R1-R8八卡并行实验v5预登记报告

## 状态

- run_id：`phase1_adv3b02_fcr_r1r8_s392002_20260902_v5`
- 当前状态：`LOCAL_VERIFIED`
- protocol_scope：Phase1 source-only；不访问Phase2 support/query/truth
- implementation_commit：`7f12d8bbd2cea1003b1abf4376beee049982e7f9`
- branch：`codex/adv3b02-fcr-20260901`
- predecessor：v4已完成但出现TX类别塌缩、R2-R8持续非有限FCR指标和累计49,594次安全跳步；v5使用新的不可覆盖run root。

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

R0及旧ADV3B02基线均不启动。所有row固定`seed=392002`、`epochs=200`、`model_variant=lite_d`、Meta-SSL`L_s/U_s/V=0.07/0.63/0.30`、E80卫星辅助CE、三段LEO_WEAK日程和与v4相同的source split。用户明确授权忽略既有显卡进程数量限制，每张GPU新增一个本实验row；不允许停止或影响既有进程。

## 强制初始化checkpoint

- run：`phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1`
- candidate：`S392002_ADV3B03_MU10_ALPHA20_E200`
- checkpoint：`final_ssdg.pth`
- checkpoint_epoch：`200`
- source配置：ManySig；day1/2/3；receiver1/3/4/6/8；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 远端路径：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1/S392002_ADV3B03_MU10_ALPHA20_E200/final_ssdg.pth`
- 状态：`COMPLETE / ARTIFACTS_COMPLETE`

launcher和加载器共同拒绝非392002 seed、非E200、非指定candidate、任何旧FCR状态、成熟基座张量跳过或缺失。真实checkpoint本地smoke加载成熟基座195/195个张量、skipped=0；仅36个v5新增FCR张量保持新初始化；零步普通logits差值0，CosFace margin logits差值不超过`3e-6`。

## v5修复与验证

- `identity_context`零初始化，FCR身份头继承成熟CosFace权重、尺度、margin，并在训练时接收`y_tx`。
- 反塌缩标准差加入数值稳定项，常量潜变量维度的反向梯度保持有限。
- 未激活损失使用与其数值断开的有限零项，杜绝`NaN×0`污染当前阶段。
- E1不再提前执行necessity、physics和three-axis；cross内部按当前stage只计算激活损失。
- 正式launcher仅允许R1-R8，强制初始化checkpoint和seed392002。
- 全部21个Phase1-FCR聚焦测试文件共96项通过；Python编译、`git diff --check`通过。
- 一次独立P0/P1审查发现2项launcher/init绑定问题；修复后定点复审为`Ready`，0项未闭合P0/P1。
- 本机Git Bash适配器误路由到WSL并失败，未执行本地Bash payload；远端发布后以N607原生`bash -n`验证脚本。

## 环境、路径与启动命令

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r1r8_s392002_20260902_v5`
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v5`
- row输出：`<run root>/jobs/Rk/Rk`
- launcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v5.Rk.launcher.out`
- 启动入口：`docs/automation_reports/CV-SincNet/phase1_adv3b02_fcr_r1r8_s392002_20260902_v5/launch_r1r8_remote.sh`
- 启动命令：`bash <release root>/docs/automation_reports/CV-SincNet/phase1_adv3b02_fcr_r1r8_s392002_20260902_v5/launch_r1r8_remote.sh`

## 直接技术停止规则

只在数据/query权限越界、错误row/seed/split、初始化checkpoint身份或完整加载不符、输出覆盖、错误checkout、命令不能运行、无prediction闭合、进程归属不明，或至少两个row出现同一确定性pre-prediction异常时，停止本run拥有的精确进程树并保留全部产物。低性能、负收益或既有GPU任务数量不得作为停止理由。

## 预期artifact

每个R1-R8独立保存`best_joint.pth`、`fcr_diagnostics.json`、`fcr_predictions.json`、`train.log`和`status.txt`。完成训练的row必须产生clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`四场景prediction。启动闭合只证明`RUNNING`；独立truth-last评分后才能进入`ARTIFACTS_COMPLETE/ANALYZED`。
