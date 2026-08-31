# PairBiCAD-CV2 Phase1快速矩阵预登记

## 当前状态

- 状态：LOCAL_VERIFIED。
- Run ID：phase1_pairbicad_cv2_screen24_seed392002_20260831_r1。
- 目标：在新CORE90数据包对应的Phase1 source-only划分上比较Core、BiAdv、Pair和TailGuard；不访问Phase2或目标接收机数据。
- 正式实现提交：`20032361bebf43101023b9276589fa4c4a74e90f`；本报告随后的发布记录提交不改变训练代码。

## 冻结矩阵

- 候选：CV2-B0/B1/B2/B3、CV2-D0/D1/D2/D3、CV2-T0/T1/T2/T3，共12个静态配置。
- fold：留出source receiver1和8；训练receiver分别为[3,4,6,8]和[1,3,4,6]。
- seed：392002；训练天：day1/day2/day3；共24行。
- GPU0–7每张最多2个本run直属训练进程，16个并发槽位，其余8行排队。若preflight发现无关训练进程，按总训练进程不超过2/GPU降低本run占用，不影响无关进程。
- 所有配置发布前静态解析；CV2-D0和CV2-T0静态复制CV2-B3，禁止运行中champion alias，历史D0-D3语义不变。

## 训练与选择协议

- Phase1 bicad_xr，ADV3B02双骨干，concat_sat_ce_only+LEO_WEAK。
- 数据角色L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15；训练、调度、选模和重跑仅允许source数据。
- 普通步16L+32U；每4步结构化步在4个source receiver时为24L+24U，未来5receiver refit时为30L+18U；每步48个物理样本拼接Clean/LEO视图后只做一次96行网络前向。
- LEO训练场景：leo_clear_weak、leo_low_elev_weak、leo_rain_weak。
- 收敛候选每0.5个U覆盖周期评估一次且相邻至少500updates；最后核心机制至少训练3个U覆盖周期。
- ReduceLROnPlateau：factor0.3、patience3、min_lr1e-6。
- 技术安全上限：12个U覆盖周期或24小时；命中时写NOT_CONVERGED_SAFETY_STOP，不得写成科学收敛。
- SWAD按S_DG及Clean/LEO/receiver floor的0.50pp窗口准入。
- 双对抗复用一次backbone前向，判别器step先于encoder step，判别器LR为1.5倍；conditional和z_dom→TX梯度比独立控制。
- Pair：128维投影、hinge容忍半径0.05、权重0.02。
- TailGuard：lambda_rex=0.02、lambda_cvar=0.05、tail20%、困难组上限30%。

## 本地实现与验证

- 设计：docs/superpowers/specs/2026-08-31-pairbicad-cv2-design.md。
- 计划：docs/superpowers/plans/2026-08-31-pairbicad-cv2.md。
- 追踪：analysis/phase1_pairbicad_cv2_traceability.md。
- 主要实现：config、convergence、swad、adversarial_game、gradients、tailguard、trainer、train_ssdg、CV2 launcher和analyzer。
- code/tests/phase1_bicad_xr共454项全部通过；仅有3条既存PyTorch autocast FutureWarning。
- 本地fixture checkpoint smoke为4/4通过；正式发布前仍须在N607用真实历史ADV3B02 checkpoint做一次无query smoke。
- dry-run：24行、12候选、fold1/8、seed392002；GPU0–7各映射3行，dispatcher上限2/GPU。
- 独立P0/P1审查：配置/矩阵无阻断；实际U覆盖计数、结构化24U/18U loader和动态stop update验收问题已修复，原问题定点复审均为RESOLVED。

## 正式路径与命令

- 本地环境：ssr-gpu。
- N607账户：普通账户szu2070436088，禁止管理员账户。
- release根：/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_cv2_screen24_20260831_r1。
- run根：/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pairbicad_cv2_screen24_seed392002_20260831_r1。
- 输入：/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl。
- 启动：使用code/scripts/launch_phase1_pairbicad_cv2_screen24_20260831.py，传入上述run ID、release根、run父目录、ssr-gpu Python和ManySig.pkl。

## 预期artifact与停止规则

每行必须生成bicad_xr_final.pth、Clean和三种LEO评估JSON、严格重建信息、训练遥测、worker状态及ARTIFACTS_COMPLETE.json。只有数据越权、错误candidate/fold/receiver/day/seed、输出冲突、错误release/CWD、命令无法运行、确定性重复异常、进程归属不清或无法产生合法checkpoint/四场景artifact时，才可精确停止对应run进程树并保留partial artifact。不得因中间或最终性能低而停止、重启、热补丁或选择性重跑；不得影响无关进程。
