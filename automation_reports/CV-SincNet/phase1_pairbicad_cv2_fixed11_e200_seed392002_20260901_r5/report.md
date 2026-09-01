# Phase1 PairBiCAD-CV2正式E200矩阵r5

## 当前状态

- 状态：`RUNNING`。
- run ID：`phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r5`。
- Git提交：`5f785287935f2b58f4e7a4f95b37341de2a176a0`；已push并独立核对远端分支OID一致。
- r4已固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，输出根和partial artifact保留；r5使用全新release和不可覆盖run根。

## 故障修复

r4的两条`CV2-B0`行均完成200epoch并保存final checkpoint，但启动器在训练返回0后只检查正式artifact是否存在，没有调用正式评估流程，因此确定性缺少`checkpoint_runtime.json`、`diagnostics.json`、clean和三种LEO评估。

r5恢复完整闭合：定位非空final checkpoint；从独立`source_loro_selection.json`和200行`metrics_epoch.jsonl`绑定终止update与E200身份；严格重建模型和trainer runtime；仅使用held-out source receiver执行clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；逐场景写JSON和日志；再次验证runtime、重建键、source-only访问标记与四场景闭合，最后才写`ARTIFACTS_COMPLETE.json`。任何环节失败均写技术失败，不伪造完成。

## 冻结矩阵与协议

- 候选：`CV2-B0/B1/B2/B3/D0/D1/D2/D3/T0/T1/T2/T3`，12种。
- fold：1、8；seed：`392002`；共24行。
- 每行从头训练200epochs；终止方式仅为`epochs=200`，命令中无`--bicad_optimizer_updates`且不使用6500updates。
- ManySig源域receiver集合`[1,3,4,6,8]`；fold1训练`[3,4,6,8]`并留出receiver1，fold8训练`[1,3,4,6]`并留出receiver8；训练日为day1/day2/day3。
- 源域角色比例：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；严格Phase1 source-only，不访问Phase2、target、support、query或truth。
- 现行增强协议：`concat_sat_ce_only`、`lambda_sat_cons=0`，三种`LEO_WEAK`课程为`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- GPU0—7每卡最多2个本run训练进程；16行并发、8行排队，不超过用户指定容量。

## 本地与远端验证

- 新增回归测试先复现“训练成功但未调用正式评估”的缺口，修复后worker必须实际调用`evaluate_final_checkpoint`才允许完成。
- `code/tests/phase1_bicad_xr`完整测试通过：476项通过，仅3条既存PyTorch弃用警告。
- launcher、测试和远端release入口均通过编译；`git diff --check`通过；dry-run读回24行、12候选、fold1/8、seed392002、全部E200、每GPU2槽和8行排队。
- 历史真实checkpoint无query smoke通过：严格重建missing/unexpected/shape mismatch均为0，optimizer step完成，clean及三种LEO前向均有限，所有target/Phase2/support/query/truth访问标记为false。
- r5正式release对r4的真实`CV2-B0-F1-S392002` E200 checkpoint执行独立闭合smoke，结果`PASS`：`runtime_valid=true`、四场景齐全、`source_only=true`。smoke使用复制到独立日志目录的checkpoint，不修改r4 partial artifact。
- Luna一次聚焦P0/P1审查结论：`无P0/P1`。

## Release、路径与正式命令

- release归档：`E:\\type10-7\\local_artifacts\\phase1_pairbicad_cv2_e200_5f785287.tar.gz`。
- 单一归档本地/远端SHA256：`0190c01045aac24bb903eaac036a26638a016e17464adc5e45f9a596d5e8a30b`，已独立核对一致。
- 远端release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_cv2_e200_5f785287`。
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r5`。
- dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r5.dispatcher.log`。
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD为上述release；使用普通账户`szu2070436088`，禁止管理员账户。
- 正式命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/launch_phase1_pairbicad_cv2_screen24_20260901.py --run-id phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r5 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs --code-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_cv2_e200_5f785287 --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --gpu-capacities 0:2,1:2,2:2,3:2,4:2,5:2,6:2,7:2`。

## 预期artifact与停止规则

每行必须闭合200epoch telemetry、final checkpoint、严格checkpoint/trainer runtime重建、final/EMA/SWAD一次`V_select`选择、`checkpoint_runtime.json`、`diagnostics.json`、clean与三种LEO弱场景独立JSON/日志以及`ARTIFACTS_COMPLETE.json`。

仅允许因数据/query越权、错误candidate/fold/receiver/day/seed/epoch、输出冲突、错误release/CWD、命令无法运行、无合法prediction/artifact闭合、同一确定性异常至少重复两行或进程归属不清而停止精确run进程树。低性能、中间指标、缺少非必要receipt/hash或报告字段不得停止、重启、热补丁或选择性重跑。若出现预登记系统技术失败，必须保留partial artifact，在本地Git修复并验证后以新release、新run ID重新发布；不得原地重启。

## 正式启动证据

- 启动时间：2026-09-01 11:11 CST；dispatcher PID`3472323`。
- dispatcher父PID为1；`/proc/3472323/cwd`独立读回精确指向release`phase1_pairbicad_cv2_e200_5f785287`，cmdline精确绑定r5、ManySig、普通账户Python和GPU容量`0:2,...,7:2`。
- 直属worker为16个，16个`train.log`已创建；GPU0—7各有2个本run计算进程，共16个，启动检查时利用率6%—40%、显存679—1,271MiB，处于数据加载/早期计算阶段。
- `ARTIFACTS_COMPLETE=0`、`TECHNICAL_FAILURE=0`；在本run目录未检出Traceback、RuntimeError、ValueError、OOM或final-artifact确定性异常。
- 判定：`RUNNING`。启动绑定和资源分配符合冻结矩阵；不得因早期利用率或性能停止、重启或热补丁。

## 2026-09-01 15:43只读监控

- 直连预检通过：普通账户`szu2070436088`、项目根和8张RTX 3090可见；未使用管理员账户。
- dispatcher PID`3472323`仍存活，父PID为1，cmdline继续精确绑定release`phase1_pairbicad_cv2_e200_5f785287`、r5 run根、ManySig和GPU容量`0:2,...,7:2`。
- 24行目录全部存在；`ARTIFACTS_COMPLETE=16/24`，`TECHNICAL_FAILURE=0`。已完成16行均有200行`metrics_epoch.jsonl`。
- 8个直属worker仍绑定未完成行：`CV2-D3-F1/F8`、`CV2-T1-F1/F8`、`CV2-T2-F1/F8`、`CV2-T3-F1/F8`，seed均为`392002`，fold均为1或8。
- 运行中trainer按周期checkpoint持续增长：D3已到`u56500`，T1已到`u36500`，T2已到`u46500`，T3已到`u31000`；这些行在结束前尚未写最终metrics，符合该入口的写出方式。
- GPU2—5处于轻量计算/阶段切换，GPU6—7约90%—92%利用率；GPU0—1检查瞬间空闲，但对应T1 checkpoint刚更新，未形成无进展证据。
- 全run的`train.log`中未检出Traceback、CUDA OOM、`RuntimeError:`或`ValueError:`确定性指纹。判定保持`RUNNING`，无需用户操作，不停止、不重启、不热补丁。

## 2026-09-01 16:02进度与ETA

- `ARTIFACTS_COMPLETE=18/24`、`TECHNICAL_FAILURE=0`；D3的fold1/8已完成`63000/63000 updates`、E200和四场景闭合。
- 剩余6行均在运行，统一计划终点为`63000 updates`：T1约`42000/42500`，T2约`55500/56000`，T3约`37500/37500`。
- 相对15:43检查，T1约推进5500—6000、T2约推进9000—9500、T3约推进6500—7000 updates；checkpoint时间更新至16:01—16:02，未出现停滞。
- 按这段真实吞吐估算，T2预计约16:20前后结束；最终瓶颈为T1/T3，预计约17:10—17:25达到训练终点。D3从最后source-LORO记录到`ARTIFACTS_COMPLETE`约需100秒，因此24行全部artifact闭合预计在17:15—17:30。
- 全量读取24行日志、metrics、runtime和四场景JSON并完成分析、报告与Git发布还需约45—75分钟；若无技术异常，最终`ANALYZED`预计约18:00—18:45。

## 2026-09-01 16:25只读监控

- `ARTIFACTS_COMPLETE=20/24`、`TECHNICAL_FAILURE=0`；T2的fold1/8已完成`63000/63000 updates`、E200和四场景闭合。
- 剩余4个直属worker精确绑定T1与T3的fold1/8；T1已到`49000/50000`，T3两行均到`45500`，计划终点仍为`63000`，checkpoint和source-LORO记录持续增长。
- GPU2、3、6、7仍有本run活动；GPU0、1、4、5已随对应行完成或阶段结束释放。未检出确定性致命错误。
- 按16:02—16:25实际速率，T1预计约17:05—17:12完成，瓶颈T3预计约17:15—17:22完成；24行artifact闭合预计保持在17:20前后。
