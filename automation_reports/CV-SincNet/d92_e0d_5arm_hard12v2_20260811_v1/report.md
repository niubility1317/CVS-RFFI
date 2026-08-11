# D92-E0D-5arm-Hard12-v2实验报告

## 1.基本信息

|字段|内容|
|---|---|
|实验ID|`D92-E0D-5arm-Hard12-v2`|
|run ID|`d92_e0d_5arm_hard12v2_20260811_v1`|
|日期|2026-08-11|
|operator|Codex primary；N607唯一runner|
|当前状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|
|协议|`p2_min_v1`|
|证据范围|`DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN`|
|唯一晋级候选|`E0_FULL_ONLY`|

## 2.目标与假设

目标是在不改变D92的288维表示、B稳健中心、旧/新任务0.5/0.5协方差、F0查询头和通用fail-closed机制的前提下，删除D46昂贵的K折LOO融合，使注册计算显著下降，并检验full-only几何能否同时恢复或提高`H_old_new`。

首轮`D92-BE-2x2-Hard12-v1`真实N607结果表明，关闭E后相对D92_FULL的mean ΔH仅为−0.0114pp，10个performance outer中8个非负，而注册wall中位降低54.04%、增量peak降低30.15%。B0没有稳定性能或资源收益，因此本轮固定B开启，只研究E0路径内的D几何。

本轮主假设：`E0_FULL_ONLY`删除block与LOO后，K5/K10注册组件拟合从E0_FUSION的24/44次降为2/2次，并可能避免support-LOO代理误选，从而同时提升H和降低注册资源。

## 3.冻结五臂

|臂|注册后且K>2计算图|K5/K10 fit|角色|
|---|---|---:|---|
|`D92_FULL`|full/block LOO融合+Fisher/Pareto|48/88|同run原方法参考|
|`E0_FUSION`|full/block classwise support-LOO soft融合|24/44|E0控制|
|`E0_FULL_ONLY`|D92 full主拟合|2/2|唯一晋级候选|
|`E0_BLOCK_ONLY`|D92 block3主拟合|2/2|几何解释臂|
|`E0_FIXED50`|full/block主拟合+RMS固定0.5/0.5融合|4/4|LOO解释臂|

D切换只允许在`registered && K>2`生效。`DA0_REG0`以及K1/K2的`DA0_REG1`五臂必须精确走D92_FULL；query仍逐样本面对全部注册类，不能使用query真值、role、class quota或全局重排。

## 4.Hard12-v2冻结矩阵

Selection SHA256：`2e3b3333a4a325bd0443a31065d3340d6a650a3e89620951a786637e6bce8d3a`。

该矩阵与首轮Hard12-v1的outer交集为0，只作覆盖约束的伪盲压力筛选，不替代完整Target125。

|outer_key|role|K/new|Hard|
|---|---|---:|---:|
|`rx_20_1__seed_713103__k_10__new_20`|performance|10/20|0.554637096774|
|`rx_20_1__seed_713103__k_5__new_20`|performance|5/20|0.674294354839|
|`rx_20_1__seed_713105__k_5__new_20`|performance|5/20|0.707963709677|
|`rx_3_19__seed_713102__k_10__new_20`|performance|10/20|0.686693548387|
|`rx_3_19__seed_713106__k_10__new_10`|performance|10/10|0.630947580645|
|`rx_3_19__seed_713106__k_10__new_5`|performance|10/5|0.459375000000|
|`rx_7_14__seed_713102__k_10__new_5`|performance|10/5|0.243346774194|
|`rx_7_14__seed_713103__k_5__new_20`|performance|5/20|0.666129032258|
|`rx_7_7__seed_713104__k_1__new_20`|liveness|1/20|0.781653225806|
|`rx_7_7__seed_713105__k_10__new_10`|performance|10/10|0.316229838710|
|`rx_8_8__seed_713104__k_10__new_20`|performance|10/20|0.499899193548|
|`rx_8_8__seed_713104__k_1__new_20`|liveness|1/20|0.854838709677|

规模：12outer×5arm=60job；每job固定`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`三场景，共180个scene-arm单元。receiver覆盖`20-1:3,3-19:3,7-14:2,7-7:2,8-8:2`；seed覆盖`713102:2,713103:3,713104:3,713105:2,713106:2`。

## 5.晋级门

只有`E0_FULL_ONLY`可晋级，且必须全部满足：

- 60/60job、8/8 shard完整，10个performance outer先场景等权再同outer配对；
- 五臂`DA0_REG0`状态与预测精确一致；两条K1的`DA0_REG1`状态与预测精确一致；
- query fit/update/selection/truth/role/quota/global-reassignment全部为0；
- 相对`E0_FUSION`：mean ΔH>0且至少8/10 outer非负；
- 相对`D92_FULL`：mean ΔH≥0.005且至少8/10 outer非负；
- 相对`D92_FULL`：old-balanced、old-floor、seen-new均不下降，forgetting不增加；
- FULL_ONLY的K5/K10 fit精确为2/2；
- 注册wall中位相对E0_FUSION至少下降40%，相对D92_FULL至少下降60%；
- 增量peak不高于E0_FUSION，query MAC逐项不变。

任一门失败即`NO_D_GEOMETRY_PROMOTION`。通过也只允许进入完整Target125确认，不构成正式性能主张。

## 6.本地实现与验证状态

|项目|状态|证据|
|---|---|---|
|冻结规格与计划|PASS|Git commit`47c71a47`|
|D模式科学核心|LOCAL_VERIFIED|同一D92 builder内实现四种registered D aggregator；独立核心聚焦15项通过|
|Hard12-v2 builder与runner|LOCAL_VERIFIED|机械聚焦7项、既有BE回归7项、`py_compile`均通过；selection SHA由完整canonical payload真实复算|
|严格分析器|LOCAL_VERIFIED|先写失败测试；7项通过，除two-state fit外同时核对真实after组件inventory|
|primary集成回归|PASS|E0D核心、机械层、分析器及既有BE相关回归共47项通过；随后分析器新增inventory红灯/修复并7项通过|
|首轮独立P0/P1审查|REVISE_CLOSED_FOR_REREVIEW|`P0=0,P1=4`；K1真实计数、跨shard distinct-outer共享失败账本、selection一次复算和发布预注册均已闭合|
|当前运行源码commit|PASS|`7d11a7012ab62058db40f878f925c38160311311`|
|Hard12-v2一次selection audit|PASS|D92/R5各125outer、R5`DA0_REG1`375scene；HiGHS optimal；12行、coverage、v1交集0和输入SHA精确；receipt SHA256=`4c7579337cb18dfb640891d77d3d327c8d2f7e9f3a96e637effb4a383be748f5`|
|P1修复后总回归|PASS|E0D、selection audit和既有BE相关共54项通过；`py_compile`、`bash -n`、`git diff --check`通过|
|独立P0/P1复审|APPROVE_RELEASE|`P0=0,P1=0`；独立59项通过；原4项P1全部闭合|
|真实checkpoint无truth smoke|待执行|N607发布前必过|

已纳入本轮Git提交的核心文件：

- `code/scripts/probe_d92_registration_balanced_covariance.py`；
- `code/cvsrffi/stage2_d92_e0d_slim.py`；
- `code/cvsrffi/stage2_d92_e0d_query_evaluation.py`；
- `code/scripts/run_d92_e0d_prediction.py`；
- `code/cvsrffi/stage2_d92_e0d_hard12.py`；
- `code/scripts/run_d92_e0d_hard12v2.py`；
- `code/cvsrffi/stage2_d92_e0d_analysis.py`；
- `code/scripts/summarize_d92_e0d_hard12v2.py`；
- `code/scripts/audit_d92_e0d_hard12v2_selection.py`；
- `configs/stage2_d92_e0d_5arm_hard12v2_v1.json`及对应测试。

## 7.N607发布预注册

|字段|冻结值|
|---|---|
|本地测试环境|Conda`ssr-gpu`|
|服务器Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|project|`/home/szu2070436088/2510044040/CV-SincNet`|
|source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_source_snapshot_20260811_v1`|
|context|`/home/szu2070436088/2510044040/CV-SincNet/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json`|
|smoke|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_truthfree_smoke_20260811_v1`|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_5arm_hard12v2_20260811_v1`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0d_5arm_hard12v2_20260811_v1`|
|GPU|8个shard固定映射GPU0–7，每卡一个本run进程，`cuda:0`进程内可见|
|CPU threads|每shard 2|
|重试权限|无fresh-run retry；失败保留artifact并回主agent|
|运行源码commit|`7d11a7012ab62058db40f878f925c38160311311`|
|运行闭包|`E:\type10-7\code\snapshots\d92_e0d_runtime_closure_7d11a701.tar.gz`；3519772B；SHA256=`36fc9df5e174ecd87863dcb6663afb6875d5f07ca6d17282648adfa38a7f32df`|
|method lock|`configs/stage2_d92_e0d_5arm_hard12v2_v1.json`；2177B；SHA256=`b80f967e1fc070a730a7b193f691036339930af022682fe2fca81c2e4d229f86`|
|launch|`automation_reports/CV-SincNet/d92_e0d_5arm_hard12v2_20260811_v1/launch.sh`；3396B；SHA256=`ed4e7fc9ba34bb1e30def280d0f5790c96ac117a7875e1d8e5d320dbd87feaa7`；`bash -n`通过|

同步映射固定为：

|本地文件|N607目标|
|---|---|
|`E:\type10-7\code\snapshots\d92_e0d_runtime_closure_7d11a701.tar.gz`|`runs/d92_e0d_source_snapshot_20260811_v1/d92_e0d_runtime_closure_7d11a701.tar.gz`|
|`configs/stage2_d92_e0d_5arm_hard12v2_v1.json`|`runs/d92_e0d_source_snapshot_20260811_v1/configs/stage2_d92_e0d_5arm_hard12v2_v1.json`|
|`automation_reports/CV-SincNet/d92_e0d_5arm_hard12v2_20260811_v1/launch.sh`|`runs/d92_e0d_source_snapshot_20260811_v1/launch.sh`|

精确服务器启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_source_snapshot_20260811_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

launch解包后工作目录为`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_source_snapshot_20260811_v1/code`。它依次执行运行闭包import、prepare、GPU0真实sealed-checkpoint truth-free smoke，再把shard0–7固定映射到GPU0–7。发布只核对运行闭包、method lock、launch三项，不做整树SHA、重复数据验证或额外签名。

预期输出：`matrix_manifest.json`、60份`job_receipt.json`、120份before/after immutable prediction COMMIT、60份独立score、每scene fit/resource audit、8份shard summary、truth-free smoke receipt和取回后的严格分析目录。

## 8.健康停止与成功判据

性能数值不得触发提前停止。仅在以下技术条件触发停止：

- query泄漏、错误checkout/运行闭包、输出覆盖或其他P0；
- 至少两个不同outer在产生prediction前出现相同确定性异常指纹；8个shard使用同一run-root共享账本和`SYSTEMIC_TECHNICAL_FAILURE_STOP.json`协调停派，同outer不同arm不重复计数；
- launcher级不可恢复故障或prediction闭合缺失。

停止前必须绑定本run的PID/CWD/cmdline，只停止本run进程树，保留partial artifacts并标记`NO_PERFORMANCE_RESULT`。正常成功条件为8/8 shard PASS、60/60job receipt、120/120 prediction COMMIT且0失败。

## 9.已知风险

- 历史D0只是方向先验，本轮可能证明FULL_ONLY性能不增；这属于有效负结果。
- 当前block路径仍执行dense 288维求解，因此本轮不预称BLOCK_ONLY更快。
- 旧`estimated_lda_fit_macs`未覆盖D46/D62额外注册图；本轮不使用该字段自证节省，以实际组件fit、wall和peak为主。
- Hard12-v2虽然与v1零交集，但仍是历史困难度选出的development-only压力矩阵；任何晋级都必须完整Target125确认。

## 10.结果区

2026-08-11首次落地时，runner把method lock同步到`source_root`根目录，而冻结launch只读取`source_root/configs/`。driver PID`1803467`在`prepare`、真实checkpoint smoke和任何GPU shard启动前因`test -f`失败退出；`launch_driver.out`与`launch_driver.err`均为0B，smoke/output/logs均未创建，8张GPU保持空闲。

因此本run没有prediction、score或性能结果，禁止进入性能分析。远端source root及三份已同步文件原样保留，不覆盖、不删除、不重启。证据已取回到`E:\type10-7\local_artifacts\d92_e0d_5arm_hard12v2_20260811_v1`，SSH/SCP无残留连接。修复仅改变交付映射和run路径，不改变方法、矩阵、阈值或输入；后续使用新run ID`d92_e0d_5arm_hard12v2_20260811_v2`。
