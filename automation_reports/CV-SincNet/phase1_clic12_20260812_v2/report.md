# Phase1 CLIC 12臂训练v2预注册与运行报告

## 1. 状态与目的

- 实验ID：`phase1_clic12_20260812_v2`
- 时间：2026-08-12（Asia/Hong_Kong）
- 当前状态：`LANDED / REMOTE_STATIC_VERIFIED / READY_TO_LAUNCH`
- 目的：保持v1科学方法、12臂矩阵、数据、seed、epoch、loss和GPU映射不变，仅修复launcher对`argparse store_true`开关多传字面量`true`的发布缺陷，重新取得真实训练checkpoint。
- v1：`phase1_clic12_20260811_v1`已封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；唯一launch=1，12/12在训练前同指纹`unrecognized arguments: true`退出，0 checkpoint、0 terminal receipt、GPU已释放。v1不得重试。

## 2. 根因、修复与证据

- 根因：`--use_sat_consistency`在`train_ssdg.py`中是无值`store_true`开关；v1 launcher错误写成`--use_sat_consistency true`，后一个token被argparse视为未识别参数。
- 最小修复：只删除该开关后的`true`，并把默认run ID改为v2；科学配置仍为`use_sat_consistency=true`。
- 防回归：dry-run的12条子命令现在逐条送入真实`train_ssdg.build_arg_parser().parse_args(...)`，不再只做文本计数。
- 修复commit：`543a1dfb`（`fix: parse CLIC launcher booleans before launch`）。
- 变更文件仅：`code/scripts/launch_phase1_clic12_20260811.sh`、`code/tests/test_phase1_clic.py`。
- 本地`ssr-gpu`：精确RED复现`unrecognized arguments: true`；修复后launcher test通过；完整CLIC为`164 passed`；`py_compile`、`bash -n`、12臂dry-run通过。

## 3. 冻结科学合同

与v1完全相同：F1—F6×C/G共12臂；C=`raw_phase_control`，G=`complex_local_invariant_curvature`；seed=`7281164`；40epoch；batch=128；AdamW；lr=`2e-4`；`L_base=clean CE+0.10×KL(clean-stopgrad→single-LEO)`；三种source LEO weak场景；final-only checkpoint；旧机制全关闭；target/query/target truth/role/正式unknown零训练访问。

GPU映射保持：0=`F1C,F5G`；1=`F1G,F5C`；2=`F2C,F6G`；3=`F2G,F6C`；4=`F3C`；5=`F3G`；6=`F4C`；7=`F4G`。

## 4. N607合同

- 发布commit：`543a1dfb`；必须从该commit构造干净Git archive，不得带入未完成Task7工作树。
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic12_20260812_v2_543a1dfb`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic12_20260812_v2`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic12_20260812_v2`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/phase1_clic12_20260812_v2_outer.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 启动：从release的`code`目录执行launcher；`RUN_ID=phase1_clic12_20260812_v2`，其余路径同v1。
- v2唯一launch调用最多1，fresh-run retry=`NO`。启动前确认v2 release/run/log/outer均不覆盖、8GPU资源允许、12 warm-start存在。

## 5. 健康与停止规则

启动后核outer PID、12个candidate PID/CWD/cmdline/run-root/GPU、`pids.tsv`和日志增长。仅在P0/安全、wrong checkout/hash/CWD、覆盖、launcher-wide确定性故障，或至少两个不同candidate在有效checkpoint前同一确定性异常指纹时停止本run拥有的进程树。不得用accuracy、floor、AUROC、u-gap或其他性能值停止或调参。

## 6. 预期工件与后续评价

每臂必须产生`final_ssdg.pth`、`phase1_clic_terminal_receipt.json`和完整log。训练完成后执行clean、三种LEO weak、fixed400 source-proxy、PAIR、G deployment bundle和目标域盲态unknown评测。每个实验指标都必须包含叠加LEO weak的目标域测试；核心目标是未知类拒识与域泛化。未达性能门应保存`passed=false`，不误记为技术失败。

## 7. 运行回填

- archive SHA/bytes：`4C5A96C37078E7016B838D6CA4694C07811166D90E2F259424C001EEEC5EEB33`，266844160 bytes；SCP=1，远端SHA/bytes闭合。
- release静态门：通过；远端py_compile、`train_ssdg.py --help`、`bash -n`、12行dry-run（C6/G6、`lambda_sat_cons=0.10`）和release无pycache通过。
- launch时间/次数：待runner
- outer与candidate PID/GPU：待runner
- 首波技术健康：待runner
- 最终状态：待runner
