# RIEI/DRIFT全source receiver Phase1实验报告

- run_id：`phase1_riei_drift_newsplit_allsource_s392002_20260901_r1`
- 当前状态：`RUNNING`
- 候选矩阵：RIEI=`RIEI_C06_sum_featnorm1e4`、DRIFT=`DRIFT_N02_raw_cap4000`，各1行。
- 冻结代码提交：`2df2a33689fcd75587424e68afec44c0e13015d7`
- 本地环境/CWD：`ssr-gpu`；`E:\type10-7\github_publish\CVS-RFFI-repo`
- N607环境/CWD：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；不可变release目录`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1_2df2a336`
- 数据输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1`
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1`
- GPU：RIEI→GPU0；DRIFT→GPU1。
- 停止规则：固定200epoch，不按updates或性能提前终止；仅在数据/query越权、错误seed/day/receiver、输出碰撞、错误checkout、无prediction闭合、launcher级故障或两行出现相同确定性异常时，精确绑定并停止本run进程树，保留全部产物。
- 预期artifact：每行`best_by_val.pt`、`metrics.json`、clean与`leo_clear_weak`/`leo_low_elev_weak`/`leo_rain_weak`逐场景结果及日志；run根包含`matrix_manifest.json`和`launch_receipt.json`。

## 冻结矩阵与协议

|方法|seed|L_s/U_s/V|训练source receivers|GPU|
|---|---:|---|---|---:|
|RIEI|392002|0.07/0.63/0.30|[1,3,4,6,8]|0|
|DRIFT|392002|0.07/0.63/0.30|[1,3,4,6,8]|1|

两行均不使用fold或`--wisig_source_holdout_rxs`。训练使用day1–day3、单一source验证集`V=0.30`、200epoch和source V选模。星地增强固定为真实`clean+satellite`拼接CE-only，`lambda_sat_cls=0.68`、`lambda_sat_cons=0`，E80开始卫星辅助CE，使用三阶段`LEO_WEAK`课程。checkpoint冻结后才测试全部目标receiver、day1–day4的clean和三种LEO场景；目标结果不得反馈选种、调参或重训。

## 启动命令

```text
PYTHONPATH=<release>/code:<release> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m cvsrffi.phase1_baseline_fold_matrix --run-id phase1_riei_drift_newsplit_allsource_s392002_20260901_r1 --project-root <release> --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1 --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1 --python-bin /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --gpu-ids 0,1 --all-source --execute
```

## 本地验证与审查

- TDD：全source构造测试先因功能不存在失败，实施后通过；GPU默认P1测试先复现失败，修复后通过。
- 聚焦测试：`8 passed`；Python编译和`git diff --check`通过。
- 独立P0/P1审查发现并修复1个全source默认GPU数量P1；仅针对原问题定点复审后`READY`。
- 命令级dry-run：2行，完整source receivers均为`[1,3,4,6,8]`，无holdout参数。
- Git push与远端OID回读：`VERIFIED`，远端`work/cvs-active`=`2df2a33689fcd75587424e68afec44c0e13015d7`。

## Release、smoke与正式启动

- release映射：本地`E:\type10-7\release_archives\phase1_riei_drift_newsplit_allsource_s392002_20260901_r1_2df2a336.tar.gz`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1_2df2a336.tar.gz`。
- 唯一归档SHA256本地/远端一致：`b2c5cfd161d4d716c88a8da23e4a5974bc5da6248e7d7be0cc40bb5daeff90c5`；远端解包和关键入口编译`PASS`。
- N607预检：普通账户直连`PASS`；GPU0/1空闲；新release、run、log路径启动前均不存在。
- 真实checkpoint无query smoke：`PASS`。RIEI_C06的epoch186 checkpoint载入132个有限tensor；DRIFT_N02的epoch76 checkpoint载入136个有限tensor；未读取数据集或目标query。
- 正式启动时间：2026-09-01约16:45 CST。
  - RIEI：PID`3637386`，GPU0，row=`RIEI_ALLSRC_S392002`。
  - DRIFT：PID`3637387`，GPU1，row=`DRIFT_ALLSRC_S392002`。
- 启动后回读：两个PID的PPID/PGID/SID、CWD、完整cmdline、run-root和`CUDA_VISIBLE_DEVICES`绑定正确；两行均已完成E1并进入E2，日志增长且已写出E1的`best_by_val.pt`；未见`Traceback`、`RuntimeError`、CUDA OOM、`Killed`或`AssertionError`。
