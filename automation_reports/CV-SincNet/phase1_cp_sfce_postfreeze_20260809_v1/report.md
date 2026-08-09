# phase1_cp_sfce_postfreeze_20260809_v1实验报告

## 1.预注册

- 状态：`LOCAL_VERIFIED / READY_FOR_N607_RELEASE / NO_PERFORMANCE_RESULT`
- 日期：2026-08-09
- 负责人：`/root`；唯一N607 runner：`/root/n607_geosat_lite_runner`
- 目标：对已完整闭环的`phase1_cp_sfce12_20260809_v2`执行一次final-only postfreeze，形成6折C/G同配对clean、三LEO场景、source proxy连续诊断和机械聚合裁决。
- 训练输入：12/12 E40、final checkpoint和terminal receipt齐全；C合同`CONTROL_ARM_NOT_APPLICABLE`，G合同6/6通过；训练阶段没有性能读取。
- 边界：本run零训练、零fit、零校准、零threshold sweep、零checkpoint选择；不读取query真值决定运行或重试。

## 2.冻结矩阵与方法

执行42个固定步骤：12个clean feature export、12个source-only LEO feature export、12个source proxy连续诊断、6个CPU串行C/G pair。GPU映射复用训练矩阵，每卡最多2个export；pair在CPU串行执行，F6只在同一postfreeze root内读取F1–F5不可变pair JSON后封存六折聚合。

每折严格绑定：

- canonical training root leaf=`phase1_cp_sfce12_20260809_v2`；
- candidate=`F{fold}{C/G}_CP_SFCE12`；
- checkpoint=`<training-root>/<candidate>/final_ssdg.pth`；
- clean/LEO/proxy manifest checkpoint路径与字节SHA一致；
- `classification_head_contract=dual_cvsincnet_tx_logits_v1`；
- local4 source TX顺序、class/logit order、strict checkpoint load、physical row order和NPZ SHA；
- LEO行级view=`single`，manifest source profile=`satellite`、TTA=`none`、三场景、固定seed、两天和六RX；
- matrix ID、canonical postfreeze root、fold→pair/source-TX及F6 prior同run绑定。

## 3.本地版本

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 实现commit：`3accce402f3bbce7712889227593baf7fe8b1409`
- 独立实际diff复核：`P0=0，P1=0，ALLOW`

|文件|SHA256|
|---|---|
|`code/scripts/eval_phase1_cp_sfce_pair.py`|`ca3796a04503efb166b13968758baa65a7310e41bccdbd52e1c1465db6e6b083`|
|`code/scripts/launch_phase1_cp_sfce_postfreeze_20260809.sh`|`3008b7e7b753a136c82f77cf681fc1096aee069ff9da5ccbcfd1f8819ac65e57`|
|`code/tests/test_phase1_cp_sfce_postfreeze.py`|`8c77a81d5f9461fbbe1672ca393c71cae9750087cffbac2ae51f9e5269fe75fa`|
|`analysis/phase1_cp_sfce_design_20260809.md`|`71cd359b0d2ade823abca973147e283524f4643cc59d3316d455f531c6eb97ee`|

验证：`py_compile`通过；CB+CP focused 38项通过；`bash -n`通过；dry-run=`12/12/12/6=42`；v1训练root负测exit3；C/G整组交换、错误head、错误checkpoint路径和跨root prior均拒绝；`git diff --check`通过。

## 4.N607发布

- run ID：`phase1_cp_sfce_postfreeze_20260809_v1`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce_postfreeze_20260809_v1_3accce40`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce_postfreeze_20260809_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cp_sfce_postfreeze_20260809_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cp_sfce_postfreeze_20260809_v1.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- training root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2`

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce_postfreeze_20260809_v1_3accce40/code && nohup setsid env POSTFREEZE_RUN_ID=phase1_cp_sfce_postfreeze_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce_postfreeze_20260809_v1_3accce40/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce_postfreeze_20260809_v1_3accce40/code/scripts/launch_phase1_cp_sfce_postfreeze_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cp_sfce_postfreeze_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 5.健康、产物与裁决

- 只按执行故障、P0、无输出或至少两个candidate同一确定性异常停止；不按性能值停止。
- expected：12 clean NPZ、12 LEO NPZ、12 proxy JSON/CSV、6 pair JSON、12 candidate logs、6 pair logs、PID/completion/manifest。
- NPZ与checkpoint只留远端；本地只回收小JSON/CSV/log/receipt/manifest。
- fresh-run retry：`NO`。caller timeout先只读核是否landed，不重复启动。
- 完整后机械报告五类非补偿门：技术绑定；clean 6/6四floor相对C不低于-2pp；LEO 18/18四floor相对C不低于-2pp且6/6 fold三场景等权overall不低于0、18格等权overall不低于0；proxy每折AUROC不降且FAR不升；所有artifact/路径/hash闭合。
- 任一门失败即`REJECT_CP_SFCE_PERMANENT`，不得调`lambda/gamma`、投影规则、场景、采样、阈值或节点；不得进入Phase3。

## 6.运行回填

- 状态：`ARTIFACTS_COMPLETE / ANALYZED / REJECT_CP_SFCE_PERMANENT`
- 固定commit：`3accce402f3bbce7712889227593baf7fe8b1409`；本地归档及远端归档SHA：`5fbec9c03addc9b89eaff69226c0195a6b3d1be2d69591af21691cb5667d8e50`。
- 归档成员采用双口径：commit/worktree LF SHA为eval=`ca3796a04503efb166b13968758baa65a7310e41bccdbd52e1c1465db6e6b083`、launcher=`3008b7e7b753a136c82f77cf681fc1096aee069ff9da5ccbcfd1f8819ac65e57`、test=`8c77a81d5f9461fbbe1672ca393c71cae9750087cffbac2ae51f9e5269fe75fa`、design=`71cd359b0d2ade823abca973147e283524f4643cc59d3316d455f531c6eb97ee`；Windows`core.autocrlf=true`归档/远端CRLF SHA为eval=`c35cb44cf4349975814d3f941b6f01cba0e3009289572d7d14c8cbdb101eaeee`、launcher同上、test=`f74db1a41f9143257c5ed4aebeae7c9aa5b91fc9cb5ae8891d73e85a67261015`、design=`014dd68963ee3d79b963286f94f95d7ecd788cf744c3a2ebf7a2762e5cea516a`；未改远端代码。
- 远端release已落地且`release/code`存在、无`release/code/code`；compile、help、`bash -n`、`DRY_COUNT=42`通过。唯一启动launcher PID=`166322`，CWD=`<release>/code`；GPU映射复用训练：GPU0=F1C/F5G、1=F1G/F5C、2=F2C/F6G、3=F2G/F6C、4=F3C、5=F3G、6=F4C、7=F4G。
- 42步结构：12/12 clean NPZ、12/12 source-only LEO NPZ、12/12 proxy JSON+CSV、6/6 pair JSON；clean每臂2400行（source1600/target_old400/proxy_unknown400），LEO每臂1600行（source-only）；12 candidate日志、6 pair日志、`candidate_pids.tsv`存在；结构核均无Traceback/OOM/ERROR指纹。NPZ与checkpoint未下载。
- 只读metadata核验：clean角色/行数、LEO`channel_view=single`、satellite profile/TTA none、三场景、两天六RX、strict checkpoint load和`dual_cvsincnet_tx_logits_v1`均闭合；pair JSON schema和technical binding字段存在，未读取数值性能。
- 2026-08-09 14:17:52直连preflight恢复：普通账户、项目根、8张RTX 3090均可见；8卡均0%/1MiB且无compute process。retrieval-only runner确认run匹配进程为0、历史PID 166322不存在，随后以direct SCP补齐21/21件小artifact；未使用bridge、未覆盖既有文件、未下载NPZ/checkpoint、未重启或重跑。最终本地51件（含历史`retrieval_partial_manifest.json`）：12/12 proxy JSON、12/12 CSV、12/12 candidate log、6/6 pair JSON、6/6 pair log、`candidate_pids.tsv`和outer log各1；19个`.out`的Traceback、RuntimeError、CUDA OOM和ERROR指纹均为0。最终本地`ssh.exe`、SCP进程及到N607/bridge TCP22连接均为0。

## 7.同折结果与非补偿门

本表只比较同一折、同一source TX集合、同一final checkpoint阶段的C/G配对。四元组顺序均为`overall/min-class/min-day/min-RX`；所有数值为`G-C`百分点。该run属于Phase1 source-only DG，`K-shot`、`seen_new_acc`、`H_old_new`和注册状态均为`N/A`，proxy unknown只用于冻结guardrail，不能形成Phase3正式unknown声明。

|折/候选对|source TX|clean Δpp（overall/min-class/min-day/min-RX）|clear Δpp|low-elev Δpp|rain Δpp|proxy AUROC C→G|proxy FAR C→G|裁决|
|---|---|---:|---:|---:|---:|---:|---:|---|
|F1C vs F1G|20-15,20-19,6-15,8-20|+0.125/+0.250/+0.122/+0.755|-0.368/-1.389/-0.621/-1.031|+0.368/-0.694/+0.481/+0.000|+1.172/+3.125/+1.190/+1.266|0.579289→0.562773|0.452500→0.375000|`REJECT_CP_SFCE_PERMANENT`|
|F2C vs F2G|14-10,20-19,6-15,8-20|+0.125/+1.000/+0.000/+1.509|-1.287/-5.556/-1.477/-2.062|+0.551/+1.562/-0.870/+6.742|+0.977/+4.688/+1.786/+3.797|0.518352→0.531777|0.867500→0.867500|`REJECT_CP_SFCE_PERMANENT`|
|F3C vs F3G|14-10,14-7,6-15,8-20|-0.062/+0.250/+0.122/+0.000|+0.551/+0.694/+0.311/+1.031|+1.654/+0.000/+1.740/+2.247|+1.172/+1.562/+3.125/+0.000|0.600463→0.633945|0.417500→0.437500|`REJECT_CP_SFCE_PERMANENT`|
|F4C vs F4G|14-10,14-7,20-15,8-20|+0.000/+0.000/+0.000/-0.377|+2.022/+6.944/+2.048/+4.124|+0.551/+3.125/+0.962/+5.618|+0.977/+4.688/+0.694/+2.532|0.552255→0.548604|0.592500→0.642500|`REJECT_CP_SFCE_PERMANENT`|
|F5C vs F5G|14-10,14-7,20-15,20-19|+0.812/+4.750/+0.856/+0.755|+2.757/+5.642/+2.174/+7.216|+3.125/+1.562/+1.213/-6.742|+2.148/+16.406/+6.052/-3.947|0.612602→0.589055|0.355000→0.500000|`REJECT_CP_SFCE_PERMANENT`|
|F6C vs F6G|14-7,20-15,20-19,6-15|+0.125/+1.750/+0.367/+0.755|+3.125/+6.250/+5.405/+4.478|+1.103/+0.781/+3.365/-2.247|+2.344/+4.688/+1.042/+6.329|0.548427→0.563605|0.335000→0.412500|`REJECT_CP_SFCE_PERMANENT`|

|冻结门|结果|关键证据|
|---|---|---|
|technical binding|通过|6/6折通过|
|clean 6/6四floor≥-2pp|通过|6折全部通过|
|LEO 18/18四floor≥-2pp|失败|F2 clear的min-class=-5.556pp、min-RX=-2.062pp；F5 low-elev/rain的min-RX=-6.742/-3.947pp；F6 low-elev的min-RX=-2.247pp|
|6/6折三场景等权overall≥0|通过|F1..F6分别为+0.391、+0.080、+1.126、+1.183、+2.677、+2.191pp|
|全局18格等权overall≥0|通过|overall=+1.274637pp；同时min-class=+3.004437pp、min-day=+1.589968pp、min-RX=+1.630590pp，仅为边际统计|
|proxy AUROC不降且FAR不升|失败|仅F2同时通过；F1 AUROC下降，F3/F6 FAR上升，F4/F5两项均退化|

## 8.最终裁决

- F6封存的机械聚合裁决与本地独立复算一致：`REJECT_CP_SFCE_PERMANENT`。
- 全局overall平均提升不能补偿LEO逐格floor和proxy guardrail失败；不得把+1.274637pp写成可晋级性能结论。
- 冻结合同要求停止CP-SFCE G分支：不得调`lambda/gamma`、梯度投影、场景、采样、阈值或节点，不得重跑该run，也不得让G进入Phase3。
- 后续Phase3协同实验只能消费另行冻结且已获准的Phase1本地artifact；若继续Phase1探索，必须作为独立新候选、新设计和新不可覆盖run ID，不得把本次拒绝改写为可调参结果。
