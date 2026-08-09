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

- 状态：`PENDING_N607`
