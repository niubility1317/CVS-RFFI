# P1-PAMR六折技术审计v2报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED`

证据边界：`TECHNICAL_ONLY / NO_PERFORMANCE_RESULT`

## 1.目标与v1定点修复

实验ID：`phase1_pamr_audit6_20260809_v2`。时间：2026-08-09。operator：Codex主控；N607唯一runner：专属实验子代理。

P1-PAMR机制、`lambda_pamr=0.05`、六折矩阵与1epoch source-train-only审计均不变。v1六折在训练前因`data_ctx`未返回局部`num_classes`而一致退出，无性能结果。v2仅修复类序绑定：显式返回local4；强制local TX序=source receipt=checkpoint训练TX序=live head4行；记录全局6→局部4映射及binding SHA。global6 head、head宽度或类序漂移均训练前fail-closed。

## 2.版本与本地验证

implementation commit：`ee31aa3345f9e5e3251bacf7de17098377b67bc0`；原PAMR commit：`79c5b245fb411cbeb33ff100cbaaeac1e471dfb0`。独立复核：`P0=0 / P1=0 / ALLOW`。

|文件|SHA256|
|---|---|
|`code/cvsrffi/phase1_pamr.py`|`a9cc20cb7109f1a1011954a3d3821174ddacf0fb866e4ce9615a6a6519edab59`|
|`code/SSDG/train_ssdg.py`|`81bdc5d295f68f2d9fd36afec8ca98abbcac27378c01982cadc163d71fad0191`|
|`code/tests/test_phase1_pamr.py`|`72285834162ca03cdc58ab91d0389d7e71139689cfe531135a07f19341952eb1`|
|`code/scripts/launch_phase1_pamr_audit6_20260809.sh`|`90dd5101ec2388f6f9c889a3860bb6c1a9bdb3a10f1c30c6a776a174c25359d1`|
|`code/scripts/launch_phase1_pamr12_20260809.sh`|`0bdd23759123aa459dbc898f8ae722c4e713cf77648cfbe8bfb8520e3ce9a8e3`|
|`analysis/phase1_pamr_design_20260809.md`|`e541eea10490c77ba16431885bf21df6faa37b1d3e1692b5df95e0b750140e6c`|

本地`ssr-gpu`：py_compile通过；CCPC+PAMR focused pytest共43 passed；launcher `bash -n`通过；dry-run为6/12；`git diff --check`通过。

## 3.N607路径与唯一命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_audit6_20260809_v2_ee31aa33`
- run/log：`/home/szu2070436088/2510044040/CV-SincNet/{runs,logs}/phase1_pamr_audit6_20260809_v2`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr_audit6_20260809_v2.launch.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_audit6_20260809_v2_ee31aa33/code && nohup setsid env RUN_ID=phase1_pamr_audit6_20260809_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_audit6_20260809_v2_ee31aa33/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl GEOSAT_CKPT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1 RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr_audit6_20260809_v2 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr_audit6_20260809_v2 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_audit6_20260809_v2_ee31aa33/code/scripts/launch_phase1_pamr_audit6_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr_audit6_20260809_v2.launch.out 2>&1 < /dev/null & echo $!
```

## 4.健康门

F1G…F6G各1epoch、GPU0…5。成功要求6/6 exit0与`TECHNICAL_AUDIT_COMPLETE`；每折local4/head4/class-order binding通过、每个source TX均有valid anchor与active hinge、raw PAMR梯度nonzero≥1且nonfinite=0、shared relation receipt完整；全部性能评估固定`SKIPPED_TECHNICAL_AUDIT / NO_PERFORMANCE_RESULT`。错hash/P0/覆盖、OOM/CUDA、至少2fold同指纹、映射/梯度/coverage异常或无进展即技术停止；不看性能，retry=`NO`。只回收小artifact，不下载checkpoint。
