# P1-PAMR六折技术审计v2报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`ARTIFACTS_COMPLETE / TECHNICAL_AUDIT_COMPLETE / NO_PERFORMANCE_RESULT`

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

## 5.N607技术终态与artifact（2026-08-09）

状态已由`LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED`更新为`ARTIFACTS_COMPLETE / TECHNICAL_AUDIT_COMPLETE / NO_PERFORMANCE_RESULT`。direct N607 preflight通过：普通用户`szu2070436088`、项目根可见、8卡均0%/约1MiB；v2 release/run/log/outer此前均不存在，v1未触碰。固定commit`ee31aa3345f9e5e3251bacf7de17098377b67bc0`以无prefix git archive落地，archive SHA256=`043cd042ef8c3ce89538a812ee953dfee2a44841bea85a6d3bf1b9e00f5940b5`（261468160 bytes）。远端LF成员SHA：`phase1_pamr.py=58c84330820caead3e748608ccdde27526ab9148a823ce5b6011e075755fa87d`、`train_ssdg.py=b1e3d6082bd707ef4aa79a128153ed5c4cc14423bdbae1f065733ad2d1632969`、`test_phase1_pamr.py=1b1477e35cdb62aa387015d908459f3ccf6d2f29864e6ffc9f48b09436615c57`、`launch_phase1_pamr_audit6_20260809.sh=90dd5101ec2388f6f9c889a3860bb6c1a9bdb3a10f1c30c6a776a174c25359d1`、`launch_phase1_pamr12_20260809.sh=0bdd23759123aa459dbc898f8ae722c4e713cf77648cfbe8bfb8520e3ce9a8e3`、`phase1_pamr_design_20260809.md=35c6927e6f35dbc35998f777e7e6c3c1d9d59130a314d8c3617780c394dc15ba`；Windows工作树差异仅CRLF/LF归档口径，未远端改码。远端py_compile、help、两份`bash -n`与dry-run=6通过。

唯一启动一次，精确命令见§3；detached launcher PID=`4148966`，child为`F1G=4148974/GPU0`、`F2G=4148976/GPU1`、`F3G=4148978/GPU2`、`F4G=4148980/GPU3`、`F5G=4148982/GPU4`、`F6G=4148984/GPU5`。launcher exit=0；运行后进程均退出，GPU回到0%/约1MiB。实际冻结baseline为F1C–F6C，终态receipt与远端sha256sum逐项一致：F1C=`4d515204f2cea62c5b82313a01b722b3b3d13a3e4fe647ff4b723b69e8a0c040`、F2C=`29c7d7ca31d80d90d7c0235fa234707b05866914dc0acdae5c44505af1bbd76d`、F3C=`39c6cdd65aade504efdea956db02cc5e762aee299a9e9319c07ed6fb839434b7`、F4C=`32d956f44f60844471ba2ef04526c5f40cad0f8bc8acb7249be6035aa85005e4`、F5C=`2b9381546878b19e7e8e2106a82b0d0a4672a3012ef79bd7f28eadfd03b75a9f`、F6C=`573ca9d039a8c854f9c0927b5b5c303ab8eeaf527ccd42cd0d764b81e630de6f`；ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。

|fold|local TX顺序|binding SHA前8位|data_ctx/head/source roles|valid anchors|active hinges|raw grad nonzero/nonfinite|terminal|
|---|---|---|---:|---:|---:|---|---|
|F1G|20-15,20-19,6-15,8-20|`aad91590`|4/4/4|3827|2689|1/0|`TECHNICAL_AUDIT_COMPLETE`|
|F2G|14-10,20-19,6-15,8-20|`6b4490b5`|4/4/4|3815|3091|1/0|`TECHNICAL_AUDIT_COMPLETE`|
|F3G|14-10,14-7,6-15,8-20|`44a24367`|4/4/4|3824|3114|1/0|`TECHNICAL_AUDIT_COMPLETE`|
|F4G|14-10,14-7,20-15,8-20|`d3b5b42b`|4/4/4|3827|2902|1/0|`TECHNICAL_AUDIT_COMPLETE`|
|F5G|14-10,14-7,20-15,20-19|`693cc5a1`|4/4/4|3764|3359|1/0|`TECHNICAL_AUDIT_COMPLETE`|
|F6G|14-7,20-15,20-19,6-15|`91f34fc7`|4/4/4|3738|3305|1/0|`TECHNICAL_AUDIT_COMPLETE`|

六折均严格满足`LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER`，per-TX valid anchor与active hinge均有覆盖，shared-gradient technical contract通过；metrics/receipt中的性能评估均固定`SKIPPED_TECHNICAL_AUDIT / NO_PERFORMANCE_RESULT`，不作性能解读或晋级判断。远端run闭环为6 folds、6 metrics CSV、6 metrics JSONL、6 final checkpoint、6 PAMR config receipt、6 PAMR terminal receipt、6 training completion receipt、6 log；final checkpoint仅保留远端，未下载。

小型artifact已回收到`E:\type10-7\automation_reports\CV-SincNet\phase1_pamr_audit6_20260809_v2\artifacts`，共58个文件（manifest entry=57，另含manifest），无`.pth/.npz`。manifest SHA256=`bc50969ac7845241d7fd75e2f6bba47c513d4ed9556cb6048dd90554583d7776`；completion.tsv（header+7行）SHA256=`7bd73b5735c466586c2097f65de78e69bc96b4aef27b6ff0e07d2c32e2e9182c`；传输临时tar SHA256=`c771fdabeba9cf8856f49ebb226110cf9200f0f6eb98389d4b246c638c5eab51`（78975 bytes），已删除并确认`TEMP_ABSENT`。manifest逐项本地哈希/bytes/lines核验57/57通过；远端manifest哈希与本地一致。SSH进程、TCP/22均为0；无重试、无远端代码修改、无性能结论。
