# P1-PAMR六折技术审计报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED`

证据边界：`TECHNICAL_ONLY / NO_PERFORMANCE_RESULT`

## 1.目标与冻结机制

实验ID：`phase1_pamr_audit6_20260809_v1`。时间：2026-08-09。operator：Codex主控；N607唯一runner：专属实验子代理。

P1-PAMR用同physical clean观测的detached raw-cosine分类margin作为标量边界，只在LEO观测丢失“正确类—最难异类”角margin时产生hinge。按TX等权聚合，固定`lambda_pamr=0.05`。无EMA或外部teacher、无新head、无阈值、无显式z对齐、无RX/domain标签、无GRL/MMD/CORAL；proxy、held和LEO评估行均不进入训练、校准或选择。

本run仅执行F1G…F6G各1epoch source-train-only技术审计，GPU0…5各一条。审计首个有效且active-hinge batch记录raw未缩放PAMR梯度及共享encoder与base loss的梯度余弦/范数比；不读取source-val、LEO、tail、leakage或heldout性能，固定输出`SKIPPED_TECHNICAL_AUDIT / NO_PERFORMANCE_RESULT`。

## 2.版本与本地验证

Git commit：`79c5b245fb411cbeb33ff100cbaaeac1e471dfb0`。独立复核：`P0=0 / P1=0 / MERGE / ALLOW`。

|文件|SHA256|
|---|---|
|`code/cvsrffi/phase1_pamr.py`|`919cebe847553eaf60b7e65eacdf52dae8c733ee8d5649007d57720ec8f415c7`|
|`code/SSDG/train_ssdg.py`|`6c7a8d4f2b153f64f83b323b55b51bf838ba42f705d6dda7779f8375282be181`|
|`code/tests/test_phase1_pamr.py`|`c340be9ed92cbb3bfe8d13bbb8677ce60d1deff6dd3417035c3b71683a112823`|
|`code/scripts/launch_phase1_pamr_audit6_20260809.sh`|`90dd5101ec2388f6f9c889a3860bb6c1a9bdb3a10f1c30c6a776a174c25359d1`|
|`code/scripts/launch_phase1_pamr12_20260809.sh`|`0bdd23759123aa459dbc898f8ae722c4e713cf77648cfbe8bfb8520e3ce9a8e3`|
|`analysis/phase1_pamr_design_20260809.md`|`1c05506bea377c4d494dccb027679d46dad01ad68ec9573200f57ba6e9c38ec5`|

本地`ssr-gpu`验证：py_compile通过；CCPC+PAMR focused pytest共41 passed；两份launcher `bash -n`通过；dry-run分别精确6行和12行；`git diff --check`通过。

## 3.N607路径与唯一命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_audit6_20260809_v1_79c5b245`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr_audit6_20260809_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr_audit6_20260809_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr_audit6_20260809_v1.launch.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_audit6_20260809_v1_79c5b245/code && nohup setsid env RUN_ID=phase1_pamr_audit6_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_audit6_20260809_v1_79c5b245/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl GEOSAT_CKPT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1 RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr_audit6_20260809_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr_audit6_20260809_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_audit6_20260809_v1_79c5b245/code/scripts/launch_phase1_pamr_audit6_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr_audit6_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 4.健康门与artifact

启动前核commit归档、六文件hash、ManySig与6个GeoSat-C checkpoint、目标路径不存在、GPU与活动进程。仅错checkout/hash、P0、输出覆盖、OOM/CUDA、至少2fold相同确定性异常、raw PAMR梯度None/nonfinite、共享梯度异常或无进展触发技术停止；不读取性能。retry=`NO`。

成功要求6/6 exit0、`TECHNICAL_AUDIT_COMPLETE`、每折至少一个raw非零有限PAMR梯度、每个source TX均有valid anchor与active hinge、共享梯度关系receipt完整、所有性能评估固定跳过。只回收小metrics/log/receipt/manifest，不下载checkpoint。通过仅授权新的40epoch完整run，不构成晋级或性能结论。
