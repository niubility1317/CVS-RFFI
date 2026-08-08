# P1-PAMR六折技术审计报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

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

## 5.远端落地、首波停止与partial artifact（2026-08-09）

状态已由`LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED`更新为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。direct N607只读preflight通过：普通用户`szu2070436088`、项目根可见、8卡均0%/约1MiB，目标release/run/log/outer此前均不存在；远端临时包清理后复核`TEMP_TAR_ABSENT`，本run进程为空、GPU仍空闲。

落地使用固定commit`79c5b245fb411cbeb33ff100cbaaeac1e471dfb0`的无prefix git archive；远端archive SHA256=`dedce09706f0bcc5a5eb8914f0aadfde55ad267a2b6507fbf50ab3dd75041768`（261447680 bytes）。远端LF字节口径成员SHA：`phase1_pamr.py=be0e68a23cdece8d21fb592c4092ebda7700648b37e7ffdff1c4cc9fa2d4e2b7`、`train_ssdg.py=afc2bb479cc77c6b5e826d6e0e65bdee47d1314cd902b1099469ae2b029d1ef1`、`test_phase1_pamr.py=9b8b0dd6445d3363abbf2884dfb9057cb14a2f379b3538b489ddee9f30943025`、`launch_phase1_pamr_audit6_20260809.sh=90dd5101ec2388f6f9c889a3860bb6c1a9bdb3a10f1c30c6a776a174c25359d1`、`launch_phase1_pamr12_20260809.sh=0bdd23759123aa459dbc898f8ae722c4e713cf77648cfbe8bfb8520e3ce9a8e3`、`phase1_pamr_design_20260809.md=ddfce181a2b6b7c99a055bb1ee2cb59d50ab4f165453be195f605d2af2b84429`；与Windows工作树SHA差异仅为CRLF/LF归档字节口径，未改远端代码。远端py_compile、help、两份`bash -n`与dry-run（6/12行）均通过。

唯一启动一次，CWD为`<release>/code`，launcher detached PID=`4135436`，GPU映射为`GPU0:F1G(4135441), GPU1:F2G(4135443), GPU2:F3G(4135445), GPU3:F4G(4135447), GPU4:F5G(4135449), GPU5:F6G(4135451)`；精确命令见§3。六个child均在首个训练telemetry前exit=1，未生成fold run目录、metrics或技术receipt；outer launch log为0 bytes。

|fold|PID/GPU|exit|错误指纹|data_ctx num_classes|head rows|source role count|
|---|---|---:|---|---|---|---|
|F1G|4135441/GPU0|1|`PAMRConfigurationError`（train_ssdg.py:5919）|`NOT_PRODUCED`|`NOT_PRODUCED`|`NOT_PRODUCED`|
|F2G|4135443/GPU1|1|同上|`NOT_PRODUCED`|`NOT_PRODUCED`|`NOT_PRODUCED`|
|F3G|4135445/GPU2|1|同上|`NOT_PRODUCED`|`NOT_PRODUCED`|`NOT_PRODUCED`|
|F4G|4135447/GPU3|1|同上|`NOT_PRODUCED`|`NOT_PRODUCED`|`NOT_PRODUCED`|
|F5G|4135449/GPU4|1|同上|`NOT_PRODUCED`|`NOT_PRODUCED`|`NOT_PRODUCED`|
|F6G|4135451/GPU5|1|同上|`NOT_PRODUCED`|`NOT_PRODUCED`|`NOT_PRODUCED`|

完整指纹为`cvsrffi.phase1_pamr.PAMRConfigurationError: P1-PAMR requires source TX role count to equal local classifier class count`，Traceback固定落在`code/SSDG/train_ssdg.py:5919`，六个不同fold同指纹（6/6），触发预注册系统性技术停止；未发kill，进程自然退出。因异常发生在data context/head/source-role telemetry之前，三项均如表记录为`NOT_PRODUCED`，不得从配置推断数值。`completion.tsv`为header+7行（6 child+launcher），launcher exit=1；未生成性能值，NO retry。

小型partial证据已回收到`E:\type10-7\automation_reports\CV-SincNet\phase1_pamr_audit6_20260809_v1\artifacts`（不含checkpoint/NPZ）：`runs/phase1_pamr_audit6_20260809_v1/manifest.json`（5757 bytes，SHA256=`6e3d36f4dfc81b1ce10acca79007d311441641c4f3757913d300c2bfffc84745`）；`logs/phase1_pamr_audit6_20260809_v1/completion.tsv`（1588 bytes，SHA256=`0b3dfc8fb11f7216bc48be350995c2f0fa76e3061bd87f207fb57b07dbcc6028`）；`pids.tsv`（1346 bytes，SHA256=`46e61ac83ebbace06f5ccc3eb422196857ab6b03e4fc759a6280eae7c9f911ce`）；六份fold traceback各680 bytes、SHA256=`14558b616053bb593c891dd684b03f2343fe3d82d1b456fe448fb45588e25b63`；outer为空文件（SHA256=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`）。远端临时tar仅用于传输，SHA256=`f74fc402a2c8026f6e029371429512c47def99dd58c3c6b12fd02254d7c33bd5`、2886 bytes，已删除并确认不存在；本地SSH进程与TCP/22均为0。该run保留partial、无性能结论、无重试。
