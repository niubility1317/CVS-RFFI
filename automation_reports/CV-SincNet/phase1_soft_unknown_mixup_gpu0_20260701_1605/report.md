# Phase1 soft unknown mixup GPU0双实验报告

## 基本信息

|字段|内容|
|---|---|
|run_id|`phase1_soft_unknown_mixup_gpu0_20260701_1605`|
|时间|2026-07-01|
|操作者|Codex|
|目标|验证方案A的3-TX软未知mixup是否提升地面开放世界拒识能力,并观察旧类/卫星压力指标是否保持稳定|
|GPU策略|只使用GPU0,并发2个训练实验|
|epoch设置|每个实验`--epochs 200`,其中`--label_epochs 150`,`--pseudo_epochs 50`|

## 协议边界

本批实验属于Phase1 source-only ground training验证,使用`ManySig.pkl`和`tx_rx_day_1_7_2`源域切分,不使用目标接收机`R_t`样本、Stage2 support/query或unknown query调阈值。结果只能作为地面训练开放世界表征与后续Stage2候选选择依据,不能直接写作在轨部署成功。

## 假设与比较目标

|候选|假设|比较目标|
|---|---|---|
|`SOFTUNK_A_BALANCED_E200`|中等强度3-TX软未知mixup能降低虚拟未知被旧类接受率,同时旧类和satellite stress不明显下降|当前`phase1_accept_domain_verify_20260701_130328`中R17/R20附近候选|
|`SOFTUNK_A_STRONGISO_E200`|更强mixup/vacuum/source episode隔离能进一步压低未知接受率,代价可能是旧类紧致性过强|`SOFTUNK_A_BALANCED_E200`和旧R17/R20候选|

## 关键参数

|候选|`lambda_open_world_feat`|`lambda_zid_compact`|`lambda_soft_unknown_mixup`|mixup count/order|soft CE|soft vacuum|`source_episode_mixup_weight`|source cap|
|---|---:|---:|---:|---|---:|---:|---:|---:|
|`SOFTUNK_A_BALANCED_E200`|0.0018|0.022|0.0030|16/3|0.80|0.35|0.75|34|
|`SOFTUNK_A_STRONGISO_E200`|0.0022|0.026|0.0040|24/3|0.60|0.50|1.00|32|

开放世界和`zid_compact`从epoch121开始、warmup10;proxy unknown和soft unknown mixup从epoch141开始、warmup10,确保200epoch内实际生效。

## 本地文件与验证

|文件|用途|
|---|---|
|`code/cvsrffi/losses.py`|已在`b430263`提交中加入3-TX软未知mixup和source episode mixup隔离|
|`code/SSDG/train_ssdg.py`|已在`b430263`提交中加入训练接线、CLI和telemetry|
|`code/scripts/launch_phase1_soft_unknown_mixup_gpu0_20260701.sh`|本批GPU0双实验launcher|

|验证命令|结果|
|---|---|
|`conda run --no-capture-output -n ssr-gpu python -m pytest tests\test_soft_unknown_mixup_losses.py -q`|通过,2 passed|
|`conda run --no-capture-output -n ssr-gpu python -m py_compile code\cvsrffi\losses.py code\SSDG\train_ssdg.py`|通过|
|`conda run --no-capture-output -n ssr-gpu python code\SSDG\train_ssdg.py --help`|通过,新增参数可见|
|`bash -n code/scripts/launch_phase1_soft_unknown_mixup_gpu0_20260701.sh`|通过|
|`bash code/scripts/launch_phase1_soft_unknown_mixup_gpu0_20260701.sh --dry-run`|通过,仅生成两个GPU0候选,均为`--epochs 200`和`--soft_unknown_mixup_order 3`|

Git发布仓库提交:

|commit|内容|
|---|---|
|`b430263`|实现3-TX软未知mixup loss、训练接线和单测|
|`d5ad5df`|新增GPU0双实验launcher和本报告|

## N607预检与占用

|检查|结果|
|---|---|
|`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`|通过,直连`N607`,项目根存在,GPU可见|
|GPU0|预检显示0%利用率、10MiB显存;后续进程检查未见GPU0训练进程|
|GPU1-7|已有旧`phase1_accept_domain_verify_20260701_130328`矩阵训练进程,本批不干预|

## 同步计划

|本地路径|远端路径|
|---|---|
|`code/cvsrffi/losses.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/losses.py`|
|`code/cvsrffi/losses.py`|`/home/szu2070436088/2510044040/CV-SincNet/cvsrffi/losses.py`|
|`code/SSDG/train_ssdg.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py`|
|`tests/test_soft_unknown_mixup_losses.py`|`/home/szu2070436088/2510044040/CV-SincNet/tests/test_soft_unknown_mixup_losses.py`|
|`code/scripts/launch_phase1_soft_unknown_mixup_gpu0_20260701.sh`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_soft_unknown_mixup_gpu0_20260701.sh`|

远端验证:

|命令/检查|结果|
|---|---|
|`sha256sum cvsrffi/losses.py code/cvsrffi/losses.py code/SSDG/train_ssdg.py code/scripts/launch_phase1_soft_unknown_mixup_gpu0_20260701.sh`|根包与`code`包`losses.py` hash一致|
|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile cvsrffi/losses.py code/cvsrffi/losses.py code/SSDG/train_ssdg.py`|通过|
|远端`pytest tests/test_soft_unknown_mixup_losses.py`|未执行,远端`CVS-RFFI`环境缺少`pytest`模块|
|内联Python soft-unknown mixup反向传播烟测|通过,输出`INLINE_SOFT_UNKNOWN_MIXUP_OK 4.0 3.0`|
|`bash -n code/scripts/launch_phase1_soft_unknown_mixup_gpu0_20260701.sh`|通过|
|远端launcher `--dry-run`|通过,仅两个GPU0候选|

## 远端启动命令

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
RUN_ID=phase1_soft_unknown_mixup_gpu0_20260701_1605 STAGE2_MAX_ACTIVE_PER_GPU=2 bash code/scripts/launch_phase1_soft_unknown_mixup_gpu0_20260701.sh
```

|候选|远端日志|远端输出目录|PID|
|---|---|---|---|
|`SOFTUNK_A_BALANCED_E200`|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_soft_unknown_mixup_gpu0_20260701_1605/SOFTUNK_A_BALANCED_E200.out`|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_soft_unknown_mixup_gpu0_20260701_1605/SOFTUNK_A_BALANCED_E200`|2632783|
|`SOFTUNK_A_STRONGISO_E200`|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_soft_unknown_mixup_gpu0_20260701_1605/SOFTUNK_A_STRONGISO_E200.out`|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_soft_unknown_mixup_gpu0_20260701_1605/SOFTUNK_A_STRONGISO_E200`|2633189|

## 启动健康检查

|时间点|结果|
|---|---|
|启动前|GPU0为0%利用率、10MiB显存,仅Xorg,无训练计算进程|
|启动后约1分钟|两个PID均存活,GPU0约96%利用率、4308MiB显存;日志进入E004-E006附近,无Traceback/OOM/unrecognized参数|
|启动后约4-5分钟|两个PID均存活,GPU0约96%利用率、4436MiB显存;两个候选均到E009/200;错误扫描为空|

启动后日志确认`[CONFIG-LOSS]`中包含`lambda_soft_unknown_mixup`,并出现`[SOFT-UNK-MIX]`与`[SOURCE-EP]`指标行。由于本批将open-world、zid compact、source episode从epoch121启动,proxy/soft unknown从epoch141启动,早期E001-E009的mixup指标为0属于预期。

## 观察指标

重点观察`train/soft_unknown_mixup_virtual_accept_rate`、`train/soft_unknown_mixup_energy`、`train/source_episode_mixup_overflow_rate`、`train/ow_feat_vacuum_violation_rate`、`train/zid_compact_pos_angle_p95_deg`、clean/test strict UDU、receiver floor、`leo_clear_weak`/`leo_low_elev_weak`/`leo_rain_weak` mean/floor和joint-safe checkpoint状态。

## 风险与停止条件

风险是隔离过强导致旧类准确率、satellite stress或joint-safe指标下降。若启动日志出现unrecognized argument、Traceback、OOM、NaN、无`[EPOCH-BEGIN]`或GPU0无利用率,先看日志根因,不得删除旧run/log或中断GPU1-7旧任务。
