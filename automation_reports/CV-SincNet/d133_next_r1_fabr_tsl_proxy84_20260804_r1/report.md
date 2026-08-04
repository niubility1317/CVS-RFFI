# D133 NEXT-R1 FABR-TSL proxy84实验报告

## 实验身份与状态

|字段|值|
|---|---|
|experiment ID|`d133_next_r1_fabr_tsl_proxy84_20260804_r1`|
|时间|2026-08-04（Asia/Hong_Kong）|
|operator|主agent负责方法/结果；唯一Luna/max runner负责N607落地与artifact|
|状态|`LOCAL_VERIFIED / CODE_REVIEW_PASS_P0_0_P1_0 / NOT_RELEASED / NO_NEW_PERFORMANCE_RESULT`|
|目标|在不改变NEXT-R1方法、84行矩阵或晋级门的前提下，验证D132唯一NumPy/PyTorch ABI修复后完整六臂结果|

D132在首个fold、0/84 prediction前因远端`torch2.1.0+cu121/numpy2.2.5`的`torch.from_numpy`边界故障退出，严格为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，未score且不续跑。只读诊断证明`from_numpy`失败、`torch.frombuffer`通过。D133仅将IQ复制路径改为连续float32 buffer→`torch.frombuffer`→reshape→clone→device，不改变样本、数值、方法、矩阵、梯度或评分。

## 冻结方法、矩阵与判据

|项目|值|
|---|---|
|candidate|`NEXT-R1 FABR-TSL/r1`，不新增候选|
|fold/K|7 receiver×6 seen-class LOCO×`K∈{1,5}`=84行|
|arms|`R0Q/R0F/R0L/R1Q/R1F/R1L`|
|K1|F/L逐logit严格alias Q；精确top tie技术拒绝|
|K5比较|`R1Q-R0Q`、`R0L-R0F`、`R1L-R1F`|
|晋级|各比较`ΔH>0`、总正确数增加，且`ΔA_retained/ΔA_held_proxy/ΔF_retained>=0`|
|负结果|完整后立即关闭，不调参、不重复矩阵|

## 本地版本与验证

- Git实现提交：`14ccbca0`；前序完整runner：`1e2a31a4`；D132终态报告：`602d132c`。
- 修复文件：`code/cvsrffi/stage2_next_r1_real.py`，SHA256=`9a22319f9791b14690029f04b78f3af6e6b26c01607c7d94c68353ec968edd21`。
- `ssr-gpu`联合54项聚焦测试通过；`py_compile`与`git diff --check`通过。
- 独立Terra/max复核：`P0=0、P1=0、CODE_REVIEW_PASS`。
- 根目录`E:\type10-7`不是Git仓库；本报告同步进入Git承载面。

## N607冻结输入与路径

|字段|值|
|---|---|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d133_next_r1_fabr_tsl_proxy84_20260804_r1`，发布前必须ABSENT|
|source/CWD|`RUN_ROOT/source`|
|output/log|`RUN_ROOT/output`必须ABSENT；`RUN_ROOT/logs/predict.log`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|checkpoint SHA|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|selected IQ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`|
|selected IQ SHA|`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|IQ receipt|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.receipt.json`|
|receipt SHA|`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|L_s join|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`|
|L_s SHA|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|
|GPU|runner预检后选择空闲物理卡；`CUDA_VISIBLE_DEVICES=<GPU>`且CLI `--device cuda:0`|

## 冻结命令

~~~text
CUDA_VISIBLE_DEVICES=<preflight-selected-GPU> PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_next_r1_proxy84.py predict --run-id d133_next_r1_fabr_tsl_proxy84_20260804_r1 --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/d133_next_r1_fabr_tsl_proxy84_20260804_r1/output --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --selected-iq /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz --selected-iq-sha256 e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --selected-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.receipt.json --selected-receipt-sha256 a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59 --ls-join /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz --ls-join-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --device cuda:0 --microbatch-size 8

/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_next_r1_proxy84.py score --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/d133_next_r1_fabr_tsl_proxy84_20260804_r1/output
~~~

predict detached启动后核验PID/CWD/cmdline/run-root/GPU/log增长；只有84行completion/manifest/逐行artifact/四SHA闭合后才单独score。技术停止只允许P0/安全、错误hash/checkout、覆盖风险、launcher-wide确定性故障或至少2个不同row在prediction前相同异常；绝不按性能停止。retry authority=false。

## 预期artifact与完成表

预期：84行row JSON/NPZ、plan、preregistration、manifest、completion、predictions、truth-side sealed file、独立score、resource/forward/smoke receipts、logs/control。

|candidate|K/arm|A_retained|A_held_proxy|H_proxy|F_retained|总正确数|资源|结论|
|---|---|---:|---:|---:|---:|---:|---|---|
|`NEXT-R1 FABR-TSL/r1`|待84行完成|—|—|—|—|—|—|`NO_NEW_PERFORMANCE_RESULT`|

