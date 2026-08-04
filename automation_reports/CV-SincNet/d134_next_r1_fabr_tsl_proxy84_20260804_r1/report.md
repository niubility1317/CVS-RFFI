# D134 NEXT-R1 FABR-TSL proxy84实验报告

## 实验身份

|字段|值|
|---|---|
|experiment ID|`d134_next_r1_fabr_tsl_proxy84_20260804_r1`|
|日期|2026-08-04|
|operator|主agent（Sol/high）；独立复核Terra/max；唯一N607 runner为Luna/max|
|状态|`LOCAL_VERIFIED / CODE_REVIEW_PASS_P0_0_P1_0 / NOT_RELEASED / NO_NEW_PERFORMANCE_RESULT`|
|目标|修复D133唯一Torch→NumPy出边界缺陷后，执行不变的NEXT-R1 84行六臂矩阵|

D132、D133均在0/84前技术退出，没有性能结果。D132证明远端`torch.from_numpy`不可用；D133证明真实tap/logits均为float32、正确shape且全finite，但`Tensor.numpy()`产生object dtype/TypeError。D134仅把pre、logits和逐样本gradient三处转换改为严格float32/finite tensor→`detach().cpu().tolist()`→`np.asarray(float32)`，保持float32位值、shape、样本和参数展平顺序，不改变方法、矩阵、参数或门槛。

## 冻结实验与判据

- 唯一candidate：`NEXT-R1 FABR-TSL/r1`；42 receiver-held×class-LOCO folds×`K={1,5}`=84行。
- 六臂：`R0Q/R0F/R0L/R1Q/R1F/R1L`；K1 F/L严格alias Q；精确top tie技术拒绝。
- K5只比较`R1Q-R0Q`、`R0L-R0F`、`R1L-R1F`。
- 每项必须`ΔH>0`、总正确数增加，且`ΔA_retained/ΔA_held_proxy/ΔF_retained>=0`；完整负结果立即关闭，不调参、不复跑。

## 本地版本与验证

|字段|值|
|---|---|
|ABI修复提交|`273831e5`|
|D133终态报告提交|`e2934385`|
|修复文件|`code/cvsrffi/stage2_next_r1_real.py`|
|文件SHA256|`532cf86b0276de2d992051a9154bc0b48c9c7c5419d44c1cbac7894dbbe6fbf3`|
|测试|`ssr-gpu`联合55项通过；`py_compile`、`git diff --check`通过|
|独立复核|`P0=0、P1=0、CODE_REVIEW_PASS`|

根目录`E:\type10-7`不是Git仓库；本报告同步到Git承载面。D134发布前在新source执行一次2行真实forward＋四block gradient短smoke，证明所有Torch/NumPy边界闭合；该smoke不读query truth、不评分、不改变矩阵。

## N607冻结输入与路径

|项|值|
|---|---|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d134_next_r1_fabr_tsl_proxy84_20260804_r1`，必须ABSENT|
|source/CWD|`RUN_ROOT/source`|
|output/log|`RUN_ROOT/output`必须ABSENT；`RUN_ROOT/logs/predict.log`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`，SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|selected IQ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`，SHA=`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|IQ receipt|同目录`d106_ls_received_iq.receipt.json`，SHA=`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|L_s join|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`，SHA=`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|
|GPU|预检后选择空闲物理卡，`CUDA_VISIBLE_DEVICES=<GPU>`且CLI `--device cuda:0`|

## 冻结命令

~~~text
CUDA_VISIBLE_DEVICES=<preflight-selected-GPU> PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_next_r1_proxy84.py predict --run-id d134_next_r1_fabr_tsl_proxy84_20260804_r1 --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/d134_next_r1_fabr_tsl_proxy84_20260804_r1/output --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --selected-iq /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz --selected-iq-sha256 e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --selected-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.receipt.json --selected-receipt-sha256 a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59 --ls-join /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz --ls-join-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --device cuda:0 --microbatch-size 8

/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_next_r1_proxy84.py score --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/d134_next_r1_fabr_tsl_proxy84_20260804_r1/output
~~~

predict detached启动后立即核验PID/CWD/cmdline/run-root/GPU/log；仅84行completion、manifest、逐行artifact和四SHA闭合后独立score。停止只限P0/安全、错误hash/checkout、覆盖风险、launcher-wide确定性故障或至少2个不同row在prediction前相同异常；不得按accuracy/H/floor停止。retry authority=false。

## 预期结果表

|candidate|K/arm|A_retained|A_held_proxy|H_proxy|F_retained|总正确数|资源|结论|
|---|---|---:|---:|---:|---:|---:|---|---|
|`NEXT-R1 FABR-TSL/r1`|待84行完成|—|—|—|—|—|—|`NO_NEW_PERFORMANCE_RESULT`|

