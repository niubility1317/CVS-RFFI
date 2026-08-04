# D135 NEXT-R1 FABR-TSL proxy84实验报告

## 实验身份

|字段|值|
|---|---|
|experiment ID|`d135_next_r1_fabr_tsl_proxy84_20260804_r1`|
|日期|2026-08-04|
|operator|主agent；唯一N607 runner为Luna/max|
|状态|`LOCAL_VERIFIED / CODE_REVIEW_PASS_P0_0_P1_0 / NOT_RELEASED / NO_NEW_PERFORMANCE_RESULT`|
|目标|纠正D134 wrapper预创建output的纯发布缺陷，执行完全不变的NEXT-R1 84行|

D132、D133分别因NumPy/Torch入、出边界在0/84退出；两处代码已由提交`273831e5`修复，D134真实2行forward＋四block gradient smoke已在N607 GPU0通过。D134正式predict仅因wrapper在启动前创建了`RUN_ROOT/output`，触发不可覆盖保护并在0/84退出。D135不改源码、方法、矩阵、参数、输入、GPU策略或评分，只冻结正确目录约束：runner仅创建`RUN_ROOT/{input,source,logs,control}`，`RUN_ROOT/output`必须ABSENT并由predict自身原子创建。

## 冻结实验

- 唯一candidate：`NEXT-R1 FABR-TSL/r1`。
- 42 receiver-held×class-LOCO folds×`K={1,5}`=84行；六臂`R0Q/R0F/R0L/R1Q/R1F/R1L`。
- K1 F/L严格alias Q；精确top tie技术拒绝。
- K5比较：`R1Q-R0Q`、`R0L-R0F`、`R1L-R1F`；各自必须`ΔH>0`、总正确数增加，且retained、held-proxy、floor不下降。
- 完整负结果立即关闭，不调参、不复跑。

## 版本、验证与输入

|项|值|
|---|---|
|源码提交|`273831e5`；D135 release提交在本报告提交后记录|
|real.py SHA|`532cf86b0276de2d992051a9154bc0b48c9c7c5419d44c1cbac7894dbbe6fbf3`|
|本地验证|联合55项通过；`py_compile`、`git diff --check`通过；Terra复核`P0=0、P1=0`|
|N607真实smoke|D134同源码已通过logits`(2,6)`、z160`(2,160)`与4个gradient block float32/finite；D135不重复|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|checkpoint|原D134固定路径；SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|selected IQ|原D134固定路径；SHA=`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|IQ receipt|原D134固定路径；SHA=`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|L_s join|原D134固定路径；SHA=`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|

## N607路径与冻结命令

run root=`/home/szu2070436088/2510044040/CV-SincNet/runs/d135_next_r1_fabr_tsl_proxy84_20260804_r1`，发布前必须ABSENT。runner只创建`input/source/logs/control`；启动前必须断言`test ! -e RUN_ROOT/output`，不得`mkdir output`。

~~~text
CUDA_VISIBLE_DEVICES=<preflight-selected-GPU> PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_next_r1_proxy84.py predict --run-id d135_next_r1_fabr_tsl_proxy84_20260804_r1 --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/d135_next_r1_fabr_tsl_proxy84_20260804_r1/output --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --selected-iq /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz --selected-iq-sha256 e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --selected-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.receipt.json --selected-receipt-sha256 a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59 --ls-join /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz --ls-join-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --device cuda:0 --microbatch-size 8

/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_next_r1_proxy84.py score --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/d135_next_r1_fabr_tsl_proxy84_20260804_r1/output
~~~

predict必须detached启动并核验PID/CWD/cmdline/GPU/log；84行completion/manifest/逐行artifact/四SHA闭合后才独立score。只按协议/安全/确定性技术故障停止，绝不按性能停止。retry authority=false。

## 结果表

|candidate|K/arm|A_retained|A_held_proxy|H_proxy|F_retained|总正确数|资源|结论|
|---|---|---:|---:|---:|---:|---:|---|---|
|`NEXT-R1 FABR-TSL/r1`|待84行完成|—|—|—|—|—|—|`NO_NEW_PERFORMANCE_RESULT`|

