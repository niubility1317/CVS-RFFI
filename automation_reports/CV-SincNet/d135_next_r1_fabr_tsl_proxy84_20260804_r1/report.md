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

## N607 runner终态（2026-08-04）

|字段|值|
|---|---|
|状态链|`LOCAL_VERIFIED → LANDED → RUNNING → STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|
|release/source提交|release=`12005afef4599eda5ab86dd224001bce9de619dc`；implementation=`273831e5d3b7f80d7e7595df2a012c7490ede349`|
|archive|`d135_next_r1_source_12005afe.tar.gz`，52,782,770 B，SHA256=`68660118733c0c1e4136f200f9dcd08b7686b59cc77b6711dbc381e2ece26a2d`；5241成员、`bad_entries=0`|
|远端落地|仅创建`input/source/logs/control`；启动前`OUTPUT_ABSENT_PRELAUNCH_PASS`；predict自行创建`output`；七关键文件hash精确、`PY_COMPILE_PASS_7_7`|
|启动|2026-08-04 14:53:16；PID=`1024196`；物理GPU0；退出后GPU0=`0%`、`1 MiB/24576 MiB`|
|覆盖率|`0/84` rows；`plan.json`声明row_count=`84`；prediction artifacts=`0`；completion/manifest=`ABSENT`；score=`NOT_RUN`|
|日志|`logs/predict.log`，1437 B，SHA256=`7E112AC86158D42AB455086659D5CA67EA5390A4979573DD5EB56ED851885E56`|
|控制证据|`main.pid` SHA256=`7DFB27D724FDCFCEA0AD080839CABBA6A6BDE87882F80EB5BDE9AB788C9F7A1C`；`command.txt` SHA256=`F739BBFEBFEBE65357417D8D6C7AAE1B69FF6AB15521CEC1EB808DE3FE243192`；`land_guard.txt` SHA256=`FAEC341354CC03777F42E5C523BD8D072B0A2E3A1F570519558FE0A5A31E6A3C`|
|输出证据|`output/plan.json` SHA256=`5BFA6C17C74C0A44DA8B2BB132A42FF898224C5DD25FF3E83B7D6E997608FD2D`；`output/preregistration.json` SHA256=`8315946FAD54FCC1912373DA2B937396EE3C159D1BA3B63EE43E9780506DCC19`；`output_state.txt` SHA256=`F25162B31FFA567A301F79DDFC2FB8D4F3678AFD78678FC21DC0EF02BA38B89A`，记录无prediction/score|
|SSH清理|所有短连接退出；本地无`ssh/scp`残留、无`ESTABLISHED remote22`|

### 同行证据结果

|candidate|机制/类别|receiver/TX split|K/arms|覆盖|性能字段|终态判定|
|---|---|---|---:|---:|---|---|
|`NEXT-R1`|FABR-TSL|7 receiver×6 seen-class LOCO|K=1,5；六臂|0/84|未产生prediction、truth或score|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

predict成功创建`output/plan.json`与`output/preregistration.json`后，在首个fold前于`build_next_r1_phase1_assets → _select_candidate`退出：`NextR1AssetSelectionError: no frozen FABR block passed all Phase1 selection gates`。该确定性异常不产生方法性能证据；按预注册`retry authority=false`，不重启、不调参、不评分、不覆盖本run。发布阶段曾因archive的`source/`前缀误在空`source`目录下形成`source/source`，已在D135 run root内核验并精确清理后按`tar -C RUN_ROOT`重新解包，最终七关键hash与编译均通过；不影响正式predict的输入或协议。D135后续若需修复FABR资产选择，必须新建不可复用run ID并重新完成本地验证、落地和报告登记。
