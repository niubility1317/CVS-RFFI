# D106真实集成N607 runner交接r6

状态：`PREREGISTERED / LOCAL_RELEASE_READY / NOT_LANDED`

## 1.运行身份

- run ID：`d106_real_integration_44e33eab_20260801_r6`
- release source commit：`44e33eab9bcc9352456e5f3a8ae85405c603a36c`
- candidate：`D106-RDCE/GTSM-r3-SCATTER02`
- protocol：`p2_min_v1`
- sole launch owner：N607专属Terra Max runner
- retry：`NOT_AUTHORIZED`
- performance truth：`NOT_OPENED`

r4与r5均已封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。r6不得修改、续写、删除或重用其路径。

## 2.绝对release映射

|本地绝对文件|SHA256|远端绝对目标|
|---|---|---|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d106_real_integration_source_44e33eab.zip`|`91c5a30b156972482476b4befdae4bbbffbb66a0b1a14ad5205f58fb8f17b6fe`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_44e33eab_20260801_r6/input/release/source.zip`|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d104_split_4a1e23cc.zip`|`b1884cf1a7e287aa489a2b591fc5688a7e655c6b541f6f90eafcf71cf476372e`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_44e33eab_20260801_r6/input/release/d104_split.zip`|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d106_real_integration_fixture_44e33eab_r6.json`|`ee90561420b3d41351c0b49dc34922094cb751d8414722971ccc0f0b2e023e00`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_44e33eab_20260801_r6/input/d106_real_integration_fixture_44e33eab_r6.json`|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d106_train_held_disjoint_receipt.json`|`ee7005fcc99d703dac2f3e529e39426587ffa8967d19c15cf848c98f5295d961`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_44e33eab_20260801_r6/input/d106_train_held_disjoint_receipt.json`|

## 3.Source archive硬门

逐entry验证：

|Entry|SHA256|
|---|---|
|`source/code/scripts/run_d106_real_integration.py`|`3bb8acb3c48ad371c6c0b51f20fbefb0821445f2b7ecfaecd54de71e8a39de27`|
|`source/code/cvsrffi/stage2_d106_phase1_tap.py`|`5a63a5935748f17a1efcbf4069d5c80c1d99a8e813330a2c3a15895483c53e9b`|
|`source/code/baseline_origin_sat_view.py`|`fa7221ae505a51a2afc2a51b857675ac4a5384b004d5a4f36e10dafc9d4f8ace`|
|`source/code/model_dual_cvsincnet.py`|`11b56f2a763eb49de21d0a566d8e4420538cb7af310c9338f4e8d4a21c42c235`|
|`source/code/model.py`|`afc6e6266a09fd5f5be967fed85254c6c92fa0241a0336fd5ffa3eb12aa1c417`|
|`source/configs/d106_candidate_runtime_manifest_20260801.json`|`0e8bc733ce9650aea3463da90242f97e969210ca8a95983fee032f1474f87cb2`|
|`source/configs/d106_rdce_method_lock_20260801.json`|`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`|

解压后在`source/code`执行无数据import smoke，必须同时成功导入`baseline_origin_sat_view`、`model`、`model_dual_cvsincnet`，并断言`D105_CANDIDATE_RUNTIME_MODEL_FILES`精确为上述三文件。任何缺失或第四个本地依赖均在创建`output`前停止。

## 4.绝对启动合同

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_44e33eab_20260801_r6`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`<run-root>/source`
- GPU：物理GPU0，进程内`cuda:0`
- log：`<run-root>/logs/run.out`
- output：`<run-root>/output`，启动前必须为`ABSENT`

唯一允许的子命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_44e33eab_20260801_r6/source
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_44e33eab_20260801_r6/source/code/scripts/run_d106_real_integration.py --fixture /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_44e33eab_20260801_r6/input/d106_real_integration_fixture_44e33eab_r6.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_44e33eab_20260801_r6/output --device cuda:0
```

script、fixture和output必须全为绝对r6路径；发现`../`、r4/r5或其他run root立即停止，不得启动。

## 5.健康与完成门

重新执行direct preflight、r6预创建ABSENT、四份传输SHA、archive/D104 entry、fixture canonical/22字段/9绝对路径、依赖import smoke、198+个Python编译、checkpoint/source pool/salt SHA、GPU和`output=ABSENT`。不得继承r4/r5门禁结论。

以一次有界SSH命令`nohup`分离并记录PID；立即核验PID、CWD、cmdline、run root、GPU、log和output。不得启动第二次。只按protocol/safety P0、wrong hash/path、overwrite风险、确定性入口异常或主进程失联且无completion marker停止精确run-owned进程树。禁止按性能停止、`pkill`、覆盖、自动retry或删除partial artifact。

成功必须出现`selected_ls_iq`、`strict_tap`、`rdce_asset`、`d106_real_integration_result.json`和`COMPLETED.json`。完成后核验全部canonical bytes、SHA和禁用访问标志，拉回小型receipt/marker/log/SHA清单，不拉回大型IQ/tap/wire；最后确认本地SSH及N607/bridge TCP22均为NONE。
