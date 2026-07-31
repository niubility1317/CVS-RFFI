# D106真实集成N607 runner交接r7

状态：`PREREGISTERED / LOCAL_RELEASE_GO / NOT_LANDED`

## 1.运行身份

- run ID：`d106_real_integration_dba10236_20260801_r7`
- release source commit：`dba10236889a45b11f2f10dab3596aff7e218df0`
- candidate：`D106-RDCE/GTSM-r3-SCATTER02`
- protocol：`p2_min_v1`
- sole launch owner：N607专属Terra Max runner
- retry：`NOT_AUTHORIZED`
- performance truth：`NOT_OPENED`

本地release独立终审：`P0=0/P1=0/P2=0 / GO`。该结论不代表已落地、技术完成或产生性能结果。

r4、r5和r6均已封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。r7不得修改、续写、删除或重用其路径。

## 2.绝对release映射

|本地绝对文件|SHA256|远端绝对目标|
|---|---|---|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d106_real_integration_source_dba10236.zip`|`1eae03c8a63ede8241c4b3cb7331994ffb32e571608774e1dd874d30c928a585`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/release/source.zip`|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d104_split_4a1e23cc.zip`|`b1884cf1a7e287aa489a2b591fc5688a7e655c6b541f6f90eafcf71cf476372e`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/release/d104_split.zip`|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d106_real_integration_fixture_dba10236_r7.json`|`d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d106_real_integration_fixture_dba10236_r7.json`|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d106_train_held_disjoint_receipt.json`|`ee7005fcc99d703dac2f3e529e39426587ffa8967d19c15cf848c98f5295d961`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d106_train_held_disjoint_receipt.json`|

## 3.Source archive硬门

|Entry|SHA256|
|---|---|
|`source/code/scripts/run_d106_real_integration.py`|`4ec55ad0c22ed1176a2fedb898dff353c9307b76e67c99e59c4337e190e4b375`|
|`source/code/cvsrffi/stage2_d106_phase1_tap.py`|`5a63a5935748f17a1efcbf4069d5c80c1d99a8e813330a2c3a15895483c53e9b`|
|`source/code/baseline_origin_sat_view.py`|`fa7221ae505a51a2afc2a51b857675ac4a5384b004d5a4f36e10dafc9d4f8ace`|
|`source/code/model_dual_cvsincnet.py`|`11b56f2a763eb49de21d0a566d8e4420538cb7af310c9338f4e8d4a21c42c235`|
|`source/code/model.py`|`afc6e6266a09fd5f5be967fed85254c6c92fa0241a0336fd5ffa3eb12aa1c417`|
|`source/configs/d106_candidate_runtime_manifest_20260801.json`|`ba8e96a925d9dc69be50fcf53af7fcbffe6391d9d51558a48b34848bff8cc901`|
|`source/configs/d106_rdce_method_lock_20260801.json`|`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`|

解压后在`source/code`执行无数据import smoke，必须同时成功导入`baseline_origin_sat_view`、`model`、`model_dual_cvsincnet`，并断言`D105_CANDIDATE_RUNTIME_MODEL_FILES`精确为上述三文件。结果入口必须导出`RDCE_RANK=3`、`Z_DIM=160`和`_validated_roundtrip_rdce_rank`；任何entry缺失、SHA漂移、第四个模型依赖或旧`asset.rank`读取均在创建`output`前停止。

## 4.绝对启动合同

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`<run-root>/source`
- GPU：物理GPU0，进程内`cuda:0`
- log：`<run-root>/logs/run.out`
- output：`<run-root>/output`，启动前必须为`ABSENT`

唯一允许的子命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/source
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/source/code/scripts/run_d106_real_integration.py --fixture /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d106_real_integration_fixture_dba10236_r7.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output --device cuda:0
```

script、fixture和output必须全为绝对r7路径；发现`../`、r4/r5/r6或其他run root立即停止，不得启动。

## 5.健康与完成门

重新执行direct preflight、r7预创建ABSENT、四份传输SHA、archive/D104 entry、fixture canonical/22字段/9绝对路径、依赖import smoke、全archive Python编译、checkpoint/source pool/salt SHA、GPU和`output=ABSENT`。不得继承r4/r5/r6门禁结论。

以一次有界SSH命令`nohup`分离并记录PID；立即核验PID、CWD、cmdline、run root、GPU、log和output。不得启动第二次。只按protocol/safety P0、wrong hash/path、overwrite风险、确定性入口异常或主进程失联且无completion marker停止精确run-owned进程树。禁止按性能停止、`pkill`、覆盖、自动retry或删除partial artifact。

成功必须出现`selected_ls_iq`、`strict_tap`、`rdce_asset`、`d106_real_integration_result.json`和`COMPLETED.json`。结果必须有`rdce_rank=3`，且四个禁用访问/性能标志保持false。完成后核验全部canonical bytes、SHA和禁用访问标志，拉回小型receipt/marker/log/SHA清单，不拉回大型IQ/tap/wire；最后确认本地SSH及N607/bridge TCP22均为NONE。
