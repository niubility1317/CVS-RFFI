# D106真实集成N607 runner交接r5

状态：`PREREGISTERED / LOCAL_RELEASE_READY / NOT_LANDED`

## 1.运行身份与失败隔离

- run ID：`d106_real_integration_deefd57c_20260801_r5`
- release source commit：`deefd57c4185a5343f87772be78b5038c37e6217`
- candidate：`D106-RDCE/GTSM-r3-SCATTER02`
- protocol：`p2_min_v1`
- sole launch owner：N607专属Terra Max runner
- retry：`NOT_AUTHORIZED`
- performance truth：`NOT_OPENED`

`d106_real_integration_deefd57c_20260801_r4`已封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。r5不得修改、续写、删除或重用r4的任何路径。

## 2.本地release资产

|本地绝对文件|SHA256|远端绝对目标|
|---|---|---|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d106_real_integration_source_deefd57c.zip`|`6d30d85b624ca2a94d8b5fcde4be0ba4d32d36e87c0e21d07fc793fee65e21a2`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5/input/release/source.zip`|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d104_split_4a1e23cc.zip`|`b1884cf1a7e287aa489a2b591fc5688a7e655c6b541f6f90eafcf71cf476372e`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5/input/release/d104_split.zip`|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d106_real_integration_fixture_deefd57c_r5.json`|`931fc13330f5f525af71c93c05fa2ba8f604a5235367e80a6be2ce57191e25f6`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5/input/d106_real_integration_fixture_deefd57c_r5.json`|
|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d106_rdce_gtsm_20260801_r1\artifacts\d106_train_held_disjoint_receipt.json`|`ee7005fcc99d703dac2f3e529e39426587ffa8967d19c15cf848c98f5295d961`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5/input/d106_train_held_disjoint_receipt.json`|

source archive的关键entry及SHA与r4交接相同；落地后仍须逐项重新验证，不能继承r4结论。远端checkpoint、source pool和salt也须重新读取SHA。

## 3.绝对路径启动合同

- project root：`/home/szu2070436088/2510044040/CV-SincNet`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5/source`
- GPU：`CUDA_VISIBLE_DEVICES=0`，进程内`cuda:0`
- log：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5/logs/run.out`

唯一允许的子命令为：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5/source
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5/source/code/scripts/run_d106_real_integration.py --fixture /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5/input/d106_real_integration_fixture_deefd57c_r5.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5/output --device cuda:0
```

`--fixture`和`--output-dir`必须都是绝对路径。启动前runner必须机械解析命令，逐字确认两个参数以`/`开头，且都在r5的精确run root内；发现`../`、r4路径或其他run root时停止，不得启动。

## 4.落地、健康和完成门

执行direct preflight；direct仅因TCP路径不可用时才使用已验证bridge。不得使用`N607-admin`。远端r5 run root必须在创建前为`ABSENT`；精确创建`input/release`、`input/d104_split`和`logs`，逐份SCP后立即断开。

启动前重新验证：

1.四份传输文件SHA；
2.source archive关键entry、D104 split关键entry和198个Python文件编译；
3.fixture canonical bytes、22个字段、9个绝对路径/SHA绑定及release commit；
4.checkpoint、source pool和salt现场SHA；
5.`output=ABSENT`、GPU占用和无同run进程。

使用一次有界SSH命令`nohup`分离并记录PID。立即核验PID、CWD、cmdline、run root、GPU映射、log增长和output。不得启动第二次。

只按protocol/safety P0、wrong hash/path、overwrite风险、确定性入口异常或主进程失联且无completion marker停止精确run-owned进程树。禁止按性能停止，禁止`pkill`、覆盖、自动retry或删除partial artifact。

成功必须出现：

- `output/selected_ls_iq/*`
- `output/strict_tap/*`
- `output/rdce_asset/*`
- `output/d106_real_integration_result.json`
- `output/D106_REAL_INTEGRATION_COMPLETE.json`

完成后读取完整log，核验result/completion canonical bytes、SHA和全部禁用访问标志，拉回小型receipt、marker、日志与SHA清单；不默认拉回大型IQ/tap/wire。最后确认本地`ssh.exe`、N607和bridge TCP22均为NONE。
