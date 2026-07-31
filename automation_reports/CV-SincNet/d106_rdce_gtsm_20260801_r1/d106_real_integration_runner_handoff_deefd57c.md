# D106真实集成N607 runner交接

状态：`PREREGISTERED / LOCAL_RELEASE_READY / NOT_LANDED`

## 1.唯一运行身份

- run ID：`d106_real_integration_deefd57c_20260801_r4`
- release source commit：`deefd57c4185a5343f87772be78b5038c37e6217`
- candidate：`D106-RDCE/GTSM-r3-SCATTER02`
- protocol：`p2_min_v1`
- sole launch owner：N607专属Terra Max runner
- retry：`NOT_AUTHORIZED`
- performance truth：`NOT_OPENED`

## 2.本地release资产

|本地文件|SHA256|远端目标|
|---|---|---|
|`artifacts/d106_real_integration_source_deefd57c.zip`|`6d30d85b624ca2a94d8b5fcde4be0ba4d32d36e87c0e21d07fc793fee65e21a2`|`runs/d106_real_integration_deefd57c_20260801_r4/input/release/source.zip`|
|`artifacts/d104_split_4a1e23cc.zip`|`b1884cf1a7e287aa489a2b591fc5688a7e655c6b541f6f90eafcf71cf476372e`|`runs/d106_real_integration_deefd57c_20260801_r4/input/release/d104_split.zip`|
|`artifacts/d106_real_integration_fixture_deefd57c.json`|`74b2367f82a682a41f46447b089ec85bb21433b39ec8205167356909e3cd0ff1`|`runs/d106_real_integration_deefd57c_20260801_r4/input/d106_real_integration_fixture_deefd57c.json`|
|`artifacts/d106_train_held_disjoint_receipt.json`|`ee7005fcc99d703dac2f3e529e39426587ffa8967d19c15cf848c98f5295d961`|`runs/d106_real_integration_deefd57c_20260801_r4/input/d106_train_held_disjoint_receipt.json`|

source archive必须按原布局包含并逐entry核验：

|Archive entry|SHA256|
|---|---|
|`source/code/scripts/run_d106_real_integration.py`|`79120dff92f42225c22769ef7d6821ed5792e51c033075c073f291b0005bedfc`|
|`source/code/model_dual_cvsincnet.py`|`11b56f2a763eb49de21d0a566d8e4420538cb7af310c9338f4e8d4a21c42c235`|
|`source/code/model.py`|`afc6e6266a09fd5f5be967fed85254c6c92fa0241a0336fd5ffa3eb12aa1c417`|
|`source/configs/d106_candidate_runtime_manifest_20260801.json`|`09d7b350ef97c9b5d26382549e6bb42f488f928f1854223c9ba59578783003d5`|
|`source/configs/d106_rdce_method_lock_20260801.json`|`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`|

## 3.N607权威路径

- project root：`/home/szu2070436088/2510044040/CV-SincNet`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r4`
- Conda Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- checkpoint：`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- upstream source pool：`runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json`
- selection salt：`runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json`
- GPU：`CUDA_VISIBLE_DEVICES=0`，进程内`--device cuda:0`
- CWD：`<run-root>/source`
- output：`<run-root>/output`，启动前必须不存在
- log：`<run-root>/logs/run.out`

## 4.落地和启动边界

runner必须先执行本地只读direct preflight；direct失败且仅为TCP/路径故障时才使用已验证lab bridge。不得使用`N607-admin`。

远端run root必须在创建前确认为`ABSENT`。创建精确`input/release`、`input/d104_split`和`logs`目录后，逐个SCP四份release资产；每次连接完成即断开。source archive解压到run root，D104 split解压到`input/d104_split`。启动前完成：

1.四个传输文件SHA复核；
2.五个关键source archive entry路径和SHA复核；
3.fixture canonical bytes、字段集合、release commit和所有远端绝对路径复核；
4.checkpoint、source pool、selection salt实际SHA复核；
5.入口与依赖模块`py_compile`；
6.`output`仍为`ABSENT`；
7.GPU占用记录；默认每GPU最多两个训练进程，本任务仅占GPU0一个lane。

精确子命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r4/source
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_d106_real_integration.py --fixture ../input/d106_real_integration_fixture_deefd57c.json --output-dir ../output --device cuda:0
```

以一次有界SSH命令使用`nohup`分离，stdout/stderr写入`logs/run.out`并记录主PID。启动后立即核验PID、CWD、cmdline、run root、GPU映射、log增长和output子目录。不得启动第二个相同run ID。

## 5.健康检查和终止语义

本任务不读取accuracy、H、BA、floor或任何性能字段。只在以下条件停止精确run-owned进程树：

- protocol/safety P0；
- wrong checkout/hash、路径越界或overwrite风险；
- 入口确定性异常、零产物或不可恢复的fixture/runtime绑定失败；
- 主进程失联且无completion marker。

终止前必须绑定主/子PID、CWD和cmdline；先发送有界graceful termination，只对仍存活且已证明属于本run的PID升级。禁止`pkill`、删除partial output、覆盖原run或自动fresh retry。

成功所需文件：

- `output/selected_ls_iq/*`
- `output/strict_tap/*`
- `output/rdce_asset/*`
- `output/d106_real_integration_result.json`
- `output/D106_REAL_INTEGRATION_COMPLETE.json`

完成后读取完整log，核验result/completion canonical bytes和相互SHA，检索query/target/source-held/performance禁用标志，拉回小型receipt、marker、日志和SHA清单。大型IQ/tap/wire不得默认拉回。最后确认本地`ssh.exe`和N607 TCP22连接均为NONE，并返回结构化handoff：`LANDED/RUNNING/ARTIFACTS_COMPLETE`状态、PID/exit、GPU、文件SHA、异常指纹和证据边界。
