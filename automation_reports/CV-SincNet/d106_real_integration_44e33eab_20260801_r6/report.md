# D106真实集成r6预登记报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 1.身份与目的

- run ID：`d106_real_integration_44e33eab_20260801_r6`
- 时间：2026-08-01
- operator：主agent本地release；N607专属Terra Max runner为唯一launch owner
- release source commit：`44e33eab9bcc9352456e5f3a8ae85405c603a36c`
- 目的：执行no-query真实DATA→checkpoint→RDCE闭环；不访问source-held、Target或性能truth。

r4和r5均为已封存技术失败，r6不修改或复用其路径。

## 2.依赖闭包修复

D105 exact checkpoint loader的权威本地模型依赖为：

1.`baseline_origin_sat_view.py`
2.`model.py`
3.`model_dual_cvsincnet.py`

D106 construction closure、runtime manifest和入口分别绑定三文件集合及实际SHA；三文件实际缺失、runtime字段缺失、SHA漂移或D105声明集合变化都会在IQ extraction前fail closed。相关entry+tap测试80通过1跳过，独立复审为`P0=0/P1=0/P2=0`。

## 3.Release资产

|资产|SHA256|
|---|---|
|`d106_real_integration_source_44e33eab.zip`|`91c5a30b156972482476b4befdae4bbbffbb66a0b1a14ad5205f58fb8f17b6fe`|
|`d104_split_4a1e23cc.zip`|`b1884cf1a7e287aa489a2b591fc5688a7e655c6b541f6f90eafcf71cf476372e`|
|`d106_real_integration_fixture_44e33eab_r6.json`|`ee90561420b3d41351c0b49dc34922094cb751d8414722971ccc0f0b2e023e00`|
|`d106_train_held_disjoint_receipt.json`|`ee7005fcc99d703dac2f3e529e39426587ffa8967d19c15cf848c98f5295d961`|

source archive的7个关键entry及SHA已本地逐项复核；直接从zip执行三模型import smoke通过。r6绝对路径、唯一launch、r4/r5隔离和archive三模型entry测试5/5通过。

## 4.N607预登记

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_44e33eab_20260801_r6`
- CWD：`<run-root>/source`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：物理GPU0，进程内`cuda:0`
- output：`<run-root>/output`，启动前必须`ABSENT`
- retry：`NOT_AUTHORIZED`

精确映射、entry、import smoke、启动命令、健康门和artifact回收语义见`d106_real_integration_runner_handoff_44e33eab_r6.md`。本段保留交runner前的历史预登记语义；r6最终N607事实见§5，且始终没有性能结果。

## 5.N607最终状态

2026-08-01，唯一runner完成direct preflight、落地、SHA/compile/checkpoint/source-pool/salt核验和三模型remote import smoke。主进程PID为`3051253`，启动时CWD精确指向r6/source，cmdline与handoff绝对命令一致，绑定物理GPU0；进程完成selected-IQ、strict-tap和RDCE wire构建后，在正式结果封装前退出。

技术异常为`AttributeError: 'D106RDCEAsset' object has no attribute 'rank'`，位置为`run_d106_real_integration.py:347`。异常指纹为`3bcc4e695401738b66dc6e2842cf2094ae9a402c120e6626a10bc915a4a4b1d4`。这是asset/result字段合同缺陷，不是性能失败；未访问query、source-held truth或Target，也未计算任何性能指标。

|证据|值|
|---|---|
|run log|712B；SHA256=`25842d3ac70bcd5e772ba43b3f9abc4311b2a9a99f710471cca47bc729f94922`|
|technical failure receipt|SHA256=`95b39a19b63000822a96bec8973ad4311d3161cd11b6c039fe926834d4d3c614`|
|SHA manifest|SHA256=`d4ed8d9e54590f61dda96868413af066574de60f3b2afb6b6c17bfab69d92704`|
|正式result|`ABSENT`|
|正式`output/COMPLETED.json`|`ABSENT`|
|partial|selected IQ 1,509,068B；strict tap 1,069,690B；RDCE wire 2,576B，均远端保全|

runner未打开或拉回三份大型partial，只回收14份小型控制证据到`artifacts/remote_44e33eab_r6/`。退出后run-owned进程为0，GPU0无compute app；`SSH_PROCESSES=NONE`且N607/bridge TCP22连接均为0。r6不得重启、覆盖或改写为性能结果；修复必须本地完成并使用新run ID。
