# D106真实集成r6预登记报告

状态：`LOCAL_RELEASE_READY / REVIEW_PENDING / NOT_LANDED`

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

精确映射、entry、import smoke、启动命令、健康门和artifact回收语义见`d106_real_integration_runner_handoff_44e33eab_r6.md`。handoff独立复审和artifact commit完成前不得交runner；当前无r6 N607或性能结果。
