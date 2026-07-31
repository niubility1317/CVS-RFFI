# D106真实集成r5预登记报告

状态：`LOCAL_FIX_IMPLEMENTED / REVIEW_PENDING / NOT_LANDED`

## 1.身份与目的

- run ID：`d106_real_integration_deefd57c_20260801_r5`
- 时间：2026-08-01
- operator：主agent本地修复；N607专属Terra Max runner为唯一launch owner
- release source commit：`deefd57c4185a5343f87772be78b5038c37e6217`
- 目的：以新run root重新执行no-query真实集成；不得访问source-held、Target或性能truth。

r4因相对`--fixture`在入口立即失败，已单独封存为`NO_PERFORMANCE_RESULT`。r5不修改或复用r4。

## 2.本地修复

|文件|作用|
|---|---|
|`d106_real_integration_runner_handoff_deefd57c_r5.md`|冻结r5绝对fixture/output命令和运行边界|
|`artifacts/d106_real_integration_fixture_deefd57c_r5.json`|把所有run-local路径绑定到全新r5 root|
|`tests/test_d106_real_integration_handoff.py`|机械验证fixture canonical/字段/绝对路径及handoff命令无`../`|

fixture SHA256为`931fc13330f5f525af71c93c05fa2ba8f604a5235367e80a6be2ce57191e25f6`。source zip、D104 split、disjoint receipt和方法代码均保持r4已验证的相同字节，不因启动合同修复而改变方法。

## 3.服务器预登记

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_deefd57c_20260801_r5`
- CWD：`<run-root>/source`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：物理GPU0，进程内`cuda:0`
- log：`<run-root>/logs/run.out`
- output：`<run-root>/output`，启动前必须为`ABSENT`
- retry：`NOT_AUTHORIZED`

精确子命令、四份release SHA、健康停止和artifact回收语义见r5 handoff。独立复审达到`P0=0/P1=0`且修复提交后才允许交接；当前没有N607 r5证据或性能结果。
