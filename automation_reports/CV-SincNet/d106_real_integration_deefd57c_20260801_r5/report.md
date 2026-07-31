# D106真实集成r5预登记报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

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

精确子命令、四份release SHA、健康停止和artifact回收语义见r5 handoff。独立复审达到`P0=0/P1=0/P2=0`，修复提交为`0dc2484b`后由唯一runner执行。

## 4.执行结果

r5在完整落地门通过后执行唯一launch。PID为`3036632`，3秒内退出。绝对CWD、script、fixture和output命令机械复核全部通过，证明r4的相对路径故障已修复。新异常为：

```text
ModuleNotFoundError: No module named 'baseline_origin_sat_view'
D105Phase1BundleError: D105 checkpoint loader dependencies are unavailable
```

source archive只包含`model.py`和`model_dual_cvsincnet.py`，遗漏D105 exact checkpoint loader显式声明的第三个本地模型依赖`baseline_origin_sat_view.py`。异常指纹SHA256为`c2c635b22c82725c409aee83abe68f14d1d35679058ca94173f786073b5f9ee0`。

|证据|结果|
|---|---|
|log|1,754B，SHA`53d4b8a6de0c20d882a13027c8fba2baa8ae5697366f25b2a7c2e1d9acd53f6f`|
|失败receipt|SHA`245c87a5333aecb8b13178cb81e2dbaea4c9563e0cf58f0973cde8d2eb4776c3`|
|SHA清单|SHA`e6d076e157f447cd7fcf732b7879f063d7885959625e0fe515bd4ef02993028b`|
|partial output|`selected_ls_iq`完成；IQ未打开、未拉回|
|正式result/completion|均`ABSENT`|
|进程/GPU|run-owned process=0；GPU0无compute app|
|SSH|`SSH_PROCESSES=NONE`、`N607_OR_BRIDGE_TCP22=NONE`|

12份小型证据封存在方法报告`artifacts/remote_deefd57c_r5/`。r5不得重启、续写或覆盖。该失败不包含性能结果，也不否定DATA、DA或HEAD机制。

## 5.r6修复方向

本地修复必须以D105代码中的`D105_CANDIDATE_RUNTIME_MODEL_FILES`为依赖权威，同时封存`baseline_origin_sat_view.py`、`model.py`和`model_dual_cvsincnet.py`；D106 construction closure和runtime需按实际SHA绑定三者，source archive缺任一entry都必须在IQ提取前fail closed。完成测试、独立复审和新commit后，只能使用新run ID发布。
