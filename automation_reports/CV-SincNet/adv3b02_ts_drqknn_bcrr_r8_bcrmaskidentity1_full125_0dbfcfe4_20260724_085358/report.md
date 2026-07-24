# ADV3B02/r8-bcrmaskidentity1完整125发布报告

- run ID：`adv3b02_ts_drqknn_bcrr_r8_bcrmaskidentity1_full125_0dbfcfe4_20260724_085358`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r8-bcrmaskidentity1`
- Git commit：`0dbfcfe43c178068d91d6bbd07d22bd14bd959cf`
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 独立review：`MERGE / P0=0 / P1=0 / P2=0`

完整目标、矩阵、发布输入、正式命令、健康止损和性能裁决合同以根报告为准：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r8_bcrmaskidentity1_full125_0dbfcfe4_20260724_085358\report.md`。

|字段|当前值|
|---|---|
|Git commit|`0dbfcfe43c178068d91d6bbd07d22bd14bd959cf`|
|run ID|`adv3b02_ts_drqknn_bcrr_r8_bcrmaskidentity1_full125_0dbfcfe4_20260724_085358`|
|remote PID/exit|`1689183 / 1`|
|prediction/score|`0/1000 / 0/1500`|
|archive/coverage/parity|`NOT_GENERATED / NOT_GENERATED / NOT_GENERATED`|
|性能裁决|`NO_PERFORMANCE_RESULT`|

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`

## Runner终态

direct预检、输入SHA、安全解包、`py_compile`和`matrix --help`均通过。唯一PID=`1689183`在任何row前以exit=`1`自然退出；submitted/completed/succeeded/failed/active=`0/0/0/0/0`，prediction/score/scene=`0/0/0`，archive/coverage/parity均未生成。唯一首源为runner提前创建`<run>/artifacts`，而冻结launcher要求`--run-root`首次进入时不存在，故抛出`ADV3B02LauncherError: matrix root must be fresh`。这不是方法性能结果；新run只修正该启动布局，不改变科学commit或矩阵。
