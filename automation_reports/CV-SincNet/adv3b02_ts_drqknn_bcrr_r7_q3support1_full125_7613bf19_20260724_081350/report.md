# ADV3B02/r7-q3support1完整125发布报告

- run ID：`adv3b02_ts_drqknn_bcrr_r7_q3support1_full125_7613bf19_20260724_081350`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r7-q3support1`
- Git commit：`7613bf1918e8af225c8cded66f0b1c406ac82942`
- 创建时间：`2026-07-24T08:13:50+08:00`
- operator：主agent；唯一N607 launch owner为单一`gpt-5.6-terra high`runner
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

目标、冻结矩阵、本地闭合、发布输入、N607命令、健康止损和裁决合同以根报告为准；根报告路径为`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r7_q3support1_full125_7613bf19_20260724_081350\report.md`。commit-bound源码包为`source_7613bf19.zip`，SHA256=`37cc6c6dc6db40611a79ab998488703f14cb6cb5f3639ff866d7265b5b33f615`，35,611,636B，含3,994个tracked regular blob、4,548个含目录ZIP成员；method lock SHA256=`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`。

|字段|当前值|
|---|---|
|Git commit|`7613bf1918e8af225c8cded66f0b1c406ac82942`|
|run ID|`adv3b02_ts_drqknn_bcrr_r7_q3support1_full125_7613bf19_20260724_081350`|
|remote PID/exit|`1667766 / 1`|
|prediction/score|`0/1000 / 0/1500`|
|archive/coverage/parity|`NOT_GENERATED / NOT_GENERATED / NOT_GENERATED`|
|性能裁决|`NO_PERFORMANCE_RESULT`|

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`

## Runner终态

9/125个row已提交并完成健康检查，0成功、9失败；两个自然结束row均在prediction前出现`SVRNBCRStateError: BCRR masked cross-view degeneracy`，其余7个run-owned worker由唯一runner健康止损。最终row receipt=`0/125`、prediction=`0/1000`、logical score=`0/1500`、scene=`0/375`，archive/coverage/parity均未生成；主PID=`1667766`，exit=`1`。首源support-only输入已回收且不含query、truth或prediction；本run没有性能结果。
