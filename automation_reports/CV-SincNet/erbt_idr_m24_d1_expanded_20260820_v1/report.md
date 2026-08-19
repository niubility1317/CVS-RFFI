# ERBT-IDR M2.4 D1扩展矩阵实验报告

## 启动前登记

|字段|值|
|---|---|
|run ID|`erbt_idr_m24_d1_expanded_20260820_v1`|
|当前状态|`LOCAL_VERIFIED / PREREGISTERED / NOT_YET_LANDED`|
|候选|仅`M24-D1-PHYSICAL256-F1`；D2–D10在诊断K10三个场景均触发support-harm整体回退，无候选具备扩展资格|
|矩阵|5个receiver：`20-1`、`3-19`、`7-14`、`7-7`、`8-8`；3个真实method seed：`7282101`、`7282102`、`7282103`；4条件：`K1/new20`、`K5/new20`、`K10/new20`、`K10/new5`；合计60个row、180个场景单元|
|协议|`p2_min_v1`、`VALIDATED_ONCE`；只读复用既有base feature cache；不重验received IQ；query不更新状态|
|prediction边界|全部60个row先完成不可变prediction且D1逐query parity为零差异，之后scorer才允许接入truth|
|环境／CWD|N607 CPU闭式执行；使用不可变Git release；`PYTHONPATH=code`|
|输出|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_expanded_20260820_v1`，不可覆盖|
|资源|默认2个CPU worker；不创建GPU计算进程，不干预既有任务|
|停止规则|错误cache身份、协议越界、输出覆盖、任一D1 parity差异、prediction不完整或scorer接线错误；不因性能停止|
|预期artifact|60个row receipt、60个不可变prediction、prediction matrix index、60个same-row score、60个four-state score、scored matrix index|

## 本地验证

M2.4聚焦及M2.3相邻测试45项通过。真实`3-19/K10/new5` base-cache-only D1回归达到660/660逐query一致；推理态7677B，`persistent_update_state_bytes=0`，不需要RF overlay或ground component持久状态。

## 证据边界

该矩阵用于刻画D1在receiver、method seed、support/query seed、new-class draw和K条件上的研发稳定性。它不是新增模块收益矩阵，也不是独立fresh confirmation；D2–D10的未晋级状态不因D1扩展而改变。
