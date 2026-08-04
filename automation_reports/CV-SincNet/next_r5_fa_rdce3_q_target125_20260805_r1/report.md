# NEXT-R5 FA-RDCE3→qKNN Target125实验报告

## 1.身份与状态

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r1`
- 日期：2026-08-05
- 当前状态：`DESIGN_REOPENED_BY_USER / IMPLEMENTING / NOT_LANDED / NOT_LAUNCHED`
- 候选：`NEXT-R5-FA-RDCE3-Q-Target125`
- 主agent：`gpt-5.6-sol/high`
- 科学实现与独立审查：不同`gpt-5.6-terra/max`agent
- 后续唯一N607 runner：方法、矩阵、命令和路径冻结后使用`Luna/max`
- Git worktree：`E:\fa125wt`；branch=`codex/next-r5-fa-q-target125-20260805`
- 起点commit：`70d775bc`

## 2.用户覆盖与实验目标

用户在2026-08-05明确要求“跑FA-RDCE3+qKNN的125实验”，因此覆盖此前只允许Target5的工作目标，但不改变`p2_min_v1`、received-IQ、support/query物理ID互斥、query零fit/update/selection、全注册类逐query竞争和truth-after-prediction-seal规则。

本实验不调FA的rank、`rho`、Wiener系数、量化或qKNN参数，不运行CER、RPPF或D92-Lite新头。历史formal D92、D62、D91、SVRN只在完全同键时作结果侧比较，不参与FA拟合或选择。

## 3.冻结矩阵草案

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed={713102,713103,713104,713105,713106}
(K,new)={(10,5),(10,10),(10,20),(5,20),(1,20)}
5×5×5=125 outer jobs
每outer覆盖3个leo_*_weak场景×4状态
=375 scene rows / 1500 state prediction surfaces
```

四状态统一为：

|状态码|中文主名称|主指标|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|old BA、old-floor、总正确数；seen-new/H=`N/A`|
|`DA1_REG0`|域适应后/新类注册前|同上|
|`DA0_REG1`|域适应前/新类注册后|old BA、seen-new、H、all-floor、总正确数|
|`DA1_REG1`|域适应后/新类注册后|同上|

K1已由完整Proxy24证明FA负收益，固定严格旁路并以alias receipt闭合；K5/K10使用同一闭式FA，唯一新锁为`FA_FIT_K={5,10}`。K10不改rank、`rho`、Wiener系数或量化；公式闭合不代表收益。K5/K10的FA只由REG0 6个old类support拟合一次并在REG1逐bit复用。

## 4.因果比较与裁决

- 域适应前后：固定注册状态比较`DA1_REG0−DA0_REG0`和`DA1_REG1−DA0_REG1`。
- 注册前后：固定DA状态比较`REG1−REG0`，但REG0的seen-new/H保持`N/A`。
- 主结果按完整125 outer、同row、同query根报告；同时完整列出receiver、seed、K/new、scene和逐class old floor。
- 不按中间accuracy、H、floor停止、重跑或选择slice；只有协议/安全或重复确定性prediction前异常可以技术停止。

## 5.本地实现与验证状态

|项目|当前状态|
|---|---|
|Target125既有input/materializer/scorer复用边界|并行审计中|
|FA K10/K1方法锁|已冻结：K5/K10 fit，K1严格旁路|
|四状态125 matrix/runner/CLI|待实现|
|query零fit/update/selection聚焦负测|待实现|
|真实checkpoint无truth smoke|待执行|
|独立代码复核|待执行|
|Git提交与release archive|待完成|

## 6.发布前必要字段

当前报告不授权N607启动。必须补齐：Git commit和文件SHA、实际CLI、Conda/Python、CWD、D92/D108输入root、checkpoint和FA资产、prepared plan/context SHA、不可覆盖run root、GPU/shard映射、log/PID/output、expected artifacts及系统性技术失败停止规则。

预期artifact至少包括：`target125_plan.json`、`target125_context.json`、真实smoke receipt/prediction、8个prediction shard manifest、合并prediction manifest、FA resource/state reuse receipt、truth-open event、truth catalog、score manifest、coverage和completion receipt。

## 7.证据边界

本地设计、代码、测试、landed或RUNNING均不是性能结果。只有125/125 outer、375/375 scene和1500/1500四状态prediction完整封存并由独立truth-side scorer闭合后，才能报告性能；partial artifact只作技术诊断，不进入性能比较。
