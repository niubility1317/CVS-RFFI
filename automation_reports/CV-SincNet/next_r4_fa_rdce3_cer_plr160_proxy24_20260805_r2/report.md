# NEXT-R4 FA-RDCE3×CER-PLR160 Proxy24 r2预注册报告

状态：`PREREGISTERED / NO_PERFORMANCE_RESULT`

## 1.实验身份与边界

- 实验ID：`next_r4_fa_rdce3_cer_plr160_proxy24_20260805_r2`
- 日期：2026-08-05
- 操作者：主agent负责协议、矩阵与最终裁决；Luna/max仅负责后续已冻结发布后的机械落地与证据回收
- 候选：`NEXT-R4-FA-RDCE3-CER-PLR160`
- 协议：`p2_min_v1`
- 语义：`phase1_seen_class_loco_directional_proxy`；不构成正式Stage2-C新类注册声明
- 目标：在不改变r1科学候选、输入或矩阵的前提下，闭合r1首row的精确float32 top tie技术故障

r1已记录为：0 prediction、`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。首个逻辑行是`DA0_REG0`（域适应前/新类注册前）、qKNN基座Q、K1；异常为`TIE_UNRESOLVED: direct qKNN final float32 top tie`。r1不产生任何性能结论，r2不回写、不覆盖、不续跑r1。

## 2.四状态与冻结矩阵

|状态码|唯一中文主名称|REG0指标约束|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|`seen_new_acc`、`H_old_new`=`N/A`|
|`DA1_REG0`|域适应后/新类注册前|`seen_new_acc`、`H_old_new`=`N/A`|
|`DA0_REG1`|域适应前/新类注册后|报告old BA、seen-new、H、all-floor及总正确数|
|`DA1_REG1`|域适应后/新类注册后|报告old BA、seen-new、H、all-floor及总正确数|

冻结值全部沿用r1：receiver=`1-1、18-2`；每receiver 6个held-class；K=`1、5`；K1为K5逐类support前缀；逻辑行=`2×6×2=24`；唯一prediction=`144`；含K1 alias的arm记录=`192`。不改logit、参数、seed、receiver、held-class、K、阈值、矩阵、capsule、split、Phase1资产或qKNN locks。

## 3.r2唯一变化

r2只登记一个机械变化：将决策总函数封存标识为`SEALED_CLASS_HANDLE_UTF8_ASC_V1`，用于处理r1首row的精确float32 top tie。其实现字节、运行commit与archive在Terra完成后回填；本报告不制定、比较或选择其它tie规则，不改变任何logit、参数、数据、矩阵或资源配置。

四态比较仍固定为`DA1_REG0−DA0_REG0`、`DA1_REG1−DA0_REG1`、固定DA下的注册效应及必要的交互项。所有query仍逐样本面对全部注册类；不得读取query真值/role、真实batch类别数、class quota或global reassignment。

## 4.同一封存输入及SHA

r2沿用r1已验证输入；received-IQ字节、物理ID、receiver/TX集合、场景分配、K、support/query划分和`p2_min_v1`均不变，不触发数据重验。以下SHA全部沿用：

|输入或锁|SHA256或标识|
|---|---|
|D105 checkpoint|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|selected received-IQ|`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|received-IQ receipt|`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|strict tap|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|FA asset manifest（本地）|`0dedb5ae2c6052820f44c1d9d986ff29222ac16765618cd470529671cbcb6fd8`|
|FA asset manifest（远端）|`dd602359d9ff28aaf9084a09c2d2e4fc9d6daf3383bc7268492b2eb58ede196d`|
|capsule metadata v2|`5b30f6dda514797beb984b5ed01995cfc64b8cb5d1a7367da241a468b6ab8272`|
|validator receipt|`d0ac04930dd02a7d4c2dfe98c41d8933301e89095cc3cc5afad028f3bc499c64`|
|qKNN K1/K5 lock|`13b5496c3580b16a6660dee4fc8cd0f41874a41144b357c7c08c99b4d80e91fc`|
|Phase1 LODO authority|`b49cdc9f99094372412fd76d647cec58495a486eeb978fbd72f10e85f0f0e26a`|
|INT8 margin audit|`024a5024c06d710fbf4ddfee5326aacc89dc2ab2c74d3a9b866af09957efd9e3`|
|method-lock原始字节|`35530428ecfe77982043a3b29f3f2275c5bfb66fa1da523f64c8c01030bc7311`|
|capsule_id|`9df82b4af19898748bc5a27c039cd7e04d1f7c53fc3aa4e082c6308a3eb32a26`|
|split_id|`a5ccbba48980228a6dfb42b86116262a33184877d5ed4cafe11d406b74d05d96`|

12个FA wire、旧类聚合、qKNN锁及其逐文件SHA均按r1清单原样复用；r2不重建、不改写、不追加输入。

## 5.发布前PENDING项

|字段|r2状态|
|---|---|
|r2运行代码Git commit|`PENDING`|
|runtime archive路径、SHA、大小|`PENDING`|
|本地聚焦验证与独立复核|`PENDING`|
|Conda/Python、精确server command、CWD、GPU|`PENDING`|
|fresh remote root|`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r4_fa_rdce3_cer_plr160_proxy24_20260805_r2`；创建前必须`ABSENT`，不得复用r1 root|
|启动后PID、CWD、cmdline、GPU和日志增长证据|`PENDING`|

预期artifact为prepare package、truth sidecar、prediction、manifest、resource receipt、score和完整日志；这些是待运行证据，不是当前结果。

## 6.停止规则与结果表

停止规则与r1相同：仅因协议/安全违规、错误checkout或hash、覆盖风险、prediction闭合缺失，或至少两个不同row在产生prediction前出现同一确定性异常指纹，才停止run-owned进程树并保留partial。不得按accuracy、H、BA、floor或任何性能值停止；fresh-run retry未授权。

|run|四态|prediction/score|性能指标|判定|
|---|---|---|---|---|
|`next_r4_fa_rdce3_cer_plr160_proxy24_20260805_r2`|四态固定，尚未运行|`PENDING`/`PENDING`|不填写、不估计|`NO_RESULT_PENDING`|

## 7.范围与版本约束

本任务仅新增本报告及其根镜像；不改r1、不改代码/测试/目标文档、不运行实验、不选择tie规则、不添加gate。Terra正在维护的4个tie核心/测试文件不在本报告所有权内。Git工作树只提交本报告文件；根镜像不在Git。
