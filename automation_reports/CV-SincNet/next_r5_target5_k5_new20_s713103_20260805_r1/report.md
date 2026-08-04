# NEXT-R5 K5 FA-RDCE3→qKNN Target5实验报告

## 1.身份与当前状态

- run ID：`next_r5_target5_k5_new20_s713103_20260805_r1`
- 日期：2026-08-05
- 当前状态：`LOCAL_PLAN_VERIFIED / NOT_LANDED / NOT_LAUNCHED`
- 候选：`NEXT-R5-K5-FA-RDCE3-Q`
- 主agent：`gpt-5.6-sol/high`
- 科学计划实现与独立审查：`gpt-5.6-terra/max`
- 后续唯一N607 runner：冻结全部命令与路径后使用`Luna/max`

## 2.目标、假设与比较

目标是在`p2_min_v1`下，用一次最小正式Target实验判断R2中K5 FA-RDCE3的source-held正收益能否转移到Target。FA公式、rank-3资产、量化、`rho=sqrt(3)`、Wiener系数和FP16动态状态`a[3]`全部不变，输出直接进入qKNN；不包含K1、CER、H、D92-Lite、K10或额外new-count。

主要同row比较：

|比较|主指标|
|---|---|
|`DA1_REG0−DA0_REG0`|old BA、old-floor|
|`DA1_REG1−DA0_REG1`|old BA、seen-new、H、all-floor、总正确数|

池化主项均不低于0且H与总正确数严格增加，才记为`TARGET5_SCREEN_POSITIVE_NOT_MULTI_SEED_PROMOTABLE`。任一池化主项为负即关闭FA-RDCE3，不调参、不重跑该seed。历史formal D92只在完全同键时连接比较，不重跑。

## 3.冻结矩阵与指标命名

```text
5 receivers × seed713103 × K5/new20 × 3 leo_*_weak场景 × 4状态
= 5 jobs / 15 scenario rows / 60 state prediction surfaces
```

receivers固定为`20-1、3-19、7-14、7-7、8-8`；场景固定为`leo_clear_weak、leo_low_elev_weak、leo_rain_weak`。plan只允许seed713103，`fallback_allowed=false`；同键检查失败时技术停止并另建不可覆盖revision，不在同一plan换seed。

|状态码|中文主名称|指标约束|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|old BA、old-floor；seen-new/H=`N/A`|
|`DA1_REG0`|域适应后/新类注册前|old BA、old-floor；seen-new/H=`N/A`|
|`DA0_REG1`|域适应前/新类注册后|old BA、seen-new、H、all-floor|
|`DA1_REG1`|域适应后/新类注册后|old BA、seen-new、H、all-floor|

每场景的旧类query根在四状态完全相同；新类query根在REG0为`N/A`，在两个REG1状态完全相同。`DA1_REG1`必须复用`DA1_REG0`的同一FA state binding。

## 4.本地实现与验证

|文件|用途|
|---|---|
|`code/cvsrffi/stage2_next_r5_target5_plan.py`|不可变5-job/60-state计划、canonical receipt、old/new query根与FA状态binding验证|
|`tests/test_stage2_next_r5_target5_plan.py`|矩阵、协议负标志、seed越权、深层不可变、query根、FA复用和覆盖闭合测试|
|`docs/STAGE2_RD_GOAL_20260731.md`|当前活动目标、四状态命名和Target5边界|

本地验证：

```text
conda环境：ssr-gpu
python -m pytest -q tests/test_stage2_next_r5_target5_plan.py
结果：5 passed
python -m py_compile code/cvsrffi/stage2_next_r5_target5_plan.py tests/test_stage2_next_r5_target5_plan.py
结果：PASS
git diff --check
结果：PASS
计划receipt：9f52666aff31f7c22324ee9aa1524004404311603698a7be6dcbdc80c3f0e397
独立复审：P0=0、P1=0、P2=0、RELEASE
```

## 5.发布前尚未闭合的必要项

以下字段尚未冻结，因此本报告当前不授权N607 handoff或启动：

- 复用D106真实包materializer的Target5预测入口和独立truth-side scorer；
- seed713103的`capsule_id/split_id/old_query_id_root/new_query_id_root/receiver/K/new_count/scenario`完全同键只读检查；
- 真实checkpoint无truth smoke与聚焦query零fit/update/selection负测；
- Git实现提交、文件SHA和本地到远端映射；
- N607 preflight、GPU分配、精确CWD、服务器命令、不可覆盖output/log路径、PID和expected artifacts。

冻结后必须在本节补齐：环境`ssr-gpu`、精确服务器命令、CWD、GPU、log、PID、output、prediction/manifest/resource/completion/score路径及系统性技术失败停止规则。性能低不能触发健康早停。

## 6.已知风险与结束条件

- 当前本地没有真实Target包；真实IQ由N607上既有D106 materializer展开，若同键或三场景覆盖不闭合则fail-closed。
- NEXT-R4 ProxyRow硬编码6类留1类，不能直接冒充Target5；只复用FA fit/transform和direct-qKNN机制。
- 本计划RELEASE只证明结构与binding正确，不是落地、执行完成或性能证据。
- 完整60-state prediction封存后才允许truth-side score；禁止按receiver、scene或中间性能重跑。
