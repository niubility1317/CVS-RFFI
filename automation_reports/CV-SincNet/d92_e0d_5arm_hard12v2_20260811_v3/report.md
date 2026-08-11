# D92-E0D-5arm-Hard12-v2运行时契约修复实验报告

## 1.基本信息

|字段|内容|
|---|---|
|实验ID|`D92-E0D-5arm-Hard12-v2`|
|run ID|`d92_e0d_5arm_hard12v2_20260811_v3`|
|日期|2026-08-11|
|operator|Codex primary；N607唯一runner待交接|
|当前状态|`LOCAL_VERIFIED_APPROVED_FOR_N607`|
|协议|`p2_min_v1`|
|证据范围|`DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN`|
|唯一晋级候选|`E0_FULL_ONLY`|
|前序run|v1为配置落点错误；v2为审计字段契约错误；二者均在shard前停止且`NO_PERFORMANCE_RESULT`|

## 2.目标、根因与修复

本run继续执行已经冻结的五臂科学矩阵，不改变方法、输入或门限。v2真实checkpoint truth-free smoke证明E0D私有fit-audit遗漏了通用D81 evaluator资源汇总所需的`before/after_center_shift_l2_max`和`before/after_effective_sample_size_min`。该遗漏在任何prediction落盘前触发`KeyError`，不是算法性能结果。

最终修复提交`96a48fc333c0b356bd89c470877dbb80edd34a84`恢复这4个既有审计字段，并要求before/after都存在合法`d81_transform_audit`：原样读取有限的`center_shift_l2_max`和每类effective sample size最小值，同时核对schema、support行数、class count、K-shot、query rows=0和无outer/query访问；缺失或不合法一律fail closed。真实K5非均匀权重回归锁定最小ESS=`4.42187`，不再用K或0伪造审计值。预测state、score、D几何、B/E开关、fit计数、query图和truth-side scorer均未改变。

## 3.冻结科学矩阵与门

- 五臂：`D92_FULL`、`E0_FUSION`、`E0_FULL_ONLY`、`E0_BLOCK_ONLY`、`E0_FIXED50`。
- selection SHA256=`2e3b3333a4a325bd0443a31065d3340d6a650a3e89620951a786637e6bce8d3a`；与Hard12-v1零交集。
- 12个outer×5臂=60个job；每outer固定clear/low_elev/rain，共180个scene-arm。10个K5/K10 outer进入性能门，2个K1 outer只作strict alias和liveness。
- 五臂`DA0_REG0`状态/预测精确一致，K1的`DA0_REG1`严格别名一致；query逐样本在所有注册类竞争，禁止truth、role、quota、fit、selection、update和global reassignment。
- FULL_ONLY相对E0_FUSION：mean ΔH>0，至少8/10非负，注册wall中位至少下降40%，增量peak不增加。
- FULL_ONLY相对D92_FULL：mean ΔH≥0.005（即0.5个百分点），至少8/10非负；old-balanced、old-floor、seen-new均不下降，forgetting不增加；注册wall中位至少下降60%。
- K5/K10 two-state组件fit：FULL=48/88、FUSION=24/44、FULL_ONLY=2/2、BLOCK_ONLY=2/2、FIXED50=4/4；query MAC精确一致。

## 4.本地实现与验证

|项目|状态|证据|
|---|---|---|
|v2真实失败复现|PASS|`KeyError: before_center_shift_l2_max`；固定K1 smoke；无prediction/score/shard|
|TDD红灯|PASS|新增契约断言在修复前精确复现KeyError|
|相关回归|PASS|`ssr-gpu`环境中36项D92-E0D/runner/probe测试通过|
|静态检查|PASS|修复文件`py_compile`及`git diff --check`通过|
|独立P0/P1复审|PASS|复审HEAD=`75c11546`；P0=0、P1=0；`APPROVE_RELEASE`|

本轮不重复数据验证、不做整树SHA或额外签名。

## 5.冻结交付物

|交付物|本地路径与身份|
|---|---|
|runtime archive|`E:\type10-7\code\snapshots\d92_e0d_runtime_closure_96a48fc3.tar.gz`；3525455B；912个Git跟踪成员；SHA256=`599914382516f3cf66a142b5420a524ef09887bdb58057951fef9af2b84c82a1`|
|method lock|`configs/stage2_d92_e0d_5arm_hard12v2_v1.json`；2177B；SHA256=`b80f967e1fc070a730a7b193f691036339930af022682fe2fca81c2e4d229f86`|
|launch|`automation_reports/CV-SincNet/d92_e0d_5arm_hard12v2_20260811_v3/launch.sh`；3519B；SHA256=`6f2f6dffd88faf964ac509a09b11b7c90a0020bccd18c7ac4a8caac85e908e05`；`bash -n`通过|

## 6.N607预注册

|项目|冻结值|
|---|---|
|python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|project|`/home/szu2070436088/2510044040/CV-SincNet`|
|source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_source_snapshot_20260811_v3`|
|working directory|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_source_snapshot_20260811_v3/code`|
|context|`/home/szu2070436088/2510044040/CV-SincNet/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json`|
|smoke|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_truthfree_smoke_20260811_v3`|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_5arm_hard12v2_20260811_v3`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0d_5arm_hard12v2_20260811_v3`|
|GPU|8个shard固定映射GPU0–7，每卡一个本run进程，进程内`cuda:0`|
|expected artifacts|60 job receipt、120 COMMIT、120 prediction artifact、60 score、180 fit audit、180 resource audit、8 shard summary|

同步映射：archive→`source_root/d92_e0d_runtime_closure_96a48fc3.tar.gz`；config→`source_root/stage2_d92_e0d_5arm_hard12v2_v1.json`；launch→`source_root/launch.sh`。

精确远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_source_snapshot_20260811_v3 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

launch先执行运行闭包import、prepare 60任务和GPU0真实sealed-checkpoint truth-free one-shot smoke；只有该smoke完整PASS才启动8个shard。

## 7.健康停止与成功判据

只在P0协议/安全违规、错误闭包/路径、输出覆盖、query泄漏、共享stop marker，或至少两个不同outer在prediction前出现同一确定性异常指纹时停止本run。绝不按H或accuracy停止。停止前绑定本run PID/CWD/cmdline，仅终止本run进程树，保留partial artifacts；fresh retry=false。

技术成功要求真实checkpoint smoke PASS、60/60 job、8/8 shard PASS、failed=0、异常指纹为空、最终GPU/run进程/SSH连接释放。完整artifact取回前禁止读取性能作决策。

## 8.结果区

待N607 artifacts完整取回后补充五臂聚合表、逐outer配对、资源表、异常、严格门和最终裁决。
