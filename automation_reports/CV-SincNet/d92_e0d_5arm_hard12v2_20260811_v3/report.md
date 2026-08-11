# D92-E0D-5arm-Hard12-v2运行时契约修复实验报告

## 1.基本信息

|字段|内容|
|---|---|
|实验ID|`D92-E0D-5arm-Hard12-v2`|
|run ID|`d92_e0d_5arm_hard12v2_20260811_v3`|
|日期|2026-08-11|
|operator|Codex primary；N607唯一runner|
|当前状态|`ANALYZED / NO_D_GEOMETRY_PROMOTION`|
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
|expected artifacts（实际文件布局）|60 job receipt、120 COMMIT、120 prediction artifact、60 score、120 fit-audit文件（每文件3场景，共360行）、120 state-level resource-audit文件、8 shard summary|

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

### 8.1运行与artifact闭环

- 真实checkpoint truth-free smoke为`D92_E0D_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS`；query truth、fit、selection和update均为false。
- 60/60 job完成；8/8 shard summary为PASS且failed=0；120个COMMIT、120个prediction artifact、60个score完整；stderr与异常指纹为空。
- v3 matrix manifest SHA256=`d9f8b4aae89a05d544ac00fa129863031509a7190f55fb753061d29a5c0a0fef`；method lock与selection身份在manifest、job receipt和分析器中一致。
- 120个fit-audit文件各包含clear/low_elev/rain 3行，共360条scene-state审计；120个resource-audit文件对应60个job的`DA0_REG0`/`DA0_REG1`两个状态。预注册曾误写“180/180文件”，这是文件布局口径错误，不是缺失运行，不补跑。
- 五臂`DA0_REG0`state/prediction精确一致；两条K1的`DA0_REG1`严格别名一致；fit计数、query MAC和query零访问均通过冻结分析器检查。
- 完整artifact位于`E:\type10-7\local_artifacts\d92_e0d_5arm_hard12v2_20260811_v3`；冻结分析位于其`analysis_r1`子目录。
- 独立结果复核逐值复算summary、gates和10行paired rows，结论为P0=0、P1=0。

### 8.2五臂聚合结果

下表只汇总10个performance outer的`DA0_REG1`结果；H、old BA、old floor、seen-new和forgetting单位均为百分比。unknown rejection、defer与coverage在本实验中为N/A。wall和peak是注册阶段配对资源统计，query MAC五臂完全相同。

|候选|机制|H|old BA|old floor|seen-new|forgetting|wall中位(ms)|增量peak中位(MiB)|K5/K10 fit|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D92_FULL|B+E+full/block LOO-soft fusion|63.153|65.139|36.167|62.250|16.417|2979.98|2.311|48/88|原D92对照|
|E0_FUSION|E关闭，保留full/block LOO-soft fusion|63.129|65.139|36.333|62.217|16.417|1479.67|1.512|24/44|效率对照|
|E0_FULL_ONLY|E关闭，仅full主几何，无LOO|63.489|65.611|35.333|62.425|15.944|92.03|1.574|2/2|唯一晋级候选；严格门失败|
|E0_BLOCK_ONLY|E关闭，仅block3主几何，无LOO|62.383|64.639|36.333|61.317|16.917|92.00|1.582|2/2|性能下降|
|E0_FIXED50|E关闭，full/block固定0.5/0.5，无LOO|63.051|65.222|36.500|62.008|16.333|191.06|1.469|4/4|未超过FULL_ONLY|

### 8.3FULL_ONLY配对效应

|参考臂|mean ΔH(pp)|H非负outer|Δold BA(pp)|Δold floor(pp)|Δseen-new(pp)|Δforgetting(pp)|wall下降|peak变化|query MAC变化|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|D92_FULL|+0.336|9/10|+0.472|−0.833|+0.175|−0.472|97.70%|−700KiB|0|
|E0_FUSION|+0.360|9/10|+0.472|−1.000|+0.208|−0.472|95.41%|+46KiB|0|

FULL_ONLY相对D92_FULL的10个outer配对如下；所有指标来自同一outer和同一候选，不拼接边际最值。

|outer|K|Cn|ΔH(pp)|Δold BA(pp)|Δold floor(pp)|Δseen-new(pp)|Δforgetting(pp)|wall下降|peak变化(KiB)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`rx_20_1__seed_713103__k_10__new_20`|10|20|+0.734|+0.556|+1.667|+0.833|−0.556|97.74%|−1044|
|`rx_20_1__seed_713103__k_5__new_20`|5|20|+0.141|+0.556|0.000|+0.083|−0.556|95.96%|−752|
|`rx_20_1__seed_713105__k_5__new_20`|5|20|+0.373|+1.389|+3.333|−0.500|−1.389|96.02%|−748|
|`rx_3_19__seed_713102__k_10__new_20`|10|20|+0.650|+0.556|−1.667|+0.667|−0.556|97.65%|−544|
|`rx_3_19__seed_713106__k_10__new_10`|10|10|−0.246|−0.556|−3.333|0.000|+0.556|97.99%|−72|
|`rx_3_19__seed_713106__k_10__new_5`|10|5|+0.005|0.000|0.000|0.000|0.000|97.84%|−652|
|`rx_7_14__seed_713102__k_10__new_5`|10|5|+0.688|+1.389|−5.000|−0.333|−1.389|97.80%|−212|
|`rx_7_14__seed_713103__k_5__new_20`|5|20|+0.093|−0.278|−3.333|+0.500|+0.278|95.21%|−928|
|`rx_7_7__seed_713105__k_10__new_10`|10|10|+0.050|−0.278|0.000|+0.333|+0.278|97.63%|−640|
|`rx_8_8__seed_713104__k_10__new_20`|10|20|+0.869|+1.389|0.000|+0.167|−1.389|97.86%|−1216|

### 8.4严格门与裁决

|门|观测|阈值|结果|
|---|---:|---:|---|
|FULL_ONLY mean ΔH vs E0_FUSION|+0.360pp|>0|PASS|
|FULL_ONLY H非负outer vs E0_FUSION|9/10|≥8/10|PASS|
|FULL_ONLY mean ΔH vs D92_FULL|+0.336pp|≥0.500pp|FAIL|
|FULL_ONLY H非负outer vs D92_FULL|9/10|≥8/10|PASS|
|Δold BA vs D92_FULL|+0.472pp|≥0|PASS|
|Δold floor vs D92_FULL|−0.833pp|≥0|FAIL|
|Δseen-new vs D92_FULL|+0.175pp|≥0|PASS|
|Δforgetting vs D92_FULL|−0.472pp|≤0|PASS|
|wall下降 vs E0_FUSION|95.41%|≥40%|PASS|
|wall下降 vs D92_FULL|97.70%|≥60%|PASS|
|peak变化 vs E0_FUSION|+47,104B|≤0B|FAIL|
|query MAC、fit计数、state/prediction parity、query协议|全部精确|全部精确|PASS|

冻结分析器裁决为`NO_D_GEOMETRY_PROMOTION`。FULL_ONLY证明了D46的K折LOO融合可以把注册fit从K5/K10的24/44（E0_FUSION）或48/88（D92_FULL）降到2/2，并把wall降低95%–98%，同时H平均略有提高；但它没有达到预注册的+0.5个百分点H门，并牺牲old-class floor，因此不进入完整Target125确认。BLOCK_ONLY和FIXED50也不晋级。本轮属于有效的“效率显著、性能门未全过”负晋级结果。

## 9.三轮回顾与第四轮去向

本轮是D92完整部件消融、D92-BE Hard12-v1和D92-E0D Hard12-v2组成的第三个已完成探索轮。回读目标、`项目.md`、conversation index、D69/D70历史生命周期行交换报告及本run完整score后，确认下一轮仍同时评价域适应前后旧类、注册后seen-new、H、逐类floor和forgetting，并保持LEO_weak-only、no clean/source、no query truth/role/quota/global assignment。

进一步分解表明，FULL_ONLY相对D92_FULL在10个performance outer上的old→new仅增加0.083个百分点；四个floor退化outer的old→old错误反而下降0.764个百分点。最严重的floor下降5个百分点outer中，old→new、old→old均下降且old BA上升。因此停止“统一压低新类分数”和按历史困难类定向修补。D69/D70已否决跨注册状态旧行拼接/替换，故也不采用DA0_REG0旧head锚定。

第四轮只检验同一个DA0_REG1 joint state内的full/block旧类contrast融合：保持FULL_ONLY的旧类组均值和全部新类行，只用旧support RMS对齐并混合block旧类contrast。fresh Hard12-v3与v1/v2零交集；冻结设计见`docs/superpowers/plans/2026-08-11-d92-e0ocf-hard12v3.md`。在该回顾落盘前未发布第四轮实验。
