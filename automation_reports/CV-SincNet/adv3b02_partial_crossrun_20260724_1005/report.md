# ADV3B02完整与部分矩阵跨run性能报告

- 分析ID：`adv3b02_partial_crossrun_20260724_1005`
- 时间：2026-07-24
- 范围：ADV3B02-TS-DRQKNN-BCRR的r2至r8所有已产生score的full125启动
- 分析状态：`ANALYZED`
- 正式性能状态：r8为`COMPLETE_MATRIX_PERFORMANCE`；r2至r6为`PARTIAL_MATRIX_DIAGNOSTIC`；r7为`NO_PERFORMANCE_ROWS`
- 数据状态：`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`

## 统计规则修复

此前“未达到125/125即完全不输出性能”的规则会丢失124/125这类高覆盖run的研发信息。本次增加两层结果：

1. `COMPLETE_MATRIX_PERFORMANCE`：只有125/125、1000/1000prediction、1500/1500score和375/375场景闭合的run可形成正式完整125裁决。
2. `PARTIAL_MATRIX_DIAGNOSTIC`：技术失败run仍对已完成row输出四臂、分层和协同统计，但必须同时报告覆盖率、缺失cell和选择性缺失风险，不得冒充完整125或用于正式推广。

该修复不改变方法、数据、runner、scorer、truth绑定或full125晋级门，只改变失败run的分析可见性。

## 证据与复算

从N607各原run根只读提取`row_receipt.json`、四臂score和matrix completion；未读取或修改数据、checkpoint、support、query状态或远端方法文件。

|文件|SHA256|
|---|---|
|`adv3b02_partial_scores_20260724_1005.tar.gz`|`93a7f24461f6c68e9fbafab6f99457fbbbe2f0334dc4c40554acd5045938d17c`|
|`partial_matrix_diagnostic.json`|`d3d6bb6897f59afc764ebaa39b7c2dd2c929482ab5198667e34d7c43d7781185`|
|`four_arm_summary.csv`|`8b516733e657dced1bb43aa7546e220f034e3a86a46c1c55adafc4425cf6050f`|

复算工具：`code/scripts/summarize_adv3b02_partial_matrix.py`。JSON保留逐run、逐row以及receiver、scene、K、seed、new-count分层；CSV保留所有run的四臂总表。

总表按完成row做算术平均；每个row的BA是注册类逐类准确率宏平均，floor/min-old/min-new分别是该row全部类/旧类/新类最低准确率，再跨row平均。old→new和new→old先在row内对三个场景平均，再跨row平均。`I_syn`分别在同一row或同一场景slice的四臂H上计算后聚合。

## 所有产生性能artifact的启动

表中部分run的均值来自不同完成子集，不能直接按H横向排名。最右列保留其覆盖边界。

|run|完成row|H M0|H M_DA|H M_OTHER|H M_JOINT|J old-after|J seen-new|J floor|J forgetting|mean I_syn|正协同slice|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|r2-affine-bcr2|8/125|21.007%|21.011%|21.007%|21.011%|32.396%|15.656%|0.625%|33.160pp|0.000pp|0/24|
|r2-zidtotal1|6/125|20.491%|20.484%|20.491%|20.484%|31.250%|15.347%|1.111%|33.333pp|0.000pp|0/18|
|r2-zidtotal1-bindfix1|29/125|25.803%|25.857%|25.803%|25.857%|41.475%|19.612%|0.920%|32.021pp|0.000pp|0/87|
|r3-q2f32-bcr2|24/125|24.815%|24.870%|24.815%|24.870%|42.211%|18.208%|0.833%|32.188pp|0.000pp|0/72|
|r4-q2f32-bcr3|35/125|26.076%|26.074%|26.076%|26.074%|41.183%|19.798%|1.667%|31.143pp|0.000pp|0/105|
|r5-qzero1|49/125|29.429%|29.433%|29.429%|29.433%|44.444%|22.957%|1.905%|30.476pp|0.000pp|0/147|
|r6-matchedaudit1-prepfix1|124/125|29.318%|29.291%|29.318%|29.291%|43.123%|23.491%|2.285%|29.991pp|0.000pp|0/372|
|r7-q3support1|0/125|NA|NA|NA|NA|NA|NA|NA|NA|NA|0/0|
|r8-bcrmaskidentity1-artifactsfresh1|125/125|29.267%|29.241%|29.267%|29.241%|43.060%|23.440%|2.267%|30.044pp|0.000pp|0/375|

prelaunch阻塞或0score的r2原始发布、r4原始发布、r6原始发布和r8 parent没有性能row，不进入均值表。

## r6的124/125四臂详细表现

缺失cell为`receiver=20-1,seed=713102,K=5,new=20`。其余124个row包含372个场景slice。

|arm|old-before|old-after|old gain|seen-new|H|BA|floor|min-old|min-new|forgetting|old→new|new→old|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|M0|72.610%|43.082%|-29.527pp|23.538%|29.318%|29.232%|2.312%|11.223%|2.634%|29.527pp|44.046%|29.320%|
|M_DA|72.668%|43.123%|-29.545pp|23.491%|29.291%|29.231%|2.285%|11.250%|2.594%|29.545pp|44.066%|29.333%|
|M_OTHER|73.143%|43.082%|-30.060pp|23.538%|29.318%|29.232%|2.312%|11.223%|2.634%|30.060pp|44.046%|29.320%|
|M_JOINT|73.114%|43.123%|-29.991pp|23.491%|29.291%|29.231%|2.285%|11.250%|2.594%|29.991pp|44.066%|29.333%|

组件结论：

- `M_DA−M0`：H下降0.0268pp，seen-new下降0.0470pp，floor下降0.0269pp；DA没有独立正收益。
- `M_OTHER−M0`：注册后old、new、H、BA、floor和双向混淆逐项相同；OTHER没有注册后独立收益。
- `M_JOINT=M_DA`的注册后性能；联合没有超过DA。
- 372/372个slice的`I_syn=0`，正协同0，负协同0。
- OTHER只改变old-before/forgetting基准，未改变注册后统一决策。这不能证明BCRR改善分类。

## r6分层结果

以下均为M_JOINT；完整四臂分层位于`partial_matrix_diagnostic.json`。

### receiver

|receiver|row|old-after|seen-new|H|floor|forgetting|I_syn|
|---|---:|---:|---:|---:|---:|---:|---:|
|20-1|24|34.444%|27.111%|29.211%|1.667%|32.975pp|0|
|3-19|25|31.900%|15.473%|20.632%|3.667%|25.122pp|0|
|7-14|25|54.089%|22.577%|31.100%|2.800%|32.289pp|0|
|7-7|25|51.456%|24.253%|32.365%|1.333%|27.511pp|0|
|8-8|25|43.378%|28.187%|33.145%|1.933%|32.178pp|0|

### K

|K|row|old-after|seen-new|H|floor|forgetting|I_syn|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|25|32.400%|14.680%|20.073%|1.400%|33.711pp|0|
|5|24|41.759%|15.382%|22.137%|0.139%|31.678pp|0|
|10|75|47.133%|29.023%|34.653%|3.267%|28.211pp|0|

### new-count

|new-count|row|old-after|seen-new|H|floor|forgetting|I_syn|
|---:|---:|---:|---:|---:|---:|---:|---:|
|5|25|52.311%|41.613%|45.148%|6.333%|23.033pp|0|
|10|25|46.333%|28.087%|34.351%|2.933%|29.011pp|0|
|20|74|38.934%|15.816%|22.224%|0.698%|32.673pp|0|

### scene

|scene|slice|H M_JOINT|mean I_syn|正/零/负|
|---|---:|---:|---:|---:|
|leo_clear_weak|124|30.425%|0|0/124/0|
|leo_low_elev_weak|124|27.911%|0|0/124/0|
|leo_rain_weak|124|28.575%|0|0/124/0|

五个seed的M_JOINT H分别为30.359%、28.994%、29.117%、29.042%和28.985%；所有seed的`I_syn=0`。

## r6与r8的共同row核验

r6和r8共有124个相同receiver/seed/K/new-count row。对这些共同row：

- 四臂的old-after、seen-new、H、BA、floor、min-old和min-new差值全部严格为0。
- r6相对r8的M_JOINT H、old-after、seen-new、floor和`I_syn`均为`0.0000pp`。
- 唯一缺失r6 row由r8补齐后，r8完整125的M_JOINT为：old-before 73.104%、old-after 43.060%、seen-new 23.440%、H 29.241%、BA 29.167%、floor 2.267%、min-old 11.200%、min-new 2.573%、forgetting 30.044pp、old→new 44.127%、new→old 29.241%。
- r8的375/375个slice仍全部`I_syn=0`。

因此r6的124-row诊断足以提前识别“DA无独立增益、OTHER注册后无效、联合无协同”；完整r8则把该结论升级为无缺失的完整125性能裁决。

## 跨revision匹配比较

将各部分run只与r8中的相同row比较，可消除“先完成了哪些row”造成的均值偏差：

|run|共同row|M_JOINT H相对r8|old-after|seen-new|floor|I_syn|
|---|---:|---:|---:|---:|---:|---:|
|r2-affine-bcr2|8|-0.0002pp|+0.0347pp|-0.0104pp|0|0|
|r2-zidtotal1|6|-0.0248pp|+0.0463pp|-0.0417pp|0|0|
|r2-zidtotal1-bindfix1|29|+0.0096pp|+0.0287pp|-0.0057pp|0|0|
|r3-q2f32-bcr2|24|0|0|0|0|0|
|r4-q2f32-bcr3|35|0|0|0|0|0|
|r5-qzero1|49|0|0|0|0|0|
|r6-matchedaudit1-prepfix1|124|0|0|0|0|0|

r3至r6在其共同row上的注册后性能与r8完全相同。早期表中H从24.9%上升到29.3%主要来自不同run完成子集，而不是revision带来的方法提升。

## 性能裁决

`REJECT_CURRENT_REVISION / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`

完整r8与高覆盖r6共同证明：

1. DA相对M0降低H、seen-new和floor，没有净独立收益。
2. OTHER在注册后与M0逐项相同，没有独立收益。
3. JOINT没有超过DA或OTHER。
4. mean`I_syn=0`，正协同为0/375。
5. old-after 43.060%、seen-new 23.440%、floor 2.267%，同时存在30.044pp遗忘，远未形成可推广性能。

下一revision必须优先修复实际decision geometry：DA需要产生净正确决策增益，BCRR需要在注册后改变预测并独立改善H/floor；在这两项发生前，继续修改量化审计、runner或控制面不会改善方法性能。
