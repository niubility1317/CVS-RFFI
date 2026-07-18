# D55中心化LOO难度截距补偿报告

## 1.状态与目标

- 状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；operator Codex；105/105行完成，exit0，elapsed73.673s。
- receiver20-1、seed713101、K10/new5、3场景×5fold；本地、无N607、无125。
- 目标：在D46上仅补偿support-LOO困难类，联合改善rain old与low-elev new floor。

## 2.公式、协议与验证

`d_c=sum_g w_gc CE_gc`，`Delta b_c=d_c-mean(d)`，`W_D55=W_D46`，`b_D55=b_D46+Delta b`。无系数/温度/阈值/clip/扫描/类ID/角色/scene/receiver/query；K1/K2精确D46 fallback。复用`VALIDATED_ONCE p2_min_v1`，support-only，clean/source/query truth/quota/count/global assignment禁止。

D55定向7项、D46＋D55联合23/23、`py_compile`通过；额外适配仅136 MAC-equivalent、0比较。成功须保持D46 after/new/min-new并改善H/forget/joint/floor，且无场景交换伤害；否则停止，不跑第二seed/formal/125。

完成后必须报告7候选、3场景、逐类、15fold、matched版本、20epoch、混淆、补偿分布、量化、资源、artifact SHA。

## 3.执行锁

- 实现`afa49cb7`；clean worktree`E:\type10-7\code\snapshots\d55wt`；脚本SHA`9dc956749a9f545e6bad98136b6f466203fb6f7e7c6f3d00c08cdb86d07e1637`；clean测试7/7；输出启动前不存在。
- exact command沿用D54全部数据/授权/hash/runtime/device/mode/candidate参数，仅替换脚本为`probe_d55_centered_loo_class_difficulty_intercept.py`、arm为`--d55-arm centered_loo_class_difficulty_intercept`、输出为本报告目录下`centered_loo_class_difficulty_intercept`。

## 4.结论先行

D55完整跑完，但相对当前最强D46出现全面退化：before old从92.22%降至83.33%，after old从81.67%降至70.56%，seen-new从84.67%降至69.33%，同一行H从82.33%降至68.46%，forgetting由10.56pp升至12.78pp。15/15个outer-row预测哈希全部改变；new→old混淆增加11次，new→new混淆增加12次。D55不晋级、不跑第二seed、不进入formal/125。

该结果不是量化误差。D55 INT8与matched FP32的before/final support及outer argmax变化计数均为0，最大分数绝对误差仅0.001935。失败来自机制尺度失配：注册后中心化截距补偿L2均值1.0032、单类绝对偏移最大0.8731，直接以support LOO-CE的原始数值改变logit截距，导致全部类别边界重排。

## 5.七候选完整同排性能

所有指标均由相同候选的15个outer rows联合计算；`min-old/new`是跨类别均值中的最弱类别，`row floor`另见汇总artifact。Phase2本轮不包含unknown拒识，coverage/rollback/defer均为N/A，不以缺失字段作性能声明。

|候选|机制|before old|after old|seen-new|H|forget|joint|min-before|min-after|min-new|old→new/new→old/new→new|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|B3_SINGLE_IQ_DIAG_FFTRF|B3单IQ对角FFTRF|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|低于D46|
|D42-D40-HNBR-INT8-NEGATIVE|HNBR旧负路线|85.56%|85.00%|15.33%|25.16%|0.56pp|0.00%|66.67%|63.33%|0.00%|2/0/0|新类失效|
|D42-D41-BEC-INT8-NEGATIVE|BEC旧负路线|86.11%|20.56%|78.67%|31.50%|65.56pp|0.00%|76.67%|0.00%|36.67%|142/0/32|旧类崩塌|
|D42-PROTOnet-CDA-ZID160|ProtoNet-CDA|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|弱基线|
|D42-USLDA-FP32-MATCHED|D55 matched FP32|83.33%|70.56%|69.33%|68.46%|12.78pp|23.33%|76.67%|63.33%|46.67%|26/19/27|与INT8完全一致，负结果|
|D42-USLDA-INT8|D55中心化LOO难度截距|83.33%|70.56%|69.33%|68.46%|12.78pp|23.33%|76.67%|63.33%|46.67%|26/19/27|主候选，负结果|
|Z0_SUPPORT_ONLY|support-only原型|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|弱基线|

## 6.主候选三场景表现

|场景|before old|after old|seen-new|H|forget|joint|min-before|min-after|min-new|old→new/new→old/new→new|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|leo_clear_weak|91.67%|81.67%|86.00%|82.67%|10.00pp|40.00%|90.00%|70.00%|80.00%|3/5/2|三场景中最好，但相对D46的after/new/H仍分别低8.33/12.00/10.90pp|
|leo_low_elev_weak|85.00%|63.33%|58.00%|59.78%|21.67pp|10.00%|80.00%|50.00%|10.00%|18/4/17|主要失效场景；new最弱类仅10%，forget比D46高11.67pp|
|leo_rain_weak|73.33%|66.67%|64.00%|62.92%|6.67pp|20.00%|50.00%|40.00%|50.00%|5/10/8|forget较D46低6.67pp，但以before/after/new分别下降16.67/10.00/16.00pp为代价|

## 7.逐类别性能

类别仅以封存哈希短前缀标识；O0—O5和N0—N4不承载旧/新之外的语义分支。

|类别|哈希前缀|before|after|变化/表现|
|---|---|---:|---:|---|
|O0|cls_1f33|76.67%|63.33%|-13.34pp|
|O1|cls_33bb|90.00%|86.67%|-3.33pp，旧类最佳|
|O2|cls_75aa|76.67%|66.67%|-10.00pp|
|O3|cls_8b02|90.00%|73.33%|-16.67pp|
|O4|cls_a53c|86.67%|63.33%|-23.34pp，旧类最差退化|
|O5|cls_f8df|80.00%|70.00%|-10.00pp|

|新类别|哈希前缀|seen-new|表现|
|---|---|---:|---|
|N0|cls_09f8|86.67%|与N4并列最佳|
|N1|cls_1c2a|46.67%|全局新类瓶颈|
|N2|cls_b8fb|53.33%|第二弱类|
|N3|cls_d3af|73.33%|中等|
|N4|cls_f608|86.67%|与N0并列最佳|

## 8.十五折完整表现

|场景|fold|before|after|new|H|forget|joint|before/after/new floor|old→new/new→old/new→new|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|83.33%|91.67%|70.00%|79.38%|-8.33pp|50.00%|50/50/50%|0/2/1|
|clear|1|100.00%|66.67%|100.00%|80.00%|33.33pp|0.00%|100/0/100%|2/0/0|
|clear|2|75.00%|75.00%|80.00%|77.42%|0.00pp|50.00%|50/50/50%|1/2/0|
|clear|3|100.00%|75.00%|90.00%|81.82%|25.00pp|50.00%|100/50/50%|0/1/0|
|clear|4|100.00%|100.00%|90.00%|94.74%|0.00pp|50.00%|100/100/50%|0/0/1|
|low|0|83.33%|58.33%|70.00%|63.64%|25.00pp|0.00%|50/50/0%|4/1/2|
|low|1|75.00%|41.67%|50.00%|45.45%|33.33pp|0.00%|50/0/0%|5/0/5|
|low|2|91.67%|75.00%|50.00%|60.00%|16.67pp|0.00%|50/50/0%|2/2/3|
|low|3|100.00%|75.00%|60.00%|66.67%|25.00pp|0.00%|100/50/0%|3/0/4|
|low|4|75.00%|66.67%|60.00%|63.16%|8.33pp|50.00%|50/50/50%|4/1/3|
|rain|0|83.33%|83.33%|40.00%|54.05%|0.00pp|0.00%|50/50/0%|0/5/1|
|rain|1|58.33%|58.33%|50.00%|53.85%|0.00pp|0.00%|0/0/0%|1/3/2|
|rain|2|83.33%|75.00%|70.00%|72.41%|8.33pp|50.00%|50/50/50%|1/0/3|
|rain|3|83.33%|66.67%|80.00%|72.73%|16.67pp|50.00%|50/50/50%|2/0/2|
|rain|4|58.33%|50.00%|80.00%|61.54%|8.33pp|0.00%|0/0/50%|1/2/0|

## 9.与当前最强D46的同折比较

|指标|D46|D55|D55-D46|
|---|---:|---:|---:|
|before old|92.22%|83.33%|-8.89pp|
|after old|81.67%|70.56%|-11.11pp|
|seen-new|84.67%|69.33%|-15.33pp|
|H|82.33%|68.46%|-13.88pp|
|forgetting|10.56pp|12.78pp|+2.22pp，变差|
|joint floor|23.33%|23.33%|0.00pp|
|min-before class|80.00%|76.67%|-3.33pp|
|min-after class|53.33%|63.33%|+10.00pp|
|min-new class|73.33%|46.67%|-26.67pp|
|old→new/new→old/new→new|25/8/15|26/19/27|+1/+11/+12|

min-after的单项改善不能作为晋级证据：它与after均值下降11.11pp、new均值下降15.33pp及min-new下降26.67pp同时发生。D55把错误从一个旧类瓶颈扩散到全部类别边界，并没有形成可接受的联合改善。

## 10.训练过程

20epoch均为support-only，所有epoch的`query_rows_used_sum=0`。loss从1.0320单调主趋势降至0.1027，support accuracy从95.14%升至100%，说明优化过程本身收敛；outer性能恶化揭示support内拟合与独立query泛化脱节。

|epoch|loss mean|support acc|gradient norm|
|---:|---:|---:|---:|
|1|1.0320|95.14%|1.0838|
|2|0.8014|95.97%|0.8706|
|3|0.6235|97.78%|0.6909|
|4|0.5005|97.50%|0.5407|
|5|0.4160|97.78%|0.4363|
|6|0.3540|98.19%|0.3698|
|7|0.2991|98.61%|0.3155|
|8|0.2610|98.89%|0.3014|
|9|0.2339|99.03%|0.2570|
|10|0.2161|99.03%|0.2359|
|11|0.1903|99.58%|0.2206|
|12|0.1744|99.31%|0.2027|
|13|0.1606|99.72%|0.1860|
|14|0.1527|99.86%|0.2058|
|15|0.1424|99.72%|0.1740|
|16|0.1314|100.00%|0.1665|
|17|0.1268|99.72%|0.1705|
|18|0.1151|99.72%|0.1474|
|19|0.1099|99.86%|0.1314|
|20|0.1027|100.00%|0.1354|

## 11.补偿机制审计

|阶段|weighted difficulty min/mean/max|补偿L2 mean/max|单类绝对补偿mean/max|补偿和最大绝对误差|系数变化L2均值|
|---|---|---|---|---|---:|
|before|0.5026/0.9241/1.5652|0.5030/0.7540|0.3789/0.6066|6.66e-16|2.62e-7|
|final|0.5356/1.3934/2.2124|1.0032/1.2903|0.6741/0.8731|3.55e-15|2.61e-7|

中心化约束数值闭合，系数变化仅来自float重中心化舍入；因此实现没有偏离预注册公式。问题在于LOO-CE的量纲不是可直接加入判别logit的校准截距。final阶段补偿L2约为before的2倍，并在rain达到场景均值1.1723，恰与跨场景outer退化一致。禁止对这一失败路线继续做alpha/clip/threshold扫描，因为那会把固定公式探针变成开发集调参。

## 12.量化、资源与协议

- INT8与matched FP32：before outer argmax变化0、final outer argmax变化0、margin sign flip0；score最大绝对误差min/mean/max为0.000383/0.000906/0.001935。
- 资源：36次closed-form LDA fit；总适配估算1,077,328,106 MAC，其中D55仅新增136 MAC-equivalent、0次比较；每query 6,624 MAC；2016个可训练参数；persistent state 8,583B；registry state 941B；峰值CUDA显存22,886,912B；20epoch/20optimizer steps。
- 协议：runtime`cuda:0`；deployment coefficient int8、intercept float16；query rows/features/labels/role/quota/true-count/global assignment/dependent optimization均为0/false；source/clean访问false；dense query graph0B。
- 数据：复用已验证`p2_min_v1`capsule，本轮方法变化未触发也不应触发数据重验证。

## 13.产物与哈希

输出目录：`E:\type10-7\automation_reports\CV-SincNet\d55_centered_loo_difficulty_probe_20260719\centered_loo_class_difficulty_intercept`。

|artifact|bytes|SHA256|
|---|---:|---|
|`training_log.jsonl`|11,312,514|`339eb0de948dc2ede4e75e0f09b8ceea8ac1c7cac25bedf17efd965839f2b179`|
|`support_audit.json`|313,588|`ab433505166bbb29123df1fc8877ff4f3ce38198cc0076e9419df45f9ef0d03f`|
|`resource_audit.json`|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|`geometry_audit.json`|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|`selection.json`|2,992|`dec4ae1b4ecfb1bbad7d7da5bf411ed3bc83227e450a306c2404577a39053d0a`|
|`RECEIPT.json`|4,940|`b1d5592755f70d088d849111b41c6bbf2ded378b97ca5f42fe25e53894f1aba5`|
|`D55_PROBE_METADATA.json`|1,901|`bab385ea68134af4640872a5e2f8bc675a7f9610d645df981e6508ec9cddbd08`|
|`full_performance_summary.json`|110,143|`8edf0c1c656a9a5740b46de5008b2d0d5c8d6d51c6e4fa27ebf84f30f87303fc`|

## 14.验收判定与缺陷

|门槛|要求|D55|判定|
|---|---:|---:|---|
|K10 after old|≥92%|70.56%|失败，差21.44pp|
|K10 min-old|≥88%|63.33%|失败，差24.67pp|
|K10 seen-new|≥92%|69.33%|失败，差22.67pp|
|联合优于D46|保持主指标并改善H/forget/floor|主指标全面下降|失败|
|协议闭合|query/source/clean/Oracle/quota均禁用|全部闭合|通过|

具体缺陷不是“效果不好”这一笼统标签，而有三项可观测表现：第一，raw CE量级造成过大的类间截距偏移；第二，support accuracy达到100%时outer after/new仍仅70.56%/69.33%，显示强烈的support泛化错位；第三，low场景集中产生18次old→new和17次new→new混淆，N1/N2分别只有46.67%/53.33%。

## 15.下一步决策

D55停止。下一候选必须与原始CE量级截距补偿正交，也不得回到D52—D54已经否决的median centroid residual/norm/spectral transport。可继续研究support-only、无超参的离散拓扑证据，例如使用LOO混淆关系的有向图结构或秩信息决定结构性收缩，但不能把raw CE幅值再次映射为logit偏置；新公式必须在执行前预注册，并继续同时报告域适应前后旧类、新类、H、forgetting、逐类floor及全部15折。
