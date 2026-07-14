# qKNN极轻型5/10/20新类目标模式优化报告

本文件镜像`E:\type10-7\automation_reports\CV-SincNet\qknn_extreme_light_5_10_20_20260714_215758\report.md`，用于Git版本承载。任务状态：`DESIGN_AND_PROTOCOL_PREREGISTRATION`。

## 成功门槛

|指标|门槛|
|---|---:|
|`old_acc`|>=88%|
|`min_old_class_acc`|>=85%|
|5类`seen_new_acc`|>=90%|
|10类`seen_new_acc`|>=88%|
|20类`seen_new_acc`|>=83%|

正式确认覆盖5个target receiver、至少5个独立seed、3个正式LEO场景；使用嵌套真实ManyTx TX集合和开发后锁定的统一K。默认冻结ADV3B02，1-view，adapter参数不超过50k，适配不超过20epoch，无query图，持久状态不超过128KB。禁止query真实角色、类别quota、query标签拟合和跨K/seed/新类规模拼接最佳结果。

## 当前执行边界

根目录`项目.md`已先加入第10.3.1节；根目录不是Git仓库，本仓库通过`docs/cvs_stage2c_extreme_light_goal_20260714.md`承载协议增量。当前尚未launch，后续需先完成ManyTx覆盖审计、本地实现与`ssr-gpu`验证，再执行N607新鲜preflight、同步、launch、完整日志分析与独立确认。

## 2026-07-14 22:17更新

ManyTx覆盖审计、嵌套5/10/20真实TX清单、开发/确认seed、K候选和资源上限已预注册。新增support-only对角度量余弦头：最大26类6,938参数、约27.1KB FP32状态、20epoch、1-view、无backbone梯度、无query图、无role/quota Oracle。新增20类exporter和resume-safe smoke/dev/confirm matrix runner。本地`py_compile`、6项pytest、exporter语法/dry-run、36-row matrix dry-run和端到端runner smoke均PASS。

N607 22:17新鲜inventory显示8张GPU各有1个约470MiB的RIEI训练进程。本任务按每GPU最多2个训练实验的许可，只计划在GPU0/1/2各增加1个20新类feature export，不干预现有任务。首次远端输出根为`runs/cvs_qknnv42_extreme_light_20new_features_20260714`，日志根为`logs/cvs_qknnv42_extreme_light_20new_features_20260714`。

## 2026-07-14 22:29覆盖审计与v2修复

首次三场景feature export本身完成，但逐receiver×TX审计发现旧清单按所有day合计覆盖筛选，而launch实际只读取`day_index=0,equalized_index=1`。其中`18-1`在`20-1`只有11个样本，`10-1`在两个receiver为0，不能满足20-shot+20-query，故首次artifact降级为`NON_LAUNCH_DIAGNOSTIC`，未启动smoke。

v2清单改为按实际slice逐receiver至少40个样本预筛；111个TX合格，嵌套20类为`1-16,1-18,18-10,14-11,8-3,18-8,10-10,16-19,20-12,4-10,13-14,2-5,1-8,19-13,19-9,3-8,19-8,11-19,2-16,19-6`，每类每receiver实际均为50个样本。新输出使用独立`v2_day0_eq1`根，保留旧artifact不覆盖。修复后15项相关pytest、JSON解析、exporter语法/dry-run和36-row matrix dry-run均PASS。

## 2026-07-14 22:40 CUDA遥测启动修复

smoke首次执行完成12个基线row；24个极轻型row在训练前统一因`reset_peak_memory_stats`早于CUDA context初始化而失败，无性能指标生成。已在本地加入目标device零长度tensor初始化，随后再重置峰值显存计数器；`ssr-gpu`下`py_compile`和15项相关pytest PASS。resume-safe重跑会保留已完成基线row，仅重跑24个极轻型row。

## 2026-07-14 22:43 smoke完成

36/36 rows和36个完成日志均通过审计，0协议违规、0非有限loss。极轻型对角头相对单qKNN显著提升old/new/H，同时将20类状态压至27,752B、head评分压至6,912MAC/query；但成功只集中于`20-1`，`8-8`的旧类floor及5/10类new仍未过门槛，不能进入正式确认。下一轮先在`8-8`测试5epoch强prototype anchor和FFT权重`0/0.5/1.0/1.5/2.0`，以同时改善跨seed稳定性并进一步压缩适配成本。

## 2026-07-14 22:56五epoch结果与K30扩展

五epoch稳定性sweep 36/36完成但联合通过0；最终support accuracy只有约59.5%–89.7%，证明直接从20压到5epoch会欠拟合。`z_id`-only最差，FFT96仍有必要。由于所有选定新类在实际slice的每个receiver均有50样本，开发K可合法扩展到30并保留20-query；下一步只在最弱`8-8`上测试20epoch FFT1.5/2.0的K30，再决定是否扩大完整开发矩阵。

## 2026-07-14 23:02 K30结果

K30 12/12 rows完成但联合通过0。FFT2.0在`8-8`的5/10/20类old均值为89.86/85.97/85.42%，new为87.50/85.58/92.71%；平均值与seed稳定性改善，但最低旧类仍仅约69%–78%。下一轮固定K30、20epoch和FFT2.0，测试prototype anchor5/20与feature noise0.01/0.05，仍不使用adapter60或角色Oracle。

## 2026-07-14 23:09正则结果与零训练prototype路线

K30正则化24/24完成但联合通过0；anchor5/20近似无效，noise0.05明显伤害old/floor。下一步实现零训练support prototype余弦头：0epoch、0参数、最大26类状态26,624B、6,656MAC/query，不依赖adapter60，并扫描FFT权重以判断去掉可训练头后能否改善逐类floor。

## 2026-07-14 23:17零训练prototype结果与闭式ridge路线

prototype sweep 30/30 rows完成但联合通过0。FFT2.0在5/10/20类的`old/new/H`均值仅为`70.69/71.50/70.85%`、`63.89/68.25/65.79%`、`63.61/70.88/66.86%`，最低旧类均值均为24.17%；单prototype不具备足够判别能力。

下一机制为闭式support-only多类ridge线性头：0epoch、0梯度、不更新ADV3B02，不使用query适配、query图、old/new角色或类别配额。最大26类状态26,728B、逐query约6,682MAC。先固定`8-8`、K30、两开发seed、5/10/20类，扫描`λ∈{1e-4,1e-3,1e-2,1e-1,1,10}`；确认seed仍隔离。本地`ssr-gpu`下33项相关pytest PASS。
