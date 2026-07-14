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

### Ridge诊断launch记录

run ID=`qknn_extreme_light_ridge_k30_20260714_2325_v1`；Git commit=`6cf54b5`。三个同步文件远端SHA256与本地一致；N607直接preflight PASS。当前GPU0–3各有1个RIEI进程、GPU4–7空闲，本轮用`--device cpu`执行，不占用GPU。输出根`runs/qknn_extreme_light_ridge_k30_20260714_2325_v1`，日志根`logs/qknn_extreme_light_ridge_k30_20260714_2325_v1`；36-row dry-run PASS，后续以3个shard执行。

## 2026-07-14 23:34闭式ridge结果与低秩margin路线

36/36 rows完成，0失败、0协议违规、108条闭式trace。最佳5/10/20类同机制row的`old/floor/new/H`分别为`85.83/72.50/88.17/86.95%`、`83.89/61.67/82.33/83.07%`、`81.11/56.67/89.54/85.09%`，联合通过0。ridge虽把适配压到毫秒级，但性能弱于20epoch对角头。

下一机制加入rank8/16低秩残差度量与对所有类对称的CosFace margin`0.05/0.1/0.2`，仍为20epoch、无query适配/图、无角色/配额Oracle。最大rank16、26类为15,130参数、60,520B状态、约15,130MAC/query。本地35项相关pytest及36-row dry-run PASS。

### 低秩margin诊断launch记录

run ID=`qknn_extreme_light_lowrank_k30_20260714_2340_v1`；Git commit=`0ac6917`。N607直接preflight、远端SHA256、`py_compile`和36-row dry-run均PASS；三个shard分别绑定空闲物理GPU4/5/6，每个12 rows。输出根`runs/qknn_extreme_light_lowrank_k30_20260714_2340_v1`，日志根`logs/qknn_extreme_light_lowrank_k30_20260714_2340_v1`。

## 2026-07-14 23:46低秩margin结果

36/36 rows完成，0失败、0协议违规、2,160条loss trace。最佳统一rank8/margin0.05在5/10/20类上的`old/floor/new/H`为`88.75/70.83/89.00/88.82%`、`84.17/65.00/86.33/85.15%`、`85.56/57.50/92.33/88.80%`，联合通过0。失败集中在`6-15↔1-18`和`14-7↔14-11`相邻边界。下一步先更新`项目.md`，再测试只增加一次性support enrollment前向、而query保持1-view的三场景support增强。

### Support增强诊断launch记录

`项目.md`已先更新，Git镜像和实现提交=`90fab8c`。run ID=`qknn_extreme_light_support_aug_k30_20260714_2355_v1`，共24 rows；N607直接preflight、远端SHA256、`py_compile`和dry-run PASS。8张GPU各有1个RIEI任务，三个shard绑定GPU4/5/6，未超过每卡2任务。每row物理K=30、support view=3、query view=1。

## 2026-07-14 23:51 Support增强诊断结果

24/24 rows完成，0失败、0协议违规、0 support/query重叠、0 query训练/选模、1,080条loss trace无非有限值。完整artifact、日志和汇总已拉回`local_artifacts/qknn_extreme_light_support_aug_k30_20260714_2355_v1*`。仅`el_diag_aug3_e20/new20/seed713102/rx8-8/K30`单row通过：`old=93.06%`、`最低旧类=86.67%`、`new=91.25%`、`H=92.14%`；同一arm在两个开发seed上的联合统计如下。

|arm|新类|old均值|最低旧类均值|new均值|H均值|联合通过|
|---|---:|---:|---:|---:|---:|---:|
|el_diag_aug3_e20|5|90.56%|79.17%|89.33%|89.74%|0/2|
|el_diag_aug3_e20|10|86.53%|60.83%|85.92%|86.19%|0/2|
|el_diag_aug3_e20|20|89.17%|75.83%|92.67%|90.80%|1/2|
|el_lowrank_r8_m0p05_aug3_e20|5|90.14%|76.67%|92.50%|91.23%|0/2|
|el_lowrank_r8_m0p05_aug3_e20|10|89.03%|75.83%|87.00%|87.85%|0/2|
|el_lowrank_r8_m0p05_aug3_e20|20|87.64%|64.17%|94.17%|90.72%|0/2|

三场景support enrollment能显著提高部分row，但没有形成跨5/10/20规模、跨开发seed的统一候选。对角头最大6,938参数、27,752B状态、6,912MAC/query；三support view只增加一次性enrollment成本，持续query保持1-view。

为补齐同切分基线，将另跑`baseline_single_qknn`的6-row配对诊断：`rx8-8×seed713101/713102×5/10/20×K30`。输出根预注册为`runs/qknn_extreme_light_baseline_k30_20260715_0000_v1`；本地`ssr-gpu`下6-row dry-run PASS。完成后若仍不存在统一候选，则按gate停止扩大到5 receiver和独立确认seed。

## 2026-07-15 00:04单qKNN基线结果与开发gate

`baseline_single_qknn`6/6 rows完成，0失败、0协议违规、0 support/query重叠、0 query训练/选模；artifact、日志和汇总已拉回`local_artifacts/qknn_extreme_light_baseline_k30_20260715_0000_v1*`。

|方法|新类|old均值|最低旧类均值|new均值|H均值|状态上限|MAC/query|联合通过|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|单qKNN基线|5|83.47%|60.83%|80.83%|82.10%|113,862B|90,112|0/2|
|单qKNN基线|10|81.67%|55.83%|76.75%|79.08%|163,758B|131,072|0/2|
|单qKNN基线|20|80.28%|55.00%|79.96%|80.07%|263,538B|212,992|0/2|
|对角头+3-view support、20epoch|5|90.56%|79.17%|89.33%|89.74%|12,332B|3,072|0/2|
|对角头+3-view support、20epoch|10|86.53%|60.83%|85.92%|86.19%|17,472B|4,352|0/2|
|对角头+3-view support、20epoch|20|89.17%|75.83%|92.67%|90.80%|27,752B|6,912|1/2|

相对单qKNN，support增强对角头的20类状态减少89.47%，head MAC减少96.75%，old/new/H显著提高；但5/10/20规模和两个seed没有一个统一通过组合，旧类floor仍是主要硬失败。

按预注册gate，不运行`713106–713110`确认seed，不扩大到5 receiver，也不放宽query、角色、配额、dense graph、query TTA或adapter60权限。当前结论为diagnostic-negative：资源目标达成，性能目标未达，不能晋升。

## 2026-07-15 00:12冻结source-logit逐样本特征周期

新周期复用`项目.md`允许冻结的source classifier bank：把每个物理样本同一次ADV3B02前向产生的6维`tx_logits`作为逐样本特征，与160维`z_id`和96维FFT拼接。source classifier/backbone均冻结，不使用query角色、query-batch统计、query标签或配额，不增加backbone前向。

预注册`rx8-8×K30×seed713101/713102×5/10/20`，扫描统一logit权重`0.25/0.5/1.0/2.0`，三场景support enrollment、FFT2.0、20epoch和query 1-view保持不变。run ID=`qknn_extreme_light_source_logits_k30_20260715_0012_v1`，共24 rows；本地编译、17项pytest和dry-run PASS。只有统一权重在全部规模和两个seed通过，才扩大receiver；确认seed继续封存。

### 冻结source-logit结果与source-bank anchor计划

24/24 rows完成，0失败、0协议违规、1,440条loss trace。权重0.25的5/10/20类`old/floor/new/H`均值为`90.97/80.00/89.50/90.04%`、`86.81/57.50/86.42/86.58%`、`89.44/77.50/91.92/90.60%`；仅三个20类单seed row通过，仍无统一候选。20类资源为7,100参数、28,400B状态、7,074MAC/query，不增加backbone前向。

下一轮固定logit0.25和source/target prototype blend0.25，扫描冻结source prototype余弦anchor strength`0.05/0.1/0.25/0.5`。run ID=`qknn_extreme_light_source_anchor_k30_20260715_0020_v1`，24 rows；19项pytest、编译和dry-run PASS。推理仍为全部注册类统一逐样本argmax，无query角色门控。

### K30 source-bank anchor结果与新目标边界

24/24 rows完成，0失败、0协议违规、1,440条loss trace。strength0.05的5/10/20类`old/floor/new/H`均值为`91.11/80.00/89.83/90.31%`、`86.81/57.50/85.92/86.32%`、`89.58/78.33/92.17/90.80%`，无统一通过组合；更强source anchor反而损害10/20类old/floor。

用户随后把正式门槛提高为K10下`old>=95%`、floor`>=88%`、5/10/20新类`>=92/90/86%`，且matched K5四项指标相对K10均不得下降超过3个百分点。该K30结果仅作目标变更前历史诊断，不作为新目标成功或选模证据。

## K10/K5审计实现

新增K10主协议配置，固定`support_pool_max_k=10`和新门槛；新增matched K5/K10审计器，逐场景核验K5 support是K10子集、query完全相同，以及`old/floor/new/H`四项drop均不超过3pp。K10绝对门槛与K5稳健性必须同时通过。22项相关pytest和编译PASS。

### K10/K5 feasibility计划

run ID=`qknn_extreme_light_k10k5_feasibility_20260715_0032_v1`，`rx8-8×2开发seed×5/10/20新类×K5/K10×2arms=24 rows`。比较logit0.25对角头与轻source-anchor0.05；固定3-view support、query 1-view、20epoch、冻结ADV3B02和无adapter60。Git commit=`a0fe369`，计划绑定空闲GPU0/1/2。

### K10/K5 feasibility结果

24/24 rows完成，0执行失败、0协议违规。36个matched receiver×seed×scenario×新类规模单元均满足K5 support严格嵌套K10、每类恰为5/10个物理ID、query ID列表完全相同，嵌套与query一致性违规为0；但K10为0/24 row联合通过，36个matched场景单元中0个同时满足K10绝对门槛与K5四指标drop门槛。

|arm|新类|K5 old/floor/new/H|K10 old/floor/new/H|全局通过|
|---|---:|---|---|---:|
|logit0.25对角头|5|80.28/57.50/75.33/77.59%|83.75/65.83/84.17/83.83%|否|
|logit0.25对角头|10|76.67/55.83/76.58/76.61%|81.11/62.50/83.42/82.22%|否|
|logit0.25对角头|20|77.92/53.33/85.17/81.37%|80.00/56.67/90.17/84.76%|否|
|logit0.25+source anchor0.05|5|79.72/55.83/74.00/76.55%|83.89/65.00/84.00/83.83%|否|
|logit0.25+source anchor0.05|10|77.08/56.67/76.33/76.69%|80.97/62.50/83.33/82.12%|否|
|logit0.25+source anchor0.05|20|78.75/53.33/84.83/81.66%|80.28/57.50/90.17/84.92%|否|

当前瓶颈是K10冻结表示在LEO弱域下的旧类整体与逐类可分性，source anchor没有实质收益。下一步继续封存确认seed并限制在单开发receiver，探索同一物理view上的低成本RF统计特征；不增加query权限、不使用角色/配额Oracle、不启用adapter60。

## 同view RF-stat32开发计划

新增32维增益归一化RF统计描述子，输入与冻结backbone和FFT96使用同一个后信道单物理view，逐样本提取IQ/幅度统计、高阶复矩和短时自相关；不使用query batch、标签、角色或配额。它与FFT96组成128维`fft_rf_features`，query仍为1-view，只新增`O(T)`统计计算；20新类预计head状态约32KB、约7.5k MAC/query。

本地编译、24项相关pytest、launcher语法和24-row dry-run PASS。预注册特征根=`cvs_qknnv42_extreme_light_20new_features_rf32_20260715_v1`，K10 screen=`qknn_extreme_light_rf32_k10_screen_20260715_0045_v1`：`rx8-8×2开发seed×5/10/20新类×K10×4个aux权重=24 rows`。只按K10选统一权重，再对锁定候选补matched K5；确认seed继续封存。

### RF-stat32结果与multi-prototype计划

三场景各导出9,800行`z_id160+FFT96+RF-stat32`，完整日志错误扫描PASS。K10 screen 24/24 rows完成、0失败、0权限违规、0联合通过。权重0.5/1/2/4下，5类`old/floor/new/H`依次为`80.28/60.83/79.33/79.70%`、`80.69/62.50/82.50/81.47%`、`82.78/65.83/84.00/83.27%`、`84.31/68.33/84.17/84.18%`；10类最佳为`80.00/61.67/82.75/81.31%`，20类最佳为`78.06/54.17/90.83/83.95%`。RF统计不能修复旧类可分性。

下一run=`qknn_extreme_light_multiproto_k10_screen_20260715_0054_v1`，比较每类2/3个确定性球面prototype以及FFT/FFT+RF，24个K10开发rows。该头0epoch、0参数、support-only、无query图；20新类三prototype状态分别79,872B/89,856B，MAC/query分别19,968/22,464。K5与确认seed继续封存。

### Multi-prototype结果与repair-canonical1计划

24/24 rows完成、0失败、0权限违规、0联合通过。最佳3prototype+FFT+RF在5/10/20类的`old/floor/new/H`仅为`70.14/41.67/73.83/71.82%`、`68.33/40.83/69.08/68.56%`、`68.19/40.83/79.92/73.51%`，显著弱于对角头；该路线终止。

下一轮使用1-view逐样本`repair_canonical1`：盲DC/CFO残差校正、RMS归一化和限幅后执行同一次冻结backbone前向，0参数、0epoch、无query batch。预注册特征根=`cvs_qknnv42_extreme_light_20new_features_repair1_20260715_v1`，K10 screen=`qknn_extreme_light_repair1_k10_screen_20260715_0100_v1`，比较FFT/FFT+RF与权重2/4共24 rows。35项相关pytest、编译、launcher语法和dry-run PASS；K5与确认seed继续封存。

### repair-canonical1结果

三场景各导出9,800行且view count=1；K10 screen 24/24 rows完成、0失败、0权限违规、0联合通过。最佳FFT+RF权重4在5/10/20类的`old/floor/new/H`仅为`38.06/11.67/29.50/32.53%`、`34.03/5.83/30.75/31.98%`、`24.58/3.33/34.29/28.14%`。盲平均相位步破坏了判别结构，该路线终止。

全部RF、multi-prototype、repair1 artifact/日志/summary已拉回`local_artifacts/*20260715*`对应目录。当前仍无统一K10候选，不运行K5、不扩receiver、不解封确认seed。下一合法方向限制为冻结backbone前极少参数support-only可学习接收校正；仍须≤20epoch、≤50k参数、1-view、逐样本推理。
