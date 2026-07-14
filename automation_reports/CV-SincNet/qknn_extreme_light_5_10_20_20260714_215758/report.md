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

## 2026-07-15 01:20 support-only微型IQ前端预注册

现有`train_target_adapt.py`的logit/LoRA/feature residual适配器只面向冻结6类source分类头，无法合法表示5/10/20个新增TX；历史`IQResidualPreAdapter`又是source-only训练语义，不能直接冒充Stage2-C target support适配。因此本周期新增独立的`support_only_micro_iq_residual_v1`：冻结ADV3B02全部参数，只训练`Conv(2→8,k5)+depthwise Conv(8,k5)+Conv(8→2,k1)`的恒等初始化残差前端。其可训练参数仅154个，FP16参数状态308B，单query新增34,816MAC；持续推理仍为单物理IQ view、逐样本argmax，不建立query图，也不读取old/new角色或类别配额。

训练只使用合法K10 support物理ID及其预注册`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`三个support view，优化目标为全注册类对称的support prototype交叉熵，加冻结基础特征anchor和输入残差约束；query IQ、query标签和query统计不进入优化。适配20epoch，学习率`5e-4`，batch size128，temperature18，feature anchor0.05，input residual0.02。每轮保存完整loss/CE/anchor/residual/support accuracy轨迹、adapter状态、support/query物理ID哈希、峰值显存、适配时长与逐scenario导出哈希。

第一步仅运行开发receiver`8-8`、seed`713101`、20新类、K10的单cell机制烟测，与同一物理切分的identity-only单qKNN和当前原始单view对角头比较。若该cell没有明显提高old/floor，先检查完整loss与逐类混淆，再决定是否使用seed`713102`和嵌套5/10类；不得直接扩5 receiver或解封`713106–713110`确认seed。性能判定仍使用K10绝对门槛`old≥95%`、旧类floor`≥88%`、5/10/20新类分别`≥92/90/86%`；只有锁定统一K10候选后才补matched K5四项≤3pp下降审计。

原始IQ缓存通过既有特征导出器新增显式`--include_raw_iq`生成，保存的是与冻结backbone和FFT完全相同的单个后信道view。目标特征根预注册为`runs/cvs_qknnv42_extreme_light_20new_features_rawiq_20260715_v1`，微型适配输出根预注册为`runs/qknn_extreme_light_micro_iq_20260715_v1`，日志根为`logs/qknn_extreme_light_micro_iq_20260715_v1`。实现文件为`code/export_spaceborne_features.py`、`paper_reproduction/scripts/train_export_cvs_micro_iq_adapter.py`和对应测试；本地`ssr-gpu`下34项相关pytest已PASS。远端同步、命令、PID、GPU和输出哈希将在launch前后补录。

### 原始IQ缓存launch记录

2026-07-15 01:24直接N607 preflight PASS；8张GPU各有1个约470MiB的RIEI训练进程，项目盘剩余7.6TB。依据每GPU至多2个训练实验的规则，本次三场景短导出绑定物理GPU0/1/2，各卡增加1个进程且不干预现有任务。三个预注册输出根启动前均不存在。同步前远端文件哈希等于上一轮已知版本，无远端独有改动；同步后远端`sha256sum`与本地一致，远端`py_compile`和`bash -n`PASS。

精确命令为`cd /home/szu2070436088/2510044040/CV-SincNet; mkdir -p logs/cvs_qknnv42_extreme_light_20new_features_rawiq_20260715_v1; nohup env OUT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_qknnv42_extreme_light_20new_features_rawiq_20260715_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_qknnv42_extreme_light_20new_features_rawiq_20260715_v1 INCLUDE_RAW_IQ=1 GPUS=0,1,2 bash paper_reproduction/scripts/export_cvs_qknnv42_extreme_light_20new_20260714.sh > logs/cvs_qknnv42_extreme_light_20new_features_rawiq_20260715_v1/driver.out 2>&1 < /dev/null &`。Python环境由launcher固定为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；checkpoint为`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。预期每个scenario输出一个含9,800行`raw_iq[*,2,256]`、`z_id160`和`FFT96`的NPZ，并保留完整scenario日志。

原始IQ导出driver PID=`817616`，三场景均完成；三个NPZ均为9,800行，`raw_iq[9800,2,256]`、`features[9800,160]`、`FFT[9800,96]`和manifest`raw_iq_included=true`核验PASS。三份scenario日志共1,191行，错误扫描为空。NPZ SHA256依次为clear=`2791aab7776ba7f367c3abde8e33a82a9a86115f798c6430eb9864838becfa01`、low-elev=`1e454646f13003afa9afa155f186588a3c0b4a685402aafd41016fa6db6cfb3f`、rain=`796a6cda8f006e570d8f4e27457f85bc418373ce825557d4c40161424937ba16`。

微型适配单cell绑定物理GPU3，精确命令预注册为`CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/train_export_cvs_micro_iq_adapter.py --config paper_reproduction/configs/cvs_qknnv42_extreme_light_20new_stage2c_k10_rawiq_20260715_n607.json --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --out_root runs/qknn_extreme_light_micro_iq_20260715_v1 --receiver 8-8 --new_count 20 --seed 713101 --k_shot 10 --epochs 20 --hidden 8 --kernel_size 5 --alpha 0.20 --learning_rate 5e-4 --weight_decay 1e-4 --temperature 18 --feature_anchor_weight 0.05 --residual_weight 0.02 --batch_size 128 --device cuda:0`。日志=`logs/qknn_extreme_light_micro_iq_20260715_v1/rx8-8_new20_seed713101_k10.log`；预期输出包含20epoch完整loss trace、FP16 adapter状态、training manifest、三场景适配特征和resolved qKNN配置。

首次PID=`819572`在训练开始前因脚本`sys.path`顺序使顶层旧`cvsrffi`遮蔽`code/cvsrffi`而退出，错误为`ModuleNotFoundError: cvsrffi.checkpoint_loading`；GPU未进入训练、输出run目录未生成，故该失败不是实验性能证据。已在本地把`CODE_ROOT`置于`REPO_ROOT`之前，并在`ssr-gpu`下完成`py_compile`、`--help`和3项微型adapter测试PASS；修复后将覆盖失败日志重新启动同一预注册cell。

后续两个重试仍在训练前/导出边界暴露环境兼容问题：retry1只因全缓存包含clean source行而错误地把空scenario算入target场景；retry2完成20epoch后因N607的PyTorch2.1与NumPy2.2.5不兼容，`torch.from_numpy/tensor.numpy`对象不能交给NumPy2的`savez`。两次错误日志均保留，分别未进入训练和未完成feature artifact，不作为性能证据。修复采用target role限定的scenario审计以及显式buffer/list兼容桥；本地新增兼容测试后，retry4 PID=`824967`在新根`runs/qknn_extreme_light_micro_iq_20260715_v2`完成，以保留v1失败轨迹。

retry4实际资源为154参数、FP16 tensor状态308B、状态文件2,806B、34,816MAC/query；ADV3B02可训练参数和梯度更新均为0，适配wall time5.76s、峰值CUDA分配258,141,696B。三个输出各1,506行，单行adapter+backbone批量导出延迟为0.051–0.089ms；20epoch loss从3.05765降至3.01836，但support prototype accuracy仅由35.64%到35.00%，说明该极小前端没有在support上形成强判别分离。完整loss trace、support/query物理ID、三场景哈希和resource audit均已落入training manifest。

为保持总适配上限20epoch，后续不再叠加另一个20epoch对角头；先用0epoch、0梯度的`el_proto_aux2p0`头评估微型IQ表示。精确命令为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/run_cvs_extreme_light_matrix.py --config runs/qknn_extreme_light_micro_iq_20260715_v2/micro_iq_rx_8-8_new_20_seed_713101_k_10/resolved_qknn_config.json --output-root runs/qknn_extreme_light_micro_iq_head_20260715_v1 --log-root logs/qknn_extreme_light_micro_iq_head_20260715_v1 --mode smoke --arms el_proto_aux2p0 --receivers 8-8 --seeds 713101 --k-grid 10 --new-class-counts 20 --device cpu`。该头只保存26类prototype，持续query仍为1-view且总训练epoch仍为20；若该单cell未明显改善old/floor，则不扩大第二seed或其它规模。

首个微型IQ候选的0epoch头结果为`old=65.00%`、旧类floor`=18.33%`、new20`=65.17%`、H`=65.06%`，显著低于原始冻结特征；最弱旧类为`14-10=18.33%`，最弱新类为`10-10=11.67%`。当前adapter仅产生`4.22e-5`输入MSE且support accuracy停在35%，属于适配幅度不足的欠拟合。该超参数组合终止，不扩第二seed、5/10类或K5。

在不改变权限、架构和资源的前提下，预注册一个更强但仍154参数的优化候选：`alpha=0.5`、学习率`5e-3`、feature anchor`0`、residual weight`0.001`，其它receiver`8-8`、seed`713101`、new20、K10、3support-view、20epoch、query1-view不变。输出新根=`runs/qknn_extreme_light_micro_iq_20260715_v3`，避免覆盖既有v1/v2证据；日志=`logs/qknn_extreme_light_micro_iq_20260715_v1/rx8-8_new20_seed713101_k10_strong.log`。仍只配0epoch`el_proto_aux2p0`头，总适配不超过20epoch。若support accuracy仍未明显上升或同cell old/floor不优于原始冻结特征，则终止154参数输入前端，而不继续盲扫学习率。

### 154参数微型IQ前端强候选结果

强候选PID=`827378`完成20epoch，最终loss=`2.84467`、support prototype accuracy=`38.33%`、输入残差MSE=`0.001306`。随后使用0epoch`el_proto_aux2p0`头在同一`8-8/new20/seed713101/K10`cell评估，结果为`old=64.44%`、旧类floor`=23.33%`、`new20=63.58%`、`H=63.99%`，最弱新类准确率为`10.00%`。该结果不优于弱候选，更远低于同切分原始冻结表示；说明154参数共享前端即使加大更新幅度，也无法修复类别条件相互冲突的LEO表征重叠。

因此`support_only_micro_iq_residual_v1`路线按预注册gate终止：不补第二开发seed、不补5/10类、不运行K5、不扩receiver、不解封确认seed。对当前正式checkpoint的只读结构审计显示`id_feature_key='feat_joint'`；直接开放`id_feature_head`实际需要180,000参数，`id_late_feature`为288,480参数，`id_norm_late_feature`为289,685参数，均超过50k上限。历史报告中的49,536/67,008/67,485来自旧架构自检，不能套用到当前checkpoint。下一合法机制改为在`feat_joint`路径的既有Linear层旁挂恒等初始化低秩LoRA，只训练LoRA参数，冻结包括CosFace/source分类器在内的全部原始ADV3B02参数；仍限制20epoch、FP16状态≤128KB、query1-view和逐样本推理。这不是历史60epoch全量`id_norm_late_feature`复现，而是面向当前checkpoint的结构化压缩adapter候选。

## 2026-07-15 01:50 support-only feat-joint LoRA预注册

新机制`support_only_feat_joint_lora_v1`只旁挂于`id_proj(160→160)`、`pa_proj(320→160)`、`id_gate(160→160)`和`joint_proj(320→160)`四个实际参与`feat_joint`的Linear层。rank8、alpha8时共12,800个可训练LoRA参数、FP16 tensor状态25,600B、每query新增12,800MAC；原始checkpoint参数全部`requires_grad=False`，不修改CosFace/source分类器，不新增query batch状态或第二物理view。相对直接开放180,000参数的`id_feature_head`，参数压缩92.89%；相对289,685参数的历史同checkpoint`id_norm_late_feature`，参数压缩95.58%，epoch由60压至20。

首个机制cell固定`receiver=8-8`、`new20`、开发seed`713101`、K10、3个预注册support view和query1-view；训练20epoch，rank8、alpha8、学习率`1e-3`、weight decay`1e-4`、temperature18、冻结特征anchor0.05。输出根预注册为`runs/qknn_extreme_light_support_lora_20260715_v1`，日志为`logs/qknn_extreme_light_support_lora_20260715_v1/rx8-8_new20_seed713101_k10.log`。适配后只配0epoch`el_proto_aux2p0`评估，保证总适配仍为20epoch。若同cell old/floor没有明显优于原始冻结表示，则不扩第二seed、5/10类、K5、其它receiver或确认seed；若明显改善，再在开发范围内锁定统一rank/alpha后逐级扩展。

本地实现文件为`paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py`，复用原始IQ缓存但query持续只做一次ADV3B02+LoRA前向。本地`ssr-gpu`下`py_compile`、6项LoRA/micro单元测试和31项极轻型相关测试均PASS。

### LoRA单cell launch记录

Git提交=`c0a3499`。2026-07-15 01:45直接N607 preflight PASS；GPU4显存占用10MiB且无本路线进程，项目盘在本轮前次检查剩余7.6TB。同步前新LoRA脚本和测试在远端不存在，micro exporter为本路线自有旧版本；同步后两份脚本及测试SHA256与本地一致。远端`py_compile`和checkpoint直接LoRA结构审计PASS；CVS-RFFI环境未安装pytest，因此没有把`No module named pytest`误判为实现失败。

精确启动命令为`cd /home/szu2070436088/2510044040/CV-SincNet && mkdir -p logs/qknn_extreme_light_support_lora_20260715_v1 && nohup env CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py --config paper_reproduction/configs/cvs_qknnv42_extreme_light_20new_stage2c_k10_rawiq_20260715_n607.json --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --out_root runs/qknn_extreme_light_support_lora_20260715_v1 --receiver 8-8 --new_count 20 --seed 713101 --k_shot 10 --epochs 20 --rank 8 --alpha 8 --learning_rate 1e-3 --weight_decay 1e-4 --temperature 18 --feature_anchor_weight 0.05 --batch_size 128 --device cuda:0 > logs/qknn_extreme_light_support_lora_20260715_v1/rx8-8_new20_seed713101_k10.log 2>&1 < /dev/null &`。预期输出为20epoch完整loss trace、仅LoRA权重的FP16状态、training manifest、三场景适配特征、资源审计和resolved qKNN配置。
