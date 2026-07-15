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

### feat-joint LoRA结果与同头基线

PID=`837133`完成。20epoch loss由`3.02877`降至`1.26051`，support accuracy由35.77%升至61.03%，适配耗时2.67s、峰值CUDA分配172,918,784B；12,800参数、FP16 tensor状态25,600B、实际状态文件29,060B、12,800MAC/query均通过资源gate，全部原始checkpoint参数保持冻结。三场景各导出1,506行，artifact哈希与training manifest逐一一致；完整90行训练日志包含20条epoch记录且无非有限值、Traceback、RuntimeError、CUDA OOM或Killed。

使用相同0epoch`el_proto_aux2p0`头时，原始冻结表示同cell为`old=64.17%`、floor`=18.33%`、`new20=64.75%`、`H=64.42%`、最低新类`=15.00%`；feat-joint LoRA提高到`old=73.06%`、floor`=51.67%`、`new20=83.33%`、`H=77.80%`、最低新类`=38.33%`。因此LoRA确实修复了一部分support/query几何，而不是无效适配；但离正式`95/88/86%`门槛仍差21.94/36.33/2.67pp，不能扩确认矩阵。

完整本地证据已拉回`E:\type10-7\local_artifacts\qknn_extreme_light_support_lora_20260715_v1*`和`qknn_extreme_light_support_lora_head_20260715_v1*`；微型IQ的v1失败轨迹、v2/v3完整20epoch日志及artifact也已拉回对应`qknn_extreme_light_micro_iq_20260715_*`目录。全日志审计区分了训练前import/scenario错误、训练后NumPy2导出错误与两个真正完成候选，没有把环境失败计为性能结果。

## late+feat-joint压缩LoRA预注册

为把历史`id_norm_late_feature`的有效调整位置压缩到当前上限内，新增`late_feat_joint`scope：在现有四个feature-head LoRA之外，再覆盖`id_backbone.t_proj(96→160)`、`f_proj(32→160)`、`pa_proj(64→160)`和`fuse(321→160)`，仍不改任何原始权重。rank4、alpha4总计11,012参数、FP16状态22,024B、11,012MAC/query，比首个rank8 feature-head LoRA还少13.97%；相对289,685参数全量`id_norm_late_feature`压缩96.20%。

第二机制cell仍固定`8-8/new20/seed713101/K10`，20epoch、学习率`5e-4`、anchor0.2，并加入仅在三份匹配support view之间计算的cosine一致性权重0.5；query不参与一致性且持续1-view。输出根预注册为`runs/qknn_extreme_light_support_lora_late_20260715_v2`，日志为`logs/qknn_extreme_light_support_lora_late_20260715_v2/rx8-8_new20_seed713101_k10.log`。先配同一0epoch prototype头与v1及原始表示比较；只有old/floor继续明显改善才进入第二开发seed和5/10类。新增scope本地7项focused测试、25项其余极轻型测试和`py_compile`均PASS。

Git提交=`00e66cb`；2026-07-15 02:01直接N607 preflight PASS，GPU4为空闲10MiB。远端同步后脚本SHA256=`e0578a634860478a1db0a663cd12e9e7817f0b6cad3f66ad4daa938fdd7ec9e1`、测试SHA256=`b51f0277e5be13032b060c8751d5dd9f7757b474a9e04f6718e8f9780fc4c2c0`，`py_compile`和checkpoint直接11,012参数审计PASS。精确命令为`CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py --config paper_reproduction/configs/cvs_qknnv42_extreme_light_20new_stage2c_k10_rawiq_20260715_n607.json --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --out_root runs/qknn_extreme_light_support_lora_late_20260715_v2 --receiver 8-8 --new_count 20 --seed 713101 --k_shot 10 --epochs 20 --scope late_feat_joint --rank 4 --alpha 4 --learning_rate 5e-4 --weight_decay 1e-4 --temperature 18 --feature_anchor_weight 0.2 --view_consistency_weight 0.5 --batch_size 126 --device cuda:0`。

### late+feat-joint LoRA结果

PID=`843203`完成，20epoch最终loss=`1.43997`、support accuracy=`58.46%`、feature anchor漂移=`0.40782`、三view一致性loss=`0.03095`。资源为11,012参数、22,024B FP16 tensor状态、27,778B状态文件和11,012MAC/query。0epoch prototype头得到`old=72.22%`、floor`=48.33%`、`new20=83.25%`、`H=77.32%`、最低新类`=46.67%`；相对feature-head-only v1的`73.06/51.67/83.33/77.80%`没有联合增益，因此更宽late scope按gate终止，不扩seed或类别规模。

## 10epoch LoRA+10epoch对角头组合预注册

feature-head-only LoRA已经把同头基线floor从18.33%提升到51.67%，说明adapter方向有效，但0epoch单prototype头仍是明显容量瓶颈。下一候选把20epoch总预算等分：先用同一rank8 feat-joint LoRA适配10epoch，再用现有全类对称`el_diag_aug3_e10`对角度量+余弦头训练10epoch。两阶段都只读相同合法support，query不进入任何训练或选模；总适配严格等于20epoch。20新类预计总可训练参数约19.9k、LoRA FP16加head FP32状态约54.0KB、持续query新增约19.9kMAC，仍低于50k/128KB并远低于单qKNN。

首个组合cell固定`8-8/new20/seed713101/K10`。LoRA参数为scope=`feat_joint`、rank8、alpha8、学习率`1e-3`、anchor0.05、10epoch；输出根=`runs/qknn_extreme_light_support_lora10_20260715_v4`、日志根=`logs/qknn_extreme_light_support_lora10_20260715_v4`。随后`el_diag_aug3_e10`输出根=`runs/qknn_extreme_light_support_lora10_head10_20260715_v4`。只有该组合相对20epoch LoRA+0epoch prototype在old/floor上继续明显提高，才进入第二开发seed和5/10类；确认seed继续封存。

10epoch LoRA最终loss=`1.53279`、support accuracy=`53.72%`。叠加10epoch对角头后得到`old=76.94%`、floor`=48.33%`、`new20=92.42%`、`H=83.96%`、最低新类`=63.33%`。组合路线把new20推到目标以上并把old提高3.88pp，但floor相对20epoch LoRA+prototype反而下降3.34pp，离old/floor门槛仍差18.06/39.67pp；按预注册gate暂不扩seed/5/10类。

首次组合评估暴露资源审计缺口：head metrics只记录6,938个head参数和6,912MAC/query，未把cache manifest中的12,800个LoRA参数、25,600B FP16状态和12,800MAC/query相加。已在本地严格增加Stage2-C support-adapter provenance分支，只有support-only、无query更新/标签、无角色/配额Oracle、query1-view、≤20epoch且原始checkpoint零参数/零更新的cache才能计入；38项相关测试PASS。同步后将只重跑同一现有cache的评估生成审计修正版，不重新训练或覆盖v4。

### v4资源审计修正版与当前gate

Git提交=`5e3996e`；远端runner/test SHA256与本地一致，`py_compile`和直接support-adapter资源审计PASS。使用既有v4 cache重跑到独立根`runs/qknn_extreme_light_support_lora10_head10_20260715_v4_audit`，性能逐位保持`76.94/48.33/92.42/83.96%`，证明审计修复没有改变算法结果。修正后的每scenario资源为head6,938参数+LoRA12,800参数=`19,738`参数，持久状态=`53,352B`，逐query总MAC=`19,712`；LoRA适配1.58s、head每scenario适配0.65–1.81s，训练峰值CUDA约172.9MB，head峰值约21.8MB，逐query实测上限0.030ms。相对单qKNN的20类263,538B状态和212,992MAC/query，状态减少79.76%，决策MAC减少90.75%。

v2/v4训练日志、三个NPZ、head结果、审计修正版和原始同头基线均已完整拉回`E:\type10-7\local_artifacts\qknn_extreme_light_support_lora_late_20260715_v2*`、`qknn_extreme_light_support_lora10_20260715_v4*`及`qknn_extreme_light_raw_proto_head_20260715_v1*`。全量日志扫描：late LoRA132行含20条epoch、LoRA10共82行含10条epoch、两个head各33行，错误和非有限值均为0；所有训练manifest声明的三场景NPZ SHA256逐一匹配。

当前开发结论仍为性能gate失败：LoRA压缩与分段预算已经证明极轻量、快速且能显著提高new20，但无法把old整体和最弱旧类同时拉到95%/88%。因此不扩第二seed、5/10类、K5、其它receiver或确认seed。下一合法研发方向应是全类对称的support hard-class/DRO联合LoRA-head训练，在同一20epoch内直接优化最弱类margin；仍不得引入old/new角色、类别配额、query拟合、query TTA或dense query图。

## class-symmetric hard-class DRO LoRA预注册

为直接处理旧类floor但不读取old/new角色，LoRA prototype CE新增两个全类对称项：CosFace support margin`0.1`，以及按当前support class-average CE经softmax动态加权的hard-class DRO，temperature=`5`。每个注册类无论旧/新均以同一公式竞争权重；权重只来自合法support loss，query和角色标签均不可见。继续使用rank8 feat-joint LoRA12,800参数，不增加部署参数或MAC。

首个DRO cell固定`8-8/new20/seed713101/K10`，20epoch、alpha8、学习率`1e-3`、feature anchor0.1、匹配三support-view一致性0.2、batch126、query1-view。输出根=`runs/qknn_extreme_light_support_lora_dro_20260715_v5`，日志=`logs/qknn_extreme_light_support_lora_dro_20260715_v5/rx8-8_new20_seed713101_k10.log`；随后仍用0epoch prototype头，避免超过20epoch。只有floor相对v1的51.67%和old相对73.06%同时改善才扩开发矩阵。

本地脚本/测试SHA256分别为`5b7cc08aad709626145f8efb9a5f5fc6a0ac879c91812ccd753ce524fed65b98`和`4e3008a35e0e5d15e0eb0d808c5f31f290aa45159160d1082d3abb2dd9aa816e`；`py_compile`、8项focused测试、35项runner/resource测试和`git diff --check`均PASS。

### hard-class DRO结果

PID=`852229`完成20epoch。DRO loss由`11.46986`降至`4.66735`，但普通prototype CE仅由`4.48536`降至`3.29195`，support accuracy只由35.77%升至48.59%，低于无DRO v1的61.03%；feature anchor漂移达到0.50532，说明temperature5把更新过度集中到少数support难类。0epoch prototype头结果为`old=71.94%`、floor`=41.67%`、`new20=83.83%`、`H=77.40%`、最低新类`=38.33%`，old/floor均低于v1，未通过机制gate。

该cell仍满足资源与权限约束：12,800参数、25,600B LoRA FP16状态，含26类prototype后的持久状态52,224B、逐query总MAC19,456，适配4.11s、峰值CUDA约172.6MB。完整93行训练日志含20条epoch、错误/非有限值为0，三个NPZ哈希全部匹配manifest；本地证据位于`E:\type10-7\local_artifacts\qknn_extreme_light_support_lora_dro_20260715_v5*`。

结论是support上的最难类并不稳定对应query最弱类；尖锐DRO放大了support小样本噪声并损害整体几何。当前不继续盲扫DRO temperature/margin，也不扩矩阵。到此同一合法cell已经覆盖微型IQ前端、feat-joint LoRA、late LoRA、10+10epoch组合和hard-class DRO：同一v4组合row达到`old=76.94%/floor=48.33%/new20=92.42%/H=83.96%`；floor最高的v1 row为`old=73.06%/floor=51.67%/new20=83.33%/H=77.80%`。两者与95/88%目标均存在结构性差距。确认seed、K5、5/10类和5receiver继续封存；下一步若保持现有资源与协议，应转向改善冻结ADV3B02的source-only域不变表示，再回到本极轻量adapter验证，而不是继续在K10 support上增加选模自由度。

## 2026-07-15 02:40闭式对角高斯head预注册

在申请扩大到source-only基座训练之前，补做一个仍属于当前冻结表示权限内、机制上与余弦prototype不同的闭式诊断：每类只从合法support估计256维均值和对角方差，类方差向全类pooled within-class方差收缩，再以逐样本对角高斯似然分类。该head为0epoch、0可训练参数，不读取query特征进行拟合，不使用query标签、old/new角色、类别配额或query图；query持续为1个物理view。20新类26个注册类时，部署状态保存26组均值、逆方差和log-determinant偏置，约53,352B；保守估算每query约26,906个标量MAC/操作，均低于128KB并远低于单qKNN的212,992MAC/query。

首轮只在既定开发cell`receiver=8-8`、`seed=713101`、`new20`、`K10`上比较9个预注册组合：variance shrinkage=`0.75/0.90/0.97`，logdet weight=`0/0.25/0.5`；统一使用原始冻结`z_id160+FFT96`和三场景support enrollment，aux权重2.0。输出根=`runs/qknn_extreme_light_gaussian_k10_screen_20260715_v6`，日志根=`logs/qknn_extreme_light_gaussian_k10_screen_20260715_v6`。只有old和floor相对当前v4或v1形成明确联合改善，才考虑第二开发seed与嵌套5/10类；K5、其它receiver和确认seed继续封存。

本地实现涉及`extreme_light_adapter.py`、`cvs_method_runner.py`和matrix runner，新增参数范围校验、query batch扩展不变性与资源上限测试。`ssr-gpu`下47项相关pytest和`git diff --check`均PASS。

直接N607 preflight于2026-07-15 02:24 CST PASS；GPU4–7空闲10MiB，GPU0–3各有1个约470MiB既有进程且不干预，项目盘剩余7.6TB。三个实现文件同步后远端SHA256与本地一致，远端`py_compile`PASS；首次dry-run因漏设`PYTHONPATH`在import前退出，未创建实验结果，不属于性能失败。补充`PYTHONPATH=.`后9-row dry-run PASS。实际命令为`cd /home/szu2070436088/2510044040/CV-SincNet && PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/run_cvs_extreme_light_matrix.py --config paper_reproduction/configs/cvs_qknnv42_extreme_light_20new_stage2c_k10_20260715_n607.json --output-root runs/qknn_extreme_light_gaussian_k10_screen_20260715_v6 --log-root logs/qknn_extreme_light_gaussian_k10_screen_20260715_v6 --mode smoke --arms el_gauss_aug3_fft_w2p0_s0p75_l0,el_gauss_aug3_fft_w2p0_s0p75_l0p25,el_gauss_aug3_fft_w2p0_s0p75_l0p5,el_gauss_aug3_fft_w2p0_s0p9_l0,el_gauss_aug3_fft_w2p0_s0p9_l0p25,el_gauss_aug3_fft_w2p0_s0p9_l0p5,el_gauss_aug3_fft_w2p0_s0p97_l0,el_gauss_aug3_fft_w2p0_s0p97_l0p25,el_gauss_aug3_fft_w2p0_s0p97_l0p5 --receivers 8-8 --seeds 713101 --k-grid 10 --new-class-counts 20 --device cpu`。该闭式screen不占GPU；每row预期输出`metrics.json`、逐类明细、split manifest、loss trace和resource字段。

### 对角高斯结果与冻结基座结论

9/9 rows完成、0失败、0权限违规；9个row的全部必需artifact均非空，27条scenario loss trace全部有限，9份完整日志共54行且错误扫描为0。所有row均为`receiver=8-8/new20/seed713101/K10`，状态53,352B、逐query估算26,906MAC、0epoch、0参数、query1-view。

|候选|old|最低旧类|new20|H|最低新类|support accuracy|适配耗时|
|---|---:|---:|---:|---:|---:|---:|---:|
|shrink0.75/logdet0|71.67%|50.00%|71.17%|71.41%|11.67%|83.46%|3.92ms|
|shrink0.75/logdet0.25|72.50%|46.67%|71.08%|71.77%|11.67%|83.08%|3.03ms|
|shrink0.75/logdet0.5|73.06%|43.33%|70.75%|71.87%|13.33%|82.44%|2.07ms|
|shrink0.90/logdet0|71.94%|40.00%|70.00%|70.94%|11.67%|80.38%|2.05ms|
|shrink0.90/logdet0.25|71.11%|33.33%|69.83%|70.45%|11.67%|80.26%|2.08ms|
|shrink0.90/logdet0.5|70.83%|31.67%|69.33%|70.05%|11.67%|80.26%|2.55ms|
|shrink0.97/logdet0|68.89%|25.00%|68.33%|68.58%|13.33%|78.08%|2.54ms|
|shrink0.97/logdet0.25|69.17%|25.00%|68.00%|68.55%|13.33%|78.21%|2.42ms|
|shrink0.97/logdet0.5|69.44%|25.00%|68.08%|68.74%|13.33%|78.21%|3.45ms|

该机制没有通过扩展gate。最佳floor50.00%仍低于v1 LoRA+prototype的51.67%；最佳old73.06%对应floor43.33%、new20仅70.75%，也明显弱于v4的76.94/48.33/92.42%。增加pooled收缩或logdet权重持续损害floor，说明问题不是prototype忽略类内尺度，而是冻结表示中的跨类均值重叠和support/query类条件漂移。该路线终止，不补第二seed、5/10类、K5、其它receiver或确认seed。

至此在冻结ADV3B02约束内，轻量head、同viewRF统计、多prototype、闭式高斯、微型IQ、LoRA、late LoRA、分段20epoch和全类DRO都未能接近95% old/88% floor。下一步若要继续提升性能，必须按`项目.md`先设计并验证source-only域不变基座改进，再用本套严格Stage2-C极轻量适配协议复验；不能继续在同一个K10 query cell上扩大超参数自由度。

## 2026-07-15 leave-one-view-out跨场景LoRA预注册

冻结基座范围内仍有一个此前未覆盖、直接针对support/query场景漂移而不是继续换head的机制：三场景leave-one-view-out prototype训练。对每个support view，仅用另外两个LEO support view计算该类prototype，再训练当前view靠近跨场景prototype；三个view在每个batch按相同物理support ID严格配对。该目标完全使用合法support及其场景增强，不读取query、old/new角色或类别配额；部署侧仍是原rank8 feat-joint LoRA的12,800参数、25,600B FP16状态和12,800MAC/query，不增加持续推理资源。

单机制gate继续固定`receiver=8-8/new20/seed713101/K10`，20epoch、rank8、alpha8、学习率`1e-3`、anchor0.05、leave-one-view-out prototype权重1.0、matched-view consistency0.1、batch126、query1-view。输出根预注册为`runs/qknn_extreme_light_support_lora_crossview_20260715_v7`，日志=`logs/qknn_extreme_light_support_lora_crossview_20260715_v7/rx8-8_new20_seed713101_k10.log`；训练后只配0epoch`el_proto_aux2p0`head。只有old/floor同时超过v1的73.06/51.67%，且不明显损害new20，才进入第二开发seed和5/10类；否则终止冻结基座support-adapter开发并停止在该query cell继续调参。

本地新增leave-one-view-out prototype bank、cross-view CE及参数范围审计；56项相关pytest、`py_compile`和`git diff --check`均PASS。

2026-07-15 02:34 CST直接N607 preflight PASS；GPU4–7空闲10MiB，GPU0–3各有1个约470MiB既有进程，项目盘剩余7.6TB，目标run/log根均不存在。计划绑定物理GPU4，精确命令为`cd /home/szu2070436088/2510044040/CV-SincNet && mkdir -p logs/qknn_extreme_light_support_lora_crossview_20260715_v7 && CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py --config paper_reproduction/configs/cvs_qknnv42_extreme_light_20new_stage2c_k10_rawiq_20260715_n607.json --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --out_root runs/qknn_extreme_light_support_lora_crossview_20260715_v7 --receiver 8-8 --new_count 20 --seed 713101 --k_shot 10 --epochs 20 --scope feat_joint --rank 8 --alpha 8 --learning_rate 1e-3 --weight_decay 1e-4 --temperature 18 --feature_anchor_weight 0.05 --view_consistency_weight 0.1 --cross_view_prototype_weight 1.0 --batch_size 126 --device cuda:0 > logs/qknn_extreme_light_support_lora_crossview_20260715_v7/rx8-8_new20_seed713101_k10.log 2>&1`。预期输出20epoch完整loss、leave-one-view-out CE、adapter状态、training manifest和三场景适配特征。

cross-view LoRA完成20epoch：leave-one-view-out CE由3.24263降至1.31834，support accuracy由32.44%升至59.10%，view consistency loss由0.09225降至0.02951。资源为12,800参数、25,600B FP16 tensor状态、29,060B状态文件、12,800MAC/query；适配4.998s、峰值CUDA分配175,296,512B，原始checkpoint参数与梯度更新仍为0。完整94行日志含20条epoch、错误和非有限值为0，三场景各1,506行NPZ哈希均与manifest匹配。

下一步只运行预注册的0epoch prototype head，输出根=`runs/qknn_extreme_light_support_lora_crossview_head_20260715_v7`。精确命令为`cd /home/szu2070436088/2510044040/CV-SincNet && PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/run_cvs_extreme_light_matrix.py --config runs/qknn_extreme_light_support_lora_crossview_20260715_v7/support_lora_feat_joint_rx_8-8_new_20_seed_713101_k_10/resolved_qknn_config.json --output-root runs/qknn_extreme_light_support_lora_crossview_head_20260715_v7 --log-root logs/qknn_extreme_light_support_lora_crossview_head_20260715_v7 --mode smoke --arms el_proto_aux2p0 --receivers 8-8 --seeds 713101 --k-grid 10 --new-class-counts 20 --device cpu`。

### cross-view LoRA机制gate结果

0epoch head完成，结果为`old=72.50%`、floor`=51.67%`、`new20=83.50%`、`H=77.57%`、最低新类`=46.67%`。对比普通LoRA v1的`73.06/51.67/83.33/77.80%`，cross-view目标只使new20提高0.17pp，floor持平，old下降0.56pp，未达到预注册的old/floor联合改善gate。其组合部署资源为12,800个LoRA参数、52,224B总状态、19,456MAC/query，LoRA适配4.998s、head闭式适配约1.04ms、query1-view；完整head日志6行、3条有限trace，错误扫描为0，权限审计仍为support-only、无query拟合、无角色/配额Oracle。

该路线不扩第二seed、5/10类、K5、其它receiver或确认seed。冻结ADV3B02范围内的合法开发空间已经覆盖不同输入位置、不同adapter位置、不同训练损失、不同head几何和同view统计，但old/floor始终存在两位数缺口；继续在同一K10 cell调权重会形成不受控的query选模。当前必须在改变“冻结ADV3B02原始参数”默认边界前停止，并取得用户对source-only基座改进的明确授权。

## 2026-07-15星上训练/部署资源预算与adapter60压缩结论

用户已明确允许合法support参与backbone侧适配，但要求新增模块参数小、可在地面预训练、上星后可快速适配。本节只定义资源口径和下一轮实验优先级，不宣称已达成性能目标。

### 历史完整体的资源问题

| 组件 | 已核实规模 | 星上问题 | 处置 |
|---|---:|---|---|
|`id_norm_late_feature`|289,685个更新参数；FP16权重579,370B，约565.8KiB|60epoch训练时还需梯度、优化器状态和backbone激活，不属于极轻型星上适配|不直接上星训练|
|60epoch|289,685参数连续更新60轮|快速适应延迟过高；epoch还会随support规模改变|改为最多5–10epoch并同时限制optimizer step|
|5-view TTA|每个query执行5次ADV3B02与FFT前向|持续推理延迟、能耗和激活读写近似放大5倍|上星query固定1-view；5-view只允许作地面teacher|
|FFT96|长度256的同view FFT后保留96维；FP16为192B/样本|计算和状态远小于backbone与5-view，现有消融又表明它对冻结特征有用|第一轮保留，不把它当作首要压缩对象|
|场景筛选|当前正式矩阵中每类覆盖三场景，实际筛选次数为0|无性能收益，却增加分支与发布风险|删除|
|角色/类别配额Hungarian|使用query的old/new角色与整批类别数量先验|核心问题不是算力，而是单星逐样本部署不可获得该Oracle|正式路线禁用|

### 三档资源门槛

| 口径 | 首选档 | 可接受上限 | 超限即不再称为极轻型 |
|---|---:|---:|---:|
|ADV3B02原参数更新|0|0|>0|
|backbone侧新增adapter|1,280–4,096|12,800|50,000|
|包含分类head的总可训练参数|≤11,034|≤20,000|>50,000|
|部署持久状态|≤36KiB|≤64KiB|>128KiB|
|target-support适配|3–5epoch且≤50 optimizer step|≤10epoch且≤100 step|>20epoch|
|query物理view|1|1|>1|
|新增决策MAC/query|≤12k|≤20k|>50k|
|峰值训练显存|≤128MiB|≤256MiB|>512MiB|

其中“总可训练参数≤11,034”由最多4,096个backbone侧adapter参数和已有对角余弦head的6,938个参数组成。该组合的adapter FP16权重为8,192B，head FP32状态为27,752B，合计35,944B，约35.1KiB。训练时建议使用SGD或无动量SGD，避免Adam的两组FP32一阶/二阶状态成为星上内存主项。

### 建议的具体更新参数

| 参数 | 首轮固定值 | 理由 |
|---|---:|---|
|模块|ADV3B02后段pooled/projection位置的identity-init FiLM/IA3|在backbone内改变特征分布，但不更新原权重；避免在高分辨率时序特征图上增加MAC|
|插入点|4个后段位置|若单点通道不超过256，scale+bias总参数不超过`4×2×256=2,048`|
|adapter硬上限|4,096|允许加入极少量gate，仍比历史289,685个更新参数少70.7倍|
|adapter初始化|scale=0,bias=0|使初始前向与严格ADV3B02逐点一致|
|target适配epoch|5|相对60epoch减少91.7%；如第3epoch后support-only监控无改善可提前停止|
|optimizer step|上限50|使用梯度累积将一个全类均衡support pass合并为1次更新，不再只用epoch描述资源|
|optimizer|SGD，lr=`3e-3`，weight decay=`1e-4`，gradient clip=`1.0`|首轮避免Adam状态；学习率只由开发support协议预注册|
|support view|每个epoch对每个物理support只采样1个预注册view|三场景轮换或确定性采样，不在每次step同时堆叠3–5个view|
|cross-view正则|每4个optimizer step启用1次|将平均support前向开销控制在约1.25-view等效，而不是3–5-view|
|query|1-view+1次FFT96|消除持续5-view TTA|
|head|先用0epoch prototype作资源下界；性能不足时才启用5epoch对角余弦head|不在第一次就同时打开两个可训练模块|

### 5-view与FFT96的压缩方式

5-view不应直接删掉其中的鲁棒性信息，而应改成“地面teacher、星上student”：地面teacher只使用合法地面数据或注册support的5-view前向特征/软logit，并在role/quota Oracle之前截断；不得使用query标签、query角色或Hungarian分配输出作蒸馏目标。teacher只训练上述2k–4k的FiLM/IA3 student；上星后只携带student参数，query固定1-view。这一设计将历史5-view的计算留在地面，同时保留将其多视图鲁棒性迁移给单view模型的可能性。

FFT96暂不降维。在20新类K10中，注册类总数为26，物理support为260条。`z_id160+FFT96`的单view support cache为`260×256×2=133,120B`，约130KiB FP16；转为逐维INT8后约65KiB。因此星上不应常驻三个FP16 support view，而应采用单view INT8 cache或streaming读入。FFT32只作后续独立消融，在证明不损害old/floor前不作默认。

### 预计压缩比

| 对比 | 参数压缩 | epoch压缩 | 持续view压缩 | 参数更新量压缩 |
|---|---:|---:|---:|---:|
|历史289,685参数×60epoch×5-view→FiLM4,096×5epoch×1-view|70.7倍|12倍|5倍|848.7倍|
|历史→已有rank8 LoRA12,800×10epoch×1-view|22.6倍|6倍|5倍|135.8倍|

“参数更新量压缩”按`trainable_params×epoch`计算，不等于端到端wall-clock加速，因为冻结backbone的前向/反向激活仍占主要计算。实际星上合格性必须在目标载荷上补测功耗、延迟和峰值RAM；以上数字是本项目的工程准入门槛，不是所有卫星平台的通用标准。

### 开发决策

下一轮不直接复制adapter60，也不将已失败的20epoch LoRA简单改为5epoch。先实现身份初始化的2k–4k后段FiLM/IA3，在地面用5-view teacher做预训练/蒸馏，再使用合法target support执行最多5epoch、50次更新的快速适配；单cell仍固定`8-8/new20/seed713101/K10`，query保持1-view。只有old和floor同时超过当前最强合法轻量row，才允许增加至rank2/4 LoRA或扩展开发矩阵。

## 2026-07-15 1,280参数late-FiLM实现与单cell预注册

已在Git承载面扩展`train_export_cvs_support_lora_adapter.py`，新增`late_film`路线。严格ADV3B02的`t_proj/f_proj/pa_proj.0/fuse.0`四个后段pooled/projection线性层各附加160维通道scale与bias，总参数精确为`4×2×160=1,280`，FP16 tensor状态2,560B，新增1,280个逐样本MAC/标量操作。scale/bias全零初始化，注入后的初始前向与checkpoint原输出严格一致；所有原checkpoint参数均`requires_grad=false`。

训练实现不在每个step堆叠三场景view。它先对合法三view support执行一次冻结teacher前向，缓存每个物理support的多view均值特征；随后5个epoch按`clear→low-elevation→rain→clear→low-elevation`轮换单view，对当前view做全类对称prototype CE与matched-view teacher约束。该teacher只读取support，不读取query特征/标签、old/new角色或类别配额。SGD不保存momentum，Adam的两组FP32状态被去除；实现也硬性拒绝late-FiLM的`epoch>5`、`optimizer step>50`、AdamW、stacked-view或teacher关闭配置。

20新类K10共260个物理support、batch126，因此预期每epoch3次更新、总计15次，低于50次硬上限。support前向样本等效量为`3×260+5×(260+260)=3,380`；相对5epoch每轮都用三view计算prototype和反向的`3×260+5×(780+780)=8,580`减少60.6%。持续query仍只有1次物理view前向。与0epoch 26类prototype head组合时，预计总持久状态29,184B，新增决策计算7,936MAC/query；均低于36KiB/12k首选档。

| 本地验证 | 结果 |
|---|---|
|Python编译|PASS|
|FiLM初始严格等价、参数/状态/MAC、冻结checkpoint、轮换view、teacher、step cap和CLI控制|9/9 PASS|
|Stage2-C runner资源/provenance、raw-IQ导出与上述focused回归|33/33 PASS|
|`git diff --check`|PASS|
|trainer SHA256|`74b258f27f9cef2fcea64eeb0be96da042d9c1ab1f1b3e5b5ae00b9d3b70834f`|
|test SHA256|`6891646a4ccbdc2232a54b239d9d3e89c7afa6a3efb90119464a8880ec443ba9`|

首个机制gate预注册为`receiver=8-8/new20/seed713101/K10`，run ID=`qknn_extreme_light_support_film5_20260715_v8`。训练固定5epoch、SGD、lr=`3e-3`、weight decay=`1e-4`、gradient clip=`1.0`、max optimizer steps=`50`、matched-view teacher weight=`0.25`、feature anchor=`0.05`、temperature=`18`、batch=`126`，query1-view。训练后先配已0epoch prototype head；只有同一row同时超过普通LoRA+prototype的`old=73.06%/floor=51.67%`，且new20不低于83.33%，才进入5epoch对角head或第二开发seed。该gate只用于判断表示修复是否有信号，正式目标仍是K10的`95/88/86%`及matched K5不下降3pp。

### N607上线前实体checkpoint审计

2026-07-15 08:43 CST在N607上对正式ADV3B02 checkpoint执行直接注入审计，未用mock模型。`exact_ssdg_training_architecture_v1`严格加载PASS：195个state tensor，`missing_keys=0`、`unexpected_keys=0`、`skipped_mismatch=0`。四个注入点各为320参数，总计精确1,280参数；原checkpoint可训参数=0，adapter可训参数=1,280，FP16 tensor状态=2,560B，新增1,280MAC/query。首次审计命令因远端同名顶层`cvsrffi`包的路径优先级在import前退出；将Git承载面的`code`显式放在`sys.path[0]`后实体审计PASS，这是环境路径漂移，不是模型或adapter失败。

上线前占用于08:43:49 CST复核：8张GPU各有1个既有RIEI训练进程，GPU7的既有PID=`1058292`、显存624MiB，卡总显存24,576MiB；本实验作为GPU7上的第2个短作业，不超过项目默认的每卡2个训练实验上限。项目盘剩余7.6TB，目标run/log根均不存在。同步后trainer SHA256与本地一致，远端`py_compile`PASS；远端`CVS-RFFI`环境未安装pytest，因此远端pytest未启动，本地33/33回归和上述实体checkpoint审计作为上线验证。

预注册服务器命令为`cd /home/szu2070436088/2510044040/CV-SincNet && mkdir -p logs/qknn_extreme_light_support_film5_20260715_v8 && CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py --config paper_reproduction/configs/cvs_qknnv42_extreme_light_20new_stage2c_k10_rawiq_20260715_n607.json --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --out_root runs/qknn_extreme_light_support_film5_20260715_v8 --receiver 8-8 --new_count 20 --seed 713101 --k_shot 10 --adapter_type late_film --epochs 5 --optimizer sgd --max_optimizer_steps 50 --grad_clip 1 --view_sampling_mode rotating_single --matched_view_teacher_weight 0.25 --learning_rate 3e-3 --weight_decay 1e-4 --temperature 18 --feature_anchor_weight 0.05 --batch_size 126 --device cuda:0 > logs/qknn_extreme_light_support_film5_20260715_v8/rx8-8_new20_seed713101_k10.log 2>&1`。预期仅15个optimizer step；完成后必须扫描全量训练日志、核验manifest/NPZ哈希并再运行独立0epoch prototype head，否则不作性能结论。

### 1,280参数late-FiLM机制gate结果

N607 PID=`1066435`完成5epoch，实际15个SGD optimizer step，未触发50-step cap。五个epoch按预注册view序列`0,1,2,0,1`运行，训练阶段累计0.7409s，support前向样本等效3,380，峰值CUDA分配172,684,800B，约164.7MiB。完整79行日志含5条epoch记录，错误、Traceback与非有限数都为0；最终support train accuracy=36.15%，loss=3.13951，matched-view teacher loss=0.15076。训练出的scale/bias绝对值仅约`1e-4–3.7e-3`，表明它实际保持在近identity区域。

三场景NPZ各含1,506行，本地重算SHA256与training manifest逐一一致：`clear=dca1b7d9...206b`、`low=421a7426...9a5`、`rain=2deb3d1f...e4d8`。adapter FP16 tensor状态2,560B，`.pt`文件5,828B，原checkpoint参数更新为0，SGD optimizer持久状态为0。与26类0epoch prototype head组合后，总持久状态29,184B，新增决算计算7,936MAC/query，query仍为1-view。

| 同一`8-8/new20/seed713101/K10`row | old | 最低旧类 | new20 | H | 最低新类 |
|---|---:|---:|---:|---:|---:|
|原始冻结ADV3B02+0epoch prototype|64.17%|18.33%|64.75%|64.42%|15.00%|
|1,280参数late-FiLM 5epoch+0epoch prototype|63.06%|16.67%|65.00%|63.99%|16.67%|
|12,800参数feat-joint LoRA 20epoch+0epoch prototype|73.06%|51.67%|83.33%|77.80%|38.33%|
|正式目标|95.00%|88.00%|86.00%|—|—|

late-FiLM相对原始同头基线为`-1.11/-1.66/+0.25/-0.43pp`，对old和floor无机制增益；相对20epoch LoRA则少`10.00/35.00/18.33/13.81pp`。因此不能把该结果解释为“压缩后轻微掉点”：实际是1,280个通道scale/bias在15次更新内没有产生LoRA所需的类条件几何变化。该机制gate失败，不运行5epoch可训head，不扩第二开发seed、5/10类、K5、其它receiver或确认seed。

下一合法方向不能根据该query row继续扫学习率，而应把地面与星上阶段分开：仅在source receiver/day和合法信道view上地面预训小模块，使单view student在上星前已具有多view域不变几何；星上阶段仍只用target support执行3–5epoch、最多50步校准。地面预训的选模只读source validation，不读当前target query，以避免在同一query cell上继续超参拟合。

## 2026-07-15地面source预训+星上5epoch校准预注册

新路线不增加部署参数，仍是四个ADV3B02后段投影点的1,280个FiLM scale/bias。差别是将学习分成地面与星上两个权限面：地面只读`dataset_role=source`的2,400条IQ，以`2-19`为完整留出source receiver（360条），其余6个source receiver共2,040条训练；target-old、target-new和target query行计数坚持为0。地面使用clean+`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`四view teacher，但每个epoch只训一个轮换view，避免每step四view堆叠。

`dataset_role=source`在这里只是地面数据协议边界，用于证明未消费任何target行；它不是target类别的old/new角色Oracle。星上optimizer只接收注册support的普通类别ID，不接收old/new bit；推理统一在26类上逐样本argmax，不读每类query数量、类别配额或Hungarian分配。`old_new_role_used_by_optimizer=true`或`class_quota_used_at_inference=true`任一出现都必须被Stage2-C runner拒绝。

| 阶段 | 参数/状态 | 优化器与步数 | view | 选模权限 |
|---|---:|---|---|---|
|地面预训|1,280参数；输出FP16 2,560B|AdamW，lr=`1e-3`，wd=`1e-4`，20epoch，预期320步，硬上限400|clean+3个LEO teacher；每epoch仅1个student view|仅留出source receiver`2-19`的四view最低准确率→平均准确率|
|星上support校准|加载同一1,280参数，不带地面optimizer状态|SGD无momentum，lr=`3e-3`，5epoch，预期15步，硬上限50|support三场景轮换，每epoch仅1-view|target support只参与训练；query不训练/不选模|
|星上推理|FiLM2,560B+26类prototype26,624B=29,184B|0步|query1-view+FFT96|per-sample argmax，无角色/配额Oracle|

地面loss固定为冻结6类source classifier CE+`0.25×`单view student到四view冻结teacher均值特征的cosine距离。每epoch在留出source receiver上同时评估clean和三个LEO view，按最低view准确率、平均准确率、teacher cosine依次选择best state。地面AdamW一阶/二阶状态不导出、不上星；星上只收到8个tensor的FP16文件，并严格验key、shape和有限性。

地面首run预注册为`qknn_ground_source_film20_20260715_v9`，命令为`CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/pretrain_cvs_source_late_film.py --config paper_reproduction/configs/cvs_qknnv42_extreme_light_20new_stage2c_k10_rawiq_20260715_n607.json --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --out_root runs/qknn_ground_source_film20_20260715_v9 --val_receiver 2-19 --epochs 20 --learning_rate 1e-3 --weight_decay 1e-4 --teacher_weight 0.25 --batch_size 128 --grad_clip 1 --max_optimizer_steps 400 --seed 713101 --device cuda:0`。只有留出source的最低view准确率和平均准确率均不低于原checkpoint，且clean准确率下降不超过1pp，才运行下游target-support校准。

下游run预注册为`qknn_extreme_light_sourceinit_film5_20260715_v9`，其余参数与v8一致，只新增`--init_adapter_state runs/qknn_ground_source_film20_20260715_v9/ground_source_late_film_seed_713101_valrx_2-19/ground_film_state_fp16.pt`。target query只评估这一个完整预注册管线；仍以超过LoRA v1的`old=73.06%/floor=51.67%`且`new20>=83.33%`为扩展gate。

本地`ssr-gpu`验证为四文件`py_compile` PASS、38项adapter/source-split/runner/raw-IQ回归PASS和`git diff --check` PASS。SHA256：地面pretrainer=`b7b9df0315795435d2e6fb0b3fea0e55257217413adcd0ad104afedd6ea83da7`，target trainer=`17c68d22b71396e1f1a684b27f5a4fe3719a8e65c3ae1db9c9581a6851cbd3da`，source test=`f26b3078b0637d1cf22fbe932ed6f93e870771f435c1db1227a80c78d28550ed`，target test=`f95deccf5aa1f9eb9d4a8f7c73aa6a80727dcff163a33c675b9aa1248444f7c5`。

## 2026-07-15严格`rx_light5`压缩与关键层快适应v10预注册

上一节`clean+3个场景均值teacher`的v9预注册在启动前作废，未同步、未启动、无性能结果。原因是历史已完成的source-teacher残差MLP表明“把adapter60+5-view平均特征拟合成单个向量”不足以保留目标域收益：该MLP有10,560参数、10,240MAC/sample，source holdout cosine达到0.918858，但无Oracle独立确认只有`old=71.06%/new=74.00%/H=72.01%`，相对无MLP的`70.98/74.69/72.33%`没有改善。该证据说明多View压缩必须保留逐View结构、一致性和分歧，不能继续只优化5-view均值。

### 新方法

地面端对3个正式LEO后信道缓存分别严格调用历史`rx_light5`：`rx_base`、`rx_shift_m2`、`rx_shift_p2`、`rx_cfo_m1e4`、`rx_cfo_p1e4`，共15个可审计View。每个optimizer step只选择一个正式LEO场景，但同一物理样本的5个接收侧View同时前向；loss为5-view source classifier CE、`0.25×`冻结ADV3B02的15-view平均特征teacher距离和`0.5×`当前5-view特征一致性。训练和选模只使用2,400条`dataset_role=source`，其中`2-19`完整source receiver留出；target行、target query、query标签均为0。

首选student改为严格白名单的`late_key_ft`：只更新`id_backbone.t_proj`、`id_backbone.f_proj`和`id_backbone.pa_proj.0`的weight/bias，共31,200个原checkpoint参数；`fuse.0`因单层会使总量超过50k而禁止。地面输出6个FP16 tensor；星上加载后仍只用注册support执行5epoch、最多50次无momentum SGD更新。最终保存“当前权重减严格checkpoint”的单个FP16 delta补丁，而不是同时保存地面补丁与星上补丁，因此关键层状态固定62,400B。该补丁合并回backbone后持续推理新增MAC为0；与26类prototype状态26,624B合计89,024B，约86.9KiB，低于128KiB硬上限。

|阶段|可训练参数|epoch/step|每步View|持久状态|query额外MAC|
|---|---:|---:|---:|---:|---:|
|历史adapter60|289,685|60epoch|训练口径不同；推理固定5-view|约565.8KiB仅权重|固定5次backbone|
|v10地面多View关键层|31,200|20epoch，≤400step|1个场景×严格5-view|只导出FP16关键层状态|不上星执行|
|v10星上快适应|31,200|5epoch，预计15step，≤50step|每epoch轮换1个support场景|最终delta62,400B|合并后0|
|v10星上默认推理|0|0|先执行1-view|delta+prototype=89,024B|1次backbone|

相对历史adapter60，更新参数减少`289,685/31,200=9.28`倍，星上epoch减少12倍，按`参数×epoch`计算的星上更新量减少111.4倍；地面15-view监督不计入星上训练预算，但必须报告其训练成本。

### 逐样本1→3→5自适应TTA

新增`adaptive_rxlight_tta.py`保留真实多View作为性能保险，而不是用角色或配额挑样本。每个query先执行base view；若当前样本top2 margin达到门限则在1-view停止，否则追加±2 shift形成3-view；仅当3-view margin仍低或三视图预测分歧超限时，再追加±1e-4 CFO形成完整5-view。每个样本独立决策，增加、删除或重排其它query不得改变它的view budget或预测。

门限只允许在source validation或注册support上按“相对完整5-view准确率下降≤1pp时平均backbone forward最小”选择；不得读取query标签、真实old/new/unknown角色、全批类别比例、每类quota或Hungarian分配。正式评估同时报告平均/P95 backbone forward、1/3/5-view触发率、最坏5-view上界和同一row的old/floor/new/H。自适应TTA当前已完成独立纯函数及3项测试；在v10学生完成前不使用target query搜索门限。

### v10固定超参数与晋升门

地面run预注册为`qknn_ground_source_rxlight5_keyft20_20260715_v10`：`adapter_type=late_key_ft`、20epoch、AdamW、lr=`1e-3`、wd=`1e-4`、teacher weight=`0.25`、multiview consistency=`0.5`、batch128、gradient clip1、max step400、seed713101、source validation receiver=`2-19`。地面gate要求留出source的15-view最低准确率与平均准确率均不低于严格checkpoint；若失败，不运行target-support阶段。

地面精确命令预注册为：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/pretrain_cvs_source_late_film.py --config paper_reproduction/configs/cvs_qknnv42_extreme_light_20new_stage2c_k10_rawiq_20260715_n607.json --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --out_root runs/qknn_ground_source_rxlight5_keyft20_20260715_v10 --val_receiver 2-19 --adapter_type late_key_ft --epochs 20 --learning_rate 1e-3 --weight_decay 1e-4 --teacher_weight 0.25 --multiview_consistency_weight 0.5 --batch_size 128 --grad_clip 1 --max_optimizer_steps 400 --seed 713101 --device cuda:0
```

若地面gate通过，星上run为`qknn_extreme_light_sourceinit_keyft5_20260715_v10`，固定`receiver=8-8/new20/seed713101/K10`、5epoch、SGD无momentum、lr=`3e-4`、wd=`1e-4`、gradient clip1、max step50、support三场景轮换、matched teacher weight0.25、query先做固定1-view机制审计。target-support阶段只有在同一row超过当前合法LoRA的`old=73.06%/floor=51.67%`且`new20≥83.33%`时才扩展；正式目标仍为`95/88/86%`。若1-view低于机制gate但source/support校准的自适应TTA能在平均≤3次backbone前向下通过，则单独标记为`ADAPTIVE_TTA_CANDIDATE`，不能把5-view最坏路径冒充1-view性能。

本地`ssr-gpu`验证：三个实现文件`py_compile` PASS；adaptive TTA、source split/严格15-view、FiLM/关键层状态、Stage2-C runner共46项pytest PASS；`git diff --check` PASS。根目录`E:\type10-7`不是Git仓库，`项目.md`的协议更新已镜像至本Git承载面的`docs/PROJECT_PROTOCOL.md`和`docs/source_controls/PROJECT_PROTOCOL.full.md`；N607同步前仍需完成提交、直接SSH preflight、实时GPU/进程占用与实体checkpoint31,200参数审计。

### v10 N607上线前审计

Git提交=`57528108015e49a48c35738c4b1af5d4ad0ac4ac`。2026-07-15 09:19 CST按规定执行直连只读preflight，直连身份、服务器时间、项目根和8张RTX3090可见性均PASS。实时inventory确认8张GPU各有1个RIEI训练进程，每进程约624MiB；物理GPU7现有PID=`1058292`。v10若绑定GPU7，将成为该卡第2个训练实验，不超过项目默认每卡2个实验上限，不干预既有任务。

项目盘剩余7.6TB。严格checkpoint和3个9,800行raw-IQ缓存均存在，大小分别为8,582,116B、32,602,986B、32,720,734B和32,563,774B；目标run/log根均不存在。检查后本地`ssh.exe=0`、到端口22的`ESTABLISHED=0`。

|本地文件|本地SHA256|N607目的地|
|---|---|---|
|`paper_reproduction/scripts/pretrain_cvs_source_late_film.py`|`615d51786a72226ae72f822724022f20ecc373ae2dd56f8319eadd9048451978`|同相对路径|
|`paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py`|`a6f61704ddbdba926ea2264726a8ccd7d278e224cd8ab5a058295ac7d5fad08d`|同相对路径|
|`paper_reproduction/cvs_aligned/adaptive_rxlight_tta.py`|`ced44529eb97e1c29cad20f2939d3578c9c5d23be6772dca8fa8062df67a3b78`|同相对路径；仅在后续自适应TTA评估使用|

同步只覆盖上述本轮已提交文件；不覆盖远端数据、checkpoint、既有run/log或其它并发修改。同步后必须复核远端SHA256、`py_compile`、实体checkpoint严格加载、31,200参数精确白名单和目标目录仍为空，满足后才启动地面run。

09:21 CST完成直接SCP，3个远端SHA256与表中本地值逐项一致，远端`py_compile` PASS。首次实体审计因Python的空路径把项目根同名`cvsrffi`放在`code/cvsrffi`之前而在import阶段退出，未进入模型加载；显式将`code`放入`sys.path[0]`后重跑PASS。真实checkpoint审计为`exact_ssdg_training_architecture_v1`、195个state tensor、`missing=0/unexpected=0/skipped_mismatch=0`。可训练参数精确为31,200，且仅有`t_proj/f_proj/pa_proj.0`的6个weight/bias tensor；FP16 delta62,400B、合并推理新增MAC=0、`fuse.0`与其它层冻结。目标run/log根复核仍为空。本轮SSH/SCP结束后本地SSH进程与端口22连接均为0。

### v10地面多View训练结果与source场景重复审计

N607 PID=`1087844`完成20epoch、320个AdamW optimizer step，训练wall time12.358s，峰值CUDA分配280,593,920B，约267.6MiB；既有GPU7 RIEI进程未受干预。完整日志156行、20条连续epoch记录，Traceback、Error、Exception、OOM、NaN/Inf扫描为0。状态包含白名单6个tensor、31,200个有限元素，FP16 tensor口径62,400B，`.pt`文件65,252B，SHA256=`d3d7a9598dcfa5f13261bc0ab8be97ed6946fb8c3d757a538ef04afa168b849c`；地面optimizer状态未导出。

|source receiver`2-19`留出|最低View准确率|平均View准确率|结论|
|---|---:|---:|---|
|严格ADV3B02基线|98.6111%|99.0556%|5个独立`rx_light5`接收View|
|选中关键层state|98.8889%|99.2778%|+0.2778/+0.2222pp，地面gate PASS|

独立数据审计发现3份raw-IQ缓存中的2,400条source行逐元素完全相同，而7,400条非source行在clear/low/rain间不同。因此manifest中的15个命名View实际只有5个独立source接收View，另外两组是相同source IQ的重复；不能声称获得15个独立View或3个source LEO场景的额外多样性。该事实不使5-view地面gate失效，但把地面结论严格收窄为“关键层student改善严格`rx_light5`稳健性”；三种真实不同的LEO场景只在后续target support与query阶段出现。

本地回收目录为`E:\type10-7\local_artifacts\qknn_ground_source_rxlight5_keyft20_20260715_v10`，包含完整日志、manifest、JSON/CSV loss trace和状态文件，哈希均已本地重算。manifest确认`target_rows_used=false`、`target_query_rows_used=false`、`old_new_role_used_by_optimizer=false`、`class_quota_used=false`。

地面gate通过后，唯一允许的target-support run预注册为`qknn_extreme_light_sourceinit_keyft5_20260715_v10`。固定参数：`8-8/new20/seed713101/K10`、`late_key_ft`、加载上述地面state、SGD无momentum、5epoch、lr=`3e-4`、wd=`1e-4`、temperature18、feature anchor0.2、matched-view teacher0.25、batch126、clip1、max step50、三场景`rotating_single`；预期15次更新。精确命令为：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py --config paper_reproduction/configs/cvs_qknnv42_extreme_light_20new_stage2c_k10_rawiq_20260715_n607.json --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --out_root runs/qknn_extreme_light_sourceinit_keyft5_20260715_v10 --receiver 8-8 --new_count 20 --seed 713101 --k_shot 10 --adapter_type late_key_ft --init_adapter_state runs/qknn_ground_source_rxlight5_keyft20_20260715_v10/ground_source_rxlight5_late_key_ft_seed_713101_valrx_2-19/ground_late_key_ft_state_fp16.pt --epochs 5 --optimizer sgd --max_optimizer_steps 50 --grad_clip 1 --view_sampling_mode rotating_single --matched_view_teacher_weight 0.25 --learning_rate 3e-4 --weight_decay 1e-4 --temperature 18 --feature_anchor_weight 0.2 --batch_size 126 --device cuda:0
```

训练完成后先运行固定1-view、0epoch prototype head。只有`old>73.06%`、floor`>51.67%`且`new20≥83.33%`才扩展；若固定1-view失败，只允许用source validation或注册support校准的1→3→5逐样本门控做一次预注册恢复检查，不得用query标签选择阈值。

### v10 target-support关键层快适应结果与1-view head预注册

N607 PID=`1089923`完成5epoch、15个SGD optimizer step，适配wall time0.7284s，峰值CUDA分配172,595,712B，约164.6MiB；support前向样本等效量3,380，SGD momentum/持久optimizer状态为0。5个epoch按View槽`0→1→2→0→1`轮换，每个物理support每epoch只训练1个正式场景。完整日志84行、5条连续epoch记录，错误、OOM、NaN/Inf扫描为0；最后support accuracy45.00%，该值只描述support训练，不作为query性能。

地面state以严格6-key/31,200元素方式载入，SHA256=`d3d7a959...b849c`、有限性PASS。最终state是“当前权重减严格checkpoint”的单个FP16 delta，31,200元素、tensor口径62,400B、文件65,078B、SHA256=`5bf5f0a749c3f130ddc5beeda91451ac8bfe558cafad266dfcf3a08b0967a5d1`；delta绝对值最大0.02504、L2=1.06084。与26类prototype状态合计89,024B，补丁合并推理新增MAC=0。

三场景输出各1,506行，全部feature有限且本地重算哈希与manifest一致：clear=`d318a4ef...1219`、low=`b6b59fdf...8f17`、rain=`df718687...2878`。权限审计为`support_only=true`、`query_update_forbidden=true`、`query_labels_used=false`、`old_new_role_used=false`、`class_quota_used=false`。完整artifact回收至`E:\type10-7\local_artifacts\qknn_extreme_light_sourceinit_keyft5_20260715_v10`。

下一步只运行预注册的固定1-view、0epoch prototype head，不训练新参数、不做query选模。输出根=`runs/qknn_extreme_light_sourceinit_keyft5_head_20260715_v10`，日志根=`logs/qknn_extreme_light_sourceinit_keyft5_head_20260715_v10`；精确命令为：

```bash
PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/run_cvs_extreme_light_matrix.py --config runs/qknn_extreme_light_sourceinit_keyft5_20260715_v10/support_late_key_ft_rx_8-8_new_20_seed_713101_k_10/resolved_qknn_config.json --output-root runs/qknn_extreme_light_sourceinit_keyft5_head_20260715_v10 --log-root logs/qknn_extreme_light_sourceinit_keyft5_head_20260715_v10 --mode smoke --arms el_proto_aux2p0 --receivers 8-8 --seeds 713101 --k-grid 10 --new-class-counts 20 --device cpu
```

该head必须由Stage2-C runner再次验证delta provenance、31,200参数/15步/精确白名单、1-view、无角色/配额Oracle；只有完整artifact contract通过后才能读取old/floor/new/H。

主工作树中的`cvs_method_runner.py`在本轮提交后出现其它并发未提交修改，因此不得直接同步。已从Git提交`5752810`创建detached只读快照`E:\type10-7\code\snapshots\qknn_rxlight5_v10_sync_5752810`，快照工作树clean。head运行前只从该快照SCP`paper_reproduction/cvs_aligned/cvs_method_runner.py`到N607同相对路径；SHA256=`9c661fa01bbde4726627672d89a2967c9c7d97bdfac93c912653199a60c28191`。同步后必须复核远端哈希、`py_compile`和`source_init`方法命中稀疏关键层gate，不能携带主工作树并发修改。

### v10固定1-view失败与多View压缩恢复检查

固定1-view head已完成，artifact contract验证`post_feature_adapter_mode=support_only_late_key_ft_source_init_v1`、31,200个关键层参数、15步更新、delta62,400B、补丁合并新增MAC=0、逐样本argmax、`role_oracle_used=false`、`equal_class_quota_used=false`、`query_joint/transductive=false`和`satellite_tta_view_count=1`全部PASS。

|方法|old|最低旧类|new20|最低新类|H|结论|
|---|---:|---:|---:|---:|---:|---|
|v10 source-init late-key+固定1-view|71.1111%|43.3333%（TX20-19）|71.4167%|20.0000%（TX14-11）|71.2357%|低于LoRA机制门，禁止扩展矩阵|
|此前合法LoRA v1|73.06%|51.67%|83.33%|未作为本row报告|77.80%|当前机制比较线|

固定1-view不仅未达到正式`95/88/86%`目标，也没有超过LoRA机制门；因此不运行其它receiver、seed、5/10类或K5。旧类最低5类依次为TX20-19=43.33%、TX14-7=55.00%、TX6-15=63.33%、TX14-10=70.00%、TX20-15=95.00%；新类最低5类依次为TX14-11=20.00%、TX10-10=38.33%、TX4-10=40.00%、TX1-18=45.00%、TX2-16=61.67%。这组同状态结果支持“多View是历史高性能的重要组成”这一判断，但尚不能证明多View单独足以恢复目标性能。

按预注册边界，只允许同一checkpoint、同一FP16最终delta、同一`8-8/new20/seed713101/K10`切分做一次多View恢复检查，不再改训练学习率或读取query选择超参数。新增`benchmark_cvs_adaptive_rxlight_tta.py`，对每个严格`rx_light5`视角分别构建support原型和query分数，再对1/3/5个视角分数求均值；这避免把5个特征先平均成一个向量而丢失视角分歧。固定1/3/5用于真实上界对照；自适应门限只用三场景注册support的leave-one-out分数，在预注册网格上选择相对support完整5-view下降不超过1pp且平均前向最小的组合。query标签只在门限冻结后用于报告，query特征不进入校准；角色、类别quota、query顺序和跨query状态均不进入决策。

部署状态按一个当前场景计算：5套FP16原型=`5×26×256×2=66,560B`，FP16关键层delta=62,400B，3个FP32门限=12B，总计128,972B，约125.95KiB，低于128KiB上限且余量2,100B。三种LEO场景是三个独立部署评估，不要求同时常驻三套原型。固定5-view最坏计算为5次backbone；自适应必须报告平均/P95前向和1/3/5触发率，不能只报最好准确率。新增脚本会用既有适配feature cache做base-view结构复现，预注册门槛为平均cosine≥0.9999且最小cosine≥0.999；这只验证实现一致性，不用于调门限。

本地`ssr-gpu`下新脚本及模块`py_compile`PASS，adaptive门控、view-wise prototype、support LOO、source预训练、关键层delta和Stage2-C runner共49项pytest PASS，`git diff --check`PASS。N607运行预注册为`qknn_extreme_light_sourceinit_keyft5_adaptive_tta_20260715_v10`，精确命令为：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/benchmark_cvs_adaptive_rxlight_tta.py --config paper_reproduction/configs/cvs_qknnv42_extreme_light_20new_stage2c_k10_rawiq_20260715_n607.json --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --adapter_state runs/qknn_extreme_light_sourceinit_keyft5_20260715_v10/support_late_key_ft_rx_8-8_new_20_seed_713101_k_10/adapter_state_fp16.pt --reference_config runs/qknn_extreme_light_sourceinit_keyft5_20260715_v10/support_late_key_ft_rx_8-8_new_20_seed_713101_k_10/resolved_qknn_config.json --out_dir runs/qknn_extreme_light_sourceinit_keyft5_adaptive_tta_20260715_v10 --batch_size 256 --max_accuracy_drop_pp 1.0 --device cuda:0
```

仅当固定5-view显著恢复性能，且support校准的自适应路线在不看query门限的前提下获得有意义的准确率/平均前向Pareto改善，才继续设计压缩View；否则结论为当前适配/特征本身不足，不能通过增加View掩盖机制失败。

09:41 CST直连N607 preflight再次PASS；8张GPU各有1个约624MiB既有RIEI进程，GPU7 PID=`1058292`，本次为短时只读推理且不超过每卡2实验上限。目标输出根不存在，checkpoint和最终adapter state存在。提交=`de51b32`；新脚本本地/远端SHA256均为`d74ad1d55c6e0dabf33b26c1f543bbddfbb58acb9fb4b3e72291c0e02c66835a`，远端`py_compile`PASS。首次只读inventory命令中的`$(date ...)`被本地PowerShell提前解释而只丢失时间字段，远端GPU/路径检查仍返回成功；这不是实验错误。每次SSH/SCP后本地`ssh.exe=0`且端口22连接为0，可以执行上述预注册命令。
