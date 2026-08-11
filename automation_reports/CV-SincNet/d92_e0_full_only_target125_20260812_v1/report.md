# D92 E0_FULL_ONLY完整Target125确认实验

|字段|内容|
|---|---|
|run ID|`d92_e0_full_only_target125_20260812_v1`|
|日期|2026-08-12|
|状态|`ANALYZED_NO_PROMOTION`|
|协议|`p2_min_v1`；复用`VALIDATED_ONCE`数据|
|候选|`E0_FULL_ONLY`，candidate=`d92_e0d_e0_full_only`|
|目标|验证完整Target125上是否同时保持或提高性能，并显著缩减D92注册计算|
|历史对照|原D92 retry2同排125结果，`row_metrics.csv` SHA256=`bc8070cd9235ab41eda5bafd2ec66e9afad48b6466d2066508d0bab46980fa62`|

## 1.假设与方法锁

`E0_FULL_ONLY`保留D92的288维联合特征A、ground-spectrum Cauchy robust center B、task-balanced covariance C和F0逐query全注册类分类头；关闭注册态Fisher/Pareto E，并把注册态D从full/block K折LOO-soft fusion缩为单次full主拟合。K1/K2继续走原D92 exact alias。

Hard12-v3的10个fresh performance outer上，`E0_FULL_ONLY−D92_FULL`得到H`+0.2439pp`、old BA`+0.4722pp`、old floor`+1.0000pp`、seen-new`+0.1333pp`、forgetting`−0.4722pp`，paired median wall下降`97.63%`。该结果只用于提出完整125确认假设，不替代本实验结论。

本次只跑一个冻结候选，不在Target125结果返回后选择arm、阈值、receiver、seed、K或new-count。query逐样本面对全部注册类；query truth、role、真实batch类别数、class quota、fit、update、selection和global reassignment全部禁止。

## 2.完整矩阵

完整Target125为5receiver×5seed×5slice=125个outer；每个outer固定3个互斥LEO弱场景，共375个scene-arm单元。

|维度|冻结值|
|---|---|
|receiver|`20-1`、`3-19`、`7-14`、`7-7`、`8-8`，各25outer|
|seed|`713102`–`713106`，各25outer|
|slice|`K1/new20`、`K5/new20`、`K10/new5`、`K10/new10`、`K10/new20`，各25outer|
|场景|`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|context|`target125_context.json`，SHA256=`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`|
|source packages|原D92 retry2的125个sealed package，不重做数据验证|
|shard|8；`outer_index mod 8`|
|smoke|`rx_20_1__seed_713106__k_1__new_20`，arm标识保持`E0_FULL_ONLY`|

运行产生固定适配开启条件下的`DA1_REG0`与`DA1_REG1`预测；本实验不新增DA关闭臂，因此`DA0_REG0/DA0_REG1`为`N/A`。新类准确率和H只在`DA1_REG1`报告。

## 3.预注册判据

### 3.1技术闭合

- 125/125job完成，8/8shard为`PASS`，failed=0；
- 125份score与job receipt、250份prediction/COMMIT/fit-audit/resource-audit齐全；fit-audit包含750个state-scene row；
- 真实checkpoint truth-free smoke先于shard，所有query禁止访问字段为`false`；
- 不覆盖既有输出；同一prediction前确定性异常指纹在2个不同outer出现时共享停派；fresh retry=false；
- 禁止按中间性能停止。

### 3.2完整125性能确认

新结果与原D92 retry2的125个相同`receiver/seed/K/new-count`行配对，使用两者score的同口径总体值，不把不同场景或不同outer的单项极值拼接：

- K>2的100行mean`ΔH_old_new>0`，且至少80/100行`ΔH_old_new≥0`；
- 全125行mean`Δold_acc≥0`、mean`Δold_floor≥0`、mean`Δseen_new_acc≥0`；
- 全125行mean`Δforgetting≤0`；
- 25个K1行必须保持D92 alias语义，单独报告同排差异；
- 必须同时给出receiver、seed、K/new-count、场景和per-old-class floor分解；任一总体门失败即`NO_TARGET125_PROMOTION`。

### 3.3计算与状态

- K5/K10的two-state component fit精确为2，`DA1_REG1`actual component fit精确为1；K1 exact alias计数为3；
- query MAC和永久state不因删除Fisher/LOO而增加；
- 完整125报告`DA1_REG1`注册wall、CPU time和增量peak的分布；历史D92 CSV没有同口径资源receipt，因此不虚构125行paired wall比例；
- D92的理论two-state fit为K5`48`、K10`88`，本候选均为`2`，对应组件拟合次数分别减少`95.83%`与`97.73%`。Hard12-v3的paired wall/peak只作为既有独立资源证据。

## 4.发布登记

|项目|冻结值|
|---|---|
|本地Git仓库|`E:\type10-7\code\snapshots\d92_125wt`|
|本地环境|`ssr-gpu`|
|N607 Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|远端source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_only_source_snapshot_20260812_v1`|
|远端output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_only_target125_20260812_v1`|
|远端logs root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_only_target125_20260812_v1`|
|GPU|GPU0–7各一个shard，每child CPU threads=2|
|代码commit|`ba1aeb7a`（runtime method/runner）|
|runtime archive|`E:\type10-7\code\snapshots\d92_e0_full_only_runtime_closure_ba1aeb7a.tar.gz`；4987764B；SHA256=`899e409d742c2135a2a5a09bdfb5055e918dd86d5704ac014c9c606ed92ca1b0`|
|config SHA|`13709fb300239526b1d7885bb5ceb90257ff70a0ac29d7f8e6c2a04b2f11c2c1`|
|launch SHA|`7646376b4f3e2860552ac3a084d90af4f3533e5c51c6f95a3361638d61ec9ab2`|

唯一启动命令预注册为：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_only_source_snapshot_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

发布前只要求：实际Git入口、聚焦协议负测、独立`P0=0/P1=0`、不可变run路径、真实checkpoint truth-free smoke和N607资源预检。完成后在本报告追加完整同排结果表、资源分布、异常和最终裁决。

## 5.本地实现与验证

|文件|用途|
|---|---|
|`configs/stage2_d92_e0_full_only_target125_v1.json`|单臂方法锁、125矩阵和晋级门|
|`code/cvsrffi/stage2_d92_e0_full_only_target125.py`|完整Cartesian matrix builder与严格身份校验|
|`code/scripts/run_d92_e0_full_only_target125.py`|prepare、真实smoke、8shard执行和共享技术停派|
|两份对应测试|125/375覆盖、arm/path篡改负测、smoke前置、正常分派与distinct-outer停派|

`ssr-gpu`下新封装9项通过；连同E0D slim/query、D92 probe和既有E0OCF runner的相关回归共80项通过，`py_compile`、CLI help和`git diff --check`通过。config SHA256=`13709fb300239526b1d7885bb5ceb90257ff70a0ac29d7f8e6c2a04b2f11c2c1`；selection SHA256=`e2d7a22c3f6968a661e9fc28a4b4259b33c286e1eb944a4d20bb42f0c49da67c`。

独立release review结论为`APPROVE`，`P0=0，P1=0`。审查确认单臂身份、K1 smoke、125×3覆盖、源包/seal、预测后独立评分以及跨outer共享技术停止均已闭合；N607 preflight与真实smoke属于下一执行步骤。

同步映射固定为：runtime archive→`source_root/d92_e0_full_only_runtime_closure_ba1aeb7a.tar.gz`；config→`source_root/configs/stage2_d92_e0_full_only_target125_v1.json`；launch→`source_root/launch.sh`。归档来自Git commit`ba1aeb7a`的完整`code/`树，共1296个成员，已核对包含`code/cvsrffi/__init__.py`、目标builder/runner及复用的E0OCF closure入口，且不存在`code/code`层级。

## 6.N607执行与产物闭合

唯一Runner使用普通N607账号完成一次冻结启动，未重试、未覆盖、未按性能干预。manifest SHA256为`5910674066e8bbf93684fddd6af6fd2cef7e8f208d64e403ac7e58030a2a8cc5`；shard PID为`1997150–1997157`，GPU0–7一一绑定。真实checkpoint smoke状态为`D92_E0_FULL_ONLY_TARGET125_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS`。

|闭合项|结果|
|---|---:|
|job prediction/score|125/125、125/125|
|正式prediction/COMMIT|250/250|
|正式fit/resource audit|250/250；fit内含750个scene row|
|shard|8/8 `PASS`；failed=0|
|事件|prediction start/complete=125/125；score start/complete=125/125|
|运行事件跨度|326.05秒|
|stderr/确定性异常/stop marker|0/0/0|
|终态|run进程0；8张GPU释放；SSH/TCP22残留0|

完整产物取回到`E:\type10-7\local_artifacts\d92_e0_full_only_target125_20260812_v1`。本地与远端全树一致：source snapshot为1310文件、69865562B、树SHA256=`3958fda500c46e514fb23d1385c465d519d5986a013326d6cd9e9de389d8d97f`；logs为22文件、15756B、树SHA256=`120452e5add4b335ceaa7b3eef8283e76a66f3682ad6ebdf3354d166f745c5f7`；output为2030文件、137247913B、树SHA256=`434b8a2bc26bb24e046ec01634f1f206d90fe6233e018bfa184cfcfd5c2a892a`。

完整读取534个文本日志/事件文件，共633503B；所有`.err`为空，精确异常模式为0，四类事件各125条且无畸形JSON。不是抽样或tail判断。

## 7.完整Target125同排结果

### 7.1总体结果

|候选|机制|outer/场景|H_old_new|旧类平衡准确率|旧类最低准确率|已见新类准确率|平均遗忘|裁决|
|---|---|---:|---:|---:|---:|---:|---:|---|
|原始D92|E开启；full/block K折LOO-soft fusion|125/375|61.5658%|65.5600%|36.8133%|58.9340%|15.9889%|冻结基线|
|E0_FULL_ONLY|E关闭；注册态仅full一次拟合|125/375|61.8030%|65.7200%|36.6000%|59.2573%|15.8289%|`NO_TARGET125_PROMOTION`|
|同排差值|E0_FULL_ONLY−D92|125/375|**+0.2372pp**|**+0.1600pp**|**−0.2133pp**|**+0.3233pp**|**−0.1600pp**|均值多数改善，但稳健门失败|

K>2的100行mean ΔH为**+0.2965pp**，但只有**70/100**行ΔH≥0，低于冻结的80/100门。H差值中位数为+0.2642pp，最小−2.1502pp，最大+2.1165pp。25个K1行与D92完全一致，全部指标差值为0，确认exact alias。

### 7.2冻结门

|冻结门|观测|结果|
|---|---:|---|
|完整125闭合|125/125|PASS|
|K>2 mean ΔH>0|+0.2965pp|PASS|
|K>2非负H行数≥80/100|70/100|**FAIL**|
|全125 mean Δ旧类准确率≥0|+0.1600pp|PASS|
|全125 mean Δ旧类floor≥0|−0.2133pp|**FAIL**|
|全125 mean Δ已见新类≥0|+0.3233pp|PASS|
|全125 mean Δ遗忘≤0|−0.1600pp|PASS|
|fit计数|K1=3/3；K5/K10=2/1|PASS|
|query禁止访问|全部为false|PASS|

因此本轮不能把E0_FULL_ONLY替换为通用D92默认方法；这是预注册判据失败，不以总体H均值为由改门。

### 7.3K/new-count分解

|slice|行数|H非负行|ΔH|Δ旧类准确率|Δ旧类floor|Δ已见新类|Δ遗忘|
|---|---:|---:|---:|---:|---:|---:|---:|
|K1/new20|25|25/25|0.0000pp|0.0000pp|0.0000pp|0.0000pp|0.0000pp|
|K5/new20|25|15/25|+0.0671pp|−0.0889pp|**−1.0000pp**|+0.2233pp|+0.0889pp|
|K10/new5|25|14/25|+0.0051pp|−0.0222pp|0.0000pp|+0.1067pp|+0.0222pp|
|K10/new10|25|18/25|+0.4209pp|+0.2444pp|**−0.6000pp**|+0.5867pp|−0.2444pp|
|K10/new20|25|23/25|**+0.6931pp**|+0.6667pp|+0.5333pp|+0.7000pp|−0.6667pp|

收益主要集中在K10/new20；K5/new20的旧类floor和K10/new5的行级一致性是主要短板。最差行为`rx_7_7__seed_713106__k_10__new_5`：ΔH=−2.1502pp、Δ旧类准确率=−1.9444pp、Δfloor=−5.0000pp、Δ已见新类=−2.3333pp、Δ遗忘=+1.9444pp。

### 7.4receiver、场景与旧类分解

|receiver|K>2 H非负行|ΔH（全25行）|Δ旧类floor|
|---|---:|---:|---:|
|20-1|18/20|+0.4569pp|+1.2667pp|
|3-19|13/20|+0.2233pp|−0.3333pp|
|7-14|16/20|+0.4052pp|−0.8000pp|
|7-7|13/20|+0.0580pp|−0.6667pp|
|8-8|10/20|+0.0428pp|−0.5333pp|

|场景|ΔH|Δ旧类准确率|Δ已见新类|Δ遗忘|
|---|---:|---:|---:|---:|
|leo_clear_weak|+0.2089pp|+0.0667pp|+0.3300pp|−0.0667pp|
|leo_low_elev_weak|+0.2915pp|+0.3133pp|+0.2840pp|−0.3133pp|
|leo_rain_weak|+0.2217pp|+0.1000pp|+0.3560pp|−0.1000pp|

三个场景均为正向，说明问题不是某个LEO场景整体失效，而是outer/receiver/弱旧类的局部不稳。逐旧类125行均值中，`14-10`为+0.8400pp，`14-7`为+0.0533pp，`20-15`为+0.0267pp，`20-19`为−0.0667pp，`6-15`为+0.0400pp，`8-20`为+0.0667pp。尽管类均值大多不降，rowwise floor仍有42行下降、31行上升、52行不变；123/125行的最弱旧类身份没有变化，故floor失败是真实的行级弱类波动，而非最弱类换位假象。

## 8.计算量与状态

|slice|E0_FULL_ONLY two-state fit|原D92 two-state fit|拟合次数下降|注册wall中位数|CPU中位数|增量peak中位数|query MAC|state bytes|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|K1/new20|3|3|0%|9.656ms|20.238ms|968KiB|7488|18503|
|K5/new20|2|48|**95.83%**|101.409ms|308.754ms|1608KiB|7488|18498|
|K10/new5|2|88|**97.73%**|60.724ms|183.794ms|1148KiB|3168|8583|
|K10/new10|2|88|**97.73%**|76.418ms|232.612ms|1236KiB|4608|11888|
|K10/new20|2|88|**97.73%**|105.651ms|322.695ms|1996KiB|7488|18498|

全125矩阵的理论two-state component fit由原D92的7875次降到275次，下降**96.51%**。实测全行注册wall中位数为76.418ms，P90为107.626ms；增量peak中位数为1236KiB，P90约1984.8KiB。完整125历史D92没有同口径资源receipt，因此这里不伪造paired wall加速倍数；Hard12-v3已有paired wall下降97.63%的独立开发证据。

## 9.最终解释与下一步

本实验回答是：**D92的E（Fisher/Pareto）和K折full/block LOO融合可以大幅瘦身，E0_FULL_ONLY把组件拟合数减少96.51%，并让总体H、旧类均值、新类准确率和遗忘均改善；但它尚不能作为通用替代，因为旧类floor下降0.2133pp且H仅70/100行非负。**

因此保留E0_FULL_ONLY为高效率候选/对照，不恢复整套Fisher和K折LOO；下一轮如果继续，应只补一个support-only、固定强度的旧类floor guard，重点约束K5/new20、K10/new10及7-7/8-8，不扫描query结果。任何新候选仍需重新跑完整125后才能晋级。

分析产物位于`E:\type10-7\local_artifacts\d92_e0_full_only_target125_20260812_v1\analysis`：`summary.json` SHA256=`5c7395d5210db52b1a6b2969e6942769a3ae734c5e87a77f5cce61b7a322d6f2`；`paired_rows.csv` SHA256=`6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a`；`gates.json` SHA256=`febdc49f99186c7309dfc4cfe10dedaeaa13d95e6fdaf659d3bf3f65d18dc9e8`。逐receiver、seed、slice、场景、旧类和资源表均在同目录。
