# PAMF 论文实验结果总表（2026-10-28）

论文补充 run 已独立核实为 **ANALYZED，369/369 阶段完成**；200 个逻辑识别器中 119 新训练、81 复用、0 新恢复器。单视图补充 120/120 评分完整。本次仅合并已冻结的评分和正式实验表，没有新增训练、预测或目标评分。

Excel 用于筛选常用论文面板；CSV 保留全量精度与逐种子记录；historical_tables 保留历史正式面板。实验研发候选的逐版搜索不纳入。

## 1. 方法与数据口径

**PAMF：**最终完整方法：三视图 SIG-TIME＋固定时序重排＋完整 Tri-SIG-TIME SSL；E50。

**PAMF-Sup：**完整有标签骨架，无 U；一般保持自然时序。U 权限面板另列同重排监督控制。

**PAMF-PL：**PAMF-Sup＋普通学生伪标签；阈值 0.95，权重 1，自然时序。

**PAMF-CE：**纯真实标签 CE，不使用 U、KL 或跨 RX 对比；E200 无增强与 E50 结构消融分别标记。

**T1：**历史 E30 再续训 20 轮至 E50、去 As 独立 KL 的配方，保留当时 EMA 和门控；不是数据划分。

**Main split：**旧 F0；源 RX 索引 [1,3,4,6,8]，L/U/V=6300/56700/27000。

**Receiver split 1/2：**新接收机重划分，不能与历史 T1 的旧 F1/F2 合并。

**六景平均：**每个种子先对六个 practical 场景等权平均，再跨种子汇总；clean 单独列。

**SD / pp：**种子间样本标准差，ddof=1；不是置信区间。差值单位为百分点。

**DURA-L / DURA-LU：**匹配当前数据契约；恢复器训练分别使用 L 或 L＋U IQ，均不读 TX 标签。通常跨 F 种子共享一个 G。

**目标综合 / 三类综合：**目标综合为跨 RX＋双跨；三类综合还包括源 RX 的 D0 跨日，不能互换。

**共同六目标：**源接收机数量曲线统一目标 RX=[0,2,5,7,9,10]；5 源点复用 Main 分层混淆矩阵。

**空白：**未测、不适用或历史未记录；不能当作 0。单视图无增强补充仅有 clean。

**配对差：**相同模型种子逐一相减；完整 PAMF 对普通 PL 的差异包含 U 算法和固定重排。

准确率和 Macro-F1 数值列均为百分数，例如 79.90 表示 79.90%。差值和标准差单位为百分点。六类主设置每景跨日/跨接收机/双跨为 30000/126000/42000；目标综合 168000，三类综合 198000。十类为 31279/92597/30424。D0 早于训练日期，不能称未来日期预测。

## 2. 实验总目录

|实验|设置与目的|数据文件|
|---|---|---|
|常见网络架构|纯 CE、无 U、无增强；检验结构表示能力。15 种网络×8 种子，其中含单/三视图 PAMF 和严格无 SIG-TIME。|15_architecture_E200_summary.csv|
|最终主方法|PAMF 使用 Tri-SIG-TIME SSL；四种外部方法使用增强＋普通 PL0.95。11 个数据设置，三种子。|01_main_comparison.csv|
|增强与 PL 控制|五种识别器的监督/PL×有/无信道增强；PAMF 另列完整方法第五臂。|02_augmentation_PL_factorial.csv|
|接收机重划分|Main split、Receiver split 1/2；源数据量相同，F/G 合同匹配。|10_receiver_splits.csv|
|标注比例|1%、5%、10%；10% 精确复用 Main split，V 固定。|09_label_fraction.csv|
|源接收机数量|1 至 6 个源接收机；共同六目标，5 源点由 Main 的分层 CM 汇总。|08_source_receiver_curve_common6.csv|
|十类规模|10 类、16 源/16 目标；不能混用六类包数或平均 F1。|11_ten_class_scale.csv|
|七场景压力|clean＋六种 practical，各景独立，准确率和 Macro-F1 齐全。|12_seven_scenes.csv|
|DURA 泛用性|固定识别器比较 raw、匹配 DURA-L、DURA-LU；不能将恢复器用 U 的贡献全部归于 F。|13_DURA_generality.csv|
|F/G 无标签权限|完整 SSL 与同重排监督识别器×G-L/G-LU；自然序普通 PL 控制单列。|07_U_permission.csv|
|两层结构消融|纯 CE 与完整 SSL 分层；完整、单视图、时间坐标、SIG-TIME、PA、参考、矩统计、重排/去重排、参考相位。|03_two_layer_structure_ablation.csv|
|去 SIG-TIME 嵌套五组|无 SIG 基底及进一步去 PA、去参考、双删、参考相位扰动；基底三种子共享。|04_nested_noSIG_CE.csv|
|单视图补充|120 行完整评分；主比较为完整 L 骨架 Sup/PL，分支为 CE/CE＋PL。|05_single_view_supplement.csv|
|三/单视图八种子|完整 SSL、相同重排、8 个配对种子；该八种子面板 raw 为主，不扩写缺失的八种子 DURA。|06_view_comparison_8seed.csv|
|历史正式机制和成本|T1 主矩阵、三核心、21 细项、G 七项、旧 REPL 内机制/结构、源物理诊断、G 交叉、时延和训练成本；保留原方法身份。|historical_table_index.csv|

## 3. Main split 五种方法

四种外部方法为原生 Full＋信道增强＋普通 PL；PAMF 为最终完整 Tri-SIG-TIME SSL。每格均为 3 个模型种子的均值±样本 SD，DURA-LU 在同一数据设置内固定。

|方法|测试|读出|场景|准确率（%）|Macro-F1（%）|
|---|---|---|---|---|---|
|PAMF|跨接收机|raw|clean|88.2931 ± 1.0560|88.1073 ± 1.1073|
|PAMF|跨接收机|DURA-LU|clean|88.3291 ± 0.9997|88.1419 ± 1.0485|
|PAMF|双跨|raw|clean|82.7960 ± 0.9256|82.0786 ± 0.7579|
|PAMF|双跨|DURA-LU|clean|82.8587 ± 0.8229|82.1600 ± 0.6624|
|PAMF|跨接收机|raw|六景平均|71.1649 ± 0.4102|70.9726 ± 0.4145|
|PAMF|跨接收机|DURA-LU|六景平均|79.9047 ± 0.8193|79.8791 ± 0.8202|
|PAMF|双跨|raw|六景平均|68.8083 ± 0.8349|68.3027 ± 0.9183|
|PAMF|双跨|DURA-LU|六景平均|76.8291 ± 0.2904|76.3538 ± 0.3072|
|RIEI + PL|跨接收机|raw|clean|61.6315 ± 1.8861|56.5622 ± 2.5913|
|RIEI + PL|跨接收机|DURA-LU|clean|61.6772 ± 1.8471|56.6061 ± 2.5458|
|RIEI + PL|双跨|raw|clean|60.8754 ± 1.7532|54.6271 ± 2.6242|
|RIEI + PL|双跨|DURA-LU|clean|60.9246 ± 1.6835|54.6795 ± 2.5443|
|RIEI + PL|跨接收机|raw|六景平均|49.9512 ± 1.5502|44.5763 ± 2.7725|
|RIEI + PL|跨接收机|DURA-LU|六景平均|56.7258 ± 0.9063|52.3363 ± 1.2721|
|RIEI + PL|双跨|raw|六景平均|51.1085 ± 1.4550|45.6197 ± 2.6541|
|RIEI + PL|双跨|DURA-LU|六景平均|57.1098 ± 0.7170|52.2609 ± 0.9162|
|DRIFT + PL|跨接收机|raw|clean|66.1183 ± 3.1981|64.2758 ± 2.2204|
|DRIFT + PL|跨接收机|DURA-LU|clean|66.1116 ± 3.1840|64.2572 ± 2.2056|
|DRIFT + PL|双跨|raw|clean|64.9802 ± 3.0383|63.1185 ± 2.3790|
|DRIFT + PL|双跨|DURA-LU|clean|64.8968 ± 3.1205|63.0258 ± 2.4417|
|DRIFT + PL|跨接收机|raw|六景平均|56.7554 ± 2.2920|55.9695 ± 1.2762|
|DRIFT + PL|跨接收机|DURA-LU|六景平均|60.7526 ± 4.4161|58.4552 ± 2.8696|
|DRIFT + PL|双跨|raw|六景平均|56.9971 ± 2.1117|56.0678 ± 2.0126|
|DRIFT + PL|双跨|DURA-LU|六景平均|59.7844 ± 3.5349|57.1798 ± 2.1341|
|IB-RFF + PL|跨接收机|raw|clean|66.6270 ± 2.2854|65.8948 ± 2.2868|
|IB-RFF + PL|跨接收机|DURA-LU|clean|66.6471 ± 2.2355|65.9163 ± 2.2288|
|IB-RFF + PL|双跨|raw|clean|67.0214 ± 0.4977|65.7296 ± 0.7571|
|IB-RFF + PL|双跨|DURA-LU|clean|66.9929 ± 0.4943|65.7052 ± 0.7478|
|IB-RFF + PL|跨接收机|raw|六景平均|52.5705 ± 2.1678|50.5462 ± 2.0724|
|IB-RFF + PL|跨接收机|DURA-LU|六景平均|62.2090 ± 2.8589|61.7356 ± 3.1184|
|IB-RFF + PL|双跨|raw|六景平均|56.3515 ± 1.3295|54.0431 ± 1.3753|
|IB-RFF + PL|双跨|DURA-LU|六景平均|62.8447 ± 0.9827|61.3792 ± 1.4795|
|SDG + PL|跨接收机|raw|clean|74.3167 ± 1.5202|74.0556 ± 1.7109|
|SDG + PL|跨接收机|DURA-LU|clean|74.3103 ± 1.5048|74.0466 ± 1.7003|
|SDG + PL|双跨|raw|clean|72.8595 ± 0.5444|72.8676 ± 0.0750|
|SDG + PL|双跨|DURA-LU|clean|72.8690 ± 0.5406|72.8747 ± 0.0644|
|SDG + PL|跨接收机|raw|六景平均|65.3010 ± 1.1439|65.1134 ± 0.8530|
|SDG + PL|跨接收机|DURA-LU|六景平均|67.1214 ± 0.9714|67.2455 ± 1.0878|
|SDG + PL|双跨|raw|六景平均|63.5923 ± 1.6250|63.6908 ± 1.2298|
|SDG + PL|双跨|DURA-LU|六景平均|65.6581 ± 0.7571|66.0254 ± 0.9063|

## 4. PAMF 的监督、普通 PL 与完整 SSL

|方法|测试|读出|场景|准确率（%）|Macro-F1（%）|
|---|---|---|---|---|---|
|PAMF-Sup|跨接收机|raw|clean|85.9376 ± 0.8355|85.7101 ± 0.8480|
|PAMF-Sup|跨接收机|DURA-L|clean|85.9280 ± 0.8525|85.7068 ± 0.8646|
|PAMF-Sup|双跨|raw|clean|81.3286 ± 1.2704|80.6593 ± 1.4857|
|PAMF-Sup|双跨|DURA-L|clean|81.2873 ± 1.2784|80.6178 ± 1.4912|
|PAMF-Sup|跨接收机|raw|六景平均|71.2860 ± 0.8997|71.0490 ± 0.8599|
|PAMF-Sup|跨接收机|DURA-L|六景平均|77.3597 ± 0.5548|77.2094 ± 0.6281|
|PAMF-Sup|双跨|raw|六景平均|68.5269 ± 1.1324|67.9944 ± 1.1932|
|PAMF-Sup|双跨|DURA-L|六景平均|75.3356 ± 0.7096|74.7046 ± 0.9153|
|PAMF-PL|跨接收机|raw|clean|86.0384 ± 0.3582|85.8162 ± 0.3793|
|PAMF-PL|跨接收机|DURA-LU|clean|86.0754 ± 0.3370|85.8489 ± 0.3536|
|PAMF-PL|双跨|raw|clean|81.8119 ± 0.7783|81.2036 ± 0.8862|
|PAMF-PL|双跨|DURA-LU|clean|81.8921 ± 0.7928|81.2954 ± 0.9129|
|PAMF-PL|跨接收机|raw|六景平均|71.0507 ± 0.9112|70.7939 ± 0.8569|
|PAMF-PL|跨接收机|DURA-LU|六景平均|78.5161 ± 0.1221|78.5055 ± 0.0871|
|PAMF-PL|双跨|raw|六景平均|68.2852 ± 1.1248|67.7344 ± 1.1691|
|PAMF-PL|双跨|DURA-LU|六景平均|76.7503 ± 0.4657|76.4610 ± 0.5182|
|PAMF|跨接收机|raw|clean|88.2931 ± 1.0560|88.1073 ± 1.1073|
|PAMF|跨接收机|DURA-LU|clean|88.3291 ± 0.9997|88.1419 ± 1.0485|
|PAMF|双跨|raw|clean|82.7960 ± 0.9256|82.0786 ± 0.7579|
|PAMF|双跨|DURA-LU|clean|82.8587 ± 0.8229|82.1600 ± 0.6624|
|PAMF|跨接收机|raw|六景平均|71.1649 ± 0.4102|70.9726 ± 0.4145|
|PAMF|跨接收机|DURA-LU|六景平均|79.9047 ± 0.8193|79.8791 ± 0.8202|
|PAMF|双跨|raw|六景平均|68.8083 ± 0.8349|68.3027 ± 0.9183|
|PAMF|双跨|DURA-LU|六景平均|76.8291 ± 0.2904|76.3538 ± 0.3072|

完整 PAMF 与普通 PL 的变化同时包含 U 方法与固定时序重排；用户已取消 PAMF-PL (Permuted)，因此此差不能单独归因于 SSL 算法。匹配差值和同种子正差数见 14_paired_effects_summary.csv。

## 5. 纯 CE E200 网络比较

|network|seed_n|准确率（%）|total_parameters|
|---|---|---|---|
|PAMF-CE (Three-view)|8|79.7265 ± 1.5684|232993|
|PAMF-CE (Single-view)|8|79.6256 ± 1.6096|196001|
|PAMF-CE w/o SIG-TIME|8|76.9801 ± 1.6892|177025|
|Complex ResNet-18-1D|8|75.0533 ± 1.3353|7693958|
|ResNet-50-1D|8|74.7982 ± 1.6419|15966982|
|ResNet-34-1D|8|74.3247 ± 2.4851|7221766|
|DenseNet-121-1D|8|73.8738 ± 2.2741|5526086|
|ResNet-18-1D|8|73.2932 ± 1.8433|3847430|
|VGG-16-BN-1D|8|72.1685 ± 2.8374|36406022|
|BiLSTM (3 layers)|8|69.2890 ± 1.3251|14743558|
|ConvNeXt-Tiny-1D|8|65.4753 ± 1.0069|26768550|
|ResNet-18-2D|8|65.3366 ± 3.9808|11176454|
|ResNet-50-2D|8|64.5985 ± 1.8623|23517190|
|Complex ResNet-18-2D|8|64.2759 ± 3.4621|22346630|
|Transformer (6 layers)|8|53.4979 ± 22.8397|18921990|

该表双跨 clean，8 种子；单视图 E200 和三视图 E200 均自然时序、无 U、无普通/信道增强。E50 结构层具有其原 CE 增强槽，不能与 E200 数值混合。

## 6. 最终 PAMF 跨数据设置

### 接收机划分

|数据设置|测试|读出|场景|准确率（%）|Macro-F1（%）|
|---|---|---|---|---|---|
|Main split|跨接收机|raw|clean|88.2931 ± 1.0560|88.1073 ± 1.1073|
|Main split|跨接收机|DURA-LU|clean|88.3291 ± 0.9997|88.1419 ± 1.0485|
|Main split|双跨|raw|clean|82.7960 ± 0.9256|82.0786 ± 0.7579|
|Main split|双跨|DURA-LU|clean|82.8587 ± 0.8229|82.1600 ± 0.6624|
|Main split|跨接收机|raw|六景平均|71.1649 ± 0.4102|70.9726 ± 0.4145|
|Main split|跨接收机|DURA-LU|六景平均|79.9047 ± 0.8193|79.8791 ± 0.8202|
|Main split|双跨|raw|六景平均|68.8083 ± 0.8349|68.3027 ± 0.9183|
|Main split|双跨|DURA-LU|六景平均|76.8291 ± 0.2904|76.3538 ± 0.3072|
|Receiver split 1|跨接收机|raw|clean|74.0257 ± 1.4762|74.2063 ± 1.3544|
|Receiver split 1|跨接收机|DURA-LU|clean|74.0341 ± 1.5213|74.2102 ± 1.4043|
|Receiver split 1|双跨|raw|clean|71.7802 ± 0.6167|72.0858 ± 0.6312|
|Receiver split 1|双跨|DURA-LU|clean|71.6960 ± 0.6158|71.9853 ± 0.6217|
|Receiver split 1|跨接收机|raw|六景平均|60.3475 ± 0.9490|60.4069 ± 1.0845|
|Receiver split 1|跨接收机|DURA-LU|六景平均|66.5754 ± 0.7213|67.0748 ± 0.7879|
|Receiver split 1|双跨|raw|六景平均|61.4360 ± 1.4162|61.4588 ± 1.7636|
|Receiver split 1|双跨|DURA-LU|六景平均|65.9221 ± 0.5495|66.5724 ± 0.6455|
|Receiver split 2|跨接收机|raw|clean|85.5304 ± 0.5730|85.5263 ± 0.5668|
|Receiver split 2|跨接收机|DURA-LU|clean|85.4881 ± 0.5837|85.4821 ± 0.5763|
|Receiver split 2|双跨|raw|clean|80.5310 ± 0.9478|80.6139 ± 0.9661|
|Receiver split 2|双跨|DURA-LU|clean|80.5000 ± 0.9536|80.5906 ± 0.9780|
|Receiver split 2|跨接收机|raw|六景平均|68.7183 ± 1.5593|68.9161 ± 1.6899|
|Receiver split 2|跨接收机|DURA-LU|六景平均|75.2632 ± 0.4636|76.0719 ± 0.4380|
|Receiver split 2|双跨|raw|六景平均|69.1841 ± 1.6891|69.3471 ± 1.6687|
|Receiver split 2|双跨|DURA-LU|六景平均|74.5488 ± 0.7196|75.0782 ± 0.6535|

### 标注比例

|数据设置|测试|读出|场景|准确率（%）|Macro-F1（%）|
|---|---|---|---|---|---|
|Main split|跨接收机|raw|clean|88.2931 ± 1.0560|88.1073 ± 1.1073|
|Main split|跨接收机|DURA-LU|clean|88.3291 ± 0.9997|88.1419 ± 1.0485|
|Main split|双跨|raw|clean|82.7960 ± 0.9256|82.0786 ± 0.7579|
|Main split|双跨|DURA-LU|clean|82.8587 ± 0.8229|82.1600 ± 0.6624|
|Main split|跨接收机|raw|六景平均|71.1649 ± 0.4102|70.9726 ± 0.4145|
|Main split|跨接收机|DURA-LU|六景平均|79.9047 ± 0.8193|79.8791 ± 0.8202|
|Main split|双跨|raw|六景平均|68.8083 ± 0.8349|68.3027 ± 0.9183|
|Main split|双跨|DURA-LU|六景平均|76.8291 ± 0.2904|76.3538 ± 0.3072|
|1% labeled|跨接收机|raw|clean|83.1524 ± 1.4147|82.6963 ± 1.5187|
|1% labeled|跨接收机|DURA-LU|clean|83.0714 ± 1.4162|82.6099 ± 1.5212|
|1% labeled|双跨|raw|clean|79.6746 ± 0.4611|78.7324 ± 0.1838|
|1% labeled|双跨|DURA-LU|clean|79.5508 ± 0.5025|78.5860 ± 0.2201|
|1% labeled|跨接收机|raw|六景平均|69.8635 ± 0.7543|69.4996 ± 0.8509|
|1% labeled|跨接收机|DURA-LU|六景平均|74.4301 ± 0.8316|73.6562 ± 0.9224|
|1% labeled|双跨|raw|六景平均|68.4380 ± 1.0549|67.8394 ± 1.0948|
|1% labeled|双跨|DURA-LU|六景平均|71.9532 ± 0.4224|70.8591 ± 0.3668|
|5% labeled|跨接收机|raw|clean|87.3434 ± 0.3990|87.1448 ± 0.4134|
|5% labeled|跨接收机|DURA-LU|clean|87.4175 ± 0.4185|87.2238 ± 0.4321|
|5% labeled|双跨|raw|clean|82.4571 ± 1.2742|81.6706 ± 1.2995|
|5% labeled|双跨|DURA-LU|clean|82.4643 ± 1.2858|81.6950 ± 1.3164|
|5% labeled|跨接收机|raw|六景平均|70.7300 ± 0.8412|70.5044 ± 0.9134|
|5% labeled|跨接收机|DURA-LU|六景平均|77.4084 ± 0.2442|77.2785 ± 0.2725|
|5% labeled|双跨|raw|六景平均|67.6025 ± 1.3072|67.0119 ± 1.4518|
|5% labeled|双跨|DURA-LU|六景平均|75.6101 ± 0.6550|74.9359 ± 0.6850|

### 共同六目标源数量

|数据设置|测试|读出|场景|准确率（%）|Macro-F1（%）|
|---|---|---|---|---|---|
|5 source receivers|跨接收机|raw|clean|88.0744 ± 1.2780|87.8342 ± 1.3288|
|5 source receivers|跨接收机|DURA-LU|clean|88.1049 ± 1.2289|87.8644 ± 1.2770|
|5 source receivers|双跨|raw|clean|82.7574 ± 1.0135|82.1417 ± 0.8545|
|5 source receivers|双跨|DURA-LU|clean|82.7972 ± 0.9020|82.1969 ± 0.7502|
|5 source receivers|跨接收机|raw|六景平均|71.2944 ± 0.3340|71.1173 ± 0.3282|
|5 source receivers|跨接收机|DURA-LU|六景平均|78.8984 ± 0.7706|78.7859 ± 0.7692|
|5 source receivers|双跨|raw|六景平均|68.3380 ± 0.6999|67.8631 ± 0.7253|
|5 source receivers|双跨|DURA-LU|六景平均|76.0289 ± 0.4810|75.5768 ± 0.4620|
|1 source receiver|跨接收机|raw|clean|49.7256 ± 1.7856|51.2861 ± 2.0547|
|1 source receiver|跨接收机|DURA-LU|clean|49.8914 ± 0.9951|51.5918 ± 1.0849|
|1 source receiver|双跨|raw|clean|47.8259 ± 2.9368|48.4528 ± 3.1300|
|1 source receiver|双跨|DURA-LU|clean|48.1093 ± 2.8246|48.8719 ± 2.9289|
|1 source receiver|跨接收机|raw|六景平均|46.1803 ± 1.1394|47.7275 ± 0.9143|
|1 source receiver|跨接收机|DURA-LU|六景平均|53.2327 ± 1.3384|53.3869 ± 1.8236|
|1 source receiver|双跨|raw|六景平均|44.9465 ± 1.5583|45.5659 ± 1.6289|
|1 source receiver|双跨|DURA-LU|六景平均|50.6215 ± 1.1722|49.6436 ± 1.6813|
|2 source receivers|跨接收机|raw|clean|74.6306 ± 1.6731|75.1344 ± 1.3439|
|2 source receivers|跨接收机|DURA-LU|clean|74.7364 ± 1.5960|75.2296 ± 1.2717|
|2 source receivers|双跨|raw|clean|71.7315 ± 2.4580|71.6017 ± 2.2025|
|2 source receivers|双跨|DURA-LU|clean|71.8750 ± 2.4225|71.7301 ± 2.1896|
|2 source receivers|跨接收机|raw|六景平均|60.4137 ± 0.6493|61.2496 ± 0.4182|
|2 source receivers|跨接收机|DURA-LU|六景平均|64.4277 ± 0.5441|65.5952 ± 0.6454|
|2 source receivers|双跨|raw|六景平均|58.1410 ± 0.9163|58.0142 ± 0.8462|
|2 source receivers|双跨|DURA-LU|六景平均|61.5577 ± 0.0845|62.3929 ± 0.2322|
|3 source receivers|跨接收机|raw|clean|85.7744 ± 1.1855|85.7664 ± 1.2574|
|3 source receivers|跨接收机|DURA-LU|clean|85.9052 ± 1.1857|85.9006 ± 1.2572|
|3 source receivers|双跨|raw|clean|81.7148 ± 0.3857|81.2851 ± 0.4304|
|3 source receivers|双跨|DURA-LU|clean|81.6759 ± 0.3537|81.2459 ± 0.3949|
|3 source receivers|跨接收机|raw|六景平均|70.4167 ± 0.7288|70.5111 ± 0.8423|
|3 source receivers|跨接收机|DURA-LU|六景平均|74.6501 ± 0.3029|75.1888 ± 0.3207|
|3 source receivers|双跨|raw|六景平均|67.7688 ± 0.9554|67.3228 ± 1.0827|
|3 source receivers|双跨|DURA-LU|六景平均|71.1582 ± 0.5796|71.2383 ± 0.6172|
|4 source receivers|跨接收机|raw|clean|85.6420 ± 1.5747|85.4573 ± 1.6247|
|4 source receivers|跨接收机|DURA-LU|clean|85.7775 ± 1.5437|85.5994 ± 1.5927|
|4 source receivers|双跨|raw|clean|82.0398 ± 0.4140|81.4453 ± 0.4492|
|4 source receivers|双跨|DURA-LU|clean|82.0083 ± 0.4406|81.4177 ± 0.4696|
|4 source receivers|跨接收机|raw|六景平均|69.6734 ± 1.4255|69.4358 ± 1.5788|
|4 source receivers|跨接收机|DURA-LU|六景平均|74.9786 ± 0.7275|74.2433 ± 0.7474|
|4 source receivers|双跨|raw|六景平均|68.4191 ± 0.4871|67.8881 ± 0.5289|
|4 source receivers|双跨|DURA-LU|六景平均|72.7062 ± 0.3915|71.5908 ± 0.4337|
|6 source receivers|跨接收机|raw|clean|87.2318 ± 0.7741|86.8621 ± 0.8506|
|6 source receivers|跨接收机|DURA-LU|clean|87.2586 ± 0.7592|86.9151 ± 0.8275|
|6 source receivers|双跨|raw|clean|82.5407 ± 1.0963|82.0480 ± 1.1830|
|6 source receivers|双跨|DURA-LU|clean|82.4537 ± 1.0734|81.9629 ± 1.1360|
|6 source receivers|跨接收机|raw|六景平均|72.2840 ± 0.0909|72.0299 ± 0.0626|
|6 source receivers|跨接收机|DURA-LU|六景平均|75.3120 ± 0.1615|74.6016 ± 0.1390|
|6 source receivers|双跨|raw|六景平均|70.2245 ± 0.4871|69.8427 ± 0.4937|
|6 source receivers|双跨|DURA-LU|六景平均|71.7585 ± 0.2391|70.8655 ± 0.3202|

### 十类规模

|数据设置|测试|读出|场景|准确率（%）|Macro-F1（%）|
|---|---|---|---|---|---|
|10-class setting|跨接收机|raw|clean|83.5459 ± 0.6384|83.5031 ± 0.6493|
|10-class setting|跨接收机|DURA-LU|clean|83.6042 ± 0.7042|83.5692 ± 0.7106|
|10-class setting|双跨|raw|clean|82.4897 ± 0.9875|82.6477 ± 0.9752|
|10-class setting|双跨|DURA-LU|clean|82.5138 ± 1.0709|82.6752 ± 1.0599|
|10-class setting|跨接收机|raw|六景平均|62.0924 ± 0.5880|61.8414 ± 0.6270|
|10-class setting|跨接收机|DURA-LU|六景平均|72.5981 ± 0.4840|72.6150 ± 0.4804|
|10-class setting|双跨|raw|六景平均|61.5689 ± 0.5221|61.3105 ± 0.5623|
|10-class setting|双跨|DURA-LU|六景平均|70.8733 ± 1.3343|71.0095 ± 1.3622|

## 7. 两层结构消融

纯 CE 层的完整基底是自然顺序三视图；完整 SSL 层的基底是最终固定重排三视图。每层保留自身匹配的 3 个种子。矩统计和无 SIG-TIME 的旧模型在排列不影响其计算时等效复用。

|层/面板|结构|时序|测试|读出|准确率（%）|Macro-F1（%）|
|---|---|---|---|---|---|---|
|Current factorial|Full three-view|Permuted|跨接收机|raw|71.1649 ± 0.4102|70.9726 ± 0.4145|
|Current factorial|Full three-view|Permuted|跨接收机|DURA-LU|79.9047 ± 0.8193|79.8791 ± 0.8202|
|Current factorial|Full three-view|Permuted|双跨|raw|68.8083 ± 0.8349|68.3027 ± 0.9183|
|Current factorial|Full three-view|Permuted|双跨|DURA-LU|76.8291 ± 0.2904|76.3538 ± 0.3072|
|Structure SSL|Single-view|Permuted|跨接收机|raw|70.7635 ± 0.3143|70.5276 ± 0.3857|
|Structure SSL|Single-view|Permuted|跨接收机|DURA-LU|79.1487 ± 0.9335|79.0771 ± 1.0100|
|Structure SSL|Single-view|Permuted|双跨|raw|68.3519 ± 1.4032|67.8898 ± 1.5950|
|Structure SSL|Single-view|Permuted|双跨|DURA-LU|77.0208 ± 0.7119|76.5648 ± 0.8202|
|Structure SSL|w/o Temporal permutation|Natural|跨接收机|raw|70.7741 ± 0.4455|70.5200 ± 0.4844|
|Structure SSL|w/o Temporal permutation|Natural|跨接收机|DURA-LU|78.2863 ± 0.9701|78.1362 ± 1.0890|
|Structure SSL|w/o Temporal permutation|Natural|双跨|raw|68.0197 ± 0.5960|67.5149 ± 0.7133|
|Structure SSL|w/o Temporal permutation|Natural|双跨|DURA-LU|76.7319 ± 0.8771|76.2015 ± 1.0614|
|Structure SSL|w/o SIG-TIME|Natural|跨接收机|raw|67.3189 ± 0.6742|66.9291 ± 0.5473|
|Structure SSL|w/o SIG-TIME|Natural|跨接收机|DURA-LU|72.6523 ± 1.5087|72.6810 ± 1.5283|
|Structure SSL|w/o SIG-TIME|Natural|双跨|raw|66.7116 ± 0.9894|66.3648 ± 0.9347|
|Structure SSL|w/o SIG-TIME|Natural|双跨|DURA-LU|70.8098 ± 1.2712|70.8645 ± 1.3298|
|Structure SSL|Moment pooling|Natural|跨接收机|raw|70.9656 ± 0.7259|70.8769 ± 0.7190|
|Structure SSL|Moment pooling|Natural|跨接收机|DURA-LU|78.4560 ± 0.5669|78.2950 ± 0.6393|
|Structure SSL|Moment pooling|Natural|双跨|raw|68.8888 ± 0.8978|68.5682 ± 0.9579|
|Structure SSL|Moment pooling|Natural|双跨|DURA-LU|76.5848 ± 0.3759|76.0372 ± 0.4554|
|Structure SSL|w/o Time coordinate|Permuted|跨接收机|raw|71.6236 ± 0.7034|71.4293 ± 0.7988|
|Structure SSL|w/o Time coordinate|Permuted|跨接收机|DURA-LU|79.4525 ± 1.3281|79.3966 ± 1.3660|
|Structure SSL|w/o Time coordinate|Permuted|双跨|raw|69.2135 ± 0.9686|68.7825 ± 1.1165|
|Structure SSL|w/o Time coordinate|Permuted|双跨|DURA-LU|76.9844 ± 0.5312|76.4494 ± 0.6712|
|Structure SSL|w/o PA branch|Permuted|跨接收机|raw|64.3889 ± 1.1794|63.9412 ± 1.2039|
|Structure SSL|w/o PA branch|Permuted|跨接收机|DURA-LU|77.4051 ± 0.3863|77.3676 ± 0.4515|
|Structure SSL|w/o PA branch|Permuted|双跨|raw|65.3948 ± 0.5537|64.7138 ± 0.5346|
|Structure SSL|w/o PA branch|Permuted|双跨|DURA-LU|76.4548 ± 1.0080|76.1689 ± 0.9309|
|Structure SSL|w/o Reference response|Permuted|跨接收机|raw|70.9402 ± 0.5394|70.7596 ± 0.5246|
|Structure SSL|w/o Reference response|Permuted|跨接收机|DURA-LU|79.0923 ± 0.8330|79.0705 ± 0.8653|
|Structure SSL|w/o Reference response|Permuted|双跨|raw|68.8165 ± 1.3682|68.3762 ± 1.3892|
|Structure SSL|w/o Reference response|Permuted|双跨|DURA-LU|76.5476 ± 0.4378|76.0042 ± 0.5269|
|Structure SSL|Phase-randomized reference|Permuted|跨接收机|raw|71.0443 ± 0.9912|70.8866 ± 0.9993|
|Structure SSL|Phase-randomized reference|Permuted|跨接收机|DURA-LU|79.3120 ± 1.1122|79.2711 ± 1.1128|
|Structure SSL|Phase-randomized reference|Permuted|双跨|raw|69.2985 ± 0.7164|68.8644 ± 0.7231|
|Structure SSL|Phase-randomized reference|Permuted|双跨|DURA-LU|76.4722 ± 0.3417|75.8716 ± 0.4504|
|Structure CE|Full three-view|Natural|跨接收机|raw|71.3993 ± 0.4039|71.1517 ± 0.3192|
|Structure CE|Full three-view|Natural|跨接收机|DURA-L|75.6915 ± 0.5342|75.4200 ± 0.5597|
|Structure CE|Full three-view|Natural|双跨|raw|69.4440 ± 0.6574|69.0651 ± 0.6892|
|Structure CE|Full three-view|Natural|双跨|DURA-L|74.0307 ± 1.5963|73.3652 ± 1.6368|
|Structure CE|Single-view|Natural|跨接收机|raw|70.5500 ± 0.5813|70.2745 ± 0.5919|
|Structure CE|Single-view|Natural|跨接收机|DURA-L|75.4662 ± 1.1959|75.2929 ± 1.2470|
|Structure CE|Single-view|Natural|双跨|raw|68.6827 ± 0.8639|68.2118 ± 1.0227|
|Structure CE|Single-view|Natural|双跨|DURA-L|73.9571 ± 1.0761|73.4246 ± 1.0128|
|Structure CE|w/o Time coordinate|Natural|跨接收机|raw|70.8399 ± 0.5621|70.5747 ± 0.7179|
|Structure CE|w/o Time coordinate|Natural|跨接收机|DURA-L|75.2624 ± 0.3710|74.9993 ± 0.3890|
|Structure CE|w/o Time coordinate|Natural|双跨|raw|68.9075 ± 0.5084|68.4865 ± 0.4642|
|Structure CE|w/o Time coordinate|Natural|双跨|DURA-L|73.4246 ± 1.2229|72.7902 ± 1.2543|
|Structure CE|Fixed temporal permutation|Permuted|跨接收机|raw|72.1560 ± 0.8634|71.8688 ± 0.8784|
|Structure CE|Fixed temporal permutation|Permuted|跨接收机|DURA-L|77.3822 ± 0.6887|77.2484 ± 0.8453|
|Structure CE|Fixed temporal permutation|Permuted|双跨|raw|69.1238 ± 1.3557|68.6105 ± 1.4026|
|Structure CE|Fixed temporal permutation|Permuted|双跨|DURA-L|74.5274 ± 1.9123|73.8964 ± 2.0026|
|Structure CE|w/o SIG-TIME|Natural|跨接收机|raw|70.5179 ± 0.5093|70.2402 ± 0.5552|
|Structure CE|w/o SIG-TIME|Natural|跨接收机|DURA-L|75.5357 ± 0.8692|75.4303 ± 0.7828|
|Structure CE|w/o SIG-TIME|Natural|双跨|raw|70.2926 ± 0.4384|70.0036 ± 0.4799|
|Structure CE|w/o SIG-TIME|Natural|双跨|DURA-L|74.0648 ± 0.5584|73.9186 ± 0.7106|
|Structure CE|w/o PA branch|Natural|跨接收机|raw|64.1881 ± 0.7137|63.7777 ± 0.6571|
|Structure CE|w/o PA branch|Natural|跨接收机|DURA-L|72.0824 ± 0.5897|71.6001 ± 0.4444|
|Structure CE|w/o PA branch|Natural|双跨|raw|63.8147 ± 2.1981|63.0073 ± 2.4099|
|Structure CE|w/o PA branch|Natural|双跨|DURA-L|72.0922 ± 1.2521|71.6450 ± 1.1715|
|Structure CE|w/o Reference response|Natural|跨接收机|raw|70.8156 ± 0.4063|70.5747 ± 0.2659|
|Structure CE|w/o Reference response|Natural|跨接收机|DURA-L|74.9497 ± 0.7762|74.6566 ± 0.7428|
|Structure CE|w/o Reference response|Natural|双跨|raw|69.1427 ± 0.6035|68.7855 ± 0.6036|
|Structure CE|w/o Reference response|Natural|双跨|DURA-L|73.8176 ± 1.1508|73.1861 ± 1.1144|
|Structure CE|Phase-randomized reference|Natural|跨接收机|raw|71.0622 ± 0.4443|70.8403 ± 0.3037|
|Structure CE|Phase-randomized reference|Natural|跨接收机|DURA-L|75.1688 ± 0.6084|74.8900 ± 0.6204|
|Structure CE|Phase-randomized reference|Natural|双跨|raw|69.4817 ± 0.5838|69.1264 ± 0.6394|
|Structure CE|Phase-randomized reference|Natural|双跨|DURA-L|73.5163 ± 1.5001|72.8652 ± 1.4658|
|Structure CE|Moment pooling|Natural|跨接收机|raw|71.6827 ± 0.8025|71.4737 ± 0.9214|
|Structure CE|Moment pooling|Natural|跨接收机|DURA-L|75.6481 ± 0.4678|75.3874 ± 0.6154|
|Structure CE|Moment pooling|Natural|双跨|raw|70.0142 ± 0.7152|69.6814 ± 0.8425|
|Structure CE|Moment pooling|Natural|双跨|DURA-L|73.7328 ± 0.5725|73.0931 ± 0.4911|

结构贡献依赖训练目标与测试人群。对去时间坐标、去参考及相位扰动出现的持平或提升均保留；不据目标分数删除结果或改换默认网络。纯 CE 层如仅声明监督系统，优先采用同合同 DURA-L；DURA-LU 使用 U，需披露额外数据权限。

## 8. 全量数据与筛图方法

- 01–15 号 CSV 按论文问题划分；每行保留方法、划分、视图、顺序、增强、训练轮数、测试人群、读出、种子数和均值/SD。
- all_overall_per_seed.csv：全体逐种子成绩；all_overall_summary.csv：全部综合群体汇总。绘制误差线优先使用逐种子数据或对应 SD。
- all_strata_per_seed.csv.gz：逐 RX、TX、日期、RX×日期及综合结果。先解压为 CSV，再筛选 collection/panel，避免重复引用同一权重。
- confusion_matrices.jsonl.gz：每个已评分分层的完整注册类混淆矩阵；没有 IQ、目标真值向量或逐包预测。
- paired_raw_corrections.csv.gz：同包修复、损伤和净纠错计数；由原固定预测的评分产物复用。
- model_manifest.csv 和 historical_table_index.csv：模型/评分来源及历史正式数据表索引。source_row 与 score_source 用于追溯，不能把别名行当新模型。
- 16_new_run_stage_cost.csv 与 17_training_U_exposure.csv：当前新 run 各阶段 worker 时间和实际 U 曝光。历史缺字段留空；worker 小时不等于独占 GPU 小时。
- historical_alias_codebook.csv：旧 F0/F1/F2、F01–F21、T1 等代号解释。历史 F1/F2 的接收机集合与现 Receiver split 1/2 不同，保持原身份。

## 9. 完整性与边界

|collection|logical_rows|missing|controller_status|complete_scores|
|---|---|---|---|---|
|current|200|0|ANALYZED|200|
|single_view|120|0|ANALYZED|120|
|historical_repl|201|0|RUNNING|201|
|external_factorial|627|0|RUNNING|627|
|noSIG_CE|15|0|ANALYZED|15|
|shuffled_8seed|16|0|ANALYZED|16|

本包直接核对 1179 个集合行引用，125414 行逐种子综合成绩、41663 行跨种子汇总、1952282 行分层成绩；重算 1657994 份评分混淆矩阵的计数、准确率和 Macro-F1，并核对净纠错记录。集合行引用存在共享模型，不等于独立新训练次数。另保留 193 份历史正式 CSV/GZ。

旧外部/REPL 控制器的 RUNNING 是未更新的历史状态字段；本次依据每行完成文件与完整评分核实闭合，不把旧控制器文字当作仍在训练。新增补充与单视图控制器均 ANALYZED。原训练只使用源域，目标评分在预测固定后独立执行。已公开工程基准及历史选型背景保持披露，不声称本汇总形成新的独立盲测。

最终三视图结构、固定重排和完整 SSL 均按用户已指定配置保留；结果有正有负，不在本次汇总中据目标结果选模或补跑。完整8种子与主比较3种子分别列明，不能择优取代。


附加逐类别数据：18_per_class_metrics.csv.gz 共 372756 行，含 precision、recall、F1 及其计数分母；class_mapping.csv 明确 CM 类别轴。Excel 共 17 张工作表，主要数值均已从保存后的文件逐格对照；全量 CSV 保留原精度。369 个完成阶段中有 1 条训练缺少控制器 ended_at，阶段耗时留空；不影响该模型已冻结及评分完整性。
