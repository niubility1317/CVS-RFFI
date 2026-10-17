# E30 固定三行完整结果

20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r02

状态：ANALYZED／VERIFIED；三行固定E30、七视图prediction及独立truth-last全部完成。

固定判定：CLOSED_FIXED_VIEW_PCGRAD_FAILED_RECOGNITION_BENEFIT。只有一个模型seed，新增IQ A_g=0，完整联合目标未完成。

|模式|clean %|high sub %|mid sub %|low sub %|high urban %|mid urban %|low urban %|
|---|---:|---:|---:|---:|---:|---:|---:|
|ORIGINAL|82.722619|81.605357|79.336905|76.969643|78.224405|64.376786|53.458333|
|MEASURE|82.691071|81.598214|79.344048|76.973214|78.211310|64.407738|53.443452|
|PROJECT|82.776786|81.572619|79.335119|76.983333|78.200000|64.379762|53.429762|

|候选相对|clean pp|mid urban pp|low urban pp|全部固定条件|
|---|---:|---:|---:|---|
|ORIGINAL|+0.054167|+0.002976|-0.028571|{'clean': True, 'low_urban': False, 'mid_urban': True, 'suburban': True}|
|MEASURE|+0.085714|-0.027976|-0.013690|{'clean': True, 'low_urban': False, 'mid_urban': False, 'suburban': True}|

|模式|high suburban距clean pp|mid suburban距clean pp|low suburban距clean pp|
|---|---:|---:|---:|
|ORIGINAL|1.117262|3.385714|5.752976|
|MEASURE|1.092857|3.347024|5.717857|
|PROJECT|1.204167|3.441667|5.793452|

|模式|实际修正更新|额外VJP|As梯度移除RMS比例|ordinary梯度等效系数均值|
|---|---:|---:|---:|---:|
|ORIGINAL|0|0|0.000000|未测量|
|MEASURE|0|4440|0.000000|1.0|
|PROJECT|284|4440|0.035200|1.0160405804648045|

标量复算说明执行与记录自洽，不重建全部参数梯度；ordinary系数变化及两视图范数不含SupCon/U、clip或AdamW，不能当作识别有效证据。MEASURE记录的coefficient是理论值，实际修正为0。

|模式|训练wall s|预测wall s|CPU评分wall s|新增NN包前向|
|---|---:|---:|---:|---:|
|ORIGINAL|1214.760|43.129|3.254|3770644|
|MEASURE|1383.885|41.642|3.321|3770644|
|PROJECT|1373.078|41.665|3.354|3770644|

总新增NN（含失败发布、开发及共享CP smoke）11316542；父E20共享训练NN 3661082 单列。新增真实更新6660，含父共享11100；新增模型轮30，含父唯一训练50模型轮，各模型累计E30。额外训练VJP 8880；开发fullF额外VJP12、二维toy20，fullF合成更新9单列。不同GPU／并发下的wall不作为严格速度归因。

源数据：全部五RX、L6300/U56700隐藏TX/V27000，无训练域留出。完整2220step／10epoch／370pool／4440 L-U序列、原监督reference和unit记录，以及2220条视图梯度记录均完成复算。实际native args各857项，仅candidate_id/output_dir不同。源V与损失曲线仅作机制和健康解释。

七视图共3528000决策、378混淆矩阵／63分区和21个配对比较独立复算；只读已评分聚合及correctness，不重新读取raw truth／IQ／checkpoint。所有RX/TX/day结果均保留，不删困难层。

无严格匹配no-As B00、无新IQ A_g、无四因子交互或跨seed证据；不能把clean或urban局部收益解释为完整联合成功，也不根据这些目标结果调参、换父、扩seed或选择性重跑。

完整逐层结果见stratified_results.md，混淆矩阵见formal_metrics.json；曲线与七视图见figures/。

匹配 No-As 参考已在读取本轮正式目标成绩前独立登记并发布：20261018-phase1-F-crossRX-matchedNoAs-e20plus10-s2026092701-m1-r01，执行 commit 00cc9bad869d98bbca408858a4adec55887cf6c7；现已完成匹配 E30／七视图评分。普通 As 相对该参考 clean −2.522619pp，PROJECT −2.468452pp，均无 clean 净收益。当前三行判定保持关闭。

测量控制与普通更新的实际 ordinary/As 输入 MSE 逐步一致，但其浮点训练轨迹并非逐位一致，SupCon 自第 2 步、监督总损失自第 3 步出现差异；不同 GPU 和额外求导下不将亚 0.1 个百分点变化归因为稳定收益。

已补齐的 No-As 完整结果见 [匹配比较](../20261018-phase1-F-crossRX-matchedNoAs-e20plus10-s2026092701-m1-r01/final_analysis.md)。这里的最初三行矩阵不含 B00；独立补齐后，clean 净收益条件已被否定，不再是缺失证据。
