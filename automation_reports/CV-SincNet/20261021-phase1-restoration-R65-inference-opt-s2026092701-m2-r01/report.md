# R65 原权重推理优化

状态：VERIFIED / ANALYZED，两版全部七景、独立truth-last、实际所属进程退出均已核实。

固定用户指定 R65 `prune_after10` 的原273765参数，新增训练/optimizer/TX标签/F优化均0。原checkpoint完整来源沿用原7/63/30物理源契约、源E20/4940冻结，加载时严格核对原R65元数据和共享权重别名。当前是已暴露工程基准的用户指定后续优化，不称独立盲测。

|固定行|作用|FFT/IFFT|prior/CG|GPU|
|---|---|---:|---|---:|
|static_graph|保持原FP32运算顺序，图执行与device有效性检查|63|3/6|2|
|residual_graph|同图执行，再复用原始残差和删除最后无用CG状态更新|39|3/6|3|

两版均完整source1800×7和正式168000×7，原F50 seed2026092701/defaulthead，B128、源尾批8/正式尾批64。每行七预测完成后才独立CPU truth-last；复用原R65预测作IQ/features/logits/argmax比较，不重跑旧目标。无源速度或准确率筛选，负结果保留，测试结果不回流调参。

统一 InferenceEngine 同时用于真实预测与测速；每次重放复制当前输入，输出clone，数值有效性在device累计后一次主机判断。原F权重/defaulthead不更新。真实2Ls/F50首步smoke覆盖B1/8/32/64/128、不同输入替换、旧输出所有权和状态不变，通过后立即继续。

同卡source-only测native/static/residual完整G与G+F，B1/8/32/128、3轮交替，P50/P90/P95、吞吐、setup与显存分列。包括copy/clone和有效性检查，预加载GPU输入，不含host传输/loader；正式预测另报告完整产物墙钟。独立G+F五批量累计graph池内存单列，不能冒充单B1部署内存。ConvLinearMAC仍30761512/IQ，不含物理/归一化/F。

物理GPU2/3每卡最多一个本run worker、总2；shared容量锁下allowner<2且free≥18GB才占GPU，CPU线程2、独立scorer最多1。健康任务不变。新不可覆盖run/output/release；技术失败保留影响行、不自动重试、不nativefallback、不停止其他owner。

42项相关检查、CLI及原权重本机合成GPU检查通过，唯一一次独立P0/P1审查PASS。本机合成G B1约76.03→3.23ms仅为本机执行诊断，不替代N607同卡G+F或正式性能。原输入统计、原权重不变，残差复用非逐位一致。

方法commit `780615368d7675331f3d42367b62bf02ea7031b1`。命令、环境/CWD、输入输出路径、六类seed、逐行矩阵、停止规则和预期产物见[experiment.json](experiment.json)。验收分别报告R65对照、固定原R17逐景1pp容忍、lowurban70%及其他五景距rawclean≤5pp，不把容忍/压缩当作全部目标完成。


## 完整正式结果

两版均复用原R65全部273765参数，新增训练为0；原F50/defaulthead不变。每版1800×7源诊断及168000×7正式结果完整，七预测完成后独立CPU评分。没有按正式结果调参、取消或重跑。

|版本|场景|准确率%|MacroF1%|较R65 pp|较原R17 pp|修复|损伤|净修复|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|static_graph|clean|84.908929|84.635704|+0.000000|+0.322024|809|418|391|
|static_graph|practical_high|82.825595|82.414014|+0.000000|+1.506548|74869|4153|70716|
|static_graph|practical_mid|81.620238|81.219920|+0.000000|+2.042262|74772|4710|70062|
|static_graph|practical_low_suburban|80.155952|79.806279|+0.000000|+2.159524|74489|5397|69092|
|static_graph|practical_high_urban|81.743452|81.359740|+0.000000|+1.389881|75228|4592|70636|
|static_graph|practical_mid_urban|75.063095|74.980802|+0.000000|+1.377976|75511|7337|68174|
|static_graph|practical_low_urban|68.024405|68.390564|+0.000000|+0.807738|73447|10069|63378|
|residual_graph|clean|84.908929|84.635704|+0.000000|+0.322024|809|418|391|
|residual_graph|practical_high|82.825595|82.414014|+0.000000|+1.506548|74869|4153|70716|
|residual_graph|practical_mid|81.620833|81.220545|+0.000595|+2.042857|74773|4710|70063|
|residual_graph|practical_low_suburban|80.155952|79.806279|+0.000000|+2.159524|74489|5397|69092|
|residual_graph|practical_high_urban|81.743452|81.359740|+0.000000|+1.389881|75228|4592|70636|
|residual_graph|practical_mid_urban|75.063095|74.980802|+0.000000|+1.377976|75511|7337|68174|
|residual_graph|practical_low_urban|68.024405|68.390564|+0.000000|+0.807738|73447|10069|63378|

|版本|口径|原native median ms|static median ms|residual median ms|本版P95 ms|本版sample/s|
|---|---|---:|---:|---:|---:|---:|
|static_graph|G_only_B1|43.153869|7.916578|7.466102|11.904785|122.193|
|static_graph|G_only_B8|47.568548|27.366758|24.625201|34.819203|292.901|
|static_graph|G_only_B32|73.249526|49.857725|51.827274|60.868164|630.717|
|static_graph|G_only_B128|49.435147|29.599927|28.749504|34.612318|4430.944|
|static_graph|G_plus_F_B1|51.331643|8.501793|8.734149|12.373097|117.695|
|static_graph|G_plus_F_B8|55.881358|26.861323|27.933639|33.024197|298.125|
|static_graph|G_plus_F_B32|79.261395|56.653680|55.771902|65.211014|579.557|
|static_graph|G_plus_F_B128|56.481131|32.820630|31.284883|39.290818|3921.469|
|residual_graph|G_only_B1|43.298106|7.969402|7.582603|11.687994|130.350|
|residual_graph|G_only_B8|49.103664|25.811984|26.904098|35.282831|304.649|
|residual_graph|G_only_B32|73.556139|51.639327|52.672186|64.028100|601.936|
|residual_graph|G_only_B128|47.757711|25.205076|28.839870|36.500889|4288.343|
|residual_graph|G_plus_F_B1|51.696230|8.897547|8.500128|12.422083|114.426|
|residual_graph|G_plus_F_B8|59.688753|27.370296|27.946275|33.106549|287.447|
|residual_graph|G_plus_F_B32|81.722676|52.880401|53.063159|64.403883|595.679|
|residual_graph|G_plus_F_B128|60.236754|30.787574|29.604959|39.749217|4222.375|

每行计时均来自该行同卡source-only native/static/residual对照，三轮交替；包括实际InferenceEngine的输入copy、独立输出clone、有效性检查，setup另计。G+F包括完整原F默认头；输入预加载GPU，不含host传输/loader/排队。两卡存在其他owner，不以跨卡绝对值比较结构；原R17历史时间仅作背景，不混入本轮同卡加速比。

|版本|正式七景墙钟s|原R65 argmax差异条数|lowurban≥70|其他五景5pp|原R17逐景1pp|
|---|---:|---:|---|---|---|
|static_graph|348.609|0|False|False|True|
|residual_graph|338.296|1|False|False|True|

逐TX/RX/day、IQ/features/logits逐元素差异、两版原始profile、累计五批量graph池allocated/reserved及setup均保存在[results.json](results.json)，分组表见[row_metrics.csv](row_metrics.csv)。参数仍273765，FP32权重1095060字节、复用原final.pt 1388956字节，ConvLinearMAC30761512/IQ；static/reuse的FFT调用分别63/39。运行优化不把删除训练状态当减参，继承训练和原F50训练成本保持原报告。

独立部署入口为experiments.pamf_r65_inference.engine.InferenceEngine；加载原R65后eval().requires_grad_(False)，可单独恢复IQ或接原F50。graph按设备/批量缓存，每次读取当前输入；多个批量的图池占用另计。没有新模型权重副本。

完整长期性能目标继续ACTIVE。本轮是用户指定已暴露基准的执行优化，不声明新的独立盲测或逐细节TX真实性。
