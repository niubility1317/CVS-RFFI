# 双骨干完整测试结果

16个独立scratch E200模型，4组×4seed；各视图168000个固定query，112组预测先于truth评分。
旧batch_neighbor伪标签＋cosine学习率、相同44400更新预算。已有暴露代理基准；Phase2/K/新增类/H不适用。
独立重算混淆矩阵、RX/TX/day分区、四seed均值与SD、paired差值通过。结果不得回流调参或选择性重跑。

|架构|视图|准确率均值±SD（%）|Macro-F1（%）|最差RX均值（%）|
|---|---|---:|---:|---:|
|legacy_cosine|clean|80.878 ± 0.763|80.417|69.403|
|legacy_cosine|practical_high|78.971 ± 0.550|78.657|69.271|
|legacy_cosine|practical_mid|76.937 ± 0.476|76.592|67.814|
|legacy_cosine|practical_low_suburban|74.749 ± 0.478|74.340|66.183|
|legacy_cosine|practical_high_urban|76.003 ± 0.382|75.649|66.965|
|legacy_cosine|practical_mid_urban|63.499 ± 0.174|62.842|56.890|
|legacy_cosine|practical_low_urban|53.229 ± 0.328|52.315|48.231|
|raw_dual|clean|70.836 ± 1.166|69.887|58.442|
|raw_dual|practical_high|72.394 ± 1.702|72.210|56.902|
|raw_dual|practical_mid|71.803 ± 1.622|71.642|56.960|
|raw_dual|practical_low_suburban|70.976 ± 1.539|70.834|56.707|
|raw_dual|practical_high_urban|71.499 ± 1.633|71.372|56.777|
|raw_dual|practical_mid_urban|66.069 ± 1.210|65.997|54.196|
|raw_dual|practical_low_urban|59.747 ± 1.009|59.487|50.692|
|curvature_dual|clean|60.007 ± 0.932|58.714|38.454|
|curvature_dual|practical_high|64.061 ± 2.064|62.831|42.417|
|curvature_dual|practical_mid|62.934 ± 1.847|61.795|42.318|
|curvature_dual|practical_low_suburban|61.511 ± 1.703|60.418|41.450|
|curvature_dual|practical_high_urban|62.698 ± 1.881|61.547|42.433|
|curvature_dual|practical_mid_urban|56.158 ± 1.243|55.067|41.602|
|curvature_dual|practical_low_urban|49.540 ± 0.764|48.336|39.408|
|curvature_interaction|clean|57.170 ± 0.832|56.002|34.852|
|curvature_interaction|practical_high|62.713 ± 0.498|61.577|36.858|
|curvature_interaction|practical_mid|61.600 ± 0.430|60.558|37.046|
|curvature_interaction|practical_low_suburban|60.338 ± 0.329|59.368|36.889|
|curvature_interaction|practical_high_urban|61.356 ± 0.397|60.307|37.092|
|curvature_interaction|practical_mid_urban|55.045 ± 0.363|54.154|37.878|
|curvature_interaction|practical_low_urban|48.612 ± 0.394|47.634|36.643|

## 配对差值

|处理−对照|视图|均值（百分点）|SD（百分点）|正增益seed|
|---|---|---:|---:|---:|
|raw_dual − legacy_cosine|clean|-10.042|0.726|0/4|
|raw_dual − legacy_cosine|practical_high|-6.577|2.038|0/4|
|raw_dual − legacy_cosine|practical_mid|-5.134|1.845|0/4|
|raw_dual − legacy_cosine|practical_low_suburban|-3.772|1.710|0/4|
|raw_dual − legacy_cosine|practical_high_urban|-4.503|1.814|0/4|
|raw_dual − legacy_cosine|practical_mid_urban|2.571|1.038|4/4|
|raw_dual − legacy_cosine|practical_low_urban|6.519|0.781|4/4|
|raw_dual − legacy_cosine|six_view_mean|-1.816|1.509|1/4|
|curvature_dual − legacy_cosine|clean|-20.871|1.293|0/4|
|curvature_dual − legacy_cosine|practical_high|-14.910|1.965|0/4|
|curvature_dual − legacy_cosine|practical_mid|-14.004|1.813|0/4|
|curvature_dual − legacy_cosine|practical_low_suburban|-13.238|1.747|0/4|
|curvature_dual − legacy_cosine|practical_high_urban|-13.304|1.869|0/4|
|curvature_dual − legacy_cosine|practical_mid_urban|-7.340|1.350|0/4|
|curvature_dual − legacy_cosine|practical_low_urban|-3.688|0.853|0/4|
|curvature_dual − legacy_cosine|six_view_mean|-11.081|1.580|0/4|
|curvature_interaction − legacy_cosine|clean|-23.708|0.852|0/4|
|curvature_interaction − legacy_cosine|practical_high|-16.258|0.939|0/4|
|curvature_interaction − legacy_cosine|practical_mid|-15.337|0.799|0/4|
|curvature_interaction − legacy_cosine|practical_low_suburban|-14.411|0.710|0/4|
|curvature_interaction − legacy_cosine|practical_high_urban|-14.646|0.744|0/4|
|curvature_interaction − legacy_cosine|practical_mid_urban|-8.454|0.338|0/4|
|curvature_interaction − legacy_cosine|practical_low_urban|-4.617|0.279|0/4|
|curvature_interaction − legacy_cosine|six_view_mean|-12.287|0.547|0/4|
|curvature_dual − raw_dual|clean|-10.829|1.816|0/4|
|curvature_dual − raw_dual|practical_high|-8.333|3.426|0/4|
|curvature_dual − raw_dual|practical_mid|-8.869|3.129|0/4|
|curvature_dual − raw_dual|practical_low_suburban|-9.465|2.919|0/4|
|curvature_dual − raw_dual|practical_high_urban|-8.801|3.156|0/4|
|curvature_dual − raw_dual|practical_mid_urban|-9.911|2.133|0/4|
|curvature_dual − raw_dual|practical_low_urban|-10.207|1.426|0/4|
|curvature_dual − raw_dual|six_view_mean|-9.265|2.695|0/4|
|curvature_interaction − curvature_dual|clean|-2.837|1.552|0/4|
|curvature_interaction − curvature_dual|practical_high|-1.348|2.091|1/4|
|curvature_interaction − curvature_dual|practical_mid|-1.333|1.797|1/4|
|curvature_interaction − curvature_dual|practical_low_suburban|-1.173|1.645|1/4|
|curvature_interaction − curvature_dual|practical_high_urban|-1.342|1.837|1/4|
|curvature_interaction − curvature_dual|practical_mid_urban|-1.113|1.165|0/4|
|curvature_interaction − curvature_dual|practical_low_urban|-0.928|0.627|0/4|
|curvature_interaction − curvature_dual|six_view_mean|-1.206|1.522|0/4|

逐行/RX/TX见scores.csv，日期见day_scores.csv，混淆矩阵见scores.json与day_scores.json。
3200个epoch的紧凑诊断见all_epoch_diagnostics.csv；实际训练/推理成本见resources.json，共享GPU计时不等同独占性能。
四个model seed仅为描述性重复，未据此宣称统计显著性。
