# 多解耦完整测试结果

32个独立scratch E200模型，8组×4seed；各视图168000个固定query，224组预测先于truth评分。
旧batch_neighbor伪标签＋cosine学习率、相同44400更新预算。已有暴露代理基准；Phase2/K/新增类/H不适用。
独立重算混淆矩阵、RX/TX/day分区、四seed均值与SD、paired差值通过。结果不得回流调参或选择性重跑。

|架构|视图|准确率均值±SD（%）|Macro-F1（%）|最差RX均值（%）|
|---|---|---:|---:|---:|
|legacy_cosine|clean|81.215 ± 1.031|80.774|69.759|
|legacy_cosine|practical_high|79.056 ± 0.720|78.745|69.975|
|legacy_cosine|practical_mid|77.022 ± 0.609|76.677|68.422|
|legacy_cosine|practical_low_suburban|74.830 ± 0.579|74.413|66.595|
|legacy_cosine|practical_high_urban|76.043 ± 0.578|75.690|67.567|
|legacy_cosine|practical_mid_urban|63.569 ± 0.230|62.922|57.614|
|legacy_cosine|practical_low_urban|53.258 ± 0.307|52.353|48.678|
|fixed_interventions|clean|80.999 ± 0.827|80.537|69.794|
|fixed_interventions|practical_high|79.020 ± 0.603|78.700|69.610|
|fixed_interventions|practical_mid|76.949 ± 0.505|76.595|68.140|
|fixed_interventions|practical_low_suburban|74.742 ± 0.489|74.318|66.407|
|fixed_interventions|practical_high_urban|76.013 ± 0.460|75.648|67.239|
|fixed_interventions|practical_mid_urban|63.514 ± 0.153|62.855|57.211|
|fixed_interventions|practical_low_urban|53.205 ± 0.265|52.293|48.408|
|unified|clean|80.992 ± 0.674|80.541|69.919|
|unified|practical_high|78.902 ± 0.478|78.586|69.532|
|unified|practical_mid|76.834 ± 0.384|76.483|68.009|
|unified|practical_low_suburban|74.634 ± 0.359|74.214|66.298|
|unified|practical_high_urban|75.906 ± 0.362|75.547|67.120|
|unified|practical_mid_urban|63.431 ± 0.321|62.777|57.168|
|unified|practical_low_urban|53.194 ± 0.482|52.287|48.443|
|linear|clean|80.971 ± 0.990|80.498|69.681|
|linear|practical_high|79.080 ± 0.446|78.745|69.929|
|linear|practical_mid|77.030 ± 0.390|76.667|68.399|
|linear|practical_low_suburban|74.857 ± 0.371|74.429|66.552|
|linear|practical_high_urban|76.102 ± 0.305|75.728|67.504|
|linear|practical_mid_urban|63.581 ± 0.209|62.917|57.324|
|linear|practical_low_urban|53.280 ± 0.380|52.371|48.581|
|temporal|clean|81.124 ± 0.505|80.671|70.237|
|temporal|practical_high|78.836 ± 0.596|78.504|69.742|
|temporal|practical_mid|76.797 ± 0.550|76.432|68.311|
|temporal|practical_low_suburban|74.650 ± 0.536|74.219|66.581|
|temporal|practical_high_urban|75.831 ± 0.516|75.460|67.299|
|temporal|practical_mid_urban|63.413 ± 0.377|62.751|57.244|
|temporal|practical_low_urban|53.128 ± 0.507|52.217|48.440|
|receiver|clean|81.175 ± 0.727|80.710|70.177|
|receiver|practical_high|78.879 ± 0.685|78.564|69.636|
|receiver|practical_mid|76.795 ± 0.624|76.442|68.219|
|receiver|practical_low_suburban|74.612 ± 0.625|74.185|66.497|
|receiver|practical_high_urban|75.867 ± 0.562|75.507|67.190|
|receiver|practical_mid_urban|63.405 ± 0.279|62.741|57.125|
|receiver|practical_low_urban|53.163 ± 0.342|52.247|48.361|
|multi|clean|81.048 ± 0.731|80.582|69.530|
|multi|practical_high|78.980 ± 0.583|78.650|69.784|
|multi|practical_mid|76.939 ± 0.519|76.581|68.327|
|multi|practical_low_suburban|74.743 ± 0.484|74.323|66.519|
|multi|practical_high_urban|75.969 ± 0.486|75.603|67.330|
|multi|practical_mid_urban|63.488 ± 0.240|62.836|57.259|
|multi|practical_low_urban|53.171 ± 0.367|52.269|48.426|
|multi_interaction|clean|81.025 ± 0.823|80.565|69.390|
|multi_interaction|practical_high|79.017 ± 0.314|78.700|69.287|
|multi_interaction|practical_mid|77.003 ± 0.299|76.658|67.993|
|multi_interaction|practical_low_suburban|74.826 ± 0.335|74.413|66.311|
|multi_interaction|practical_high_urban|75.996 ± 0.184|75.642|66.974|
|multi_interaction|practical_mid_urban|63.533 ± 0.297|62.876|56.969|
|multi_interaction|practical_low_urban|53.263 ± 0.385|52.351|48.365|

## 配对差值

|处理−对照|视图|均值（百分点）|SD（百分点）|正增益seed|
|---|---|---:|---:|---:|
|fixed_interventions − legacy_cosine|clean|-0.216|0.405|2/4|
|fixed_interventions − legacy_cosine|practical_high|-0.037|0.205|2/4|
|fixed_interventions − legacy_cosine|practical_mid|-0.073|0.159|1/4|
|fixed_interventions − legacy_cosine|practical_low_suburban|-0.088|0.126|1/4|
|fixed_interventions − legacy_cosine|practical_high_urban|-0.031|0.172|2/4|
|fixed_interventions − legacy_cosine|practical_mid_urban|-0.054|0.115|2/4|
|fixed_interventions − legacy_cosine|practical_low_urban|-0.054|0.067|1/4|
|fixed_interventions − legacy_cosine|six_view_mean|-0.056|0.136|1/4|
|unified − legacy_cosine|clean|-0.223|0.394|1/4|
|unified − legacy_cosine|practical_high|-0.155|0.274|2/4|
|unified − legacy_cosine|practical_mid|-0.188|0.269|2/4|
|unified − legacy_cosine|practical_low_suburban|-0.196|0.251|1/4|
|unified − legacy_cosine|practical_high_urban|-0.138|0.297|2/4|
|unified − legacy_cosine|practical_mid_urban|-0.137|0.203|2/4|
|unified − legacy_cosine|practical_low_urban|-0.064|0.193|2/4|
|unified − legacy_cosine|six_view_mean|-0.146|0.245|2/4|
|linear − legacy_cosine|clean|-0.244|0.282|1/4|
|linear − legacy_cosine|practical_high|0.024|0.298|2/4|
|linear − legacy_cosine|practical_mid|0.008|0.245|2/4|
|linear − legacy_cosine|practical_low_suburban|0.027|0.246|2/4|
|linear − legacy_cosine|practical_high_urban|0.059|0.296|2/4|
|linear − legacy_cosine|practical_mid_urban|0.012|0.175|2/4|
|linear − legacy_cosine|practical_low_urban|0.022|0.131|2/4|
|linear − legacy_cosine|six_view_mean|0.025|0.230|2/4|
|temporal − legacy_cosine|clean|-0.091|0.585|2/4|
|temporal − legacy_cosine|practical_high|-0.221|0.200|1/4|
|temporal − legacy_cosine|practical_mid|-0.225|0.168|0/4|
|temporal − legacy_cosine|practical_low_suburban|-0.180|0.155|0/4|
|temporal − legacy_cosine|practical_high_urban|-0.212|0.178|1/4|
|temporal − legacy_cosine|practical_mid_urban|-0.156|0.155|1/4|
|temporal − legacy_cosine|practical_low_urban|-0.130|0.219|2/4|
|temporal − legacy_cosine|six_view_mean|-0.187|0.168|1/4|
|receiver − legacy_cosine|clean|-0.040|0.388|2/4|
|receiver − legacy_cosine|practical_high|-0.177|0.128|0/4|
|receiver − legacy_cosine|practical_mid|-0.227|0.067|0/4|
|receiver − legacy_cosine|practical_low_suburban|-0.218|0.118|0/4|
|receiver − legacy_cosine|practical_high_urban|-0.177|0.130|0/4|
|receiver − legacy_cosine|practical_mid_urban|-0.163|0.140|1/4|
|receiver − legacy_cosine|practical_low_urban|-0.095|0.127|1/4|
|receiver − legacy_cosine|six_view_mean|-0.176|0.101|0/4|
|multi − legacy_cosine|clean|-0.167|0.568|2/4|
|multi − legacy_cosine|practical_high|-0.077|0.225|1/4|
|multi − legacy_cosine|practical_mid|-0.083|0.185|1/4|
|multi − legacy_cosine|practical_low_suburban|-0.087|0.143|2/4|
|multi − legacy_cosine|practical_high_urban|-0.075|0.196|1/4|
|multi − legacy_cosine|practical_mid_urban|-0.081|0.090|1/4|
|multi − legacy_cosine|practical_low_urban|-0.087|0.098|1/4|
|multi − legacy_cosine|six_view_mean|-0.082|0.148|1/4|
|multi_interaction − legacy_cosine|clean|-0.189|0.453|2/4|
|multi_interaction − legacy_cosine|practical_high|-0.040|0.412|2/4|
|multi_interaction − legacy_cosine|practical_mid|-0.019|0.312|2/4|
|multi_interaction − legacy_cosine|practical_low_suburban|-0.005|0.248|2/4|
|multi_interaction − legacy_cosine|practical_high_urban|-0.047|0.397|2/4|
|multi_interaction − legacy_cosine|practical_mid_urban|-0.035|0.165|2/4|
|multi_interaction − legacy_cosine|practical_low_urban|0.005|0.085|1/4|
|multi_interaction − legacy_cosine|six_view_mean|-0.023|0.256|2/4|
|multi − fixed_interventions|clean|0.049|0.268|2/4|
|multi − fixed_interventions|practical_high|-0.040|0.137|1/4|
|multi − fixed_interventions|practical_mid|-0.011|0.158|2/4|
|multi − fixed_interventions|practical_low_suburban|0.001|0.113|1/4|
|multi − fixed_interventions|practical_high_urban|-0.044|0.158|1/4|
|multi − fixed_interventions|practical_mid_urban|-0.026|0.089|2/4|
|multi − fixed_interventions|practical_low_urban|-0.034|0.110|1/4|
|multi − fixed_interventions|six_view_mean|-0.026|0.123|1/4|
|multi − unified|clean|0.056|0.338|1/4|
|multi − unified|practical_high|0.078|0.271|2/4|
|multi − unified|practical_mid|0.105|0.255|2/4|
|multi − unified|practical_low_suburban|0.109|0.179|3/4|
|multi − unified|practical_high_urban|0.063|0.319|2/4|
|multi − unified|practical_mid_urban|0.057|0.249|1/4|
|multi − unified|practical_low_urban|-0.023|0.194|1/4|
|multi − unified|six_view_mean|0.065|0.243|2/4|
|multi − linear|clean|0.077|0.497|3/4|
|multi − linear|practical_high|-0.100|0.165|1/4|
|multi − linear|practical_mid|-0.091|0.147|1/4|
|multi − linear|practical_low_suburban|-0.114|0.136|0/4|
|multi − linear|practical_high_urban|-0.134|0.196|1/4|
|multi − linear|practical_mid_urban|-0.093|0.186|1/4|
|multi − linear|practical_low_urban|-0.109|0.128|0/4|
|multi − linear|six_view_mean|-0.107|0.157|1/4|
|multi − temporal|clean|-0.076|0.277|2/4|
|multi − temporal|practical_high|0.144|0.248|3/4|
|multi − temporal|practical_mid|0.142|0.234|3/4|
|multi − temporal|practical_low_suburban|0.093|0.172|3/4|
|multi − temporal|practical_high_urban|0.137|0.224|3/4|
|multi − temporal|practical_mid_urban|0.075|0.199|2/4|
|multi − temporal|practical_low_urban|0.043|0.202|2/4|
|multi − temporal|six_view_mean|0.106|0.209|3/4|
|multi − receiver|clean|-0.127|0.223|1/4|
|multi − receiver|practical_high|0.101|0.295|3/4|
|multi − receiver|practical_mid|0.144|0.246|3/4|
|multi − receiver|practical_low_suburban|0.132|0.252|3/4|
|multi − receiver|practical_high_urban|0.102|0.296|2/4|
|multi − receiver|practical_mid_urban|0.083|0.217|3/4|
|multi − receiver|practical_low_urban|0.008|0.200|3/4|
|multi − receiver|six_view_mean|0.095|0.245|3/4|
|multi_interaction − multi|clean|-0.022|0.263|2/4|
|multi_interaction − multi|practical_high|0.037|0.322|2/4|
|multi_interaction − multi|practical_mid|0.065|0.239|3/4|
|multi_interaction − multi|practical_low_suburban|0.082|0.154|3/4|
|multi_interaction − multi|practical_high_urban|0.027|0.315|2/4|
|multi_interaction − multi|practical_mid_urban|0.045|0.179|3/4|
|multi_interaction − multi|practical_low_urban|0.093|0.059|4/4|
|multi_interaction − multi|six_view_mean|0.058|0.201|3/4|

逐行/RX/TX见scores.csv，日期见day_scores.csv，混淆矩阵见scores.json与day_scores.json。
6400个epoch的紧凑诊断见all_epoch_diagnostics.csv；实际训练/推理成本见resources.json，共享GPU计时不等同独占性能。
四个model seed仅为描述性重复，未据此宣称统计显著性。
