# D92-BranchLocalRidge-v1重复基准比较结果

记录：2412。4个固定Phase1模型、1RX、3场景、4个K、5种新增类规模、5组support抽样。

单元等权平均；H先在单元内计算再平均。不同support抽样复用query，不作为独立模型重复。

|K|方法|旧类准确率|新类准确率|H|宏F1|
|---|---|---:|---:|---:|---:|
|1|D92-BranchInteraction-v1|56.82%|32.78%|40.11%|43.44%|
|1|D92-BranchLocalRidge-v1|56.58%|33.75%|40.99%|43.86%|
|5|D92-BranchInteraction-v1|74.40%|51.48%|60.17%|63.08%|
|5|D92-BranchLocalRidge-v1|74.90%|55.04%|62.97%|65.34%|
|10|D92-BranchInteraction-v1|77.08%|60.03%|67.11%|68.87%|
|10|D92-BranchLocalRidge-v1|77.30%|63.40%|69.35%|71.07%|
|20|D92-BranchInteraction-v1|79.74%|66.14%|71.95%|73.45%|
|20|D92-BranchLocalRidge-v1|80.34%|69.00%|73.96%|75.24%|

|K|Δ旧类（百分点）|Δ新类（百分点）|ΔH（百分点）|
|---|---:|---:|---:|
|1|-0.24|+0.96|+0.88|
|5|+0.50|+3.57|+2.80|
|10|+0.21|+3.37|+2.25|
|20|+0.60|+2.86|+2.01|

以上表格为新增类数大于0的联合任务。旧类保护条件另按全部单元（含旧类单独任务）计算，见per_k_all.csv。

验收标准：每个K的H和新类严格提升、旧类退化不超过1个百分点。通过：True。每个K的新类均提升：True。

完整K×新增类规模、每seed和RX×场景结果见同目录CSV。最弱类别准确率、分组宏F1与遗忘均保留，不能用总体H掩盖局部退化。

适用范围：REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS: fixed LocalRidge predictions; D92, BranchRidge and BranchInteraction references unchanged; not independent confirmation or evidence of unexposed generalization。评分不回流本候选调参或选择性重跑。
