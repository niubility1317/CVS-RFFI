# D92-BranchLocalRidge-v1重复基准比较结果

记录：9648。4个固定Phase1模型、4RX、3场景、4个K、5种新增类规模、5组support抽样。

单元等权平均；H先在单元内计算再平均。不同support抽样复用query，不作为独立模型重复。

|K|方法|旧类准确率|新类准确率|H|宏F1|
|---|---|---:|---:|---:|---:|
|1|D92-BranchLocalRidge-v1|42.06%|29.54%|33.14%|34.28%|
|1|D92-BranchRidge-v1|42.00%|29.09%|32.67%|33.95%|
|5|D92-BranchLocalRidge-v1|60.04%|46.13%|51.52%|52.82%|
|5|D92-BranchRidge-v1|57.63%|42.15%|47.79%|49.10%|
|10|D92-BranchLocalRidge-v1|64.75%|53.05%|57.87%|59.00%|
|10|D92-BranchRidge-v1|62.80%|48.32%|53.90%|55.07%|
|20|D92-BranchLocalRidge-v1|68.83%|58.45%|62.84%|63.74%|
|20|D92-BranchRidge-v1|66.52%|52.80%|58.29%|59.33%|

|K|Δ旧类（百分点）|Δ新类（百分点）|ΔH（百分点）|
|---|---:|---:|---:|
|1|+0.07|+0.45|+0.47|
|5|+2.41|+3.97|+3.73|
|10|+1.95|+4.73|+3.96|
|20|+2.31|+5.64|+4.55|

以上表格为新增类数大于0的联合任务。旧类保护条件另按全部单元（含旧类单独任务）计算，见per_k_all.csv。

验收标准：每个K的H和新类严格提升、旧类退化不超过1个百分点。通过：True。每个K的新类均提升：True。

完整K×新增类规模、每seed和RX×场景结果见同目录CSV。最弱类别准确率、分组宏F1与遗忘均保留，不能用总体H掩盖局部退化。

适用范围：REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS: fixed LocalRidge predictions; D92, BranchRidge and BranchInteraction references unchanged; not independent confirmation or evidence of unexposed generalization。评分不回流本候选调参或选择性重跑。
