# D92-BranchLocalRidge-v1重复基准比较结果

记录：9648。4个固定Phase1模型、4RX、3场景、4个K、5种新增类规模、5组support抽样。

单元等权平均；H先在单元内计算再平均。不同support抽样复用query，不作为独立模型重复。

|K|方法|旧类准确率|新类准确率|H|宏F1|
|---|---|---:|---:|---:|---:|
|1|D92|37.11%|23.67%|26.55%|28.33%|
|1|D92-BranchLocalRidge-v1|42.06%|29.54%|33.14%|34.28%|
|5|D92|44.10%|35.32%|38.42%|39.59%|
|5|D92-BranchLocalRidge-v1|60.04%|46.13%|51.52%|52.82%|
|10|D92|53.48%|42.21%|46.54%|47.89%|
|10|D92-BranchLocalRidge-v1|64.75%|53.05%|57.87%|59.00%|
|20|D92|60.24%|46.30%|51.72%|53.28%|
|20|D92-BranchLocalRidge-v1|68.83%|58.45%|62.84%|63.74%|

|K|Δ旧类（百分点）|Δ新类（百分点）|ΔH（百分点）|
|---|---:|---:|---:|
|1|+4.95|+5.86|+6.59|
|5|+15.93|+10.81|+13.10|
|10|+11.27|+10.84|+11.33|
|20|+8.59|+12.14|+11.13|

以上表格为新增类数大于0的联合任务。旧类保护条件另按全部单元（含旧类单独任务）计算，见per_k_all.csv。

验收标准：每个K的H和新类严格提升、旧类退化不超过1个百分点。通过：True。每个K的新类均提升：True。

完整K×新增类规模、每seed和RX×场景结果见同目录CSV。最弱类别准确率、分组宏F1与遗忘均保留，不能用总体H掩盖局部退化。

适用范围：REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS: fixed LocalRidge predictions; D92, BranchRidge and BranchInteraction references unchanged; not independent confirmation or evidence of unexposed generalization。评分不回流本候选调参或选择性重跑。
