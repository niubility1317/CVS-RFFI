# D92-BranchLocalRidge-v1重复基准比较结果

记录：2412。4个固定Phase1模型、1RX、3场景、4个K、5种新增类规模、5组support抽样。

单元等权平均；H先在单元内计算再平均。不同support抽样复用query，不作为独立模型重复。

|K|方法|旧类准确率|新类准确率|H|宏F1|
|---|---|---:|---:|---:|---:|
|1|D92-BranchLocalRidge-v1|56.58%|33.75%|40.99%|43.86%|
|1|D92-BranchRidge-v1|56.85%|32.73%|40.02%|43.36%|
|5|D92-BranchLocalRidge-v1|74.90%|55.04%|62.97%|65.34%|
|5|D92-BranchRidge-v1|73.33%|48.55%|57.65%|60.83%|
|10|D92-BranchLocalRidge-v1|77.30%|63.40%|69.35%|71.07%|
|10|D92-BranchRidge-v1|76.55%|56.19%|64.32%|66.49%|
|20|D92-BranchLocalRidge-v1|80.34%|69.00%|73.96%|75.24%|
|20|D92-BranchRidge-v1|79.34%|62.63%|69.58%|71.38%|

|K|Δ旧类（百分点）|Δ新类（百分点）|ΔH（百分点）|
|---|---:|---:|---:|
|1|-0.26|+1.02|+0.97|
|5|+1.56|+6.49|+5.32|
|10|+0.74|+7.20|+5.04|
|20|+1.00|+6.38|+4.39|

以上表格为新增类数大于0的联合任务。旧类保护条件另按全部单元（含旧类单独任务）计算，见per_k_all.csv。

验收标准：每个K的H和新类严格提升、旧类退化不超过1个百分点。通过：True。每个K的新类均提升：True。

完整K×新增类规模、每seed和RX×场景结果见同目录CSV。最弱类别准确率、分组宏F1与遗忘均保留，不能用总体H掩盖局部退化。

适用范围：REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS: fixed LocalRidge predictions; D92, BranchRidge and BranchInteraction references unchanged; not independent confirmation or evidence of unexposed generalization。评分不回流本候选调参或选择性重跑。
