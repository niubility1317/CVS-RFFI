# D92-BranchLocalRidge-v1重复基准比较结果

记录：7236。4个固定Phase1模型、3RX、3场景、4个K、5种新增类规模、5组support抽样。

单元等权平均；H先在单元内计算再平均。不同support抽样复用query，不作为独立模型重复。

|K|方法|旧类准确率|新类准确率|H|宏F1|
|---|---|---:|---:|---:|---:|
|1|D92-BranchLocalRidge-v1|37.22%|28.13%|30.52%|31.09%|
|1|D92-BranchRidge-v1|37.05%|27.88%|30.22%|30.81%|
|5|D92-BranchLocalRidge-v1|55.08%|43.15%|47.71%|48.65%|
|5|D92-BranchRidge-v1|52.39%|40.02%|44.51%|45.19%|
|10|D92-BranchLocalRidge-v1|60.57%|49.60%|54.04%|54.97%|
|10|D92-BranchRidge-v1|58.21%|45.69%|50.43%|51.26%|
|20|D92-BranchLocalRidge-v1|65.00%|54.93%|59.13%|59.90%|
|20|D92-BranchRidge-v1|62.25%|49.53%|54.53%|55.31%|

|K|Δ旧类（百分点）|Δ新类（百分点）|ΔH（百分点）|
|---|---:|---:|---:|
|1|+0.18|+0.26|+0.30|
|5|+2.69|+3.13|+3.20|
|10|+2.35|+3.91|+3.61|
|20|+2.75|+5.40|+4.60|

以上表格为新增类数大于0的联合任务。旧类保护条件另按全部单元（含旧类单独任务）计算，见per_k_all.csv。

验收标准：每个K的H和新类严格提升、旧类退化不超过1个百分点。通过：True。每个K的新类均提升：True。

完整K×新增类规模、每seed和RX×场景结果见同目录CSV。最弱类别准确率、分组宏F1与遗忘均保留，不能用总体H掩盖局部退化。

适用范围：REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS: fixed LocalRidge predictions; D92, BranchRidge and BranchInteraction references unchanged; not independent confirmation or evidence of unexposed generalization。评分不回流本候选调参或选择性重跑。
