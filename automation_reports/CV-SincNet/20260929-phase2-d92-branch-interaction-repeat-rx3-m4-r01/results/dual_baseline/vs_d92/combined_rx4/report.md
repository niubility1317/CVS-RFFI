# D92-BranchInteraction-v1重复基准比较结果

记录：9648。4个固定Phase1模型、4RX、3场景、4个K、5种新增类规模、5组support抽样。

单元等权平均；H先在单元内计算再平均。不同support抽样复用query，不作为独立模型重复。

|K|方法|旧类准确率|新类准确率|H|宏F1|
|---|---|---:|---:|---:|---:|
|1|D92|37.11%|23.67%|26.55%|28.33%|
|1|D92-BranchInteraction-v1|41.91%|29.14%|32.68%|33.94%|
|5|D92|44.10%|35.32%|38.42%|39.59%|
|5|D92-BranchInteraction-v1|58.77%|43.96%|49.50%|50.84%|
|10|D92|53.48%|42.21%|46.54%|47.89%|
|10|D92-BranchInteraction-v1|63.72%|50.63%|55.83%|56.97%|
|20|D92|60.24%|46.30%|51.72%|53.28%|
|20|D92-BranchInteraction-v1|67.56%|55.54%|60.49%|61.48%|

|K|Δ旧类（百分点）|Δ新类（百分点）|ΔH（百分点）|
|---|---:|---:|---:|
|1|+4.80|+5.47|+6.13|
|5|+14.67|+8.64|+11.08|
|10|+10.24|+8.42|+9.29|
|20|+7.32|+9.24|+8.77|

以上表格为新增类数大于0的联合任务。旧类保护条件另按全部单元（含旧类单独任务）计算，见per_k_all.csv。

验收标准：每个K的H和新类严格提升、旧类退化不超过1个百分点。通过：True。每个K的新类均提升：True。

完整K×新增类规模、每seed和RX×场景结果见同目录CSV。最弱类别准确率、分组宏F1与遗忘均保留，不能用总体H掩盖局部退化。

适用范围：REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS: user authorized reuse; frozen D92-BranchInteraction-v1 support-only formula, no query fitting; all four receivers pooled with equal paired-cell weight; not a new independent confirmation or evidence of unexposed generalization。评分不回流本候选调参或选择性重跑。
