共享mask未带来源域净增量。S−I的mid urban修复3、新错6，净−3／3600（−0.083333 pp）；low修复11、新错10，净+1／3600（+0.027778 pp），合并净−2／7200（−0.027778 pp）。每场景是1800个源V样本的两次信道观测，不是两个模型seed。

表中Δ间隔为默认头margin÷30的均值差；不同分层维度重叠，不能跨维度累加。新增噪声比不等于总SNR。

| 源分层 | mid n | mid 净正确 | mid Δ间隔 | low n | low 净正确 | low Δ间隔 |
|---|---:|---:|---:|---:|---:|---:|
| 新增噪声比 <3 dB | 383 | −1 | +0.000019 | 1127 | 0 | −0.000037 |
| 3–8 dB | 718 | −1 | −0.000064 | 794 | +2 | −0.000040 |
| 8–15 dB | 293 | 0 | +0.000035 | 303 | +1 | +0.000075 |
| ≥15 dB | 2206 | −1 | −0.000025 | 1376 | −2 | −0.000014 |
| LOS | 1899 | −1 | −0.000041 | 1060 | −1 | −0.000029 |
| PARTIAL | 367 | 0 | +0.000087 | 561 | 0 | +0.000069 |
| BLOCKED | 1334 | −2 | −0.000028 | 1979 | +2 | −0.000039 |
| locked | 2205 | −1 | −0.000023 | 1367 | −2 | −0.000014 |
| degraded | 715 | −3 | −0.000023 | 667 | 0 | +0.000003 |
| unlocked | 680 | +1 | −0.000023 | 1566 | +3 | −0.000033 |

整体mid间隔0.098499→0.098476，low为0.057271→0.057251；没有跨困难状态一致改善。

固定原As20 student的1765个clean正确样本，原As30、I、S均保留1765／1765。I→S的mid保持2613→2611／3530（74.022663%→73.966006%），low为2112→2114／3530（59.830028%→59.886686%）。父集合未损失不等于完整V27000无代价：S对I clean仅+0.003704 pp，对NoAs仍−0.118519 pp。

2220步trace实际实现共享：raw／As位相同率I为50.499934%、S为100%；两次RDrop仍用不同命名流，位相同率分别50.500981%／50.499798%。两行四份私有draw的seed与kept计数逐步一致，原生draw保留；142080个As按原序各用一次，物理ID、标签、domain对应通过。这些证明执行，不证明身份收益。

I对旧As的mid已净+7／3600、low净−1／3600，合并+0.083333 pp；因此不能将S对旧As的变化全归于共享mask，主比较必须是S−I。本轮只有一个模型seed，未提供可重复改善证据。

来源：[源分层](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/source_strata.csv)、[源配对纠错](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/source_paired_effects.csv)、[原As20父集合](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/P43_As20_clean_cohort.csv)、[父来源](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/parent_cohort_summary.json)、[完整训练与mask摘要](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/training_summary.json)、[先行源判定](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/stored_source_decision.json)。
