M3 的有界风险未转化为源验证增量。相对 M2，mid urban 修复13、新错9，净+4／3600（+0.111111 pp）；low urban 修复13、新错19，净−6／3600（−0.166667 pp），合并净−2／7200（−0.027778 pp）。每场景为同一1800个源V样本的两次信道观测，不是两个模型seed。

下表为M3−M2；Δ间隔表示默认头分类间隔除以30后的均值差。不同维度存在重叠，不能跨维度累加。

| 源分层 | mid n | mid 净正确 | mid Δ间隔 | low n | low 净正确 | low Δ间隔 |
|---|---:|---:|---:|---:|---:|---:|
| 新增噪声比 <3 dB | 383 | +1 | +0.000842 | 1127 | +2 | +0.000890 |
| 3–8 dB | 718 | 0 | +0.001359 | 794 | −1 | +0.001290 |
| 8–15 dB | 293 | +4 | +0.001342 | 303 | −2 | +0.000439 |
| ≥15 dB | 2206 | −1 | −0.001933 | 1376 | −5 | −0.001704 |
| BLOCKED | 1334 | +5 | +0.001270 | 1979 | +1 | +0.001069 |
| locked | 2205 | −1 | −0.001935 | 1367 | −5 | −0.001721 |
| degraded | 715 | +4 | +0.001473 | 667 | −4 | +0.001010 |
| unlocked | 680 | +1 | +0.000941 | 1566 | +3 | +0.000954 |

3–15 dB间隔上升，low该区间却净−3；≥15 dB合计净−6，间隔下降。BLOCKED局部净增不能证明遮挡损伤已被解决；新增噪声比也不是总SNR。

固定原As20 student的1765个clean正确样本，M2→M3的clean保持为1763→1764／1765（99.886686%→99.943343%）；两draw的mid保持2603→2607／3530（73.739377%→73.852691%），low为2114→2109／3530（59.886686%→59.745042%）。该子集略有clean修复，不能代替完整V27000：M3 clean对M2仍−0.007407 pp，对NoAs−0.133333 pp。

M3的实际风险CE从E21的0.875668降至E30的0.832836；同logits nominal CE从0.767574降至0.724443，四组训练CE均下降，但源mid／low未净增。风险CE与M2目标不同，不能直接横比其绝对值。此处只是训练与验证的对应观察，不作物理因果归因。

来源：[源配对表](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/source_paired_effects.csv)、[源分层表](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/source_strata.csv)、[父正确集合](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/P43_As20_clean_cohort.csv)、[父来源](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/parent_cohort_summary.json)、[风险汇总](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/risk_summary.json)、[训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.csv)、[已固定源判定](E:/type10-7/local_artifacts/cvs_FcrossRX_bounded_risk_E30_20261019_r01/retrieved_r01/analysis_r01/stored_source_decision.json)。
