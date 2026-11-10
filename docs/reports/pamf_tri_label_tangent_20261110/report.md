# 三视图 SSL 身份保护物理关系：三种子结果

本结果是已冻结的完整 Tri-SIG-TIME SSL 加有标签源域物理关系候选；不把新增关系声称为无标签损失。原方法固定局部时序顺序、三视图、同种子 E20 来源后各自续训至 E50；使用完整 PAMF 默认头和 DURA-LU。训练中 U 的 TX 标签不可见。

主口径为 Main split 的 cross_rx 与 cross_both 合计，每种子每场景 168000 包。三种子分别为 2026092701、02、03；各自七景 prediction 全固定后独立 truth-last 评分。普通 PL 采用自然时序而完整 Tri-SIG-TIME 采用固定时序，因此相对 PL 的差异不能归于物理关系。历史测试已暴露，不属于新的独立盲测。

| 场景 | 候选准确率 % | 相对 Tri pp | 对 Tri 纠错/错失（每种子平均包数） | 相对 PL pp | 对 PL 纠错/错失（每种子平均包数） |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 86.7538 | -0.2077 | 482.3 / 831.3 | +1.7242 | 6462.7 / 3566.0 |
| practical_high | 84.2002 | -0.0952 | 493.3 / 653.3 | +1.3788 | 6270.7 / 3954.3 |
| practical_mid | 82.7869 | -0.0897 | 536.7 / 687.3 | +1.3159 | 6530.0 / 4319.3 |
| practical_low_suburban | 80.9863 | -0.0766 | 567.7 / 696.3 | +1.1710 | 6814.3 / 4847.0 |
| practical_high_urban | 82.8165 | -0.0899 | 525.3 / 676.3 | +1.2143 | 6335.0 / 4295.0 |
| practical_mid_urban | 75.3216 | -0.0821 | 568.3 / 706.3 | +0.5234 | 6693.3 / 5814.0 |
| practical_low_urban | 68.2268 | -0.0429 | 631.0 / 703.0 | +0.2871 | 6850.3 / 6368.0 |

相对 Tri-SIG-TIME 的三种子在全部七景均下降，因此首要目标未达成。相对 PL 七景三种子均值均提升，但种子 03 的 practical_mid_urban 为 −0.1071 pp、practical_low_urban 为 −0.2756 pp，不能称为逐种子稳定领先。

## 分层结果

下表为每个维度按同一场景和组别对三种子求平均后的极值，用于标识差异范围；完整逐种子、逐场景、逐 TX/RX/day、逐对照的准确率、纠错和错失保存在本地产物 `local_artifacts/pamf_physics_tri_label_tangent_20261110/combined_target_per_seed_strata.csv`。不同分层有不同样本数，不应把它们作为独立实验。

| 对照 | 维度 | 平均差值最低：场景、组别、pp | 平均差值最高：场景、组别、pp |
| --- | --- | --- | --- |
| Tri-SIG-TIME | transmitter | clean、14-7、-0.9524 | practical_mid、20-19、+0.6417 |
| Tri-SIG-TIME | receiver | clean、20-1、-0.6069 | practical_high_urban、14-7、+0.2847 |
| Tri-SIG-TIME | date | clean、2021_03_15、-0.4421 | practical_mid_urban、2021_03_01、+0.0198 |
| PL | transmitter | practical_low_urban、14-7、-5.2238 | practical_high、20-15、+8.5702 |
| PL | receiver | practical_high、7-7、-2.4000 | practical_high、8-8、+7.2236 |
| PL | date | practical_low_urban、2021_03_01、-0.2833 | clean、2021_03_15、+3.2071 |

## 机制和现有工作

有限幅度时延回波用于估计各分支对可解释扰动的局部变化；标签类中心的差异方向被投影保护。它与 [FixMatch](https://arxiv.org/abs/2001.07685) 的弱增强伪标签与强增强一致性不同，也不同于 [Deep Co-Training](https://openaccess.thecvf.com/content_ECCV_2018/html/Siyuan_Qiao_Deep_Co-Training_for_ECCV_2018_paper.html) 的多模型视图交换；[信道稳健射频指纹对比学习](https://livrepository.liverpool.ac.uk/3190639/) 已研究用增强减弱信道影响。它们均不直接证明本候选有效或新颖。

已在读取本候选测试汇总前固定并启动同种子物理关系无身份投影消融 `20261110-phase1-pamf-tri-unprotected-f0-m3-r01`。该消融可比较投影的条件效应；与原 Tri 基线相比仍需注意新增物理损失造成的训练轨迹差异，不能单独证明具体组件效果。资源成本、更多接收机划分和标注比例尚未完成。
