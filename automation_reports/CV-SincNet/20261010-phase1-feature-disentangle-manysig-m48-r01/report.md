# 特征增强版多解耦三种子探索

纯 CE、无半监督、无星地训练增强；保持当前身份结构、cosine分类头与学习率日程。48行均从零训练E200/44400步，不加载旧权重。

## 实验矩阵

| 配置 | 关系流 | 弱Style | Fishr | 机制 |
|---|---|---|---|---|
| native | none | False | none / 0.0 | 无作用网络 |
| random_S | random | False | none / 0.0 | 无作用网络 |
| S | balanced | False | none / 0.0 | 无作用网络 |
| S_raw5 | balanced | False | raw / 0.05 | 无作用网络 |
| S_dir5 | balanced | False | direction / 0.05 | 无作用网络 |
| LTR_S | balanced | False | none / 0.0 | linear,temporal,receiver |
| LTR_S_A | balanced | True | none / 0.0 | linear,temporal,receiver |
| LTR_S_F3 | balanced | False | direction / 0.03 | linear,temporal,receiver |
| LTR_S_F5 | balanced | False | direction / 0.05 | linear,temporal,receiver |
| LTR_S_F10 | balanced | False | direction / 0.1 | linear,temporal,receiver |
| LTR_S_AF3 | balanced | True | direction / 0.03 | linear,temporal,receiver |
| LTR_S_AF5 | balanced | True | direction / 0.05 | linear,temporal,receiver |
| LTR_S_AF10 | balanced | True | direction / 0.1 | linear,temporal,receiver |
| joint_noR | balanced | True | direction / 源选强度 | linear,temporal,receiver |
| joint_shared | balanced | True | direction / 源选强度 | linear,temporal,receiver |
| joint_direct | balanced | True | direction / 源选强度 | receiver |

每配置seed为2026092701、2026092702、2026092703。先完成13种固定配置39行；以SAF3/5/10完整源V三seed选择强度，并选择同强度SF，未选4配置12行只保留源结果。再从零训练noR/shared/direct的9行。最终测试36行，每行clean及六practical。

固定控制逐行冻结即测试；需要源选的候选等待必要源选择，未选候选不读target。所有预测固定后独立truth-last评分，目标结果不回流任何健康训练、排序或参数。

B48为6TX×2RX×4不同物理PID，同一天；30block公平轮换。每8个原生step一次，5550关系调用、266400clean包次，不等于独立样本数。所有关系对照CE权重0.1。Style只复用nativeL原有CE。

Fishr使用当前cosine头6×160参数方向梯度；raw梯度作同流程对照。每桶RX/day/clean独立计数去偏EMA，E1–20积累，E20成熟源L定标，E21–40升权，之后固定。EMA只保存梯度统计，不是教师。

L/T预测用于筛选并在真实IQ训练；合法困难IQ不因代理预测失准被丢弃。R保留独立角色分袋、判别margin分布和尾部风险，noR对照仅关闭身份约束。共享作用核心hidden50与独立hidden48参数差约0.1612%，保持候选和真实视图预算。

最高尚未验证的科学问题是弱Style与方向Fishr是否在不损害身份间隔的情况下改善困难RX；不能用辅助loss下降替代识别收益。分支关闭仅依赖性探针，非严格因果结论。

报告中旧U/EMA/LEO配方按用户明确要求覆盖为pureCE和clean-only Fishr；DSU、困难重采样和第四网络按首轮条件延后。完整要求见[设计追溯](traceability.md)和[原报告](design_report.md)。

本地CPU/CUDA聚焦与原生训练集成验证通过，独立P0/P1审查通过。E20全部成熟关系批次中位数定标；保存每桶统计、计数、年龄及各分支加权梯度。日志重复annotate按epoch幂等，避免覆盖稀疏观测分母。详见evidence。
