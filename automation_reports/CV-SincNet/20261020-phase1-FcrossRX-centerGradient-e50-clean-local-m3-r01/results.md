# F跨RX：均值梯度消融的三种子结果

各行从零50轮（33L＋17U），默认分类头正常联合训练、RDrop1、旧门控/ownEMA/cos、LS.01与普通增强不变。无星地增强，只测试相同168000条clean样本。预测固定后独立CPU truth-last评分；三个模型种子不是三批独立物理测试样本。

|方法|seed|clean%|BASE%|整体差/百分点|14-7差|20-19差|弱TX均差|
|---|---:|---:|---:|---:|---:|---:|---:|
|CENTER_GRAD|2026092701|85.2821|85.1393|+0.1429|-0.7893|+0.5179|-0.1357|
|CENTER_GRAD|2026092702|84.1994|84.4214|-0.2220|-3.0179|-0.7893|-1.9036|
|CENTER_GRAD|2026092703|84.6655|84.7244|-0.0589|-2.0607|+1.1179|-0.4714|

全部逐seed配对差、均值、样本标准差、正差数、Macro-F1、最差RX/day、逐TX及成本见results_summary.json。与事先固定STAT_GRAD/NORM_ONLY的配对在对应三种子齐全时另存secondary_comparisons，不按目标分数择优重排。

固定科学判断：{"PASS": false, "both_weak_means_positive_and_each2of3": false, "weakmean_positive3of3": false, "other_TX_overall_worstRX_health": false, "all3_required": true, "rule": "Both weakTX mean>0 and >=2/3positive; weakmean>0 all3; overallmean>=BASE; eachotherTXmean>=BASE-.25pp; worstRXmean>=BASE-.5pp. Fixed before target; no target-driven tuning/ranking/retraining."}

CENTER_GRAD在原描述子上仅保留当前均值的导数，当前标准差仍停止梯度；其辅助描述子梯度为原梯度减批均值。它保留原来的尺度方向响应，同输入前向/掩码/温度/损失值/默认头完全相同。不声称完整重算标准差前向的总导数，也没有梯度范数匹配；与完整统计梯度的比较同时改变梯度方向和大小。无部署BN、运行统计或额外网络前向。

目标基准已暴露，本次是事先固定全部三种子及final50的探索确认，不声称新的独立盲测、Phase2迁移或卫星性能。负结果全部保留；源V不做梯度、另训头或目标反馈。
