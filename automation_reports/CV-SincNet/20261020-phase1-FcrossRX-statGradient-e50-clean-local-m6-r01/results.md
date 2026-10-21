# F跨RX：标准化统计量梯度的三种子结果

各行从零50轮（33L＋17U），默认分类头正常联合训练、RDrop1、旧门控/ownEMA/cos、LS.01与普通增强不变。无星地增强，只测试相同168000条clean样本。预测固定后独立CPU truth-last评分；三个模型种子不是三批独立物理测试样本。

|方法|seed|clean%|BASE%|整体差/百分点|14-7差|20-19差|弱TX均差|
|---|---:|---:|---:|---:|---:|---:|---:|
|STAT_GRAD|2026092701|85.4351|85.1393|+0.2958|+0.0250|-0.0357|-0.0054|
|NORM_ONLY|2026092701|85.2488|85.1393|+0.1095|+0.3786|+0.3250|+0.3518|
|STAT_GRAD|2026092702|83.9083|84.4214|-0.5131|-3.2750|-1.7357|-2.5054|
|NORM_ONLY|2026092702|84.4333|84.4214|+0.0119|-0.8071|-0.0750|-0.4411|
|STAT_GRAD|2026092703|84.5393|84.7244|-0.1851|-3.4179|+0.8643|-1.2768|
|NORM_ONLY|2026092703|84.7185|84.7244|-0.0060|-1.5786|+0.6786|-0.4500|

全部逐seed配对差、均值、样本标准差、正差数、Macro-F1、最差RX/day、逐TX及成本见results_summary.json。STAT_GRAD与NORM_ONLY之间的固定配对单独保存，不按目标分数择优重排。

固定科学判断：{"primary_STAT_GRAD_vs_BASE": {"PASS": false, "both_weak_means_positive_and_each2of3": false, "weakmean_positive3of3": false, "other_TX_overall_worstRX_health": false}, "direction_vs_NORM_ONLY": {"PASS": false}, "all3_required": true, "rule": "Primary: each weakTX mean>0 and at least2/3positive; weakmean>0 all3; overallmean>=BASE; each otherTX mean>=BASE-.25pp; worstRXmean>=BASE-.5pp. Attribution separately: STAT_GRAD-NORM_ONLY weakmean>0 and >=2/3positive. Fixed before target; no target-driven retuning/ranking/retraining."}

STAT_GRAD在原描述子中心/标准差上使用真实批统计导数；NORM_ONLY保留原停止统计量梯度的方向，只将同一描述子梯度范数缩放到完整导数的范数。两者的同输入前向、对比掩码、温度、损失值和默认头结构完全一致；控制不表示整网参数梯度或不同训练轨迹范数相等。它不是部署BN，不添加运行统计或额外网络前向。

目标基准已暴露，本次是事先固定全部三种子及final50的探索确认，不声称新的独立盲测、Phase2迁移或卫星性能。负结果全部保留；源V不做梯度、另训头或目标反馈。
