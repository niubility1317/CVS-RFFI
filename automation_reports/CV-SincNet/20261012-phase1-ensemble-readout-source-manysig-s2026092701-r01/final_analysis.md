# 三成员集成固定读出：源端零训练检验

|读出|源均值|最差RX|资格|
|---|---:|---:|---|
|MEAN_LOGIT|61.8074%|37.8704%|False|
|UNIT_MARGIN|61.9704%|37.9815%|False|
|MAJORITY|61.6889%|37.6852%|False|

源选择：None。

0训练/0拟合/0校准，复用相同15个source fixedfinal30 native组成单primaryensemble种子的5RX留出，先固定81000新source决策再评分。MEAN_LOGIT几何共识；UNIT_MARGIN每packet每模型中心化6classlogits并L2规范化；MAJORITY单packet三模型预测票+0.001固定probmean破tie，非query真实批次数/配额。全部类同公式、query独立、不update父模型或query。
与原固定probmean61.7778%对照，预登记+1pp、4/5非负、worst非劣1pp；完整15后合格最高均值才正式三primary已冻结9权重自动7views独立truthlast，不根据target成绩选择；84尚未达成。

独立核对 VERIFIED：完整15row及81000条source实际物理IDs/原始决策混淆重算；worker stdout 15项、无技术错误；0训练/优化/校准、U/target关闭。原15个source父模型完整99900步、真实配置/初始化/fitL排除heldRX及source预测核对沿用同row原报告。
