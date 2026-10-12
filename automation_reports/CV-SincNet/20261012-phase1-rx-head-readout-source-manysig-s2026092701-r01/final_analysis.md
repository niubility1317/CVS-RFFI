# 已冻结RX分类器：零训练读出源检验

|读出|RX模型源均值|RX混合源均值|Random混合源均值|资格|
|---|---:|---:|---:|---|
|GLOBAL_ONLY|62.3000%|62.3741%|61.3185%|False|
|ENTROPY_GATE|62.2481%|62.3630%|61.3704%|False|
|CONFIDENT_HEAD|62.1037%|62.4704%|61.3370%|False|

固定源选择：None。

0训练/0优化/0校准，复用10个相同fitL5040/heldV5400 fixedfinal30 native4head父模型，真实物理角色/fit IDs/配置/继承严格核对。统一源RX训练相对随机分组已有+0.9778pp混合收益，本次只区分读出；新父模型不因成绩重新训练。
GLOBAL_ONLY仅原global；ENTROPY_GATE各专家熵softmax(-4H)作packet内共同权重再与global等分；CONFIDENT_HEAD最大maxprob专家再与global等分。规则和常数结果前固定，全部类同公式，无RX或querytruth/metadata/真实批次数/配额/全局重排，不更新query或parents。
新readout再与native3固定0.5/0.5，source预测和混合先保存再source评分。完整15row后仅合格读出按source均值最高冻结；相对native3+1pp、random+0.5pp，各4/5非负，worst非劣1pp。所有负结果保留，无target，84仍待验证。

独立核对 VERIFIED：全部15row×2parents×5400=162000条源读出概率及162000条固定混合的实际物理IDs/混淆重算；完整worker stdout 15项、无技术报错；0新训练/优化/fit、U/target关闭。父模型原完整66600步及四个专家实际更新核对沿用RXHeads同row报告，加载前strictprovenance不放宽。
