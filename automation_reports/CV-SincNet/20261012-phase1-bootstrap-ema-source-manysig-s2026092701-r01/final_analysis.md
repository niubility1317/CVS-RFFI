# 固定源集成向EMA移交：源端结果

每个候选5个RX完整留出、单模型种子、从零固定30轮；heldRX同时排除L/U及teacher输入。

|方案|源V均值|最差RX|
|---|---:|---:|
|CE30|61.5222%|38.9815%|
|SELF_EMA|61.8074%|39.6481%|
|BOOTSTRAP_EMA|61.7185%|39.4074%|

预登记源资格：False。

随机学生30轮，EMA每次student更新后更新，前5轮只L；第6轮开始U，历史2pass且confidence0.90，noise0.015、KL0.5五轮ramp。bootstrap固定3个原fitRX纯CE30教师软概率权重0.8从step1111线性移交到step4440的0，后10轮纯当前学生EMA。final30 student预测，未挑EMA或epoch。
新增10模型66600更新，bootstrap独立继承源teacher90轮每RX的历史成本及U预测缓存成本，控制不加载旧教师。U真值隐藏；不访问target，不称84%已达成或新盲测。
机制采用权重EMA，依据Tarvainen与Valpola Mean Teacher (2017)，本次handoff为待证假设：[原论文](https://arxiv.org/abs/1703.01780)。

独立核对 VERIFIED：10模型完整66,600条step、30轮epoch均值/EMA更新及U消费计数；54,000条heldV原始预测及混淆；actual fitL/U IDs排除heldRX，真实匹配teacher ancestry/配置/物理角色；bootstrap概率缓存仅fitU，移交权重在step4440后精确为0；0 target、U真值未读。

|方案|首轮CE均值|末轮CE均值|末轮KL均值|末轮每批接受U均值|
|---|---:|---:|---:|---:|
|SELF_EMA|1.398561|0.018673|0.017720|125.3991|
|BOOTSTRAP_EMA|1.398469|0.018287|0.018074|125.4072|
