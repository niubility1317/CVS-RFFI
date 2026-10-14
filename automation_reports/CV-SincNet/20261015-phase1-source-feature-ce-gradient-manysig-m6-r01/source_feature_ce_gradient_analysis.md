# 同L强图身份与特征梯度：完整六模型源诊断

状态 **ANALYZED / VERIFIED**。结果支持先检查教师/学生的相对可靠性，**不支持把所有特征一致性梯度都说成有害**：教师对、学生错组的梯度余弦18/18为正；教师错、学生对组16/18为负，但该组每模型只有3至8个seen-L样本。18是6模型×3参数对象的计数，不是18个独立实验。不能由此推断实际U正确率或解释目标负结果的因果来源。

## 固定边界与真实执行

run 20261015-phase1-source-feature-ce-gradient-manysig-m6-r01；代码 da1aafdde1931fcef67ecd509e079cfa371ec166；执行 2fb6cf1ee4cdbf830f3921a19ef2346c4f1de185。固定SIG_TIME ZERO与FEATURE_MT各3预定义seed2026092701/02/03，共6个已完成scratch50/33+17/11100模型及各自真实ownEMA；父run 20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01、20261015-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s0203-m6-r01，无目标择优。source_proof核对实际数据角色/physicalIDs/终轮选择/继承，EMA从同CP严格加载epoch50/11100。没有训练初始化、另训头或替换头。

仅合法L6300，源split392005/RX1/3/4/6/8/day1/2/3、TX集合和NoExplicitCFO保持。每模型相同shuffle2026101501、strong/noise/dropout2026101502、batch256末156；teacher eval/stopgrad，student train复用原strong noise.015/dropout，拒绝BN。模型train只是只读测量模式，不执行优化器/EMA更新。原default cosine头作为固定checkpoint函数计算预测和fθ的CE导数，全部参数buffer前后逐值相同，.grad未累积，头/optimizer/model更新均0。

在同物理L强图上测trueCE smoothing.01、原half squared L2单位joint160特征一致性，以及固定诊断系数.1的导数。ZERO也评估.1诊断方向，但其原训练feature系数仍0；这不是实际更新轨迹的回放。四个对错组的trueCE与feature贡献均用完整batch分母，逐项相加恢复总梯度；只测t1PW3456、t2PW6912、SIG projection576，不是全部195041个feature参数的联合方向。原feature与导出边界仍沿用此前完整训练检查，无新模块。

## 完整分组计数

|条件/seed|教师对学生对|教师对学生错|教师错学生对|教师错学生错|
|---|---:|---:|---:|---:|
|SIG_TIME/2026092701|6259|25|8|8|
|FEATURE_MT/2026092701|6259|23|7|11|
|SIG_TIME/2026092702|6248|33|3|16|
|FEATURE_MT/2026092702|6250|32|3|15|
|SIG_TIME/2026092703|6257|32|3|8|
|FEATURE_MT/2026092703|6255|33|4|8|

每行6300样本；全体来自同一训练用L，不能把模型重复测量当成6套独立数据。教师和学生都对占绝大多数；错误教师/正确学生组很小。新代码保留空组的0向量和undefined cosine，本次全局四组均有样本，但部分batch为空组。标签只用于合法L真实CE及事后分组，不能将这些真值分组用于U训练门控。

## 梯度证据与限制

全L包数加权平均向量的余弦如下；范数比和全部批次/分组数据见results.json和source_gradient_summary.json。

|条件/seed|t1 PW|t2 PW|SIG投影|
|---|---:|---:|---:|
|SIG_TIME/2026092701|-0.078031|0.022286|-0.310699|
|FEATURE_MT/2026092701|-0.086376|0.068700|-0.340645|
|SIG_TIME/2026092702|-0.232287|-0.139442|0.082239|
|FEATURE_MT/2026092702|0.198093|0.136923|-0.051986|
|SIG_TIME/2026092703|-0.036496|-0.082814|0.100711|
|FEATURE_MT/2026092703|0.038576|-0.004426|0.138003|

整体方向跨模型/层位有正有负，不能用“普遍冲突”解释此前均值教师下降。更具体的已测模式：教师对、学生错组，两臂三个seed的三个对象均同向（18/18）；教师错、学生对组，SIG的t1/t2各3/3为负、投影2/3，FEATURE的t1/投影各3/3、t2为2/3，总16/18为负。教师/学生都错组符号也不统一。参见same_graph_source_gradient_relations.png。

这些是固定final50、同L强增强、一次固定随机视图的局部导数。L已用于训练，错误组仅3至8个样本，label smoothing、当前默认W、梯度平均抵消和模型阶段都影响数值。范数比不是贡献比例，负余弦不是实际AdamW累积冲突，更不是实际U或目标域的因果证据。不能据这些数放大权重、删掉全部一致性或宣称新TX能力。此前完整默认头三seed均值教师负结果仍为82.4359% vs82.6806%，本只读诊断未改变性能，也未完成84%目标。

## 独立复算、成本与产物

全部6行自然完成、原CPU进程已退出，failure0。独立第二进程重开全部保存PT，逐batch比较11种梯度向量的统计和按包数加权的全L平均，重计150 gradient batches、450层记录、4950向量，source分组总数37800观察（6300物理L×6模型），均VERIFIED。对应75600源包forward观察/300 forward batches；不含预检合成IQ前向。模型计时合计122.91秒CPU2threads，不含发布/数据物化/独立复算；不当作训练速度或GPU部署结论。

V27000被共用loader物化后丢弃，V forward/gradient0；U返回None，不读IQ/TX/IDs；query IQ/truth/统计0。没有目标预测/scorer或头拟合，源诊断显式只做source-only。原始PT和完整resolved_config保留远端PROJECT/runs/20261015-phase1-source-feature-ce-gradient-manysig-m6-r01/<row>；Git聚合results.json省略重复父source_config并标明public_config_view，原始完整配置仍在immutable registration/resolved_config，未改远端产物。源码复现入口见report.md/experiment.json。

局部13负例与实际SIG合成梯度、生成CPU入口compile、一次独立P0/P1、普通账户preflight、单次archive SHA对比、remotecompile、真实CP无query smoke、实际PID/CWD/argv/CPU环境/loggrowth均通过，没有新增seal/receipt或重复审查。新训练epochs0，现有每骨干总预算仍50。

## 下一步最多两项

1. 完成当前健康运行的RAW/TX_DAY_CENTERED固定02/03泛化矩阵，保持系数、默认头、轮次和逐行独立评分；首seed正向信号不代替完整3seed。
2. 下一源问题是能否用真值不可见时可观察的量区分可靠教师方向。只提出一个固定零置信差规则作为待证伪对象：teacher弱默认头置信度≥student强置信度时保留一致性；先用合法L及无梯度V核查对两类不一致组的保留情况，不拟合头、不扫阈值，不解封U。当前尚未实现/登记/启动，不能把源真值对错当U oracle，也没有据此发布新训练候选。
