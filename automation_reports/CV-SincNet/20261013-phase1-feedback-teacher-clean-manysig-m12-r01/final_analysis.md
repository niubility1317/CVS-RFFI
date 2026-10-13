# 教师目标反馈：完整三种子clean结果

|方案|模型角色|三seed均值|SD|≥84%|
|---|---|---:|---:|---|
|TEACHER_SSL|STUDENT|79.9837%|1.1863%|False|
|TEACHER_SSL|TEACHER|79.8524%|1.8178%|False|
|FEEDBACK_TEACHER|STUDENT|79.9466%|1.2438%|False|
|FEEDBACK_TEACHER|TEACHER|79.5931%|2.0344%|False|

6对联合从零轨迹，12模型各30轮/6660更新，共79920optimizerupdates；STUDENT/TEACHER均为普通cos，各168000clean，全部预测固定后独立truthlast。sourceV无前向/性能资格门槛，原L/U/V7/63/30、训练池标签10%，U TX真值隐藏；无目标拟合、旧checkpoint、EMA或星地训练。
学生实际初始seed为2026092701/02/03，教师实际初始seed为对应+10000；教师角色仅诊断，偏移seed均值即使超过84%也不替代学生primary三seed目标。完整报告实际seed及paired_model_seed，不把辅助初始化冒充primary。
该机制是MPL启发的冻结AdamW二阶度量一阶有限差分教师奖励，并非原MPL或精确虚拟AdamW优化。奖励从训练L另RX的梯度产生，中心化/统一RMSfloor/clip2；只调整teacher backbone的辅助梯度，不调整cos权重的辅助梯度。teacher普通CE与student普通CE照常训练共同cos，Phase2可丢弃W重新拟合。
对照同2模型/虚拟前向/梯度开销，仅teacherauxweight0；固定.1ramp/学生KL.25/warm5/ramp5/conf95/twopass/noise.015，无源V或target条件调参。全部负结果保留，不按测试分数重跑/挑seed；基准已暴露，不称新盲测。

独立核对VERIFIED：6联合row全部39960条实际step、79920次双optimizer更新及180条epoch完整均值/LR核对；12实际checkpoints/sourcefit IDs/noCFO/普通cos及actual-init-seed；2016000原始physicalID预测与216条指标/实际角色seed重算。每row全部5550次virtual_restore、11100虚拟前向、5550教师aux梯度和12210教师前向计数核对。teacher偏移seed未冒充primary目标。

|row|student末CE|teacher末CE|student末L准确率|U接受率|奖励RMS中位|teacheraux梯度中位|
|---|---:|---:|---:|---:|---:|---:|
|TEACHER_SSL-s2026092701|0.028371|0.030289|99.1026%|84.7904%|15.460796|2.850927|
|TEACHER_SSL-s2026092702|0.027895|0.031602|99.1132%|84.8438%|16.411863|2.613781|
|TEACHER_SSL-s2026092703|0.030075|0.030634|99.0745%|85.1836%|15.325170|2.537823|
|FEEDBACK_TEACHER-s2026092701|0.028912|0.031747|99.0956%|84.6737%|15.496222|2.770043|
|FEEDBACK_TEACHER-s2026092702|0.028202|0.032531|99.1237%|84.7820%|16.229736|2.553049|
|FEEDBACK_TEACHER-s2026092703|0.030085|0.031164|99.0463%|85.4627%|15.559425|2.389711|
