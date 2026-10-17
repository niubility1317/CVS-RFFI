# 完整 clean 保留配方：默认 EMA 的源域资格结果

状态：ANALYZED／VERIFIED。两个固定端点均完成源域预测并通过运行前固定的资格条件；整体联合目标仍 ACTIVE。此实验没有训练。

|固定端点|EMA clean 准确率 %|置信覆盖 %|覆盖内正确率 %|q 加权正确率 %|low 教学机会 draw0 %|draw1 %|资格|
|---|---:|---:|---:|---:|---:|---:|---|
|E20|97.778|89.000|100.000|100.000|38.346|39.565|PASS|
|E30|98.056|92.556|99.880|99.940|37.981|39.655|PASS|

“教学机会”是同一合法源 V 物理包上，clean EMA 默认头正确而 urban 学生默认头错误的 q 加权比例；分母是 q 权重和，不是全部 U 或目标样本数。源 V1800 只代表这一固定已使用的源面板，不是独立目标盲测；E20／E30 是同一 seed 的两个顺序端点。U IQ 与 TX 真值均未读取，不能据该结果宣称 U 伪标签真实精度。

每个 RX 的置信预测正确率：E20 全部100%；E30 RX3为99.310%，其余100%。mid urban 教学机会分别 E20 draw0/1=24.958%/22.827%，E30=24.099%/23.912%。本面板中 q 加权 teacher错误/student正确机会为0；不能据此推断其他数据没有错误。

原参考学生源 clean 准确率 E20=97.833%、E30=98.056%。所有输入复用既有完整 practical source 缓存，2draw×low_suburban/mid_urban/low_urban，0新信道调用；正式测试IQ、SNR和边界没有改变。本实验不产生 target 成绩，不替代训练后七环境评价。

6 项聚焦检查和编译 PASS；一次独立 P0/P1 审查 PASS。真实 student/ownEMA 严格加载后，在任何源 IQ 打开前先通过零query smoke；学生12602NN/端点、EMA1802NN/端点，共28808NN。参数900722、state206、原cos6×160/scale30保持，全程冻结、0优化更新、0另训头、0新增teacher训练。历史源成本自身20+10=30E/6660updates，只用于来源披露，不是新候选训练初始化。独立重算全部1800×两个端点预测、16个混淆矩阵和12个scene/draw记录，判定一致。

远端普通uid1000，CPU2threads，CUDA_VISIBLE_DEVICES空。实际PID762019、start_ticks150283570、CWD及argv与预登记一致；stdout从0增长至2812字节、stderr0；完成后PID自然退出、complete与全部源预测存在。运行墙钟19.684299s。执行代码身份c2b56188d1728c6cb189fe65c242b5a627ee3424，GitHub远端已独立读回一致；一次release归档比较和远端compile PASS。

产物：E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_source_20261018/source_r01；独立重算与CSV：E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_source_20261018/recount_r01/source_qualification.csv；16个混淆矩阵同目录confusion_matrices.json。真实checkpoint、训练日志及源IQ仍留原N607；导出只含本诊断源预测及小型元数据，源V标签合法用于冻结评估。无新训练曲线或新图，不能虚构路径。

结论：可信 clean 默认头和 urban 错误之间存在足够源域教学机会，支持进一步独立检验 C4，但没有证明参数更新后会提高识别率。旧SIG_TIME E50已有SSL正例资格而训练增益很小，不能把教师资格当效果。后继将沿用户选定完整13L17U、cos20→重置cos10、4×128、逐包完整practical/raw、原L监督.75/.25组织，保留原U门控不改；再独立对比额外普通U软监督与额外As(U)软监督，以控制额外前向。未来使用自身在线EMA，旧诊断权重不作初始化。C4训练流水线尚未实现或启动；soft_transfer.py 仅是已经验证的单一算子。

当前唯一待验证假设：可信clean教师对完整As(U)的监督，能带来超过相同成本普通U软监督的urban增量，同时保住原clean。最小设计为完整参考、参考＋额外普通U软监督、参考＋额外As(U)软监督，原有全部5源RX和E30固定；具体注册前不启动。若各源urban抽样不能稳定优于匹配普通U对照，或clean保护失败，则放弃该固定候选，不据目标结果调参或救跑。
