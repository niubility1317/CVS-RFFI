# F 原始监督份额：RX1 三组 E20 结果

源资格 **FAIL**。三组 scratch E20 均已完成，目标访问 0。

|模型|held RX1 accuracy|macroF1|seen V均值|seen V最差RX|
|---|---:|---:|---:|---:|
|A0|81.148148%|80.603394%|98.041667%|95.537037%|
|A1|85.814815%|85.689668%|98.000000%|95.592593%|
|A2|85.185185%|85.157960%|98.013889%|95.574074%|

A2−A0 accuracy +4.037037pp、macroF1 +4.554566pp；A2−A1 accuracy -0.629630pp、macroF1 -0.531708pp。

逐项判定：{'vs_A0_accuracy': True, 'vs_A0_macroF1': True, 'seen_health': True, 'A1_accuracy_retained': False, 'A1_macroF1_retained': False, 'complete_health': True}。同物理ID修复/破坏：{'A2_vs_A0': {'repaired': 314, 'broken': 96}, 'A2_vs_A1': {'repaired': 63, 'broken': 97}}。完整60epoch/13320step/nativeCSV/fullstdout已解析；5994条实际L/U批次三组逐条匹配；F每行4440次物理clip/单位映射全部保存并检查。

新60模型轮/13320更新，真实完整网络前向10639854，墙钟1157.648s；无历史权重成本，本地合成检查另列。默认头/三投影/q均按原联合训练更新，未拟合额外头。

只有一个已使用过的源RX折、一个modelseed；不等于正式clean、urban或跨seed证据。固定E20用尽不等于收敛。FAIL关闭本候选；PASS只允许另登记固定其余四RX，不能直接进入target。

逐TX/day负结果与完整曲线均保存。图：complete_training_curves.png、held_source_strata.png。正式practical和received IQ均未改动；clean/urban双突破仍未证明。


实际运行与科学状态：**VERIFIED / FAIL_SOURCE_RX1_ONLY**。实际训练代码44fc0bb689d03a65dc18a897b0e8e28324c7958a；三worker和dispatcher自然退出，无failure。单archive transfer/remotecompile、真实scratchCP no-query smoke、PID/CWD/argv/GPU/log增长已独立保存。release未热改，不重跑或覆盖任何旧研究。

基座为用户指定F：SIG_TIME_X3/CENTER_25K/CENTER_UNITS，900722参数、25MHz/IQ256/equalized1/原nativeRMS；NoExplicitCFO、原内部补偿/joint160/defaultcos联合训练保持。复用固定数学实现，不继承历史E50/RAW_HZ权重或状态。三行同scratchE20=13L+7U，全部4440更新，L每物理样本四前向、两个同IQ的RD对各.5；没有跨raw/G一致性、As、额外头/部署分支/新U规则。G是原ordinary固定研究视图，仍包含正常PA，未预称所有硬件扰动标签保持。

源契约：RX3/4/6/8 L5040/U45360/seenV21600；held源RX1 L/U不用，原V5400仅FINAL。U真值始终隐藏，V不更新状态。每行6TX×4RX×3day完整曝光、三行全部实际L/U批次序列逐条匹配。实际nativeargs只差candidate/output/use_aug；包装resolved顶层33/17是历史描述，实际nativeargs13/7与全20epoch/4440成功更新证明预算，没有延长E50。

成本：新增60model-epoch/13320updates，真实完整网络前向10639854，矩阵墙钟1157.648408s。所有L/studentU/EMA/sourceeval(含初始smoke2)/heldV分项见measured_costs.csv；历史失败与F E50成本不计作本轮父权重。本地累计3584合成网络前向/512增强输入/12合成optimizer更新另列，0localrealIQ/CP，包含512前向后失败的测试fixture；另一次包路径fixture0前向，两者修正不改实际训练数学。CPU分析/绘图未统一计时，预算耗尽不等于收敛。

完整60epoch/13320step/nativeCSV/fullstdout均已解析，有限关键loss/gradient、无跳步/U真值、两对平均CE/KL尺度匹配。F每行4440条真实优化器映射与physicalclip按step核对，默认W/gain、三路局部投影及两个q坐标有实际运动；这些机制证据不替代识别收益。未运行probe_dom_acc的空值是原占位字段，未拟合诊断头。

|源分层|A0准确率|A1准确率|A2准确率|A2−A0/pp|A2−A1/pp|
|---|---:|---:|---:|---:|---:|
|TX0|96.2222%|89.7778%|89.1111%|-7.1111|-0.6667|
|TX1|86.0000%|86.0000%|85.5556%|-0.4444|-0.4444|
|TX2|56.7778%|76.0000%|79.8889%|+23.1111|+3.8889|
|TX3|65.5556%|65.5556%|65.8889%|+0.3333|+0.3333|
|TX4|100.0000%|100.0000%|100.0000%|+0.0000|+0.0000|
|TX5|82.3333%|97.5556%|90.6667%|+8.3333|-6.8889|
|day1|80.0556%|87.0556%|85.5556%|+5.5000|-1.5000|
|day2|88.2222%|92.4444%|94.0000%|+5.7778|+1.5556|
|day3|75.1667%|77.9444%|76.0000%|+0.8333|-1.9444|

只有一个已使用过的源RX折和一个modelseed，不能替代完整跨RX或正式clean/practical证据。F由用户在已暴露A–F历史基准后指定，不声称新的独立盲测；本轮固定FINAL20，新目标分数不参与阈值、选模、调参或晋级。失败关闭raw-anchor候选，不换主行/强度/RX/seed救援；通过只允许另登记固定其余四RX，不能直接进target。clean净增益、lowurban>60%、suburban保持与跨seed均未证明，goal保持active。

复现训练入口是experiment.json的immutable CWD/Python运行experiments/cvs_raw_anchor20/runner.py；已完成root禁止重复启动。离线复算不读IQ/CP：F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_raw_anchor20.harvest --output E:/type10-7/local_artifacts/cvs_F_raw_anchor20_20261016/final --state E:/type10-7/local_artifacts/cvs_F_raw_anchor20_20261016/terminal_readback.json --cached。

实际产物：

- [完整三组训练曲线](E:/type10-7/local_artifacts/cvs_F_raw_anchor20_20261016/final/complete_training_curves.png)
- [逐TX/day默认头结果](E:/type10-7/local_artifacts/cvs_F_raw_anchor20_20261016/final/held_source_strata.png)
- [全部分层数值](E:/type10-7/local_artifacts/cvs_F_raw_anchor20_20261016/final/source_strata.csv)
- [完整训练数值](E:/type10-7/local_artifacts/cvs_F_raw_anchor20_20261016/final/training_curves.csv)
- [实际成本分项](E:/type10-7/local_artifacts/cvs_F_raw_anchor20_20261016/final/measured_costs.csv)

## 收益归因与负结果

A2相对A0修复314个、破坏96个判决，净修复218/5400，得到+4.0370pp；相对A1修复63个、破坏97个，净损失34/5400。A2相对A1的TX2提高3.8889pp、TX3提高0.3333pp，但TX5下降6.8889pp，TX0/1分别下降0.6667/0.4444pp，TX4持平。day2提高1.5556pp，day1/3分别下降1.5000/1.9444pp。它改变了错误分布，没有形成对增强控制的整体净收益。不能由这些TX分层直接推断PA、多径或其他某个物理算子是原因，更不能设置TX专属规则。

A2的seenV均值只比A0低0.027778pp，健康保持条件通过；失败来自A1保持条件：accuracy下降0.629630pp（比预定0.5pp上限多0.129630pp），macroF1下降0.531708pp（多0.031708pp）。这是预定标准下的近阈值失败，单fold/单seed不能证明稳定劣势；同样不能事后放宽阈值或补换seed将它改判为PASS。本候选关闭，不扩其余四折、正式target、混合As或更多seed，不切换A1为新主行。

完整曲线已查看：三组监督CE/KL早期下降、后期大体平台，普通增强两组仍有波动；physical grad_before_clip有限，E14按计划启用旧U，LR相同。A0的训练CE/KL更低却held RX1最差，A2的CE/KL低于A1也没有胜过A1；训练损失不能替代泛化收益。U选择量只作执行机制记录，没有TX真值，因此不报告伪标签准确率。没有由预算结束推断全局收敛。

下一步设计边界：固定F/E20继续，但不把本近阈值结果改名晋级，也不沿raw比例调参救援。现有证据只支持“普通IQ扰动在这个F源折具有净收益，固定一半raw监督未保留其全部收益”；尚不能指定其身份保持算子或声称Ag已跨RX稳定。新的联合方案必须给出独立的源域机制依据和匹配raw/G/As控制，保持原始监督份额与实际前向预算可比较；在此之前，完整clean/practical四因子实验保持未启动。本轮已完成一个可执行的独立训练组织否证，不宣称星地双突破。
