# 类1和类3的source准确率

本报告按用户最新要求仅选取B1_TAIL_LR。source指原源域验证集V，不是L_s训练集。冻结E200模型，在原source RX/day、划分与评估batch配置下重新评估；每场景27000条、每类4500条，采用原评估器、原LEO随机seed。准确率单位%。

|实验|类1 source clean|类1 source LEO均值|类3 source clean|类3 source LEO均值|
|---|---:|---:|---:|---:|
|B1_TAIL_LR|98.31|92.34|96.98|86.75|

|场景|类1准确率%|类3准确率%|
|---|---:|---:|
|clean|98.31|96.98|
|leo_clear_weak|93.78|88.80|
|leo_low_elev_weak|91.36|86.09|
|leo_rain_weak|91.89|85.36|

全部模型评估前后state_dict逐张量一致，未创建optimizer或更新参数，未构造目标loader。所有逐类correct/total与总体计数一致。原日志只保留总体指标，故本表为新增冻结评估数据。与原存档总体correct最大差异为1条/27000（0.00370个百分点）；所有差异逐场景保存在replay_deltas.csv。观察到小量重算差异，未将其强行归零，也未断言唯一数值原因。

第一次检查要求总体correct完全相同，在B0 LEO-clear相差1条时退出；第二次仍要求clean完全一致，在B1 clean相差1条时退出。两次只读评估日志保留。最终评估将新旧统计差异完整记录，以实际计数给出逐类结果；不把逐bit分数一致增设为读取源域准确率的条件。

最终脚本提交e85e8018ef6df174fb45f11aff4d9de9bb93a7d6，发布releases/a1_source_classes_e85e8018，单文件SHA256=0b9b1f905c3bfc1b302cda5fb91509867f8753879b64fdbafb748b49459ede97；使用空闲GPU2顺序完成8行，未停止任何原训练任务。

完整四场景×六类计数见all_source_class_counts.csv；source与target的类1/类3均值并列见class1_class3_summary.csv。数据为source V准确率，不代表target泛化性能。
