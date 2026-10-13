# 固定三成员监督对比集成：完整三seedclean结果

|机制|三seed均值|SD|≥84%|
|---|---:|---:|---|
|ALL_TX|81.3282%|0.7829%|False|
|CROSS_RX_TX|81.3935%|0.9426%|False|

|比较|三seed配对差均值|正向seed|
|---|---:|---:|
|ALL_TX ensemble-ALL_TX member0|0.9391%|3|
|CROSS_RX_TX ensemble-CROSS_RX_TX member0|0.8960%|3|
|ALL_TX ensemble-historical CE ensemble (sampling differs)|0.0028%|1|
|CROSS_RX_TX ensemble-historical CE ensemble (sampling differs)|0.0681%|2|

六个三成员集成组，复用六个固定成员0，另十二成员从零各30轮/6660更新，累计540模型轮/119880更新，当前新增79920更新；每组531075参数/3倍单模型成本，各成员均普通cos。sourceU全程TX标签隐藏，V不前向，无性能测试门槛。固定浅中层归一时间mean/std，温度.2，warm5/ramp10至.1。每行固定clean预测后独立truth-last，复用既有UNIFORM_SSL匹配控制原预测。属于条件样本级监督对比扩展，不声称严格接收不变、未来TX完整信息或独立盲测。

独立完整重算 VERIFIED：新增12模型79920条step、360epoch均值/LR、18个真实checkpoint/数据来源/初始化/普通cos、6个真实三成员softmax均值公式；1008000原始预测/108指标一致。六个成员0已在上轮独立审计39960更新；合计119880/540模型轮。原运行末尾汇总缺导入导致NameError，所有row预测评分已完成；恢复只读取既有固定产物，零训练、零预测、不改原输出root。
