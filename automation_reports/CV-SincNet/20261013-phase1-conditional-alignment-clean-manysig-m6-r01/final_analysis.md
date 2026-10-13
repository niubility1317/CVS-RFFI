# 浅中层TX条件矩：完整三seedclean结果

|机制|三seed均值|SD|≥84%|
|---|---:|---:|---|
|UNCONDITIONAL|79.8954%|0.8989%|False|
|TX_CONDITIONAL|79.9931%|0.8940%|False|

|比较|三seed配对差均值|正向seed|
|---|---:|---:|
|UNCONDITIONAL-UNIFORM_SSL|-0.1359%|1|
|TX_CONDITIONAL-UNIFORM_SSL|-0.0383%|2|
|TX_CONDITIONAL-UNCONDITIONAL|0.0976%|2|

6新模型从零各30轮/6660步、共39960optimizerupdates，全部sourceL6300/U56700隐藏标签；V无前向或性能资格门槛，每rowclean168000预测固定后独立truthlast，完整108条same-row指标。所有模型同普通cos，Phase2可重新拟合W；矩统计仅当前训练Lbatch计算，不保存为query运行状态。
固定time_down[B,48,128]与t2.pool[B,96,32]浅/中层时间均值及populationstd，合法L当前batch detached均值/标准差归一floor1e-4，源TX条件下RX质心平方偏差；UNCONDITIONAL省去TX分组，其余同。固定warm5/ramp10至.1，CE保护源身份类差异，不声称完整未来新TX信息或真实硬件RX因果分解。
sourceU沿已有UNIFORM_SSL同数据/目标/采样/随机性：5RX L26/26/26/25/25 TX/day平衡，U4其他RX32，warm5/ramp5/KL.25/conf95/twopass/noise.015。复用合法已完成UNIFORM_SSL三seed固定prediction作loss匹配控制，不加载它的checkpoint训练，不重跑target；冗余meta梯度诊断不在新方法执行，不冒充相同计算成本。
系数/层位沿此前未启动矩对齐核心固定；本轮不按target错例/分数调参、挑seed或选择性重跑，全部负结果保留。数学动作检验不是跨新RX性能保证，已暴露benchmark不称新盲测。

独立重算 VERIFIED：完整 39960 条更新、180 条 epoch 均值/LR、66600 条浅中层 L 对齐记录及六个真实 checkpoint/物理角色/普通 cos；1008000 个原始 clean 预测和 108 条指标重算一致。源 U 标签隐藏，V 未前向；末轮冻结，无 query 更新。

|row|末 CE|末 L 准确率|U 接受率|末加权对齐|末加权 KL|
|---|---:|---:|---:|---:|---:|
|UNCONDITIONAL-s2026092701|0.028801|99.0815%|86.0705%|0.004787|0.004392|
|UNCONDITIONAL-s2026092702|0.028007|99.1026%|86.0322%|0.005609|0.004972|
|UNCONDITIONAL-s2026092703|0.029881|99.0534%|85.6506%|0.005079|0.004349|
|TX_CONDITIONAL-s2026092701|0.031192|98.9970%|85.9086%|0.025486|0.004662|
|TX_CONDITIONAL-s2026092702|0.031093|98.9935%|85.8933%|0.026349|0.004999|
|TX_CONDITIONAL-s2026092703|0.033008|99.0182%|85.4947%|0.023288|0.004438|

登记修正：原模板 rows.seeds.model 曾残留 2026092701；每行实际 config/resolved/初始化均为对应 2701/2702/2703。本地登记已按实际来源更正，原 release 登记保留，训练配方和运行未修改。
