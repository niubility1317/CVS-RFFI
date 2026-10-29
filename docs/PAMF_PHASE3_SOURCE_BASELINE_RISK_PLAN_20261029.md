# Phase3 原模型参照的已知类风险校准

2026-10-29。独立候选 `20261029-phase3-source-baseline-risk-s2026092701-r01`。上一固定E51与跨分支读出组合已完成，监督／OE编码器的MLP未知宏拒识86.3492%／90.7619%，均NO_PROMOTION。本候选只改变源已知风险的计算参照，不重训或追改上轮结论。

固定上一组合两行全部编码器、L-only几何及200步linear／MLP读出；历史E50原11100＋222步、OE编码器额外21设备成员监督、读出原4设备OE4200监督均保留身份。本轮encoder／原六类head／readout更新、新F前向全部0。先在两条合法L特征上做冻结读出smoke，再读取Vknown和按物理ID独立对齐的原F6同IQ闭集预测。

此前校准约束newF自身正确样本拒绝损失≤1.5pp、宏FRR≤4%，为encoder闭集损伤预留.5pp。新候选对目标中原本的旧类保护直接计账：总体及每sourceRX，原F6闭集宏准确率减newF含拒识宏准确率≤2pp，同时宏FRR≤5%。每方法仍只冻结一个全局最大合法scalar阈值，`score>=threshold`接受，整组ties不可拆。校准只使用已知V标签、两模型已知预测、已知分数与RX；不使用novel分数、标签、TX或target。编码器闭集损伤≤.5pp的评分约束继续保持。本候选不会改变上轮门槛或将旧结果重新判成成功。

全部阈值文件及校准complete先写完，才读取novel特征生成30150×7个预测。保持原完整六类默认头决定接受身份，九字段预测固定后独立truth-last scorer连接未知TX。onlyMLP可推进，linear和五静态分数诊断；总体及每RX的95%未知、5%FRR、.5pp编码器损伤、2pp总损伤仍须全部通过。完整两行后判定，若两MLP均通过，优先监督编码器以减少历史OE训练成本。

复用同V和novel3属于重复source开发，既非新盲测，也非未见RX／星地／无标签适应／匿名注册成功。即使通过仍先独立source确认，正式RX24 U/Q不读。本候选两行全CPU，GPU新增0，不停止foreign。新root、一次P0/P1、focused tests、实际code/config提交、N607一次preflight／单归档对比／compile及PID/CWD/argv/log读回依最小流程完成，科学未达正常评分归档，不回流调阈值或重跑。
