# reference_response：跨TX共同位移补偿×分支贡献预测，CE拼接四seed消融

- run_id：`20261008-phase1-receiver-residual-manysig-m16-r01`
- group_id：`cvs-cross-tx-displacement-branch-contribution`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定强骨干与最基础CE+原生clean/satellite拼接；四组baseline/displacement/contribution/combined，从零训练，固定E200，全部测试并保留负结果。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 实现与局部验证

设计与固定系数见[DESIGN.md](../../../experiments/cvs_receiver_residual/DESIGN.md)。新增两条主线共四组、四seed，基础均为纯CE+原生拼接，无PL/EMA/MixStyle/GRL/正交/旧域骨干。E200×50=10000次更新，不沿用U训练器的44400次更新。E41至60线性启用残差，幅度上限25%。所有组执行相同的E40至200源L只读统计前向，成本单独记账。

CPU数值验证通过：初始骨干权重/RNG及logits逐位一致；三个实际分支贡献之和一致；持有TX不进入位移目标；辅助梯度有限且非零；单包推理不依赖批次；checkpoint严格往返；source拒绝target/truth。新增参数分别为0、5578、2173、7751。源配置登记检查、compile、source CLI及远端载荷语法检查通过。

按项目最小流程完成一次独立P0/P1审查，当前无未解决阻断项。发布解释器问题已修复：控制载荷显式使用CVS-RFFI Python，只读probe核实Torch2.1.0+cu121。审查提醒：batch零均值惩罚含单元采样噪声，不证明物理可辨识性。

## 结果与覆盖

当前尚未启动、无识别收益结论。固定全部16源模型冻结后，自动预测clean及六个完整practical视图，每视图168000相同物理ID。所有预测完成后独立scorer读truth并重算混淆矩阵/Accuracy/F1。Phase2、K、新类和H均不适用。历史benchmark已暴露，不作新盲测或正式LEO分层声明。所有负结果保留，无目标反馈调参或选择性重跑。

## 交接

已完成独立实现、固定16行预登记、本地数值/契约验证及单次独立审查。下一步：提交并push，远端CPU验证后单owner发布，核对PID/CWD/argv/GPU与日志增长，再等待训练、全矩阵冻结、全量预测与独立评分。恢复先读本run的独立readback，不重复提交。
