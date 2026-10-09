# reference_response：跨TX共同位移补偿×分支贡献预测，CE拼接四seed消融

- run_id：`20261008-phase1-receiver-residual-manysig-m16-r01`
- group_id：`cvs-cross-tx-displacement-branch-contribution`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

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

## 发布与初始化问题

远端独立读回VERIFIED：执行commit为`3712e8e4f004f21b8ffdc74ce0459ad8a2020f3d`，dispatcher PID1598058；首批四行在GPU4/5/6/7正常运行，sourcecounts6300/56700/27000、每epoch50步。远端CPU smoke通过。发布客户端因远端smoke和submit均打印JSON而解析失败，但远端已经提交；独立核实后未重复启动，本地修复stdout处理。

后续发现R1新增头初始化的`torch.manual_seed`会重置CUDA dropout RNG，四组并非完全配对的随机性控制。R1保留为`DIAGNOSTIC_RNG_CONFOUNDED`，不能据此单独宣称新增机制收益。数据角色、scratch来源及truth-last边界未改变。健康任务不停止、不热改。

正式修复矩阵另登记为`20261008-phase1-receiver-residual-manysig-m16-r02`，使用独立release和输出。只改CPU私有生成器，从零执行同一16行设计，不继承R1权重，设计在R1任何target读取前固定。CPU/CUDA RNG检查与CUDA全机制smoke通过。

## 结果与覆盖

R1已启动，完整训练和测试仍未完成；无识别收益结论。原自动收尾保持，结果需带随机性混杂声明。固定全部16源模型冻结后，自动预测clean及六个完整practical视图，每视图168000相同物理ID。所有预测完成后独立scorer读truth并重算混淆矩阵/Accuracy/F1。Phase2、K、新类和H均不适用。历史benchmark已暴露，不作新盲测或正式LEO分层声明。所有负结果保留，无目标反馈调参或选择性重跑。

## 交接

已完成独立实现、固定16行预登记、本地数值/契约验证及单次独立审查。下一步：提交并push，远端CPU验证后单owner发布，核对PID/CWD/argv/GPU与日志增长，再等待训练、全矩阵冻结、全量预测与独立评分。恢复先读本run的独立readback，不重复提交。

## 每卡4进程调度覆盖

用户新增指示“每张卡4个实验进程”，将本轮R1/R2总任务上限从每卡2个覆盖为每卡4个，包括预CUDA预约。原启动记录和策略保留为历史。新控制release只替换本轮调度父进程，按PID/start_ticks/CWD/argv接管训练子进程；已完成行跳过，失败行保留且不自动重跑。每run单launch owner，两个调度器共享GPU预约锁。训练模型/config及预算继续使用原不可变release。source矩阵全部冻结后再预测，全部预测完成后truth-last独立评分。此处为预登记，实际接管和占用须由独立读回确认。


首次容量发布部分完成：R1已接管并启动12行；R2旧dispatcher精确SIGTERM后已独立absent，但/proc退出瞬间environ权限读取竞态导致发布器中断。训练worker未signal。恢复先核实已完成部分，控制release r02修复退出检查并接管剩余状态，不重发原提交、不重跑训练。


## 容量接管VERIFIED

独立/proc身份与nvidia-smi、完成产物核实：新dispatcher PID1641984，控制commit`42c657c3757b61e7a9fb8c71ccd8d9e8c433c9d7`；模型运行commit仍为`3712e8e4f004f21b8ffdc74ce0459ad8a2020f3d`。保留12个活跃训练PID/start_ticks/CWD/argv；首seed4行已自然完成E200，未重跑。目前12行在训、0行排队，无失败。

GPU0至3各3个总实验进程，GPU4至7各4个；其中本轮R1/R2共24个，另4个既有任务。每卡已设最多4个，含预CUDA预约；显存余量约19至21GB。固定矩阵剩余任务已全部启动，因此不补开额外实验。证据：[capacity4_verified.json](evidence/capacity4_verified.json)。

新调度器使用专用handoff锁及全生命周期owner锁；跨run共享GPU预约锁。原训练PID不signal，不热改模型，不改变预算/损失/数据角色。完成16源模型后继续原冻结→112预测→独立truth-last评分。当前未完成目标测试，尚无新增机制收益结论。

## 完整测试已独立交付

原训练控制器FAILED保留，16个ownE200模型源训练完整。其全部固定测试在预测run `20261009-phase1-receiver-residual-test-manysig-m16-r03` 与评分run `20261009-phase1-receiver-residual-score-manysig-m16-r05` 完成，结果VERIFIED。[完整结果](../20261009-phase1-receiver-residual-score-manysig-m16-r05/results/full_results.md)。原R1混杂处置不变。
