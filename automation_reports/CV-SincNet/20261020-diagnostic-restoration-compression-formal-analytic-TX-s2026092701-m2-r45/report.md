# R45：两种压缩版的正式无F同TX关联

状态：**VERIFIED / WAITING_PARENT_FORMAL**，已发布。17项聚焦检查、CLIhelp、一次独立P0/P1 PASS；方法commit `a3f1137726f5e691b6466bbd781d411bdb0df62e`。


状态：PLANNED。这是 R44 两种压缩方法的事后保真评价，不是新恢复网络，也不改变 R44 的训练、选模、测试或最终权重。比较用户指定的 R38 dense_half794529、R44低秩530161和重要性剪枝442805，保留原received与clean参考。只有两版全部七场景正式预测和独立评分都完成后，才打开正式 IQ；全部112特征单元冻结、有限且物理ID对齐后，另一个CPU评分进程才连接原truth。

直接复用 R34/R39/R42 的参数自由解析变换、原训练域L/U8192固定均值/标准差（floor0.05）和坐标：mirror64、spectral4、reference24、combined92。combined对三组按sqrt(3×组维数)缩放后拼接，平方误差中三组等权；不重新校准，不根据当前成绩调整坐标。raw和三恢复器共4读出、7视图、4组，共112单元；同168000原物理样本，4704000次解析变换，不是额外独立样本或神经预测。

评分从原clean评价参考每个TX/RX/day按物理ID固定取最多2条，参考RX和day均与当前样本不同，完整六TX参考单元等权。报告同TX余弦减其他五TX均值、配对clean描述量误差、TX/RX/day、有效/缺失参考计数；这是评价统计，不拟合或替换分类头，不生成新的分类决策，不作运行时适应。query标签在RX/day内固定199次置换、参考标签不变，仅作描述性阴性对照，不称p值。沿用原数学，仅重命名读出，3候选×4组×7视图×18分组=1512记录，252阴性对照、84候选配对差全部保留，包括clean本身关联弱或负的组。

本诊断明确使用完成预测后的目标真值和clean评价参考；这些信息不进入恢复器训练、推理或部署状态。恢复器训练0TX/F的权限与本有标签事后统计分别说明。0模型checkpoint、F权重/embedding/logits、额外恢复器预测、神经训练或分类头拟合；不以关联指标筛候选或修改其他健康行。固定基准已暴露，不称新的独立盲测；描述量仍可能含RX、载荷和日期因素，关联改善不等于所有新增细节真实属于输入TX，也不证明未知TX或在轨有效性。

训练和正式预测仍由R44既有owner完成。本诊断N607普通账户、CPU2线程、空CUDA、0GPU，等待R44全部正式评分后，复用R41 score.lock控制最多一个本任务CPU评分。独立新run/release/output；源码Git固定、17项聚焦检查、一次限P0/P1审查，launcher首步用真实两条L_s IQ及固定源校准做参数自由no-query smoke。本阶段不消费checkpoint，无额外checkpoint/F加载。PASS立即等待父正式完成；一次归档SHA/编译及实际PID/CWD/argv/空CUDA/日志增长读回。父失败、输入/物理ID/校准错误、非有限或输出冲突时保留输出，不自动重试、不停止或热改健康任务。

正式场景包含clean、郊区high/mid/low和urban high/mid/low，与R44相同。精确方法commit、命令、路径、环境/CWD、种子、矩阵、停止规则和预期产物见本run的experiment.json。当前没有本诊断结果；其结果不替代七场景识别率、逐景1pp容差或原70%/5pp目标。


精确环境、命令、路径、预算见[登记](experiment.json)。


VERIFIED / WAITING_PARENT_FORMAL：普通账户新release/归档单SHA/compile已核实；真实2Ls参数自由noquery smoke PASS，0checkpoint/F/TX/V/target；实际PID/CWD/argv/空CUDA/2线程及日志增长已读回。等待双方全部正式评分后解析，无统计结果，不改变父训练或预测。
