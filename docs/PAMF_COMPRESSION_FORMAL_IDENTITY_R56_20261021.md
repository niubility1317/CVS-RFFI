# R56：R53—R55 七个新压缩版的固定无 F 身份关联

状态：LOCAL_IMPLEMENTED，未发布。22项聚焦检查、CLIhelp和一次独立P0/P1审查通过。主训练仍为指定350521父的压缩优化；本诊断不更改父训练、预测、权重或选择。

固定纳入R53三行、R54两行、R55两行，不按正式分数选择。全部三个父run完成各自七景正式预测和独立评分后，才读取已经保存的恢复IQ；不重新加载恢复器，不新增神经预测、分类头、训练或校准。CPU2、空CUDA，使用原共享score.lock；不阻塞或改变父版测试和交付。

坐标完全复用R34/R39/R47/R50的mirror64、spectral4、reference24、combined92，以及源LU8192已有均值/尺度和std下限0.05。复用R50的raw、指定350521父、R49同宽retain控制84特征单元，新算七版196单元，共280单元；新增8232000次固定解析变换，不访问F权重、logits或embedding。首步真实2Ls解析noquery smoke，无checkpoint阶段不加载checkpoint。

全部280特征、完整168000物理ID和calibration引用核实并固定后，独立truth-side才打开原测试真值，统计6TX、7RX、4day。clean参照仍为事后评价用的跨RX且跨day样本：每TX/RX/day按opaque物理ID至多取2条，完整6TX单元等权。固定199次RX/day内query标签置换、参照标签不动；不把条件对照当独立p值。

输出4536组同TX/异TX余弦、margin、正margin比例、有效/缺失/零参照和配对描述误差，756条件对照，392配对差。七版都与指定父比；R53retain_anchor另与R49同宽控制比；R53prune_anchor、R54两版、R55两版与R53prune_decay比；R55 full/half互比。结果不回流研发、父选择、宽度、损失或重跑。

该固定描述符在clean TX2已有区分不足，必须逐TX保留，不能把恢复后负margin单独当作错误身份。该诊断只提供已暴露工程基准上的同TX关联与负控制证据，不证明每个生成细节属于输入TX，不替代F50七景准确率、原1pp容差、70%/5pp目标或新独立盲测。

现有READY/EXACT_MATCH、原7/63/30和物理ID不变；数据、特征、权重、原始日志不提交。新不可覆盖run/release/output、普通账户、单归档一次SHA/compile及实际PID/CWD/argv/空CUDA/日志增长核实。错误路径/祖先/ID、父技术失败、非有限/缺失特征或输出冲突保留产物、无自动重试；不干预健康父任务。额外审批/签名/成员哈希/重复review为REJECTED_EXTRA_GATE，不等待。

精确方法commit、矩阵、命令、环境/CWD、输入/输出、预算和停止规则在experiment.json预登记。
