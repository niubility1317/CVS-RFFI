# R46 两版恢复器的正式同 TX 关联评价

状态：**VERIFIED / WAITING_PARENT_FORMAL**。固定比较用户指定 R44 group_pruned442805、R46 retain_refine442805 与 prune_refine350521。R46 训练、权重与完整七场景测试保持已发布计划；本诊断只做事后统计，不筛候选、改变损失或影响正式交付。

CPU2线程、空CUDA、0GPU。等待R46两行全部七正式预测和独立truth-last评分完成，才读取保存的新IQ。复用已经完成的R45 raw与group_pruned共56个解析特征单元，在独立新root中建立只读引用；检查原方法commit、完成标记、原source-LU校准与物理ID对应。只为R46两个新版新算56单元、2352000次解析输入；总112单元全部固定、有限且ID对齐后，另一个CPU评分进程才打开原truth。

数学保持R34/R39/R45不变：原训练L/U8192均值/标准差floor0.05、mirror64/spectral4/reference24/combined92。clean评价参考按原物理ID，每TX/RX/day最多2条，参考RX与day均不同，完整六TX单元等权；同TX余弦减其他五TX均值、配对clean描述量误差、TX/RX/day与有效/零向量/缺参考计数。三候选×四组×七场景×18分组=1512记录；199次RX/day内query标签置换、参考标签固定，仅作描述性阴性对照，252组；84候选配对比较全部保留，包括原clean信息弱或负的组。不是新分类器，不拟合/替换头，不作运行时适应或p值声明。

正式数据为原168000物理记录，覆盖clean及郊区high/mid/low、urban high/mid/low。2352000次是两个读出与相关场景的解析计算次数，不是新增独立样本。0checkpoint/F权重或embedding/logits/额外神经预测/训练/头拟合/新校准。旧R45特征与新R46恢复器来自同一原source契约，label-free恢复器训练与完成预测后的有标签统计分别说明。

当前工程基准与用户指定父的历史测试暴露背景保留，不称新盲测或干净泛化证据。目标clean参考和truth仅参与有标签事后评价，均不进入R46训练或部署。描述量可能仍混有载荷、RX与日期因素；有限同TX关联不能证明每个恢复细节来自输入TX、未知TX泛化或真实在轨有效性，也不能替代R46逐景识别率、1pp容差、70%/5pp目标。

独立不可覆盖run/release/output，普通N607账户，复用R41 score.lock/max1own scorer，0训练GPU占用。22项聚焦检查、一次限P0/P1审查，launcher首步真实2Ls IQ与原训练校准做参数自由no-query smoke，不消费checkpoint；PASS立即等待父正式完成。一次预检/归档单SHA/远端compile及实际PID/CWD/argv/空CUDA/2线程/日志增长读回。父技术失败、输入/IDs/校准错误、非有限或输出冲突时保留产物，无自动重试，不停止或热改健康任务。精确commit、命令、环境/CWD、路径与预期产物见本run登记。当前没有本诊断结果。


实际启动状态：VERIFIED / WAITING_PARENT_FORMAL：普通账户新release/归档单SHA/compile已核实；真实2Ls参数自由noquery smoke PASS，0checkpoint/F/TX/V/target；实际PID/CWD/argv/空CUDA/2线程及日志增长已读回。等待R46双方全部正式评分后解析，复用R45raw/442805的56单元，新算56单元；无统计结果，不改变父训练或预测。
