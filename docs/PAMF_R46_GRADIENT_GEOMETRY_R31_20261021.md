# R31：固定四模型的源重建与统计目标梯度关系

状态：LOCAL_VERIFIED。12项聚焦检查通过，唯一独立P0/P1审查PASS，无P0/P1；无新网络、训练更新或正式识别成绩。

本轮检查原重建目标和新增四阶统计目标在源数据上的局部梯度关系。R29已显示IQ/统计误差取舍，R30两版已固定发布；本诊断不修改其权重、系数、预算、矩阵或队列。只读同R29/R25固定四份source-final：原350521、R18阶段版、R22复频谱版、R22恒等版，不按R23目标结果增删或挑父。未来训练初始化/教师仍只用原指定350521，不继承诊断学生。

原数据7/63/30、READY/EXACT_MATCH、完整来源与source-final20/4940或20/19700契约复用。仅R07原128Ls+128Us固定不透明源位置，clean/received01/received20三视图。每视图16批，每批固定8Ls+8Us，256个位置各出现一次；无TX标签、F权重/输出、V或target访问。CPU2、一个诊断进程、零GPU，无优化器、梯度裁剪/投影、参数更新、新网络或头拟合。

对每个模型统一计算三个诊断目标：base为R22恒等配方在epoch20的pairedIQ+.05powerSTFT+.1三stage原父IQ+.02分组复FFT+.25G(clean)恒等；moment_clean与moment_teacher为R30相同11维四阶描述MSE/权重1，参考分别为配对clean与原350521同received最终IQ，全部参考detached。这个共用目标不冒充各历史模型实际训练的梯度，也不等同于R30在训练中的轨迹。

autograd.grad分别返回三种batch mean损失对唯一模型参数的梯度，未写入.grad；临时允许参数求导后在finally恢复原requires_grad。所有模型保持eval，权重/缓冲区逐位不变。共享prior权重只计一次，coarse/action/prior三个分组完整划分参数，另报告whole。norm/dot在FP64统计；任一L2≤1e-12时cosine无定义，以null记录并从冲突比例排除，不视为0。有效cosine<0只表示该源批次局部方向相反，不证明每例或TX退化；极小梯度即使有效仍需结合norm比值判断数值敏感性。

每模型每视图16批，共192条批记录；12模型/视图单元，每单元3损失+12范数+12成对cosine聚合，共324汇总行，保留有效/无效数、均值/中位/p90/p95/min/max、负cos比例及norm比均值。每模型首步2Ls执行received及clean两个完整G路径和三目标梯度，学生4输入；诊断每模型received768+cleanidentity768，总1540/模型。原父共享teacher缓存另770（2smoke+768面板），全G输入实际6930。无额外独立梯度向量导出或源真值评分；原始批统计和opaque位置仅私有，公开聚合与报告。

普通账户、新不可覆盖run/release/output、一个方法commit，一次preflight、归档两端一次SHA及远端compile。launcher真实checkpoint/2Ls无query smoke PASS立即继续，实际PID/CWD/完整命令/CPU2/空CUDA/零GPU及日志增长核实；技术失败保留、不自动重启/回退，不改健康owner。精确命令、环境/CWD、来源/预算、seed和预期artifact见experiment.json。所有未来新网络仍逐版完整七景，本诊断不产生新网络。

12项检查包括手算平行/相反/零梯度、null统计、唯一共享参数分组、参考隔离/无.grad、异常flags恢复、真实完整恢复结构的合成梯度与分组范数闭合、固定混合批覆盖、完整6930输入/192批/324汇总fixture、F/target参数拒绝和旧输出先拒。唯一独立P0/P1审查PASS，无P0/P1；通用设计审批/重复全库测试为REJECTED_EXTRA_GATE/NONBLOCKING。联合压缩与性能目标仍ACTIVE。
