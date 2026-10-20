# R55：源通道重建合并到原有剪枝权重

状态：LOCAL_VERIFIED，尚未发布。性能优先；两固定行fold_full、fold_half，均273765完整参数，不引入推理分支、矩阵或校准状态。唯一直接初始化和冻结IQ教师仍为人选R46 prune_refine350521 E20/4940、方法bbb8daf817de050b958d217b70d4c33f4aefe341；不继承R48/R49/R53/R54学生。固定weight-only GN4通道map与原273765拓扑不变，只补偿删除输入通道的相关贡献。理论依据是原切片把被删通道置零，源输入通道间相关性可用于线性重建；不依据正式分数重排父、宽度或超参数。R53 prune_decay作为固定同宽/同micro/同E20控制复用，不重复训练。

共同CPU2校准一次：seed2026102156从原6300L_s/56700U_s分别抽64条，使用原R07 received01/20，128物理样本/256冻结父G输入。校准不读取clean、TX标签、F、V或target，无optimizer更新。每个需要减输入通道的Conv采16个固定、互异的空间位置，action global_head采实际池化后输入；在float64累计未去均值的通道二阶矩。共享prior长卷积的各stage先合并二阶矩，再为同一共享权重求一个投影，保留step独立bias/GN。原权重map的GN4配额及decoder旧上游宽度偏移保持。

重建矩阵P的保留行严格为单位矩阵，被删行作岭回归，固定ridge=1e-3×保留通道平均二阶矩对角线（尺度下限1e-12）。不拟合截距，因此线性通道合并与卷积零填充边界一致。原完整卷积先保留输出行，再沿输入通道合并W×P；fold_full使用完整合并，fold_half使用原切片与完整合并各一半。线性动作头同样合并，bias和GN切片保持。投影、二阶矩和源行位置留私有run，部署仅加载既有结构的最终权重。局部通道重建误差可以核对，不代表剪枝后GN、完整G输出或TX识别必然改善；半量行用于固定比较这项局部近似的幅度，不作源/目标网格选择。

两行沿用pairedclean IQNMSE＋0.05三尺度相对功率STFT＋线性1→0.1冻结父三stage IQ教师，全部coarse/action/prior联合。固定E20/4940、global256micro256/directfloat32buffer、ResetAdamW、LR2e-4暖247/cos至2e-5、WD1e-4/clip1、FP32/TF32关闭；model2026092701、split392005、data2026102001、augmentation2026101912、formal2027。每行L126000/U1134000、1260000学生/教师IQ输入；每组件累计100、组件和300、串行140pass，F50另计。额外共用校准256次冻结G输入、L128/U128视图曝光独立列账，不重复计入两行、不冒充0数据初始化或梯度epoch。

source READY/EXACT_MATCH、原7/63/30物理划分及全部R46/R44/R38/R17/R07/R04来源复用，实际加载前仍核对原R46训练契约。原F50/defaulthead只作冻结评测，无另训头。恢复器训练用模拟受损/clean配对，属于TX无标签源域自监督，使用U_s而不读取其TX标签；不能称无clean盲学习。历史工程基准及人选父已暴露，不声称新独立盲测、干净泛化晋级或输入TX逐细节真实性。

资源仅物理GPU3(fold_full)/GPU2(fold_half)，每卡一条本任务GPU worker、跨R54/R55最多两条；守护先检查固定R54同卡行完成并实际所属PID退出，不读取前驱性能。普通账户、CPU2、共享capacity lock、同一score.lock最多一条scorer，旧健康任务不停止/重启/热改。新run/release/output不可覆盖；唯一owner codex/root/R46-fold-compensate。

31项聚焦检查与CLIhelp通过，一次独立P0/P1审查PASS；第一步实际校准和R46/F50/2Ls noquery smoke，PASS直接继续。一次preflight、一个归档两端SHA及远端compile，再核实实际PID/CWD/argv/GPU/日志增长；技术失败保留输出、无自动重试或回退，低性能不触发停止。额外签名/设计审批/成员哈希/重复审查均REJECTED_EXTRA_GATE，不等待。

所有新行固定最终E20/4940，不依据源分数取消正式测试，自动source1800×7、同卡G及G+F/B1/B32三轮成本、正式168000×7预测；每行七景预测固定后独立CPU truth-last。主要报告每景accuracy/MacroF1、相对固定同宽控制和R17/R46、逐TX/RX/day与repair/damage；时延/吞吐/显存/参数次级。原R17每景1pp为退化上限，不当作优化成功；lowurban≥70%、其他五景距rawclean84.6761904762%≤5pp及更强TX细节目标继续ACTIVE。

成本使用64源V同步预加载FP32，不含传输/前处理/loader；三G一F共享suite驻留不是独立部署显存。ConvLinearMAC排除FFT/CG/norm/激活及F；参数、MAC和实际时延分别报告。精确commit、命令、环境/CWD、路径和矩阵见experiment.json。

方法commit `6bea50605a1c2cfb49645b5c2f7ba8306b39791d`；精确命令/环境见[登记](experiment.json)。
