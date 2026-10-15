# 真实 RX1 留出：源身份风险约束 U 的特征位移

状态 **RUNNING / 启动 VERIFIED**。下文保留预登记。目标仍为实际 joint160、原默认 cosine 头联合训练后的固定 clean 三种子均值84%；本轮是有真实训练的 source-only 机制证伪，不读取 target，不宣称候选完整训练已改善。

依据：上一轮冷初值 RX-blocked80条件中 ROUTED 对同辅助梯度范数 JOINT_NORM 的身份 CE 20/20 改善，但 AdamW 位移未匹配。完整责任路由 M12 默认头81.120635%，相对匹配SSL−0.054167个百分点；最差RX+0.245833个百分点、3/3正保留。source优化方向存在不能推出持续训练泛化增益。历史CENTER25K等正路线与全部负结果保留，不重跑当前已负配方。

三条可执行方向：①本轮选择真实源 RX 风险约束 U 的有状态优化器特征增量；②责任路由的早期方向证据保留为独立模块，先解释完整轨迹差异，不靠增权/增频重复；③跨 RX 真实证据关系继续作为独立研究，但必须对照旧PA/difference/set/observational负结果，不能换名重试。本轮只有①进入实现。

机制对象是 u=theta_(L+U)^F−theta_L^F，而不是仅修改 AdamW 历史 first moment 或增加伪标签接受链。L-only 用当前 identity_loss 的真实 backward；RAW 用实际 postclip 梯度的临时原生 AdamW（不二次裁剪），两者从同一 pre-state/m/v/step 独立构造。F为实际encoder的71个参数张量，包含3个feature gain、排除唯一类别W。固定144个可见训练L锚包给出4个训练RX的默认头身份CE梯度，单位化后解 min||p−u||², g_RX·p≤0。CONE在L-only特征位置上加p；SHRINK沿原u方向，二分匹配CONE写回FP32后的实际位移范数（相对误差≤0.001）。所有模式的W及非F参数使用同一RAW位置；主模型和教师均只走原生RAW轨迹，无候选实际纠正，无另训分类头。

梯度投影已有 [GEM](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)、[PCGrad](https://arxiv.org/abs/2001.06782)、[CAGrad](https://arxiv.org/abs/2110.14048) 等先例。本轮的具体研究差异是分离U引起的有状态AdamW位移、用跨源RX身份风险约束其feature部分，并用实际存储位移范数控制检验方向。未做穷尽新颖性检索，不称全局首创；线性风险约束不保证真实非线性CE或目标泛化改善。

更近的 SSL 先例是 [Fix-A-Step，Algorithm 1](https://proceedings.mlr.press/v206/huang23c/huang23c.pdf)：它按有标签与无标签梯度的内积决定保留联合更新或只用有标签更新。因而“用 L 防止 U 伤害”本身不是新贡献。本轮检验的是实际 AdamW 增量差、分 RX 风险交集和 feature-only 投影的具体组合；SHRINK 与 L_ONLY_F 两个控制用于排除简单减弱或删除 U 的解释。只有完整证据支持时，才能进一步讨论其方法价值。

真正源折：固定 RX1 为评估域，选择来自物理索引顺序、不是按任何分数选择。完整原source角色先核验，再在模型/teacher创建前过滤所有L/U/V、calibration、probe loader，训练RX为3/4/6/8；L5040/U45360隐藏/V21600，比例7/63/30不变。原raw载体和静态15域ID映射保留，训练/校准/特征缓存不读RX1；RX1的1260个原可见L包仅在无状态读出中评价。该折checkpoint与全5RX正式训练数据契约不同，禁止用于正式all-source初始化；模型从零训练，依赖R02只是代码环境，不加载它的权重。

固定20轮、13L+7U、每轮222步、4440真实更新，原oldgate/ownEMA、LS0、NoExplicitCFO、无普通L/星地增强。过滤后U loader每轮178batch，旧legacy迭代重启保留222步：7轮U共396368次包观察（每轮56624），不是原56700完整单遍；本折只作source证伪，不冒充原全5RX同采样比较。四个probe为2896/3344/3792/4432，RAW/L_ONLY_F/SHRINK/CONE共16个临时状态，每个评价同1260个源L包及ACTUAL/PRE_W无拟合读出。8临时AdamW、4额外L backward、144锚包×4前向、1260评价包×17前向及2zero-smoke，总21998次诊断packet前向；额外L IQ读取1404。四时点不当四个独立seed，源折baseline20不当完整候选20。

资格在真实数据前固定：至少2/4 probe同时 U loss>1e-8、U的F增量/完整F步≥0.01、保留增量范数比≥0.1、CONE/RAW余弦≤0.95、存储范数匹配≤0.001、FP32 cone残差≤0.001相对raw增量范数；对RAW、SHRINK及L_ONLY_F的ACTUAL和PRE_W CE均下降至少1e-5且错误不增加。baseline final L CE≤0.5。通过只允许随后完整source-fold训练控制，不直接进入target或三seed主矩阵；不足关闭当前候选，不扫λ/阈值/源RX。所有四个probe都保存，包括失败或零U信号；不按分数停止健康训练。

实际代码/config由本次Git提交固定，唯一owner `codex/root/RX1-U-increment-20261015`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX1_U_increment_sourcefold_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_rx_u_increment20/runner.py experiments/cvs_rx_u_increment20/configs/source_fold.json`。单一不可覆盖输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-RX1-U-increment-sourcefold-s2026092701-r01/source`，普通N607/CVS-RFFI Python，GPU按预登记容量规则选一块、计入所有owner总进程≤3。依赖 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02`，完整矩阵/seed与字段见experiment.json。停止规则：Protocol/role/held-RX or U-truth leakage, nonfinite loss/gradient, undeclared task, wrong budget/actual-gradient oracle or output collision: preserve outputs, no automatic retry. No performance stop or unrelated-process intervention.

最小流程：聚焦数学/角色负例及实际CPU/CUDA构造检查、唯一独立下一run P0/P1审查、Git提交推送/remoteOID、一次N607 preflight/单archive比较/remotecompile。原生launcher先构建本折合法数据和own-scratch真实checkpoint no-query smoke，PASS直接继续；启动后独立PID/CWD/cmd/GPU/log-growth。训练和诊断冻结后报告完整source同row结果；无target预测/评分，不降低84%目标或改变整体goal active。

本地解析/角色/反向传播/实际存储范数测试共16项 PASS，包含必须优于L-only特征控制的负例；实际188参数模型 CPU 与 CUDA 合成路径各2warm+1RAW步通过，RAW实际梯度oracle逐位0差，W联合更新、source/RNG状态保持。无真实sourceIQ/checkpoint读取，不将合成结果用于科学资格。唯一直接下一run P0/P1审查 PASS，编译 PASS。


真实启动已独立读回：执行 `779e77620ea724154bfb2c899498578643234f00`，GPU 0 上所属进程/命令/CWD与GPU成员关系一致，总实验进程 1/3；log已增长，own scratch真实CP no-query smoke PASS。有效角色L5040/U45360/V21600，五个training/calibration/probe loader全部排除RX1，实际参数13L+7U/LS0/ownEMA/无L与星地增强，初始化来源为空。仍是运行状态，未宣称训练完成、机制资格或新target分数。
