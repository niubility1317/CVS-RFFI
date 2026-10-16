# 同 TX 真实 IQ 有界相干插值：E20 源域证伪

REGISTERED_NOT_RUN。独立一次 P0/P1 审查 PASS，唯一 baseline_reuse 标记问题已修复并定向读回；N607 普通账户预检 VERIFIED，尚未启动真实训练。起点1452eecc766197010c3a5971303baa9b4fa8dfeb；实际执行commit由release记录。唯一假设：真实sameTX/day跨RX观测经固定参考窗复标量对齐、插值与有界能量投影后，能提高源跨RX clean身份识别，且超过同RX插值的增量。原RXMIX为共轭混合，旧realpair为独立ordinary对监督，二者已关闭；本次研究新的IQ组合算子，不修改或复跑旧产物，也不以缩强度/换seed救援。

最新As证据：固定additive在RX6 seed01 clean+4.52、urban+23.39、low+16.27pp，但RX1seed02 clean−8.25pp，双提升不稳定。已有F+CROSS_RX固定接收状态/重算状态的单损伤完整诊断，移除新增noise、flatH、carrier或RXIQ均未恢复clean；不重复该矩阵、不据状态关联定单因果。当前仍无合格新Ag、四格交互、low>60%或suburban距clean≤2pp证据。

固定F+CROSS_RX/SIG_TIME_X3/CENTER_25K/CENTER_UNITS/NoExplicitCFO/joint160/defaultcos30，900722参数、原骨干/默认头正常联合训练。原13L+7L/U、ownEMA/旧门控/AdamW/cos、ordinary中既有PA/shift等不变；原ordinary只抽一次，每RDrop对同IQ两随机子网络，LS.01/KL1；不加新一致性、头、部署分支或As训练。

数据：原VALIDATED_ONCE carrier/split392005，源RX1/3/4/6/8、day1/2/3、6TX，原7/63/30。fit3/4/6/8 L5040/隐藏U45360/V21600；heldRX1原L1260只FINAL20读出，不做训练、EMA、校准或donor。全部实际roleIDs严格等于已核实seed02 D_raw来源。复用既有数据，不新拆V，不把1260heldL与旧5400heldV或5580/720混同。U真值不解封；donor仅当前filtered可见L，同TX/day、不同物理包；SAME保留RX，CROSS均匀选其他三个fitRX。类标签仅决定配对，所有系数/公式对所有TX相同。

IQ算子：参考窗[80,160)做donor→anchor复LS标量a；v=.75x+.25ay。保持anchor全256点L2 norm；将归一化v在实2×256维球面上从x出发的弧长限制为2asin(.1/2)，确保最终||out−x||/||x||≤.10，能量相同。不引入时移/FIR、额外噪声、PA/DAC/剪裁或未知TX反演。参考窗零能量/平行退化可返回原x并记录，样本全部监督，不重抽或丢弃。复标量/球面界仅证明数值执行，不证明标签保持；插值不等于真实硬件替换，不产生新的独立物理样本。相关度、实际变化、cap/fallback完整记录，变化量不替代识别指标。

共同base=.50R(raw)+.25R(ordinary)+.25R(raw第三对)+原ordinary浅中层SupCon。第四对权重固定.25：既有D_raw使用raw，两个新模型分别same/cross coherent(raw,partner)。三行总pair系数1.25、8L前向、共同base原raw绝对.75/ordinary.25不变；新增pair总量同尺度。原L/U主序列、初始化和全部原共同梯度必须匹配；donor私有seed20261016961+step不推进global RNG，RDrop两次前向复用一次新IQ。新视图使用anchor逻辑domain作原头调用，记录双物理来源；没有给混合IQ虚构单一真实RX。

唯一三行：reused_D_raw-RX1-s2026092702（不加载为初始化、不启动模型，原source报告/全部预测/轨迹复用）；SAME_RX-RX1-s2026092702、CROSS_RX-RX1-s2026092702（两新scratchE20）。base代码固定不可变source seed02 release05984a1229ebac7025a95081dc5fae4108276b28，profile真实激活seed02后才建图/args。技术no-query smoke只加载同角色D_raw final20默认头做2合成包，严格load，丢弃该模型；学生、EMA、optimizer、统计均从零，无E50/E200权重继承。

独立Ag主资格：CROSS相对raw clean≥+1pp、相对SAME≥+.5pp，clean宏F1非负、fitV下降≤.5pp，固定20/4440/8L/3330SupCon、最终共同base≤.5、完整合法覆盖/默认头联合更新、teacher/U/As额外调用和真值均0。比较仅1e−10pp浮点容差。clean＋六种 practical 全部读出/报告，单独Ag的practical保持标记以每场景−.5pp记录，但不阻断独立clean Ag资格；联合四格另检验，不能把单独Ag的clean改善当satellite改善。该口径在首次真实IQ运行前固定，纠正尚未登记时将practical独立保持作为Ag必要条件的过强表述。

FAIL关闭固定coherent插值，不调fraction/bound/window/RX/seed/预算救援，不事后把SAME改为获胜候选。PASS只支持独立源确认，非目标、跨seed、联合或自动晋级。完整负TX/day/场景/状态/质量、修复/新错和固定rawclean正确集合保持率保留。每新row完成自动FINAL20 clean+原6practical，各1260；实际leo __file__/25MHz/IQ256/residual/post_sync/EQoff G1/reference30/phase/IQ/quality/AGC/reflect保存。测试条件和困难样本不改，source-only，无正式target/truth服务。

新增预算40modelE/8880updates；每新行L4546560NN+原U/EMA/V/8820held，预计5823318NN但以实际计数为准；另568320partner物理曝光/新IQ构造与5040bank读取。唯一可见L ID仍5040。复用raw原20E/4440/5823318NN单列，source读取分析/绘图0NN；smoke2另列。新训练序列和原reference全量逐条核对；无法匹配则比较INVALID，不热改/自动重启。E20耗尽不是收敛。

本地合成检查：3090NN/3optimizerupdates/896成功算子输出、20负例；所有初态与共同loss/全部参数梯度逐位相同、同CPU/Python随机末态、8L/同IQ对、无As、默认头联合更新/export160、变化界/能量/相位尺度数值检查PASS。另16权限/阈值负例，U元数据隐藏getter、held1过滤45360、不泄漏或修改role scope、实际对照资格口径PASS，0额外NN/更新。NumPy2/Torch2.1本地ABI提示为已知非致命警告，不安装或修改环境。

实际命令：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_coherentmix_source_E20_20261016_r02/experiments/cvs_f_coherentmix20/runner.py；CWD同release，普通账户。动态GPU所有owner≤3/GPU、启动≥12000MiB、CPU2/loader0。一个owner/run/outputroot，不覆盖、不重复启动，不动其他健康进程。只对本run协议/来源/角色、非有限、算子界、输出碰撞或不能合法预测的系统技术失败停止pending并保留partial/健康worker；不因科学分数停训练，禁止自动重试。

一次限定P0/P1、本地commit、一次N607preflight、单release归档本远比较/远端compile、首步真实CPnoquerysmokePASS即继续、一次PID/CWD/argv/GPU/log增长与自然终态、独立读数报告/登记/Git交付。精确注册矩阵、路径、种子、成本与artifact见experiment.json。REJECTED_EXTRA_GATE：单独设计审批、报告hash等不在项目最小流程白名单，研发已授权，不新增审批。

技术恢复：r01（0b692362a0cb2136b825af12bf4b6f0ecfd9ec40）在读 CP/IQ 前因 operator.py 遮蔽标准库退出，0 训练/NN。原发布和报告保留 FAILED。r02 仅重命名 iqmix.py、更新两处导入及补测 direct-file argv 入口，不改变科学配方/种子/预算/数据角色。
