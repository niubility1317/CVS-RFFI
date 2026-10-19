# 纠错器窄窗与全窗注意力

状态：RUNNING / VERIFIED。用户完整目标保持为目标 low urban ≥70%，其他五 practical 接近原 clean 85.113889%（本对话 Agent 暂按差距≤2个百分点解释）；分别报告同物理 clean 的 IQ 差距，以及接入冻结原 F/default heads 后真实 TX 识别和净纠错。实际目标最好仍65.309127%，未达成，不以新结构、检查通过、源分数或波形代替目标成绩。

本轮依据 source V 分组证据提出结构假设：此前跨RX统计混合相对同RX主要改善RX1/6，但最弱RX3仍79.722222%，RX4下降；此前IQ摘要/语义FiLM源low/mid也未改善。这些证据不能证明原因是时间上下文不足。本轮只检验显式、随当前输入变化的时间交互，训练配方不依据目标分数调整。已有卷积感受野较宽，GroupNorm也使用帧内整体统计，不能称原网络没有全局信息，也不能将窄注意力窗称为全模型只看局部原始IQ。

两个新臂 window_attention、global_attention 均从原合法 R03 coarse20/correction20（method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe）完整继承全部旧state，只新增纠错器encoder bottleneck128通道×32位置的一个残差注意力块：PreLayerNorm，固定当前帧sin/cos位置编码，128→384 QKV、4头×32维、缩放点积softmax，再128→128投影残差；无dropout/FFN。输出投影weight/bias初始化为0，初始完整IQ应逐点等于原R03，不加载任何失败精修权重。新增66304参数，两臂均2207144可训练参数；新层只有10轮，原coarse/correction各累计30轮、60原组件epoch、串行继承50pass，预算分别披露。

两个新臂参数张量和随机初值完全匹配，只改变固定注意力mask：window_attention半径2，每位置最多关注相邻5位置，共154个允许的有序位置对；global_attention关注当前帧全部32位置，共1024对。两者都执行dense32×32矩阵运算，不把窄窗说成计算成本更低，实际耗时/显存报告。没有跨物理帧attention、RX/场景路由、伪身份bank或运行F条件，也不使用其他对话的四步迭代/末端信道一致性路线；只仅读取了其当前进展，没有发送消息或改动其文件/任务。

原63000L/U、R02缓存epochs1..10、.15clean/.60residual/.25full路由、observed＋clean等权、batch256、modelseed2026092701、dataseed2026101951保持。每臂新增L63000/U567000物理优化曝光、1260000IQ，联合E10/2470固定最终轮；AdamW重置，原网络LR3e-5、新注意力LR2e-4，共warm247/cos至各自0.1、WD1e-4、clip1、FP32/TF32关闭。日志记录两LR、新输出投影范数、QKV相对初值delta、实际非零QKV梯度步数，以及attention调用/IQ/允许位置对/实际context pair数。零输出初始化首步QKV梯度为0是预期；unit验证第二步可学习，实际计数单独验证，避免把AdamW decay变化称为学习。

基础损失保持posterior03：paired IQ NMSE＋0.1logpower＋三原F各自final160 cosine＋0.3×T²KL（T=2）的标量均值，没有新增CE/prototype/focus/style/FiLM。三F是原20261016-phase1-F-supcon-e50-clean-manysig-m9-r02 CROSS_RX seeds2026092701/02/03及原联合训练默认头，method4df7b94d0fd8ce29056bd8236364e45afa92e07f；参数/统计冻结，0新头拟合/0F优化更新，不读L/U TX真值。称冻结F引导自监督恢复，不称整个系统全无标签。复用已验证同数据契约和原父来源，方法/结构变化不重验IQ、不加哈希链。

推断只用一个当前received IQ，单次coarse/corrector前向，无restorer内部运行F、信道迭代、clean、RX、pseudo/bank、其他帧/视图、目标truth或状态更新。三套原F仅作为独立外部读出，分别argmax，不投票、不按环境拼接模型。一个恢复器训练seed不是三恢复器seed，不宣称新的独立盲测、Phase2部署或在轨性能。

源V同1800条/原F1/clean＋六practical，只作诊断，不用于参数/统计更新、早停、accuracy/NMSE资格或排序。两组固定最终E10都自动完整测试原168000物理目标的七视图/三原默认头。每行全部IQ/ID/预测固定后，独立无CUDA CPU才连接truth及评分pairedclean；source/target scorer共用score.lock。目标成绩不回流任何行的预算、配置、权重、顺序或选择性重跑，所有登记行完整报告。报告逐F/TX/RX/day、accuracy/Macro-F1、修复/损坏/净修复、距原clean差距，以及无oracle对齐的full256/prefix232/tail24 NMSE和logFFTpower均值/中位数/P90。

42项聚焦检查PASS，覆盖原R03全state继承与初始输出、同参数/初值、实际窄/全窗作用范围、零投影及QKV后续梯度、逐帧独立和eval无状态更新、checkpoint预算/来源/权限、固定双行无源筛选、七视图/三F/有限IQ后truth。单次独立P0/P1审查PASS，无确定P0/P1；QKV实际非零梯度观测已纳入该次审查。launcher首步使用实际原R03/三F、source Ls四条（RX1/3各2条）no-query smoke，验证零输出逐点等于原R03、两臂初始state相同、coarse/correction/新attention.out有限非零梯度、原头向IQ输入梯度、F无梯度及新checkpoint往返；0V/0TX/0target/0optimizer更新，PASS立即训练，不设smoke令牌。

一次N607普通账号预检VERIFIED：原R03来源合法，前序固定矩阵完整完成，新run/release不存在，GPU4/5各24124MB空闲无PID、空间≥8GB，其他owner保持。GPU4窄窗/GPU5全窗，每卡一条本run训练/预测worker，计入全owner<2、source获取≥18GB/target≥12GB；CPU2线程、独立scorer最多1。root确认某source进程实际退出才同卡启动其target，不等另一行。非法来源/权限/代码、冲突、非有限/OOM、预算/新模块未实际执行或预测失败时保存并技术失败该行，无自动重试，健康行继续；不热改/停止/重启其他任务。一个代码配置commit、不可覆盖root、一个归档本地/远端SHA比较与remotecompile，随后核实实际PID/CWD/命令/GPU/日志增长。命令、环境/CWD、路径、资源、停止规则与产物见experiment.json；原IQ、权重、预测、日志及私有运行状态不进Git。


## 实际发布与首轮核验

方法commit `d3d0d90f16271204bb0f31a788557f21ad86655a` 已推送并独立核实远端OID。一次发布归档本地/远端比较、远端compile/import和真实原R03/三F/四条Ls no-query smoke均PASS；两臂初始完整IQ逐点等于原R03、参数初值匹配、coarse/correction/新attention.out及原头到IQ输入梯度有效、F无梯度、新checkpoint往返一致；0V/0TX/0target/0optimizer更新。实际参数两臂均2207144、新attention66304，允许位置对为154/1024，PASS立即开始训练。

两条实际source worker的PID/CWD/完整命令/GPU4/5及持续日志增长已核实；三F参考缓存各63000条。训练中QKV已有实际非零梯度，不能仅由参数delta推断学习：

- window_attention：1/10轮、247/2470步，attention调用247次、IQ126000条、位置对19404000次；QKV非零梯度246步，输出投影范数0.15690157、QKV delta0.61689281。
- global_attention：1/10轮、247/2470步，attention调用247次、IQ126000条、位置对129024000次；QKV非零梯度246步，输出投影范数0.14223653、QKV delta0.73779160。

新模块已实际参与，不代表性能有效。当前仍在训练，尚无新源V或目标准确率；固定两臂最终E10后分别自动完整目标测试，source V不作资格筛选，用户70%/近clean完整目标保持未达成。
