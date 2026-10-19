# 同伪身份的同 RX 与跨 RX 统计混合

状态：RUNNING / VERIFIED。用户目标为目标 low urban ≥70%，其余五 practical 接近原 F clean 85.113889%（Agent 暂按差距≤2个百分点解释）。分开报告恢复 IQ 与同条 clean 的差距，以及接入冻结原 F/default heads 后的真实 TX 识别、修复/损坏/净修复。当前已核实目标最好 low urban 65.309127%，目标未达成。

r01首次发布因首步smoke的Torch2.1/NumPy2索引兼容错误退出，0训练worker/0优化步/0target，旧产物保留。r02仅修复显式Torch索引并使用新不可覆盖run/release，科学配方和固定目标矩阵保持；复用本候选一次独立审查，不重复全审。

本轮检验源接收机统计依赖。两个新训练配方均从合法原 R03 coarse20/correction20 严格加载全部权重，方法 b01b534f52daf9cae9e940ed6ec6f9243bccbdbe；不继承任何源失败的精修权重。samerx_style（GPU4）在同伪身份、同 RX 内取非自身 donor；crossrx_style（GPU5）在同伪身份、不同源 RX 间取 donor。两臂都保护伪身份，没有无保护混合。仅对两种规则均有合法 donor 的记录启用，固定随机抽取数量匹配 eligible、尝试概率、有效物理曝光和系数，只改变 donor RX 条件。

配对身份复用此前 consensus_native 在精修训练前用原三 F 对同 R02 source clean L/U 产生的 pseudo/accepted 字段，方法 623b3f3cb93f56fe77bf680e71570def9170ed2e。只接受三原默认头原生 argmax 一致、每头置信度≥0.9的59402/63000条（L6074/U53328），这是伪身份接受率，不是真实准确率。不读取 L/U TX 真值，不使用此前精修网络或 bank 中的 prototype 作为方法输入。来源核对原 clean L/U、原 F 种子、相同缓存句柄以及0V/0TX/0target/0headfit/Fupdate，不重新验证已 VALIDATED_ONCE IQ，不加签名/哈希链。

训练时在 correction encoder 第0/2层（32×256、96×64）混合每通道时间均值/标准差：p=0.5、lambda~Beta(0.1,0.1)、epsilon=1e-6；均值/标准差 detach。混合特征同时进入后续编码与对应 skip。observed/clean 半批复用物理配对和系数，但 donor 严格留在相应半批，不把 clean 半特征送入 observed 半。float32 lambda=1 的尝试记录但不计有效混合。无合法 donor 跳过，不退回跨身份配对。其余原基础样本不删减；逐轮记录 eligible/attempted/effective、L/U、两层实际应用与调用次数和5×5 RX donor计数。

两个新臂恢复器参数同为2140840，无新训练参数；同原R03初始化、原63000L/U、R02缓存epochs1..10、dataseed2026101951、modelseed2026092701、mixseed2026102051、.15clean/.60residual/.25full、observed＋clean等权和数据顺序。reset AdamW联合E10/2470、batch256、LR3e-5/warm247/cos至3e-6、WD1e-4、clip1、FP32/TF32关闭。新增L/U物理优化曝光63000/567000、IQ1260000；coarse/correction各累计30轮，共60组件epoch、串行继承50pass，原F50仅继承不更新。

基础损失保持原posterior03：IQ NMSE＋0.1logpower＋三原 F 各自final160 cosine＋0.3×T²KL（T=2）的标量均值。没有额外CE/prototype/focus/FiLM。三 F 为原20261016-phase1-F-supcon-e50-clean-manysig-m9-r02 CROSS_RX seeds2026092701/02/03及原联合训练默认头，来源4df7b94d0fd8ce29056bd8236364e45afa92e07f；参数/统计冻结，0新头拟合。称冻结F引导自监督恢复，不称整个系统全无标签。

按用户目标测试要求，本轮是训练前固定的双行目标比较：source V同1800条/原F1/七视图仅诊断，固定最终E10，无V早停/参数或统计更新，不用源accuracy/NMSE筛选、排序或发布一行。两个新臂各自完整测试原168000条目标clean＋六practical，三原默认头各自argmax，不投票、不按环境拼接模型。此规则不追认或重测旧失败模型。每行全部七视图IQ/ID/预测固定后，独立无CUDA CPU才接入truth及评分用pairedclean。目标成绩不回流任何行的参数、预算、权重、顺序或选择性重试，所有登记行完整报告。

推断 augmentation OFF，StyleCore eval直接调用原Core路径；只接收当前IQ，不需要RX、pseudo/bank、其他帧/视图、clean、TX或更新状态。报告accuracy/Macro-F1、逐F/TX/RX/day、净纠错与原clean差距；无oracle相位/增益/时延对齐的full256/prefix232/tail24 NMSE及logFFTpower均值/中位数/P90；实际walltime/峰值显存单独报告。一个恢复器种子、三套既有F读出，不称三恢复器种子、新独立盲测、Phase2部署或在轨性能。

46项聚焦检查PASS，覆盖身份/RX/有效曝光/半批隔离、统计停止梯度、实际浅/中层接线、推断原路径与逐帧独立、预算/来源/权限拒绝、固定双行无源选择、七视图/三F/有限IQ后truth。单次独立P0/P1审查发现进度覆盖P1，已修复且唯一一次定点复核PASS。launcher首步实际原R03/三F/source Ls同伪身份两RX四条smoke，检查初始eval逐点等于R03、真实两层应用、coarse/correction及原默认头到IQ有限非零梯度、F无梯度、新checkpoint往返；0V/0TX/0target/0optimizer更新，PASS立即训练，不设smoke令牌。

一次N607普通账号预检VERIFIED：新run/release不存在，原R03与source-only伪身份来源核实，前序完成，GPU4/5各24124MB空闲且无PID，空间≥8GB，其他owner保持。每卡最多一条本run训练/预测worker，计入全owner进程<2、source获取≥18GB/target≥12GB；CPU2线程，source/target共用一把score.lock，独立scorer最多1。root确认某行source Popen实际退出后才同一卡启动该行target，不等待另一行结束。非法来源/权限/代码、冲突、非有限/OOM、预算或预测失败时保存并技术失败该行、无自动重试，其他健康行继续，不热改/停止/重启其他任务。一个代码配置commit，一个不可覆盖run/output root，一个归档本地/远端SHA比较和remotecompile，启动后独立核实PID/CWD/完整命令/GPU/日志增长。具体命令、环境/CWD、路径、矩阵与产物见experiment.json。IQ、权重、预测、日志和本机状态不进Git；未给另一对话发指令。


## 实际发布与首轮核验

方法commit `99aa5bb976112f1469aacfefa2f1ee40f4fe537a` 已推送并独立核实远端OID。一次r02归档比较、远端compile/import、真实原R03/三F/source Ls四条两RX no-query smoke均PASS，修复了r01的Torch/NumPy索引兼容失败。两臂augmentation OFF初始输出逐点等于原R03，真实浅/中层混合次数正确，coarse/correction与原默认头到IQ有限非零梯度、F无梯度，新checkpoint往返一致，实际参数均2140840；0V/0TX真值/0target/0optimizer更新，PASS后已立即训练。

两个实际worker的PID/CWD/完整命令与物理GPU4/5对应、持续日志增长已核实；source参考缓存各63000条、三原F，无TX真值。实际两层应用和RX donor矩阵符合预登记：

- samerx_style：1/10轮、247/2470步，累计eligible/尝试/有效混合59372/29726/27151，有效L/U2821/24330；每层实际IQ应用54302次，同RX臂只含对角donor、跨RX臂只含非对角donor。
- crossrx_style：1/10轮、247/2470步，累计eligible/尝试/有效混合59372/29726/27151，有效L/U2821/24330；每层实际IQ应用54302次，同RX臂只含对角donor、跨RX臂只含非对角donor。

这是实际训练接线与预算证据，尚无新的源V或目标准确率。按固定矩阵两臂最终E10都自动完整目标测试，不以源涨分/NMSE筛选。root只有确认该source进程退出后才在同卡启动其target；source/target CPU评分共用锁。当前用户70%/近clean目标保持未达成。
