# 完整 practical：类内有界困难 CE 与相同权重随机控制 E30

状态：PLANNED，尚未启动。run 20261019-phase1-FcrossRX-withinTXHard-e30-s2026092701-m2-r01。实际代码提交在发布时固定于release_commit.txt；起点95cde0db8。

唯一假设：固定物理质量分组的负结果，尚未直接检验按当前模型在同一TX内的实际CE排序分配少量As监督是否有价值。当前原配方保留完整practical、raw→As路径、clean/raw份额、同IQ RDrop、ORD128 CROSS_RX与原U／自身EMA；本轮只改变As64的两次CE权重，不新增损失、网络或分类头，不改变KL。

对每个TX在64个As观测中有n个，k=ceil(n/2)，以两次同IQ CE01的detach平均值排序。H01为topk赋权0.75+0.25*n/k，其余0.75；R01在各TX内用独立CPU生成器打乱相同的权重直方图。各TX权重总和n，全部As都被监督，权重范围[0.75,1.25]。R01控制检验“困难关联”是否优于同幅度随机加权，不是取消As。

新增loss只作CE差值修正：delta=0.25*sum_pass sum_As((w−1)*CE01)/128。原mixed RD内加2delta，其外层系数0.5，故最终L只增加delta。原L总名义系数1.25、As CE名义0.25，各TX质量保持不代表有效梯度份额保持。排序和权重停止梯度，模型骨干／Sinc／默认cosine头正常联合更新。

## 匹配矩阵与区别

experiment.json为逐行矩阵唯一字段维护处。新增H01 HARD、R01 SHUFFLE两条独立E20＋10尾段；均严格加载原P43 B01 scratch FINAL20 student及自身EMA，重置AdamW，lr2e-4／WD1e-4／cos10／min1e-6。原P43 B00 NoAs30和B01 As30的固定读出直接复用，不重训练／预测／评分，不把不同父状态当同一初始化。原始父及代码19aedc3604fed9c193237dd92366164178af0983、固定来源路径与角色对照保留。

相比P70／P71，本次关键改变为逐TX内当前识别误差，而非物理噪声分箱的静态风险份额。相比AsLS／质量条件目标熵，本次目标仍CE01，不改变标签或温度；相比splitAdam，本次保持原生AdamW且无分量VJP／梯度相等断言。已有正负噪声、As分配、场景分层、公共传播和独立完整信道结果保留，不换名重跑。另一对话的P43受控退化诊断不重复。

## 数据和默认头

完整L6300／U56700（TX始终隐藏）／V27000，RX1/3/4/6/8、day1/2/3、split392005全部保留，无RX留出或Ldev。IQ2×256、Fs25MHz、eq索引1、NoExplicitCFO及必要内部补偿不变。F＋CROSS_RX／SIG_TIME_X3／统一Sinc／joint160／默认cosine6×160 scale30共同训练。U的batch_neighbor、strong agreement、raw ownEMA、Gaussian强视图、CE／entropy、阈值及分母保持原实际配置，无U星地路由。V仅无更新读出。

完整practical仍从raw64单次生成，几何／传播／新增噪声／接收质量／AGC／补偿依赖不变；不要求学生先正确。原ordinary仍在共享四格中启用，不冒称无增强。原RDrop两次前向使用相同生成IQ。

## 源判定与放弃条件

在读取本轮正式分数前固定并观察：H01相对原As30的固定V1800×2draw mid/low均值至少+0.5个百分点；相对R01的同指标必须为正；完整源V clean相对原NoAs30不低于−0.1个百分点。任一不满足则关闭该固定gamma0.25／tophalf配方，不扫描gamma、分数、种子或父状态补救，不扩E50。通过仅为后续源预算资格，不等于最终未知RX收益。科学负结果两行仍全部正式评分。正式clean≤0.1pp阶段判定单独报告；最终需要可重复clean正收益，不能用源微增或降低clean缩小suburban差距替代。

## 实际成本和发布

两条新增尾段共20模型轮／4440更新，每行6L=768前向/step，L1704960NN；原U学生／教师各567000NN，原生验证324002NN，共3162962NN/行。新增排序无NN／VJP；每步额外2个CE64算子、128个logit样本。另计每行源面板12600NN、正式1176002NN、两行共享真实父模型synthetic smoke1540NN，预计新增8704668样本前向；实际以产物计数与完整日志核算，不把额外CE或墙钟当0。原两父及冻结对照复用成本单列。

N607普通账户，现有CVS-RFFI/PyTorch，CPU2；每GPU所有owner合计最多3进程、空闲至少12000MB；不操作健康旧任务。CWD PROJECT/releases/cvs_FcrossRX_withinTXHard_E30_20261019_r01；命令 CVS-RFFI/python -u experiments/cvs_as_hardrisk30/runner.py；输出 PROJECT/runs/20261019-phase1-FcrossRX-withinTXHard-e30-s2026092701-m2-r01，逐行source/prediction/scoring独占，logs子目录。私有runtime不入Git。

仅本run系统技术错误停止：源角色／父权重／默认头／非有限／输出碰撞／无法合法预测或至少两行同一pre-prediction故障；保留产物核实，不能因低分重跑。一个launch owner。launcher首步真实父模型无query smoke通过立即训练。最小验证为60项协议负例、实际Torch25数学／梯度检查和7输入负例、19Python/2远端脚本AST、一次独立P0/P1、一次归档比较／编译及一次启动PID/CWD/argv/GPU/log增长核实。

## 自动评分与交付

固定clean＋六practical，每视图168000相同receivedIQ/物理ID，不改SNR、信道、困难样本。各行全7prediction先固定，再独立exec scorer连接truth，自动all7，不等待另行指令；原对照复用已有产物。source决定不读formal，collect在所有新旧prediction／score齐全后才独立truth聚合。

报告准确率及配对变化、macroF1、最差RX、RX/day/TX分层、urban三景均值、suburban绝对差距、修复／新增错误、完整成本和缺项。只有模型seed2026092701，两个源信道draw不是两个模型种子；单seed或<0.1pp微增不称稳定双提升。

预期：真实父模型smoke、逐步reference／shape／unit／hardrisk／native日志和training_sequence、source完整结果及fixedFINAL30、source_matrix_decision、各行all7prediction／independent scores、automatic_summary、最终报告和登记。目标仍ACTIVE。

实际发布代码 8d7628f3fed309a068f50ee403afcf4ac736d367；单归档传输比较与远端编译VERIFIED。真实父模型／默认头及两条synthetic hook检查PASS1540NN／0更新。dispatcher1895346，H01 PID1895457 GPU0，R01 PID1895535 GPU1；PID/CWD/argv/CUDA／实际GPU进程和log增长独立核实。状态RUNNING；未完成训练，未读取新正式分数。
