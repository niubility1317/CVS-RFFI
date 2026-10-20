# R44 44.28 万裁剪版的训练优化与进一步压缩

状态：**VERIFIED / RUNNING**，已发布且两版实际训练。27项聚焦检查、CLIhelp和一次限P0/P1审查通过。用户在 R44 完整正式结果揭示后明确要求保留 group_pruned442805 并继续优化、压缩；因此直接父由用户指定，不按 Agent 目标排名选择，不声称新的独立盲测或干净泛化。固定两行同预算比较，已有 44.28 万权重与全部负结果保留，不研发其他恢复网络。

|行|完整参数|上下文参数|较442805减参|主/谱分支宽度|
|---|---:|---:|---:|---|
|retain_refine|442805|302471|0%|12/24/36/48；8/16/24/36|
|prune_refine|350521|237927|20.84%|12/20/32/44；8/12/20/32|

唯一初始化与冻结 IQ 教师为 R44 group_pruned 源域固定最终 E20/4940，方法 `2f5247dbd5fe50f28defa38a0252d70e692934af`。原 R38/R17/R07/R02 祖先、原物理 L6300/U56700/V 与六 TX 不变，恢复器训练始终0TX标签/F权重或输出/V/target。R44源固定训练记录与checkpoint元数据核对后严格加载，不加载其他学生或 F 蒸馏来源；历史目标暴露背景保留。

retain_refine 在训练前逐状态严格保留父函数，作为同预算优化控制。prune_refine 对每个 GN4 组内按原 R44 归一化输出 filter 均方权重分数稳定选通道，使用原已验证一致 encoder/decoder/skip/GN/action global-head 映射；保留主分支最浅12通道，减少更深和谱分支通道。三步 prior 的长卷积权重继续共享，逐步 bias/GN/pointwise 独立，三 prior/六 CG 与完整接收链不变。剪枝只读权重，0IQ/标签/F/目标信息，不声称与父函数等价。

两行分别从同一父独立分叉，新增完整20轮4940更新。使用全部原L6300/U56700及原20个模拟received缓存；每行新增L126000/U1134000曝光，学生context/教师IQ各1260000输入。所有coarse/action/prior联合训练；重置AdamW、LR2e-4/warm247/cos2e-5、WD1e-4、clip1、B256/micro64，尾批24按实际分母，FP32/noTF32、model2026092701/data2026102001。父每组件累计60轮，本轮后80轮，组件轮次和240、串行训练pass累计120，原F50预训练另计。

目标为最终paired-clean IQ NMSE，加0.05的64/128/256三尺度相对功率log1p平方误差均值，再加冻结父三stage IQ NMSE均值。教师系数20轮线性1→0.1。频谱项使用复IQ、periodic Hann、hop=size/2、centerFalse、正交FFT；每条pairedclean平均窗口功率detach/floor1e-4，同时缩放预测与参考。配对clean和教师不反传。这是待验证的训练假设：减弱后期父输出约束，为配对clean的弱谱细节留下优化空间。频谱项不使用身份标签或F知识，不预先声称提升识别或逐细节真实。

术语保持“无 TX 标签的模拟配对源域自监督”，L/U都利用，U_s TX真值隐藏；源clean配对参与，不称不使用clean的盲无监督学习。部署仍只读当前原生float32 `[B,2,256]` IQ并输出同形IQ，原冻结F50及checkpoint默认头统一派生FFT、PA、时域和统计视图，0另训头。

固定最终E20/4940，无早停或按源分数筛行。每行自动source1800×7诊断和同卡原R17/442805父/本版完整链batch1成本，然后原168000物理样本×7正式场景：clean、郊区high/mid/low、urban high/mid/low。每行全部预测固定后独立CPU连接truth，原F50/defaulthead不训练。报告accuracy/Macro-F1、逐TX/RX/day、IQ/FFT/配对特征几何、repair/damage与净修复、相对原R17/指定442805父/R38的逐景差；原R17每景最多下降1pp及指定父1pp分列，70%/其他五景gap≤5pp目标分列，不把均值覆盖单景退化。两行全部正式测试，不根据目标结果调参或选择性重跑。

N607普通账户GPU0/1每卡一own、总最多二own，等待R44对应GPU实际句柄退出、容量free≥18GB/allowner进程<2；其他owner不动。CPU2/max1本任务scorer，复用R41 score.lock。独立不可覆盖run/release/output，方法Git固定；聚焦协议负测、一次限P0/P1审查；launcher首步真实R44/F50/两条Ls配对IQ noquery smoke，PASS立即按卡启动。一次普通预检、归档单SHA/远端compile和实际PID/CWD/argv/GPU/日志读回；技术失败保留受影响产物，无自动重试，不停止或热改健康行。精确commit/命令/环境CWD/路径/预算/矩阵/停止规则与预期产物见run登记。

当前没有本轮源或正式性能结果。减参是确定的结构事实，优化收益、延迟与显存必须实测；参数下降本身不完成用户性能和输入TX细节真实性要求。R45旧正式解析评价独立收尾，不作为本轮训练或预测的门槛。


方法commit `bbb8daf817de050b958d217b70d4c33f4aefe341`。精确命令、环境/CWD、路径、矩阵、预期产物见[登记](experiment.json)。


VERIFIED/RUNNING：实际442805父/F50/两Ls配对IQ noquery smoke PASS，两行全部组件有限非零梯度、teacher/F不更新、checkpoint严格重载一致。归档单SHA/compile及controller/两worker实际PID/CWD/argv/空controllerCUDA/各物理GPU0或1和GPU-PID匹配、首轮曝光及日志增长已核实。无本轮正式性能结果；每行E20后自动七正式测试和独立truth-last。


## 训练中完整日志核对

本节只覆盖当时已经生成的全部完整记录，训练及正式测试尚未结束。逐轮核对L/U曝光、学习率、衰减教师系数、四项加权损失闭合与F_inputs=0；同时扫描两版全部已写stdout。各项检查通过，未发现Traceback、OOM、Killed、warning、非有限值、提前停止或resume标记。

|行|已完整解析轮次|L曝光|U曝光|U占训练输入|IQ NMSE 首轮→最近轮|频谱误差 首轮→最近轮|
|---|---|---:|---:|---:|---|---|
|retain_refine|1至14|88200|793800|90.0%|0.02632830→0.02319195|0.00697492→0.00614539|
|prune_refine|1至14|88200|793800|90.0%|0.07831143→0.02429943|0.01738684→0.00641115|

[全部已完成轮次、损失分量与扫描计数](training_progress.json)。这是训练配方执行与重建损失的阶段性证据，不是源V或正式识别收益；不得据此选权重、缩短预算或取消任何一行测试。旧父与本轮增量预算及教师系数不同，不比较跨配方总loss来归因机制。后续完整E20正式结果将覆盖本节的阶段性状态。
