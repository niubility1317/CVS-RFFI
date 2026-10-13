# 旧门控不变的 U 邻包成对采样特征训练

状态LOCAL_VERIFIED，尚未训练。run `20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02`；唯一owner codex/root/neighbor-pair50-20261014。逐row矩阵/配置唯一来源 [experiment.json](experiment.json)。本路线不同于另一工作树单基时序残差控制。

依据为完整源诊断 `20261014-phase1-source-neighbor-pair-gate-manysig-m3-r01`：同三个固定默认学生/EMA，RANDOM精确复现360计数；全720source统计已复算，V FULL覆盖3.4099%→89.9617%，PAIRS接收正确率99.9959%、错误3/0/0，三seed均满足预先冻结的coverage≥.5/precision≥.995/drop≤.0025。没有U或目标IQ/真值进入该诊断。仅证明物理邻包可见性机制，不能推断U正确率/目标收益。方案由这一源证据支持，不依据Sinc/TIME目标分数，不使用任何旧权重初始化。

唯一新训练因子是U_s minibatch排序。RANDOM_ORDER复制原native RandomSampler逐位顺序；PAIRED_ORDER把同允许RX/day/eq且sig/base_index差1–2的非重叠物理邻包成pair，再随机shuffle所有pairs及pair内朝向。sampler只dataset.index的rx/day/eq/sig和dataset.selected/base_index，不读TX/label/IQ/域内类别身份，不按真值类别配额或RX分类路由。Dataset/collate/worker/pinning/drop_last保持，私有generator生成顺序；每次迭代与native恰好同一个globalTorch seed draw，sampler外部RNG逐位相同，初始全model状态、默认logits与joint160逐位原基线。数据初始化时没有随机状态消耗。

保留原confidence RX/day quantile .86与阈值、batch_neighbor window2/预测一致、strongagreement、EMA .999、损失权重、普通cos训练和原输入。不关闭temporal gate，不改λ/温度/margin/头。网络原Sinc/rawHz、time/frequency/PA/reference/norm/gain/内部补偿全保留，无Sinc/TIME新模块/新损失/新分支；NoExplicitCFO保留。默认ResidualPhysicalHead cosine6×160、scale30正常联合学习，单独头拟合/冻结头/refit/原型/RX读出均0。身份177025参数，丢弃类别W960后真实fθ176065，joint160/id_feat_joint导出逐位一致且W独立。

固定三模型种子2026092701/02/03，源split392005、L6300/U56700（TX隐藏）/V27000、源RX1/3/4/6/8与day1/2/3、旧6TX与物理ID完全原样；L占训练池10%。每新模型scratch50/11100update，33label＋17pseudo，AdamW lr0.0002→0.000001/cos/WD.0001/smoothing.01、L128/U256。旧32＋18谱系不混入。label33阶段仅以U loader长度确定222step，从不取U IQ；pseudo17阶段每U包恰一次、B256尾124，共963900包。采样诊断按真实0/17 active迭代记录，绝不声称label期U被消费。终轮固定，无V/target选模，无星地训练。

三组各3seed：OLD_GATE_COS50严格核验原M12固定final50默认决策后仅复用；两个新arm各3seed。主候选PAIRED_ORDER，配对RANDOM−OLD、PAIR−OLD、PAIR−RANDOM。历史对照150model-epochs单列，新6models=300epochs/66600updates；每个骨干≤50，祖先为空，自身scratch教师来源，不加载目标得分权重。主目标是固定clean默认头三seed均值≥84%，不是最高单seed/其他读出。

先训练seed01两个新arm。两者final222step labeledCE均≤.5，且PAIR17Uepochs accepted/963900≥.25才释放其余4newrows；源规则固定，不读U truth/V分数/target排序。未满足时停止释放后续row，保留首seed训练与自动评分/缺项，不改阈值重跑。源U诊断从已有gate前向读取RX/day/预测类直方图/接收/强弱一致，不新增U前向。source sampler记录每活跃pseudo迭代恰一次；额外V33/34健康检查恢复模式及Python/NumPy/Torch/loader RNG，无梯度/optimizer/EMA更新。总额外V12/324000包，头更新0。实际source冻结后以固定合法L720检查真实feature变化、W独立及feature-only一致，诊断成本另列。

每合法固定final checkpoint自动defaultclean预测→独立truth-last→同row分析；Only ORIGINAL_HEAD。clean ManySig168000包/row、旧6类7targetRX4days既有opaque IDs与IQ，不按target统计/类配额/全局重排。全9row=1512000决策/162CM指标，报告3seed均值样本SD、Macro-F1/最差RX/day、逐seed配对与正差数、修正/引入错误。其余seed是否放行只看源条件；score绝不回流任何候选/参数/停止。已暴露benchmark不称新盲测/新TX/Phase2/卫星部署。

本地合成PASS：真实56700规模/222批/尾124、RANDOM全顺序与sampler外全局RNG精确原native；PAIR全池一次/物理gap、标签访问陷阱、label阶段222Lstep但不取U；model全状态/defaultfeatures/导出/W独立及6协议负例。真实noquery CP smoke留给remote preflight，源qualification只复用已完成结果，无重复数据验证或额外authority/receipt/seal。合成验证不冒充训练或DG结果。唯一新训练集成P0/P1审查仅限下次实际运行的直接缺陷，旧源诊断代码不重复审查。

执行：普通N607既有CVS-RFFI fullFP32，线程2；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_neighbor_pair50_20261014_r02，命令 python -u -m experiments.cvs_neighbor_pair50.dispatch，实际GitHEAD见release_commit.txt。不可变输出 /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02，每row独占source/prediction/scores/日志；已有时只核实，不覆盖。共享锁按全部owner每GPU最多3实验进程且空闲≥12000MiB，真实GPU记录launch.json。code/config commit+push核对remoteOID，一archive SHAcomparison/remotecompile/realnoquery smoke，PASS立即dispatch；一次独立PID/CWD/argv/GPU/loggrowth读回。

系统技术失败（所属来源/roles/nonfinite/budget/hiddenU/sampler/defaultpredict/score）关闭新dispatch，保留所有健康row与产物，不自动retry，不操作其他owner。科学源反证失败同样保留partial。所有implementation/config/正负result/report Git交付并核对remoteOID；原IQ/权重/大NPZ/log留原存储，完成训练不称全部实验完成。

唯一新训练集成P0/P1审查PASS；独立合成原生50轮确认label33零U迭代、pseudo17完整56700/尾124，总963900，未读真实IQ/目标/权重，无第二次旧源诊断审查。


r02技术恢复：r01远端合成NumPy dtype推断失败、未进入真实训练。仅显式整数转换修复，同原环境检查PASS；生产模型/采样/优化/阈值不变、原失败保留。仅对此互操作问题一次针对性复核PASS，无全量重复审查。
