# 连续层级时序特征：首种子最小实验

依据用户附件报告路线二和 20261014-phase1-source-hierarchy-geometry-manysig-s01-m1-r01 固定源证据（L720、连续/direct64误差8.882e-16、native32误差7.153e-7；省略cross-area的feature/time相对RMS37.0827%/87.0831%）。一个核心假设：在原SIG_TIME基础上，显式保留相邻和全局的有序关系能否强化实际fθ。源几何不是性能结论。

固定seed2026092701，5row：历史OLD_GATE_COS50仅复用比较；4新从零SIG_TIME、LOCAL_GTIME、HIER_STAT、HIER_ORDER。所有新模型50轮/11100updates，33label＋17pseudo，旧batch_neighbor门控＋自身EMA.999＋原cosine schedule。默认6×160scale30 cosine头按原流程与骨干联合学习；0额外分类头/RX/day/channel诊断头拟合，不冻结骨干再训头。全部默认头clean预测固定后独立truth-last评分，每row168000包，合计840000预测/90组指标，逐row自动完成，不等全矩阵才预测。

原SIG_TIME四个8点窗分别localreset0至1；LOCAL_GTIME只将局部窗改成整32位置的单一globalclock，其他位置保持。HIER_STAT与HIER_ORDER均为四个局部globaltime签名＋两个相邻16位置＋一个全32位置描述子；共享单6D Conv+tanh，t2完整postDropout取点，原主干Dropout.1/t3/frequency/PA/reference/NoExplicitCFO/internal compensation保持。没有叠加preDropout、U权重、更多卷积核/SiLU/Sinc变更。

HIER_ORDER一阶increment7＋二阶skewarea21，三个边界增量7→8/15→16/23→24参与连续4→2→1组合。HIER_STAT同局部globaltime签名，粗上下文使用mean6/variance6/offdiagcov15，经同固定fullrank27→28lift。两层级196→160，pool32416/fθ208481、classW960，实际初态trainable参数/nativebody/W一致；局部两方法112→160，pool18976/fθ195041。层级比较匹配窗口覆盖、输出维度和可训练容量；不宣称原始量级、信息/内在秩和梯度条件也完全匹配，因此不是单独因果隔离顺序的实验。仍为真实joint160/id_feat_joint，可独立导出仅丢W、保留norm/gain/新模块。

配对：LOCAL_GTIME−SIG_TIME分辨clock变化；HIER_STAT−LOCAL_GTIME检查大窗统计/容量；HIER_ORDER−HIER_STAT比较同容量粗上下文的有序/统计表示；候选与局部/原SIG_TIME/旧门控cos的比较全部保留。预登记primary HIER_ORDER；首seed不能完成三种子84%目标，保持暴露基准背景，不称新TX/Phase2/星地部署。

固定L6300/U56700隐藏TX/V27000health-only，训练pool标注10%；split392005/sourceRX1/3/4/6/8、day1/2/3、旧6TX与physical IDs不变。新4模型无checkpoint或teacher继承，teacher只ownscratch；历史基线final50/11100/33+17来源通过既有source_proof，仅比较和真实CP2零IQ无查询smoke，不加载训练初始化。复用VALIDATED_ONCE数据，不重复数据验证。

源扩种子判据提前固定：所有4模型final L CE≤.5；各新增trainable张量数值更新，feature-only导出与原z相同、W独立，HIER_STAT/HIER_ORDER初态trainable参数及nativebody/W相同；终态固定L720候选−统计控制joint160平均L2≥1e-6。仅由该SOURCE PASS决定原样02/03；不读取target指标控制扩展，不扫参数/换seed/挑权重，不用V小涨分选模。每row仍固定终轮，目标结果不改变其他健康row。

预算4×50=200新增model-epochs/44400updates＋1历史50轮复用；CPU2threads/fullFP32。实现/config 588209f4186e3056a310458237f4e478c6c15a64。唯一独立直接P0/P1审查PASS，无发现；实际native4模型局部梯度/数值更新/默认头grad/teacherdeepcopy/exactexport/W独立、连续组合及grad与direct等价、5协议负例和可选参数计数回调通过，无源/目标IQ访问。

命令 python -u -m experiments.cvs_hierarchy50.dispatch；普通N607既有CVS-RFFI；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_hierarchy50_20261014_r01；不可变run/output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-hierarchy-oldgate-cos-e50-clean-manysig-s01-m5-r01，日志 /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_hierarchy50_20261014_r01/dispatcher.stdout.log；owner codex/root/hierarchy-20261014。共享capacity lock计入所有owner，每GPU最多3总进程且空闲≥12000MiB；GPU按发布时实际资源排队，登记实际PID/CWD/argv/GPU/log增长。一个archive单次本地/remote SHA比较和remotecompile；launcher先真实CP无查询smoke，PASS直接运行。

停止规则：角色/来源/默认头/初始化/非有限/预算/实际导出状态/physical IDs/预测或truthlast技术失败停止新派发，保留现有健康row/产物，不能自动重试或按目标早停。数学容差非有限由实际训练检查处理。预期每rowinitial/resolved/checkpoint/source_frozen/完整epoch与step日志/source_feature_witness/prediction/scores/completion；run source_features/source_qualification/analysis/completion和launch/queue_state；大型私有IQ/权重/日志/预测不上传Git，只提交实现/config/聚合结果和对应报告，独立核对remoteOID。

实际发布VERIFIED：执行7bdfdbf448b9c97ead5702f958bee293d646a09f、dispatcher2191810；4新worker2192180/2192262/2192348/2192433，GPU5/6/4/7，实际resolved四方法/t2/112或196描述子/defaultjointhead/nativefittedprobesdisabled正确。preflight/remotecompile/单archive比对PASS，PID/CWD/argv与日志增长核实。训练健康中，完整源资格/目标性能尚未完成，不据此宣称提升。
