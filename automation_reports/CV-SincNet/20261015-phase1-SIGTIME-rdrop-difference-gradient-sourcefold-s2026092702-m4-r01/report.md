# SIG_TIME＋RDrop：差分梯度机制源RX留出验证

状态 PLANNED，已登记，尚未发布。用户现只优化SIG_TIME＋RDrop，重点为RDrop机制。旧完整E50均值84.340079%的性能里程碑保留；本试不继承任何已评分权重，不读target，不据该分数调系数或选择父权重。刚实现的特征残差交换已暂缓、未发布，不纳入本试。

唯一新假设：RDrop的KL同时推动W、两次特征的共同方向和随机差分；只对随机差分反传，可能更直接降低dropout敏感性。设KL到joint160的梯度g1/g2，将其替换为(g1-g2)/2与(g2-g1)/2，局部共同分量为0。它等价于对特征中心停止梯度的surrogate；不保证实际AdamW更新后中心不动。

四条件只有KL反向不同：FULL_KL原RDrop；FEATURE_KL仅在KL计算中detach W，CE仍正常joint更新原W；SHRINK把FEATURE的梯度缩至其自身差分投影范数；DIFFERENCE执行差分投影。四条件前向CE和KL逐位相同，均alpha1、普通增强后同IQ两次native dropout、LS0.01、RAW_HZ、默认cosine6×160/scale30/eps1e-4。SIG_TIME、CFO内部处理、U、ownEMA、日程和dropout率均固定。SHRINK只匹配本地feature-gradient范数，不匹配不同训练轨迹的参数位移，不能单凭此声称纯方向因果归因。

固定模型seed2026092702为既定种子序列的下一独立源种子；四行各scratch20/13L+7U/222steps，共80model-epoch、17760更新。拟训练RX3/4/6/8，有效L5040/U45360隐藏/V21600；RX1从训练、U、teacher、calibration、probe和cache均先排除，再建立模型。只在fixed final20用默认头读1260合法RX1 L。144额外fitL anchors由既有fold代码读取但不用于本试，单列成本。U为每轮56624重复观察、每行396368；物理U仍45360。总额外L前向2273280、U观察1585472、held默认决定5040；另8个合成ownCP smoke包。

预登记科学资格：四行final L CE<=.5；DIFFERENCE累计移除共同梯度能量至少1%；它在1260 held L上的错误严格少于FULL、FEATURE和SHRINK，meanCE均不高于对照。V只健康诊断，不选终点。PASS才另行登记main；FAIL关闭固定算子，不扫alpha/层/seed/配方/轮次救跑，不访问target。

CPU和CUDA检查各51负例、5资格布尔情形、1024合成identity包PASS，198初始化state匹配；原默认头logits、前向目标和native RNG逐位相同。模型浅中层/SIG/W均有CE梯度，FEATURE/SHRINK/DIFFERENCE的CE W梯度逐位相同。差分共同梯度为0；SHRINK范数相对误差CPU4.8011e-8/CUDA1.5178e-8。0真实IQ、CP加载、优化或头拟合。这不是训练收益。真实checkpoint无query smoke在每worker首步，PASS立即训练。

最小流程：一次聚焦P0/P1、实际Git提交推送及OID读回、一个新root、一次ordinary N607 preflight、单archive SHA比较和remote compile；每GPU所有owner含preCUDA最多3进程，CPU2/loader0。先留PID再有限2s观测，不重复Popen。技术错误保留产物、停pending、不杀健康或其他任务；科学低分不重训。运行代码commit写release_commit.txt。模型结构代码依赖b3ac442e2d33a5b315fb3251c30e0ad7083566bc，仅代码，不加载旧模型。

环境ordinary N607 CVS-RFFI；CWD PROJECT/releases/cvs_SIGTIME_rdrop_difference_gradient_source20_20261015_r01；命令CVS-RFFI/bin/python -u experiments/cvs_sigtime_rdrop_route20/runner.py experiments/cvs_sigtime_rdrop_route20/configs/matrix.json；唯一owner codex/root/SIGTIME-RDrop-difference-gradient-20261015；root PROJECT/runs/20261015-phase1-SIGTIME-rdrop-difference-gradient-sourcefold-s2026092702-m4-r01。逐row配置/seed/路径/命令唯一维护于experiment.json。

文献边界：[RDrop](https://arxiv.org/abs/2106.14448)已提出dropout子模型双向KL；[Fraternal Dropout](https://arxiv.org/abs/1711.00066)已约束两次pre-softmax预测差；[CR-Aug](https://arxiv.org/abs/2205.12461)已使用一致性stop-gradient。本试的具体问题是默认cosine头下，保持同KL前向，通过joint160共同/差分梯度分解和范数控制检验跨RX学习，不声明global首创。检索是有限原文核对，不是排除全部先例的证明。

预期输出包括完整step/epoch/source角色/初末checkpoint/ownEMA/冻结、route backward审计、1260默认预测和逐day/TX/配对错误。完成后独立只读完整分析；本试明确source-only，无目标clean、practical或Phase2发布。

Independent next-run P0/P1 review: PASS. No direct defects found. Scope and checked execution paths are recorded in review_evidence.json.
