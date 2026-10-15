# SIG_TIME＋RDrop：批次中心化方向一致性源域预登记

状态 PLANNED。本批仅优化 SIG_TIME＋RDrop 的训练机制，SIG_TIME 和默认分类头保持原联合训练。原版 RDrop E20/E50 三种子 clean 均值83.942857%/84.340079%的已验证收益保留；互补掩码、跨RX证据、差分梯度、教师包络和NULL余空间固定配方的失败不重启。本批不是这些失败配方的参数救跑。

## 依据与可证伪假设

上一[源训练](../20261016-phase1-SIGTIME-rdrop-curvature-completion-sourcefold-s2026092702-m4-r01/report.md)中，全空间ISOTROPIC对原FULL修正45/新增27错，净少18错，CE从.766282降至.715820；主NULL未胜过ISOTROPIC，按原规则FAIL关闭。之后[冻结特征诊断](../20261016-phase1-SIGTIME-rdrop-feature-geometry-source-s2026092702-m4-r01/report.md)给出默认W之外的特征正迹象：whole-feature关系满足率92.4757%→93.1034%，但只有4/6TX改善；平均特征范数10.08→19.10，共同均值能量比例84.53%→90.45%。类间绝对距离间隔反而变小，不能把距离收缩当作身份几何全面改善，也不能断定共同分量就是RX噪声。

新假设：保留有效的完整预测KL，同时对去除当前训练批次共同分量后的方向施加同包双mask一致性，能否产生更好的身份学习信号。中心只在训练当前L批次瞬时计算，没有持久中心、额外网络、额外头或推理状态。它与旧每包差分梯度投影不同。

## 固定机制与对照

实际部署joint160为z1,z2，复用同一实际增强后L128输入，原生独立dropout。mu=.5×mean_i(z1_i+z2_i)，**mu参与反向传播**。normalize均使用eps1e−4。T_i=900×[sum_k pbar_ik ||A_k||²−||pbar_i A||²]，A=normalize(默认W)，pbar=(softmax(logits1)+softmax(logits2))/2；T double计算后转FP32并停止梯度。T是环境归一化特征坐标中的默认softmax Fisher迹，不是编码器参数中的Fisher。

记P=.5×mean_i[T_i ||normalize(z1_i)−normalize(z2_i)||²/160]，C=.5×mean_i[T_i ||normalize(z1_i−mu)−normalize(z2_i−mu)||²/160]。每组总目标都包含原paired smoothedCE＋原full symmetricKL＋原hiddenU；固定alpha1，附加项如下：

| 条件 | 附加项 | 用途 |
|---|---|---|
| FULL | 0 | 原版RDrop |
| POINT | P | 原ISOTROPIC正路线，独立源折确认 |
| NORM_MATCH | stop(||g_C||/||g_P||)×P | 区分方向改变与特征边界梯度力度 |
| CENTER（主候选） | C | 当前批次共同平移不敏感的方向一致性 |

g_P/g_C只对当前图中的原始joint160对(z1,z2)求导；两次autograd遍历在此停止，不累积参数grad、不更新优化器，不增加模型前向。范数匹配只在当前图中成立，不等于跨独立轨迹、模型参数、clip或AdamW位移匹配。ratio停止梯度且不截断；两范数都为0时ratio1，POINT为0而CENTER非0则技术失败。匹配相对误差≤2e−5，CENTER共同平移梯度比例≤1e−5，实际逐步记录。

在固定停止梯度T条件下，给全部z1/z2加相同向量不会改变C，两视图全部样本特征梯度之和为0。这只描述辅助项；完整CE/KL和T取值、编码器中心、共享参数Jacobian、实际AdamW步及接收机不变性均无此保证。当前batch是随机L采样，无跨RX身份配对或类专属权重。

## 数据、预算与预定判据

固定seed2026092702，四组各scratchE20=13L＋7旧U，每轮222步、各4440步。RAW_HZ、普通增强、LS.01、默认6×160 cosine头scale30/eps1e−4、AdamW lr.0002/wd.0001、cosine及自身EMA/旧U门控不变，无星地训练。无冻结骨干、换头、单独重训或拟合诊断头。辅助项对默认W没有直接梯度，CE和原KL继续联合更新W。

本批真实留出源RX3，fit RX1/4/6/8、day1/2/3、六TX。原source carrier合法元数据核对后，在模型/EMA创建前过滤训练L、probeL、U、V、calibration五loader，保留原15个RX×day静态域ID不重编号。有效L5040/U45360隐藏TX/V21600。U只看RX过滤，不读TX；V只健康监测。自己的随机初始真实checkpoint严格加载2合成包noquery smoke为worker首步，PASS立即继续。没有trained checkpoint/teacher/EMA/optimizer祖先；远端release仅为代码依赖。

固定final20学生checkpoint直接用默认头预测1260原可见RX3 L，保存逐包opaque物理ID、标签、预测、CE、margin、真类概率，独立复核完整日志/决定，不另拟合头。全部是开发源域，不能声称独立盲测、目标接收机、新TX或卫星泛化。共用helper另读取144fitL anchor但不参与拟合或训练，计入成本。

**主判据**：四行finalLCE≤.5，CENTER辅助项≥90%步骤为正；CENTER必须相对FULL/POINT/NORM_MATCH全部严格少错且meanCE不高。技术数值及过滤检查必须通过。PASS只允许另行预登记同四臂/seed/E20的RX1源折确认；FAIL关闭此固定CENTER配方，不扫alpha/seed/temperature/epoch/layer救援。

**独立预登记正路线确认**：POINT相对FULL严格少错且meanCE不高，四行健康且POINT辅助项≥90%步骤为正。此前RX1的+18错及特征证据保留，本批RX3结果作为第二开发源折确认。POINT结果不替代CENTER主判据、不救活NULL；PASS仅支持另行预登记后续工作，FAIL如实记跨折未确认。此批没有target队列，不能据某一臂源结果临时改主候选。未来正式E20/E50三模型seed须另行登记并保留同预算原版RDrop对照，冻结后自动clean预测及独立truth-last评分。

总4scratch20=80 model-epoch/17760更新。每行额外568320 L身份包次、396368循环U观察、1260heldL、1404额外L读取（含144unusedfit），四行各项合计2273280/1585472/5040/5616；另首步2合成包/行。POINT/CENTER/NORM_MATCH各有2次特征边界autograd/步，FULL没有此附加遍历；实际walltime单列，不承诺同计算成本。

## 执行与验证

CPU/CUDA各59协议负例、主资格与POINT独立判定、198同初态、实际CE/logits/nativeRNG一致、FULL原版梯度对齐PASS。解析Fisher迹、共享平移loss/梯度、norm匹配、类置换/样本置换/视图互换、同视图零loss/梯度均通过；辅助梯度进入t1/t2/SIG且对默认W直接梯度None。各1280合成包次，0真实IQ/CP加载/优化步/头拟合。源RX3元数据过滤覆盖L/U/V；U模拟项根本无TX字段。具体证据见local_verification.json。唯一一次独立P0/P1下一运行审查记录review.json，未完成前不发布。

唯一owner `codex/root/SIGTIME-RDrop-batch-center-20261016`。ordinary N607 CVS-RFFI，CPU2/loader0；每GPU计全部owner及preCUDA总实验进程≤3。CWD `PROJECT/releases/cvs_SIGTIME_rdrop_batch_center_RX3_source20_20261016_r01`；命令 `CVS-RFFI/bin/python -u experiments/cvs_sigtime_rdrop_center20/runner.py experiments/cvs_sigtime_rdrop_center20/configs/matrix.json`；不可覆盖root `PROJECT/runs/20261016-phase1-SIGTIME-rdrop-batch-center-RX3-sourcefold-s2026092702-m4-r01`。代码身份为一次Git提交，实际commit见release_commit.txt；模型代码依赖b3ac442e2d33a5b315fb3251c30e0ad7083566bc，runner显式加载default-head-only FOLD trainer。逐row预期初末CP/实际args/角色/冻结/完整日志/rdrop_center_progress/result/source_readouts与全矩阵completion/qualification，详见experiment.json。一次普通账户preflight、一个release archive本地远端SHA比较、remote compile；独立读回PID/CWD/argv/GPU/日志增长。技术错误保留产物并停止pending，不重发、不停止健康/无关进程。本source-only批不访问目标/卫星/Phase2。

文献边界：[R-Drop](https://arxiv.org/abs/2106.14448)已有同包双向KL；[LR-Drop](https://aclanthology.org/2024.lrec-main.891/)已有隐藏状态一致性；[Barlow Twins](https://arxiv.org/abs/2103.03230)和[VICReg](https://arxiv.org/abs/2105.04906)已有表征一致性及中心化/方差/协方差学习背景。仅“加feature consistency”或“减均值”不能声称全球创新，本候选也不继承VICReg防坍塌保证。待证点是当前SIG_TIME上保留有效KL，使用可微共同中心约束方向，并以边界梯度范数对照区分作用；有效性尚待本批及独立后续验证。

REJECTED_EXTRA_GATE：通用brainstorming中的额外设计/spec审批不在用户最小流程白名单，NONBLOCKING不执行。固定方案/矩阵/判据在此，0额外审批/签名/重复审查。goal ACTIVE。
