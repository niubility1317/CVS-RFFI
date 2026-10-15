# SIG_TIME＋RDrop：特征余空间曲率补全源域预登记

状态 RUNNING，四行启动 VERIFIED。只优化 SIG_TIME＋RDrop，固定原 SIG_TIME 与默认头联合训练。原版 RDrop E20/E50 三种子 clean 均值83.942857%/84.340079%的正收益保留；差分梯度、跨 RX 证据、互补掩码和包络的失败配方保持关闭。本轮新增的是原版预测一致性之外的特征学习信号，不救跑既有失败算子。

最新[同图梯度诊断](../20261016-phase1-SIGTIME-rdrop-operator-gradient-source-s2026092701-r01/report.md)证实，MEAN/CONFIDENCE/ENVELOPE同时改变强度和方向，不能把损失系数相同当作同样优化力度。本候选完整保留原双向 KL，不把它替换成教师目标。其假设是：默认六类头只观测160维归一化特征中的至多5个类间对比方向，双向 KL 对其余方向的同包 dropout 差异没有直接的类间约束。补充这些方向的稳定性可能减少部署特征中的随机依赖，也可能损害有用变化，须用公平源训练证伪。

令 u_v=normalize(z_v,eps1e−4)，其中 z 是实际部署 joint160；A=normalize(默认W,eps1e−4)，C=A−mean_rows(A)，以C的右奇异向量（sigma>sigma_max×1e−6）构造B，P=B Bᵀ，数值rank固定为5。W、B及以下曲率系数仅在辅助项中停止梯度，默认W仍由原CE和原双向KL更新，无冻结骨干、换头、重训头或拟合诊断分类器。新增项没有参数。

同一对随机前向得到 pbar=(softmax(logits1)+softmax(logits2))/2，计算停止梯度的 t_i=900×[sum_k pbar_ik ||A_k||²−||pbar_i A||²]。这是在环境归一化特征坐标u中的默认softmax Fisher矩阵迹，不是原始z或共享编码器参数中的Fisher迹。用 d_i=u1_i−u2_i，四个固定条件如下：

| 条件 | 在原 paired CE＋原 full KL＋原 U 上加入 |
|---|---|
| FULL | 0；原版 RDrop 控制 |
| SPAN | .5×mean_i[t_i ||d_i P||²/5] |
| ISOTROPIC | .5×mean_i[t_i ||d_i||²/160] |
| NULL（主候选） | .5×mean_i[t_i ||d_i(I−P)||²/155] |

三种新增度量的总迹都为t_i；它们在同一模型同一图上具有相同曲率总量。不同训练轨迹的W、概率与t会变化；本对照不匹配实际loss、参数梯度范数、clip或AdamW位移，也不声称优化强度相同。新增系数固定1，不扫alpha/rank/温度/层位。全空间对照检验普通特征一致性解释，SPAN检验继续加强已有判别方向的解释。

NULL指环境u坐标中centered-A的核；它对类间logit差异不敏感。u由归一化z得到，两视图及共享参数Jacobian不同，因此不保证原始特征中心、编码器中心或实际AdamW步不变。这个核也不是已识别的接收机干扰空间，不能把全部155维当作域噪声或声称已保留新TX信息。默认头推理、部署特征和网络架构不变。

四行固定下一预定义模型seed2026092702，从零E20=13L＋7原U，每轮222步、共4440步。旧RAW_HZ、普通增强use_aug=True、LS.01、scale30/eps1e−4默认cosine头、AdamW lr.0002/wd.0001、cosine日程、自身EMA与旧U门控保持。原生独立dropout、两次前向复用实际增强后的同L128输入，不重抽增强，无额外前向、步数或星地增强。

真实源RX1在训练/U/teacher/calibration/cache/继承路径全排除，fit RX3/4/6/8、day1/2/3、六TX。有效L5040/U45360hidden/V21600，U真值不可读，V只监测。固定final20用checkpoint默认头预测1260合法held L；保存逐包CE、margin、真类概率、day/TX和paired修正/新增错误。RX1此前已用作开发，不能称新盲测。共用helper另读取144fit L anchor但未拟合或参与训练，计入成本。0target。

科学资格预先固定：四行finalLCE≤.5；主NULL相对FULL、SPAN、ISOTROPIC同时严格少错且meanCE不高；NULL辅助项在至少90%实际L步骤为正。数学/协议错误按技术失败处理；同一科学条件失败就关闭本配方，不扫alpha、rank、layer、seed、temperature或epoch救援。PASS只进入另行预登记的固定RX3排除源确认，四臂/seed/E20不变；确认前不发布主目标测试。之后若确认通过，另行登记E20/E50三模型seed完整验证，逐row冻结后自动clean预测与独立truth-last评分，不以目标分数选方案。

本批4×scratch20=80 model-epoch/17760更新；每行568320额外L身份前向、396368循环U观察、1260held决定，四行合计2273280/1585472/5040。每行首步自己的随机初始真实CP用2合成包no-query smoke，PASS后立即继续。无任何trained CP/teacher/EMA/optimizer祖先，代码依赖不等于权重继承。

CPU/CUDA各53协议负例、7科学否决、实际198同初态、CE/logits/native RNG一致、FULL原梯度一致检查PASS。核/投影、直接Fisher迹对齐、附加metric迹、类标签置换/两视图互换、相同视图零损失和梯度通过；辅助项实际进入t1/t2/SIG且对默认W无直接梯度，CE+KL对W梯度保留。各1280合成身份前向包次，0真实IQ/CP加载/optimizer/headfit。source进度记录形状已无数据检查。一次独立下一运行P0/P1审查PASS，无P0/P1，记录在review.json。

唯一owner `codex/root/SIGTIME-RDrop-curvature-completion-20261016`，ordinary N607 CVS-RFFI，CPU2/loader0。CWD PROJECT/releases/cvs_SIGTIME_rdrop_curvature_completion_source20_20261016_r01；命令 `CVS-RFFI/bin/python -u experiments/cvs_sigtime_rdrop_completion20/runner.py experiments/cvs_sigtime_rdrop_completion20/configs/matrix.json`；不可覆盖root PROJECT/runs/20261016-phase1-SIGTIME-rdrop-curvature-completion-sourcefold-s2026092702-m4-r01；实际Git身份写release_commit.txt，原代码依赖b3ac442e2d33a5b315fb3251c30e0ad7083566bc，FOLD显式提供default-head-only trainer。预期逐row初末CP/源角色/freeze/完整日志/rdrop_completion_progress/result/source_readouts和全matrix completion/qualification，见experiment.json。一次ordinary preflight、一个release archive SHA本地/远端比较、remote compile；独立回读PID/CWD/argv/GPU/日志增长。所有owner和preCUDA总进程计入每GPU≤3。技术错误保留所有产物、停止pending、不重发或停止健康/无关进程。source-only例外不测目标/卫星。

文献：[R-Drop](https://arxiv.org/abs/2106.14448)已有同包双向KL；[Fraternal Dropout](https://arxiv.org/abs/1711.00066)已有两mask预测差异正则；[LR-Drop](https://aclanthology.org/2024.lrec-main.891/)已对hidden states、attention及output做分层一致性。因此“额外加特征一致性”不能单独声称创新。本候选待证点是保留有效KL，按当前默认头的可见/余空间与解析等迹预算补充信号，并通过方向对照检验独立作用。有限检索不能证明全球新颖，更不能证明泛化收益。

REJECTED_EXTRA_GATE：通用brainstorming技能中的额外设计审批/书面spec审批不在用户Exclusive Minimal Experiment Workflow白名单内，作为NONBLOCKING不执行；意图、固定方案及停止规则均在本最小报告，不增加审批/签名/重复审查。goal ACTIVE。

<!-- LAUNCH_VERIFIED -->
执行提交 `78ad998e9dcb5653573fa801538e914189833d20` 已推送且远端OID匹配。单次archive SHA比较、remote compile均VERIFIED。四行own初始CP smoke、独立PID/CWD/argv/GPU/CPU与日志增长、实际源角色/RX1排除及初轮rank5/Fisher/等迹审计已核实。训练和最终源评价尚未完成，未访问target。见launch_verification.json。
