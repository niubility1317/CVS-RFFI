# Phase3 层次密度发现与匿名注册（2026-10-31）

Run `20261031-phase3-hier-register-s2026092701-r01`，模型seed2026092701，唯一主 `predicted_REG1`；四列仍为 pooled/predicted×REG0/REG1。检验上轮固定sourceε未形成注册类后，层次密度发现能否改善独立未来新类识别；不按新结果换主、扫参数或重抽观测。

## 唯一方法变化

沿用原source-only r03、原F0 Tri-SIG-TIME SSL E50/11100/default6、DURA v1/R69 E20/4940、source density160 known90/OE32/whitener及两组六阈值。只把完整U的聚类改为sklearn HDBSCAN，固定min_cluster_size16/min_samples8/euclidean/brute/n_jobs1/alpha1/cluster_selection_epsilon0/eom/allow_single_clusterFalse/copyTrue/max_cluster_sizeNone。它利用密度层级选择稳定簇，不使用原sourceε切割；定义见[官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)。本地1.7.2/N6071.7.0已只读核实签名，无安装或环境修改。

仍按opaqueID排序与最小ID canonical编号，簇至少16且原predicted拒识比例严格过半才注册，全部簇成员形成Gaussian；K自由含0，不能读取真实类别数/配额。新类条件密度无1/K权重，v按n/(n+160)向原source pooled variance收缩。仍与原旧类最高margin共同竞争，同分保旧类，最大分<0拒识。

匿名状态冻结后，原source V27000仅为两种base各校准一个global注册阈值，保护known12（overall/5RX/6scene）C_K≥90%且同DURA原F0闭集−旧A_K≤2pp；30个RX×scene仅诊断。保留旧两组六阈值精确控制。旧类margin不改变，新增竞争自身不能增加旧正确接收或同一later集合拒识；本轮只检验注册部件，不用A_N单项替代完整Phase3综合晋级，后续无标签适应与拒识仍须优化。

## 合法缓存及数据

只继承上轮 `20261031-phase3-signal-register-s2026092701-r01` / code `e48f3b357eccb12e778c1eb348776e5c6b457922` 的source聚合与U公开特征：source density与当前原模型逐值相同，source_radius作为来源记录保存、newfit0且不参与HDBSCAN；U只取ids/default_logits/closed/signal_features，忽略上轮聚类分配/拒识和注册结果，重算base margin。禁止用旧score、注册校准阈值、sourceNV或Q输出拟合新簇。

祖先输入权限与模型身份核对后，actual三CP synthetic2 smoke先执行，再读source聚合/U缓存。缓存是原G10星地received→同一DURA输出→冻结原F0/r03，VALIDATED_ONCE capsule/split不变；不重读L/OE/U/Q原始IQ或执行真实网络前向。source新L读取/sourceε拟合均0，原sourceL6300与GF/U前向成本明确继承，不把复用写作零总成本。公共兼容字段dbscan_labels在本run承载HDBSCAN labels，诊断显式写算法及版本。

原split392005：L6300、U_s56700、OE6134（仅outside-six/publicRXday）、V27000（每旧类4500额外校准标签）、source NV3150（19-15/19-16/7-1各1050，truth仅scorer）。旧六类14-10/14-7/20-15/20-19/6-15/8-20；源RX1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，2021-03-08/15/23，L每class/RX/day70，原L/(L+U_s)=10%。

U384来自RX24-1/index26/2021-03-08，capsule `p3dev-star-g10-6e1292b62ab09286ebf1a0df` / split `83c5cc1fe2f2a1512dbc1287`。future Q为同RX/03-15的864物理记录：旧六类×96=576，非旧三类×96=288。U、futureQ与源角色物理互斥；clean/star复用同一Q物理记录，均是同DURA后的测试，clean不进入发现或校准。

G10 MIX01保持600km/25MHz/2.45GHz/residual/post_sync/noEQ/referenceSNR30dB/reflect、郊区城区各0.5、原覆盖面积六分量，receiver2027/scene2026103001/channel2026104001，一record一固定received。只改算法不触发数据重验/新hash；历史NV/Q曝光、单seed/RX、日期与源重合如实披露。正式RX24-18不访问。

## 冻结顺序与指标

source聚合及U匿名状态冻结→sourceV两globalτ＋旧two6τexact→全部阈值冻结→source NV与两个cached Q完整预测→独立CUDAempty CPU才读Utruth/sourceNVtruth/两Qtruth。匿名ID的Hungarian一一匹配只用U真值，不回流、不用Q重排。全部真实U到达类进入A_N，未发现/未匹配计错；unmatched匿名接受计错，不当拒识。共享默认旧流程和新流程使用局部policy，不能通过改全局身份混用run。

REG0主宏A_K/C_K/R_U/H的未知集合为三非旧类；REG1主R_U/H用U未见later集合，另报实际未注册集合和固定later前后配对，禁止跨分母伪增益。新类未来A_N、H_KN、H_OW、预期注册集合（旧六＋全部真实U到达类，含未发现）macro accuracy/CK/FRR单列，实际匿名数K另列。U簇数/纯度/混入/合并拆分不能替代未来Q识别。

沿最新用户v3完整判定：两future条件overall A_N≥80%，旧A_K较同base REG0下降≤2pp且较同DURA原F0闭集下降≤2pp，旧及预期注册宏CK≥90%，later RU clean≥95%/star≥90%；主景完整roster及旧保护/接收/later约束，scene A_N报告但不新增scene80门槛，source known12校准须通过。high缺19-15时相关指标N/A，不认证完整六景；源失败或K0也完成全部预登记矩阵。本轮没有target DA或持续多session注册完成声明，完整目标ACTIVE。

## 成本与执行

新NN/真实F0/G/signal前向/sourceεfit均0；实际三CP合成各2。新增U384的HDBSCAN和实际K匿名聚合、两套sourceV注册校准；K0则只有零占位且无有效新类阈值学习。原sourceL6300signal/U384GFsignal、sourceε、GMM、F0/r03/G训练和Q缓存成本继承。部署NN834665及实际聚合/阈值字节单列，不部署source/U逐行bank；过程耗时不当B1时延。

N607唯一owner `codex-phase3-hier-register-20261031`，不可覆盖新run/release，命令/路径/六类seed见experiment.json；singleproducer GPU实际free≥4096MiB、每卡新增本任务≤2，独立CPU scorer，保留其他owner。聚焦数学/缓存/默认回归/实际新producer→CPU贯通、一次直接P0/P1、一个code commit、单preflight/archive SHA比较/remotecompile、启动PID-CWD-argv-GPU-log核实。非法角色/祖先/非有限/输出碰撞/无法合法prediction保留，不partial truth评分/覆盖重启；UNKNOWN先核原PID/root，科学负结果不停止健康任务。
