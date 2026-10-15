# SIG_TIME＋RDrop：冻结特征跨 RX 关系诊断

状态 PLANNED。本轮仅分析四个已冻结源模型的实际部署joint160，不新增训练、分类头或目标测试。父实验 20261016-phase1-SIGTIME-rdrop-curvature-completion-sourcefold-s2026092702-m4-r01 的主NULL资格FAIL保持；全空间ISOTROPIC相对FULL少错18包、CE降低的单seed源对照正迹象保留，尚未独立确认。这里检查这组默认头观察是否伴随跨RX特征几何变化，不把新的诊断指标作为晋级或救跑门槛。

固定来源是父实验FULL/SPAN/ISOTROPIC/NULL四个final20 student，执行代码 78ad998e9dcb5653573fa801538e914189833d20。各自scratch20/4440步、fit RX3/4/6/8、day1/2/3、六TX、L5040/U45360hidden/V21600、自身EMA、0target、默认头联合训练、final-only。全部source_proof在任何CP反序列化之前执行，真实角色/继承相符；不加载全5源RX或未知来源权重。父训练4×20=80model-epoch单列，本轮0轮/0更新。

仅从原visible L6300carrier中按原生索引选每个TX/fitRX/day cell前2包，共144fitL；RX1每TX/day全部70包，共1260heldL。共1404唯一L物化一次、四模型共享。fit144属于四模型实际L5040，held1260属于原visible L且与四模型训练/U/V角色全部不交。U/V只核对既有句柄，无IQ或标签物化；target不存在loader，0IQ/truth。源RX1之前已用于开发，本诊断不是新盲测。物理ID首列是eq_i而非TX，合法标签逐包与native tx_i核对。

全部模型eval且关闭梯度，输入clean无增强/无dropout。用actual id_feat_joint160，转FP64后按统一eps1e−4归一化，要求原范数>eps。计算平方余弦距离d=clip(2−2uᵢ·uⱼ,0,4)，不拟合头、原型、变换或阈值。对每个fitanchor，只比较同一天的70个held同TX与350个held异TX距离；若同TX距离小于异TX超过1e−12记1，差绝对值≤1e−12记.5，否则0。此数值tie band在读取数据前固定，用于浮点近等关系，不是分类阈值。汇总称“关系满足率”，不能称识别准确率。

每模型10080个同身份pair、50400个异身份pair、3528000个三元比较；144anchor记录，overall＋4fitRX＋3day＋6TX＋72cell共86分组。72cell各2个fitanchor。将报告关系满足率、同/异身份距离均值及间隔和分层，不输出分类决定。成对／三元组合高度相关，不能当作独立物理样本或模型seed；模型seed仅2026092702，4个是训练条件。

成本：1404源L物化；四模型各1404实际feature前向，合计5616；每模型首步自己的真实CP做2合成包no-query smoke，合计8，PASS立即继续。无optimizer/backward/EMA/头拟合或CP写入。全部198state、nonpersistentbuffer、gradslot、特征counter及auxaudit不变。每模型独占PROJECT/runs/20261016-phase1-SIGTIME-rdrop-feature-geometry-source-s2026092702-m4-r01/ARM/config.json、features.npz、result.json。source选择、逐anchor结果和rawfeatures均保留privateartifact，不进Git。独立分析可用保存features重算全部关系，无需再读IQ或前向。

CPU14协议负例、ideal/tie/reversed精确计数、分组、标签和坐标置换/正比例缩放检查PASS；实际模型joint160在临时W扰动前后逐位一致，所有state/buffer/gradslot/counter恢复并保持。检查8合成身份前向、0真实IQ/CP/更新/头拟合。行配置与输出目录唯一性通过。一次独立新诊断审查发现1项P1：已知远端旧Torch/NumPy数组桥接不兼容。发布前改为一次tolist→NumPy转换并复用；禁用Tensor.numpy的保存→回读→关系入口检查PASS，FP32往返精确、0桥接调用、0新增模型/IQ/CP前向。该问题一次定向复核PASS，其余无P0/P1，见review.json；没有重复全审或父训练。

本诊断没有科学晋级阈值。更好的已知身份跨RX关系最多补充源端特征证据，不能证明原头收益全由特征造成、训练机制的因果效果、三seed稳定、新TX或目标泛化；较差几何也不能单独否定所有部署读出。关闭的NULL及其他失败配方保持，不扫描alpha/温度/层/seed/epoch，不启动确认或主测试。新训练方向须有独立假设和公平对照。

唯一owner `codex/root/SIGTIME-RDrop-feature-geometry-20261016`，ordinary N607 CVS-RFFI、CPU2/loader0、CUDA_VISIBLE_DEVICES空，0GPU实验进程。CWD PROJECT/releases/cvs_SIGTIME_rdrop_feature_geometry_source_20261016_r01；命令 `CVS-RFFI/bin/python -u experiments/cvs_rdrop_feature_geometry/runner.py experiments/cvs_rdrop_feature_geometry/configs/request.json`。不可覆盖root PROJECT/runs/20261016-phase1-SIGTIME-rdrop-feature-geometry-source-s2026092702-m4-r01。一个实际Git提交、一次ordinary preflight、一个release archive本地/远端SHA比较、remotecompile；独立回读PID/CWD/argv/CPU/日志或终态产物。短任务若先完成，只报告终态证据，不虚构live观测。协议/输入/状态变更/非有限/运行错误保留输出、失败且不重发，不改变健康或无关进程。预期产物和全部行见experiment.json。goal ACTIVE。
