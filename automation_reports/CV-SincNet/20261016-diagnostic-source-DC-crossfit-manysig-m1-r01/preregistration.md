# RX加性DC：源重复性预登记

REGISTERED_NOT_RUN。实际row／commit／command／environment／CWD／paths／CPU／owner／stop／artifacts见experiment.json。

# RX 加性 DC：先检验跨 TX/day 的源重复性

当前没有合格的独立 Ag。practical 配对五源折提升 urban，但 RX6/RX8 clean 保护失败；真实 cross-RX partner 和 C4 相位候选也已按原规则失败。2026-10-16 的只读原始长度核查确认：全部合法源 L6300 原始数组本来就是256×2，native center crop 不提供额外真实采样点。真实长窗裁切方向关闭，不用反射／补零伪装真实观测。

新的唯一问题是 RX 相关的加性 DC 均值差是否跨 TX、跨 day 稳定，并超出旧普通 DC 增强的幅度覆盖。原 native RMS center=false，普通 DC 只是旧增强中的一项；信号自己的均值也含调制／TX／通道作用，不能先把它解释成 RX 硬件。此处先作可证伪的源统计，不训练新模型、不拟合身份／RX 分类器、不改变部署前处理或正式测试。

只用当前源 RX1 留出实验的 fit RX3/4/6/8 合法 L5040、六个真TX、day1/2/3，每cell70包。RX1 的 IQ、U/V值及 U TX真值都不使用；复用现有 VALIDATED_ONCE source cache 的原角色，不重验数据或读取任何模型checkpoint。源缓存可能同时反序列化其他源角色容器，它们不进入统计；不构造target。新统计与原 F＋跨RX SupCon／joint160／默认cos头无模型状态继承。

每包描述为完整256点的mean(I,Q)/sqrt(mean(I²+Q²))，以原native packet-RMS为单位，不改IQ或训练归一化。先在每TX/RX/day cell取均值，再减去同TX/day的四RX均值，得到RX相关复数DC对比。18个固定fold各同时留出一个TX和一个day：只用其他5TX×其他2day估计各RX偏移，预测未参与估计的TX/day对比。零RX效应作风险参照，统计平方欧氏误差；不改窗口／坐标，不用RX分类头或任何默认头拟合。

固定源资格：总零效应风险非零，聚合预测风险至少降低10%，18折至少12折风险降低，并且完整L5040估计的RX均值对差的逐分量最大值p95大于旧ordinary实际E13 dc_offset_max。旧幅度由同合同C0保存的base config与E13实际概率比例还原，不拿默认值当执行值。最后一项只检查现有增强是否已经覆盖该尺度，不扫描新的增强强度。全部条件需通过；否则关闭本固定RX-DC合成方向，不改窗口、绝对值坐标、TX/day/RX子集或阈值救援。

PASS也只支持另登记有界IQ训练候选，仍不能称DC是物理因果硬件效应或Ag已经改善识别。当前0模型、0网络前向、0checkpoint、0更新、0As与target。CPU2线程、CUDA隐藏；读取5040个合法L IQ用于均值统计，原有E20训练预算不消耗。源判定后保存全部18fold误差、原扰动幅度、量级图及正负结果，无跨seed或正式性能声明。

精确row、commit、command、environment/CWD、输入输出/log、owner、停止规则与预期artifact在experiment.json。唯一不可变run；技术错误保留产物、不自动重试，不动健康外部任务。最小工作流按G0只使用源统计所需能力，checkpoint smoke不适用于0模型/0checkpoint统计；不为此加载无关权重、创建头或生成prediction。聚焦负例、一次限定P0/P1、一个archive比较／remotecompile及PID/CWD/argv/CPU/log核实足够。

发布前实际验证：LOCAL_VERIFIED_NOT_RUN。15协议负例、5040合成descriptor构造、18fold稳定效应PASS、TX交互／零效应／旧幅度已覆盖FAIL、同时留TX/day改值隔离均PASS；0真实IQ/CP/网络。隔离发布路径导入／单row真实config一致，且由原C0实际metadata还原DC参数幅度0.005012006678699482，通过。唯一限定P0/P1只读审查PASS，0额外IQ/CP/网络。N607普通账户preflight于2026-10-16 16:42:09 CST核实一次。当前G0不消费模型或checkpoint，noquery模型smoke不适用；没有为此加载无关权重。尚未启动，无真实DC统计结果。
