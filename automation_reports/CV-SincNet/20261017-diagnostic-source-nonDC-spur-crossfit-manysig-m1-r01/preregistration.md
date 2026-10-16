# 非 DC 窄带谱峰：源域 RX 关联的独立证伪

状态：PLANNED／本地聚焦检查已通过，尚无真实源结果。起点提交 958892a4df42ae2b1c1a398d48ea10c8846196e1；实际发布代码由 release_commit.txt 和 experiment.json 记录。当前基线仍为 F＋跨 RX 同 TX SupCon、SIG_TIME_X3、统一 Sinc、NoExplicitCFO、joint160、原默认 cosine30 联合训练、scratch E20；本统计不加载网络、不训练。

上一轮时间栅格收缩的独立 clean／宏 F1 条件失败，固定候选已关闭；原反向噪声、coherent mix、平滑频谱响应及重复差相关性方向也不重开。现有 ordinary 实际包含 DC、taper、循环移位、增益、相位、CFO、相位噪声、AWGN、多径和少量 PA；DAC 与 class signature 关闭，RDrop 复用同一 IQ。practical 作用于 raw，未再次叠加 ordinary AWGN／多径。本轮不把这些已存在操作重新称为 Ag。

唯一假设：存在非 DC 的局部谱峰，其 RX 对比可跨 TX 和日期复现，因此值得进一步检验窄带接收干扰是否构成可安全扰动的非身份线索。接收链的时钟／本振混频产物可以产生杂散，只提供物理动机：[Analog Devices 接收链讨论](https://www.analog.com/en/resources/technical-articles/2017/07/06/00/57/x-and-ku-band-small-form-factor-radio-design.html)。[WiSig 原始论文](https://arxiv.org/abs/2112.15363)说明接收机与传播会影响发射机识别，但不证明当前数据含有这些局部峰。观测关联也可能来自 TX、传播、外部干扰或已有均衡，不能声称分离了纯 RX 硬件。

仅复用已验证 source cache 的当前 fit L5040：RX3/4/6/8、TX0至5、day1/2/3，各单元70包，共72单元。必须先以原合同及 C0 实际 L 角色核对，才提取这些 IQ。原 cache 会反序列化其他源角色对象，只有选中 fit L 的值和合法标签进入计算；held RX1、U、V 的 IQ／TX真值不用，目标不访问。不使用旧 L6300 聚合系数或更改划分；不进行模型／EMA／优化器／原型状态继承。

每包原256点做 FFT256，分别使用矩形和 Hann 窗，不零填充。各 periodogram 除自身256频率单元均值，再将每个单元70包平均。这里只归一化统计量，不改变网络输入。每个候选频率 k 的局部峰值为 k−1、k、k＋1 的平均功率，背景为 k±4至7 共8点的中位数，取10log10比值；固定相对下界1e−12。候选只含 shifted bin8至247，排除 DC周围120至136，剩223个；25MHz 下每点97.65625kHz，三点局部峰不等于真实仪器窄带分辨率。

18折各排除一个 TX 与一个 day，拟合只使用其余5TX和2day的40单元。只用矩形窗拟合侧 RX 对比的全局最大值选择一个 RX/bin，固定词典序处理并列；同一个选择在两种窗的留出单元评价，不按留出值或窗选择频率。各窗的 RX 对比减去该 TX/day 四 RX 均值。对照是零 RX 对比和3个循环错配 RX，全部固定，不拟合分类器。

所有科学条件必须同时满足：两窗各折训练峰值至少6dB、训练RX对比至少3dB，且留出峰值及RX对比均至少3dB的共同成功折不少于15/18；精确相同 RX/bin 的众数不少于15/18；两窗各自风险相对零对比下降至少50%、不少于15折正收益、优于最佳错误RX对照的差至少为零风险的20%，全部TX/day/RX汇总风险均正改善。无效或零能量包不删除，直接技术失败。所有折、窗口和负分层保留。

科学失败关闭本固定非 DC 局部谱峰假设，不改 FFT、窗、背景、DC范围、频率、阈值、RX、TX或day救援。通过也只说明源统计关联；必须另登记有界、匹配功率的 IQ 干预及身份保护检验后，才能考虑新 E20。没有自动训练、目标或因子矩阵晋级；统计阳性不是 Ag 收益。

唯一 row、六类 seed、完整命令、环境、CWD、GPU策略和路径见 experiment.json。普通 N607 账户、CPU两线程、CUDA_VISIBLE_DEVICES为空；0模型、0神经网络前向、0模型checkpoint、0优化更新、0新增信道，E20预算未消耗。launcher首先做原合同与精确fitL角色检查，G0不消费模型，因此模型checkpoint smoke不适用，不读取无关权重。一次普通账户 preflight、一个归档比较、一次远端编译；唯一不可覆盖输出 root 与 owner，启动后读取实际PID/CWD/argv/环境及日志／终态。技术失败保留partial且不自动重试，不影响健康旧任务。

保存全部18折、两窗 cell PSD与局部峰、resolved config、角色使用声明、执行／完成／失败和完整stdout。聚合 NPZ 保持私有，不含IQ或physical IDs；结果可独立离线重算并生成频谱／跨折图。预期0身份准确率，不能回答默认头提高多少或low urban>60%；这些联合目标继续未完成。

本地检查通过：18个协议／无效输入负例；5,040个合成波形和33个额外统计计算；解析RX线正例通过、公共峰／TX纠缠／无谱峰反例失败；held TX与day不参与选择、两窗使用同一fit频率、FFT频率oracle、增益不变性和RX置换等变性通过。0真实IQ、0模型前向／checkpoint／更新。唯一一次独立 P0/P1 审查 PASS，无发现；这里只检查真实执行缺陷，不新增签名、封条、重复审查或额外门槛。
