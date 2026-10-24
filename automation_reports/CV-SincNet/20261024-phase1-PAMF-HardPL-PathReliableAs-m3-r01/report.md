# 主干互补与可靠信道视图组合消融

LOCAL_VERIFIED_NOT_LAUNCHED：86 项相关检查、一次独立 P0/P1 PASS；首个释放且匹配 seed/card 的真实原 E20 检查尚 PENDING，通过即丢弃探针并重载原权重继续。

固定 HPL_E20_PATH_RAS6G，三 seed2026092701/02/03。每个模型从原同 seed F21 E20/4440 独立 native10/2220＋20/4440，累计 E50/11100＝13L＋37LU；不加载原 E30 或候选权重。数据复用已有 VALIDATED_ONCE 的 F0 物理契约，L6300/U56700/V27000；严格核实父状态角色、物理ID、scratch继承和实际cfg/payload。U TX真值隐藏；V只诊断，无模型/归一化/EMA/原型/头更新。N607普通账户/CVS-RFFI、FP32/TF32off、每worker CPU2，CWD PROJECT/releases/20261024-phase1-PAMF-HardPL-PathReliableAs-m3-r01。命令、不可覆盖路径、GPU、停止规则和预期产物见experiment.json。一个实际代码commit、一个archive一次本地远端比较、一次remotecompile；提交后独立核实PID/CWD/完整命令/GPU/日志增长，健康旧任务保留。

源依据分为两项：原 E20 U960/V720 原生As视图诊断显示 raw 高纯伪类不等于增强视图容易识别，固定 .95＋同类门控采用As，其余 exactraw回退以保留原样本；源诊断D既有E30的主干特征交原默认头时，固定V有效子集准确率从Joint90.5073%提高到Complement97.5892%和Path98.5434%，完整融合均99.8493%。这些是源域机制证据，不是U真值正确率或目标成功。已知Joint和此前目标结果，当前Complement/Path/ReliableAs目标分数尚未读取；本组合在其目标结果前固定，不改父、门槛或矩阵。

组合完全保留两个独立固定配方：Path普通U CE系数full0.25＋noSig0.125＋core0.125，总CE分母1.25；noSig从同次joint扣实际sequence_pool，core再扣response_projection乘response_gain.tanh；已有头增加4次调用，完整签名/参考仍参与部署和原完整视图训练。ReliableAs只改变原mixed半批输入：原生As生成一次，currentstudent eval/noGrad得到q，raw已接受且q置信度≥.95且argmaxq==raw伪类才用As，其余精确同包raw回退。raw接受集合、硬伪类、所有六学生前向保持。额外As teacher先于临时style/joint/path hook，恢复训练模式，不更新归一化。

其余保持 .95rawPL、权重6、LS.01、ordinary/raw RDrop1、AsKL0、浅中层跨RX0.1/joint160跨RX0.05、六前向共享accepted同伪类跨RXstyle，原L/defaulthead原位联合、完整L anchor分组投影/范数帽1、原nativeCENTER_UNITS AdamW/clip5/decay。多继承共同祖先只执行一次Joint原六U loss，完整L guard只消费一次组合U梯度；不重复CE/teacher/学生损失。不新增头、参数、额外学生骨干、EMA、原型、Utruth、target或G训练；不保证AdamW步、源L损失或目标单调。

每模型新增6660步、U1701000、学生U10206000；As生成850500及额外As teacher850500，Path既有头26640次/6804000条已算特征。累计raw teacher2097900＋As850500＝2948400，学生U10602900；实际used＋fallback＝850500，物理学生视图2×used。生成/额外eval即使回退仍计成本，逐seed记录实际采用率/回退/父与增量及累计墙钟，不称相同FLOP或独占GPU小时。

真实原E20检查同时验证两个机制：L六前向768、U六前向1536、rawteacher256＋As128、源Practical原L64＋U128＝192；As精确谓词/回退/原接受集合、eval归一化和模式恢复、Path四既有头调用1024条已有特征、主干CE不走exclusive签名/参考而原头/主干梯度非零、完整joint签名辅助梯度、原CE总剂量、完整L guard显式梯度相等与临时hook退出。探针丢弃，正式行重载原E20。

GPU5/6/7固定对应三个seed，每卡最多两个本任务且计外部最多3 compute进程；派发前free≥10000MiB。只接同seed Path全部GPU预测结束的卡位，且ReliableAs已分发；Path和ReliableAs两侧pendingCUDA PID预留。首个符合条件的匹配卡做真实检查，smoke临时首row对齐，正式三seed/card分配保持。唯一CPU scorer接ReliableAs终态，两个相同系统性技术错误只停止新增派发并保留健康任务/产物；不按目标分数停训、调参、重排或选择性重跑。

最终E50学生逐row自动测试raw及固定DURA各clean＋六practical，168000/view，共三seed42cell，整row全部决策固定后独立truth-last。报告全部逐seed/景/SD/MacroF1/TX/RX/day/RX×day、纠错损伤、接受/视图/成本，并复用F20/F21同读出完整对照。固定源设计的Joint/Path/ReliableAs三项邻近比较全部保留。原raw clean对两控制均≥+1pp、六景对两控制均严格>+1.5pp，DURA不能替代主目标。G固定用户DURA v1.0 R69/266951、FP32/TF32off/reuse=True/graph=True/native=False，G不进入新F训练，不套用原F50时延；默认部署定义保持，goal ACTIVE。
