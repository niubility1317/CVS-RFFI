# 固定additive As：源RX6预登记

REGISTERED_NOT_RUN。actualcodecommit以release为准，精确执行字段见experiment.json。

# F＋跨 RX：固定 additive As 的源 RX6 确认

本轮唯一假设：完整保留原 raw／ordinary 绝对监督系数、另加 .25 R(As(raw)) 的固定组织，能在此前对 As 敏感的源 RX6 上同时保护 clean 并提高 urban，且优于追加 raw 的同尺度对照。RX1 discovery 已完整通过原单源折规则（clean +3.333333、urban +9.708995、low +5.714286 个百分点），但 low 仅33.809524%、部分 TX 明显下降；不把它当作 Ag、联合或跨 seed 成立。原 owner 的 RX1 seed02 确认计划保留，本轮不重复该矩阵。

固定 F＋跨 RX 同 TX SupCon、三路 SIG_TIME、统一 Sinc CENTER_UNITS、NoExplicitCFO、真实 joint160 和原 cosine30 默认头正常联合训练。没有新增分支、分类器、对齐损失或门控。每组 scratch E20（13L＋7L/U）、4440更新，原U/ownEMA/旧门控和cosine学习率保持。

共同 L_base=.50R(raw)+.25R(ordinary(raw))+.25R(raw第三对)+w(epoch)SupCon。三组8次L前向：D00追加raw对但系数0；D_raw追加.25R(raw)；D_As追加.25R(完整As(raw))。总pair系数1／1.25／1.25；原raw绝对.75、ordinary.25不变，但相对份额改变，不能由系数保证clean。每对R=原两次同IQ smoothedCE均值+双向KL，LS.01／KL1；SupCon只用ordinary原对，原浅中层与warm/ramp不变。

唯一科学差异沿 discovery 不变：原六场景均匀抽样，As直接作用raw且只叠加一次完整practical residual，不叠ordinary的噪声／多径。传播、噪声、质量、补偿残差的原依赖全部保留；EQoff即G1，正式测试IQ／SNR／信道和困难样本不变。有标签As全部CE，无学生置信度筛选。RX硬件session按RX/day/eq稳定，不用TX定义。

本轮held sourceRX6；fit RX1/3/4/8×day1/2/3×6TX，实际L5040/U45360/V21600，held原L1260仅固定final20默认头读出，heldL/U/V不参与训练或校准。原carrier7/63/30、split392005及已验证物理ID复用，不构造target。原数据过滤器只在本进程prepare期间参数化held/FIT并恢复，元数据过滤不读U TX；实际role IDs须与已核实同RX6训练角色一致。旧读取的144fit anchors仍不做模型前向，作为共同历史数据读取成本披露。

只加载匹配训练数据角色的旧RX6 C0做真实CP no-query合成smoke；它的4L配方不作本轮性能对照、不初始化任何学生或EMA。三组从零开始，父模型继承成本0；smoke2合成NN单列。实际新训练60modelE/13320更新，每组4546560 L NN、原U/EMA/V/held另实测；每组ordinary568320，D_As独有568320完整As。历史RX1 discovery成本60modelE/13320更新/17469956NN单列，不和本轮或跨seed混合。

每row完成即自动final clean＋六practical（各1260），默认头预测固定后独立源评分。完整20epoch/4440步native、view、Sincunit、L/U序列和TX×RX×day×scene×state×lock覆盖全部分析。另保存每包margin/30、完整原生接收状态、残余CFO／核／噪声／AGC／边界字段；元数据捕获只包裹原process返回，不重做信道，真实合成输出和draw相等已验证。固定D00 clean正确集合的保持率、各状态/质量和修复/新错全部保留。

判断完全沿 discovery（只对浮点比较使用 1e-10 个百分点数值容差，实际样本阈值不变）：相对D00 clean accuracy／Macro-F1不降、urban均值≥+3pp、low≥+5pp；相对D_raw clean及F1下降≤.5pp、urban≥+3pp、low≥+5pp；相对两组每个practical与seenV下降≤.5pp；完整4440／8L／3330SupCon、最终共同base目标≤.5、无U/teacher As、非有限或头拟合。任一失败停止固定候选扩展，不调系数、RX、场景概率、seed、阈值或预算救援；不停止其他owner健康任务。PASS只支持源RX6单seed证据，仍非Ag、正式target、跨seed或联合目标完成。

本地检查已经验证：D00与原六前向loss及所有参数梯度逐位相同、三组初始化及CPU/Python随机末状态相同、8L及sameIQ输入、默认头联合更新、export160、RX1合法／RX6拒绝、U元数据过滤与隐藏标签、scope恢复、完整physics捕获不改变输出。累计工程10020合成NN／6合成优化更新（含一次测试末端json未导入后的定向重跑），不计入正式E20；20配置/科学负例＋6角色负例另记录。未读取真实IQ／CP／target。

一次限定P0/P1；当前源码与配置Git提交、一个不可覆盖run/release、一次成功N607预检、单归档比较和remotecompile，dispatcher先真实CP smoke PASS后立即继续；所有owner每GPU≤3且启动≥12000MiBfree，CPU2，独立PID/CWD/argv/GPU/log增长核实。只对本run协议、角色、来源、输出碰撞、系统非有限或无法生成合法prediction的技术失败停止pending并保留partial；不因科学负分停止或重启健康worker，不自动重试。

精确矩阵、命令、环境/CWD、路径、GPU、owner、stop和产物由同run experiment.json维护。REJECTED_EXTRA_GATE：brainstorming的独立设计审批不属于当前项目最小流程白名单；研发已有授权，不增加审批等待。
