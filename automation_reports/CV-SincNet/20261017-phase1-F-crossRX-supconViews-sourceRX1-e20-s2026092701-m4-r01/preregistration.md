# 原跨RX同TX对比项的视图组织：固定四行源E20

状态 IMPLEMENTED／PREREGISTERED_NOT_LAUNCHED。run：20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01；owner：codex/root/F-supconViews-E20-20261017。代码/config以实际Git执行commit固定，不另设设计审批或hash链。

唯一问题：不新增loss项、系数、模型前向或部署模块，只让一部分既有浅中层跨RX同TX对比正对跨ordinary/practical，能否同时胜过ordinary-only与practical-only控制的urban身份识别，保护clean？这是As训练组织，不是IQ泛化增强Ag。频域相位、旧scene-cycle等已失败路线保持关闭。本轮与另一对话的As启停时序路线不同，不借用其模型或增益。

## 固定模型、数据和四行

F＋跨RX同TX、三路SIG_TIME、统一Sinc、真实joint160、NoExplicitCFO及必要内部补偿、既有默认cosine头正常联合训练保持。scratch seed2026092701、E20/4440更新、原13L+7U/oldgate/ownEMA/ordinary保持。有效L/U/V为5040/45360/21600，fit RX3/4/6/8、day1/2/3、原六TX；RX1原L1260只作final20源读出，U TX真值隐藏。不混同fullL6300或L_train5580/L_dev720，target不构造、不读、不参与选择。

| 行 | extra IQ | 原SupCon输入 | 执行 |
|---|---|---|---|
| Q0 | raw | ordinary | 复用sceneCycle的B00固定源产物 |
| Q_ord | 单次完整As(raw) | ordinary | 复用sceneCycle的C_IID固定源产物 |
| Q_sat | 相同单次完整As(raw) | practical | 新scratch E20 |
| Q_mix | 相同单次完整As(raw) | 64个L出现位置用As、64个用ordinary | 新scratch E20 |

两个旧控制执行commit c9a74172e5f81bb4f04ce12cd4386ad4dd953ee9，run为20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01。已从实际resolved config、native args、source_contract全部物理L/U/V IDs、fold、初始化、冻结、completion、执行audit核对：同source角色/seed、scratch E20/4440、8L、pairweight1.25、默认头联合、3330对比更新/13320项、原U/EMA，target为false。当前对应model/channel/source/design与执行commit无差异。控制权重不用于新行初始化；启动时只复核同一已存在控制元数据，不重复训练或推理。它们是历史40modelE/8880更新，不冒称本轮新增训练。

所有行保留完整raw.75＋ordinary.25监督，再加.25 extra RDrop pair，8次L网络前向、总pairweight1.25。每pair仍是原CE(label_smoothing=.01)均值＋原RDrop1KL，两个随机子网络前向使用同一已生成IQ。As从raw生成一次，不在ordinary损伤后重复叠加；practical传播、遮挡、噪声、接收质量、补偿、EQoff与session/随机流全部保持。所有合法L As无高置信门控参与监督；不新增U或teacher视图，不把不同U作负对，不平均IQ，不改正式测试SNR/channel/receivedIQ或困难样本。

原SupCon仍为time_down、t2.pool两层、两个RDrop phase、共4项原平均，temperature0.2，原E6起升至0.1 schedule。同TX不同物理RX为正对，已知不同TX为负对；无projection或诊断分类器。仅改变原项使用的maps，不附加全特征相等损失或joint/raw112对齐。Q_mix用独立default_rng(202610176001+optimizer_step)排列128个L出现位置，取前64为As，选择不依赖TX/RX/day/置信度/损失，两个layer和两个RDrop phase共享同一选择。原loader可能出现物理ID重复，本方案按出现位置计权，不能把128出现说成128独立物理样本。主loss仍按原物理出现和视图归一化。

## 固定判据与放弃条件

全部新行完成并冻结final20后立即自动做source RX1 clean＋六practical，各1260包；新行17640预测，两个旧控制17640预测仅复用。相同物理ID/scene/命名随机流，原默认头直接预测，源标签仅评分，不改变训练。正式target真值未连接。预定primary始终Q_mix，不在Q_sat/Q_mix按成绩挑选候选。

Q_mix相对Q_ord和Q_sat均须：clean accuracy及macroF1不降；low urban accuracy≥+1pp、macroF1不降；mid urban不降；urban均值≥+1pp；每个practical与seenV下降≤0.5pp。相对Q0须：clean及F1不降、low urban≥+5pp、low F1及mid urban不降、urban均值≥+3pp、同样每场景与seenV保护。原最终base L objective≤0.5及全部预算/权限为健康检查，不替代身份收益。suburban各场景绝对准确率与距自身clean差距均报告，≤2pp仍为完整联合目标的探索要求，不降低clean缩差。

任一科学条件失败即关闭固定视图组织，不改比例/层位/系数/scene概率/RX/seed或预算救跑。所有row按固定E20正常完成，不按中间成绩停机或调参。通过仅支持单独独立源seed确认；不证明Ag、稳定As、完整B00/B10/B01/B11交互、跨seed或正式联合成功。

## 验证、成本与发布

本地29个协议/科学反例及真实F合成执行PASS。新ordinary-route与旧控制loss完全一致，74组参数梯度逐一接近；同初始化、八次输入逐点相同、相同fullAs IQ和CPU/Python RNG；两条新行默认头联合更新、export160逐点一致、四项原对比输入重算一致。合成训练NN4096包＋eval NN12包＝4108、2次联合更新；128 raw、128 ordinary及512完整practical生成，额外对比重算2次单列。0真实IQ/CP/target。最初误用无Torch的F Python只造成导入失败，随后使用现有已验证Torch环境；已知NumPy桥警告保留，使用list/buffer路径，未安装或改环境。

本轮新增2scratchE20／40modelE／8880更新；复用历史2E20／40modelE／8880更新，四行每模型总预算都E20，不是续训。每新行L网络包4546560（含extra1136640），ordinary568320、完整practical568320；U/教师/V/held8820包和初始化2NN分别按实际记录。E6至20对比出现数426240：Q_sat均As；Q_mix As213120＋ordinary213120，两个phase/layer的实际捕获计数另报；没有新增NN。捕获/路由/descriptor计算/内存/生成时间不能隐入零成本声明。

一次限定P0/P1审查后，提交代码和配置。普通N607、已有CVS-RFFI Python、FP32 GPUworker/CPUdispatcher2threads/loader0，CWD PROJECT/releases/cvs_F_supconViews_sourceRX1_e20_20261017_r01。命令CVS-RFFI/bin/python -u experiments/cvs_supcon_views20/runner.py；独占root PROJECT/runs/20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01。共享capacity锁、每GPU最多3个所有owner进程、至少12000MiBfree，不干预健康旧任务。

launcher第一步加载已核对来源和同数据角色的旧B00最终20真实checkpoint，仅做2个合成包无query兼容smoke，不用作初始化；PASS立即继续。一次普通账户/路径/空间/依赖预检，一次归档传输SHA比较、remote compile，启动后一次独立PID/CWD/argv/GPU/log增长检查。仅实际角色/继承/预算/输出碰撞/非有限/不可启动/非法预测等系统技术失败可取消pending；保留产物，健康worker继续，无自动重试，无其他owner停机。

预期产物：两新行完整CP/config/args/物理角色/冻结，全部4440 native/view/Sinc-unit步和L/U顺序，实际选择索引/对比项、全部As监督覆盖与场景、原默认头source七视图预测；两个旧控制引用及成本，独立全量预测/步/路由/成本/配对修复新错/差距/报告与登记。源IQ/权重/私有ID/逐包预测/日志/runtime不入Git；聚合报告供查看。独立分析不能把anchor或梯度当收益。完整联合目标ACTIVE。
