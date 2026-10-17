# P43：同新增噪声功率的形态 × practical 因子实验

本轮仅首模型 seed2026092701，6 个独立 scratch own20+own10（累计E30）模型。基线继续 F＋跨RX同TX／SIG_TIME_X3／统一Sinc／joint160／原默认cos30，原内部门控、参考响应、旧U门控及ownEMA保持。全部原5源RX/day1/2/3、L6300/U56700隐藏/V27000，无训练RX留出；V只诊断。

唯一候选Ag：独立加性接收观测噪声，新增噪声SNR固定30dB（不是总SNR），每包噪声RMS精确匹配。WHITE为同私有高斯创新的白噪声；BG令每个复采样点的两IQ分量共用Bernoulli(p=.02)状态，激活点高斯方差为背景的50倍，随后只归一化噪声能量，不归一化信号或裁剪波形。参数对所有TX/RX相同，私有种子202610185301加累计step，不含TX标签；各epoch刷新、每对RDrop复用同一IQ。它是观测干扰候选，不反演未知TX效应，不声称WiSig已实测到该EMI或其因果来源。

6行：B00(noG/noAs)、B10_WHITE、B10_BG、B01(noG/As)、B11_WHITE、B11_BG。每row共同保留原完整cleanretain四前向（.75ordinary+.25As_or_raw），再加入.25Ag_or_raw的双前向。六行一律6×128 L NN／步，L总损失尺度1.25；common普通视图与实际完整As的绝对监督权重不被稀释，旧U目标尺度不变。关闭槽用同物理raw控制，不更改augmentation或额外合成ordinary。既有SupCon仍仅原ordinary128，两深度／两已有forward，temp.2/max.1/warm5ramp10；不叠加新一致性、对齐或诊断头。

B11首先是Ag与As的独立监督视图，不组合As(Ag(x))；先检验独立与联合收益，若失败不靠顺序／形态／功率／概率／seed／预算救跑。已有白噪声路线作为负历史保留，本轮新加入严格同成本WHITE控制以区分功率与时间尾形状，不把它换名宣称新机制。已有重复差lag1关闭规则保留：本轮不更换其统计量、窗口或数据子集救援，不开展重尾重复差探针。

因果/文献边界：原practical注入Gaussian，普通增强也含AWGN；这提供形态对照的代码依据，不意味着改成脉冲后必然更好。独立原研究对SEI比较过AWGN和impulsive环境：https://ieeexplore.ieee.org/document/7753733/ ；Bernoulli-Gaussian两状态加性噪声及平均功率公式见https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cmu2.12077 。这些只支持有物理意义的候选族，不能证明当前数据存在该分布或本模型能获益。

源判据在首次训练前固定：BG独立cleanV相对B00≥+.2pp；BG联合cleanV相对B00和B01均≥+.2pp；B01−B00及B11_BG−B10_BG在2draw×mid/low4格均≥+3pp；B11_BG−B01每格≥+1pp；B11_BG−B11_WHITE每格≥+.5pp且clean≥WHITE−.1pp；上述BG配对各源RX clean下降≤.25pp、两draw low_suburban下降≤.25pp。全部通过才称source-qualified，不称Ag或跨域性能已证明。任一失败关闭该固定BG候选，不自动扩种子。冻结旧头对新增IQ不天然不变不再被错误地视为训练后收益的必要定理，真实训练增量才是判据。

每模型预算：own20=13L+7U，再own10=10U，L仍每轮训练，AdamW cos20／strict自身学生与EMA／resetAdamW cos10，ordinary冻结实际E13。222步/轮、6660更新、L网络包5114880、Ag槽1704960 NN、原L物理观察852480，原U观察963900；6模型180model-epoch/39960更新/L NN30689280。源eval/teacher/realCP-smoke/合成smoke/正式推理/物理信道调用及墙钟另按实际计数。相对4前向旧记录每步L计算多50%，不能称总算力相同；本次六行严格匹配。

实际first launcher step：固定P41 ORD已核实同源E30默认头仅作2合成无query smoke，绝不加载到新训练；随后六个真实F合成前向/梯度各768NN，共4610 NN、0update。source旧已验证handles复用，不新增数据验证链。

所有六行固定自己的FINAL30，逐row自动原clean+全部6practical预测（每view168000）再独立CPU truth-last；无目标反馈、无改变SNR/测试信道/receivedIQ/困难样本。不因sourceFAIL跳过已预登记row正式评价；source matrix decision只用源文件，agent在其固定前不读取正式指标。报告全部6行/seed、accuracy/macroF1/worstRX/suburban绝对值和clean差、同物理错误修复与新增、固定父clean正确集合保持率、已有门控/状态/质量证据、Ag独立差/As独立差/联合交互项与WHITE形态控制。

预期科学目标仍是clean相对同预算无Ag/As有净收益、low urban>60%、各suburban距自身clean≤2pp（探索条件），并需后续固定种子重现；本首seed没有跨seed结论。

最小流程：1代码config提交并推送读回，8项聚焦算子/协议负例，launcher首步真实CP smoke后即续，一次P0/P1，只一份预登记/登记矩阵、不可覆盖run与唯一owner、一次N607普通账号preflight/单归档SHA传输/remotecompile、一次PID/CWD/argv/GPU/loggrowth，0额外审批。REJECTED_EXTRA_GATE：brainstorming的单独设计审批/固定spec流程在本项目最小工作流及用户已授权持续研发下NONBLOCKING。

技术停止仅来源/角色/非有限/错误budget/输出冲突/不能合法闭合预测/launcher系统故障，保留产物、不自动重试，只操作run-owned进程；科学低性能不触发技术停止。

## 已核实启动状态

状态：VERIFIED_RUNNING_INITIALIZED。执行提交 19aedc3604fed9c193237dd92366164178af0983 已推送并独立核对远端相同。单归档传输SHA匹配、remotecompile与12个分段实际config/args核对通过。首步真实P41 ORD无query smoke2NN PASS，六个真实F合成梯度/计数共4608NN PASS，随后自动进入训练。

队列PID900618，六worker PID900700/901260/901341/901424/901507/901588分别使用physicalGPU0至5。独立核对PID/startticks/CWD/argv/GPU和日志增长。六行actual initialization均scratch、checkpoint_sources为空、source roles EXACT_MATCH、无target contact。当前只是初始化/运行证据，尚无新增性能或收敛结论。

完整精确命令/CWD/环境/路径/seed/预算/owner/停止规则/预期artifact见experiment.json；进程与验证聚合见verification.json。未修改其他健康实验，未访问目标指标。

首轮真实证据已核实：六行均E1/222updates/LNN170496；三As行各14208次实际完整practical，另三行0；全部sourceRX/day角色匹配，U真值与target访问标记均false。实际模块均指向已钉住的 experiments/adv3b02_xuc/code/leo_practical 实现；source报告仍不是正式性能结果。详见 progress_e1.json。


## 完整分析实现与等待状态

分析代码提交 c02154d6cf9fb29fafecc4145bb6c08cb26d2490，已推送并独立核对远端一致。分析只复算已保存输出，不读取IQ/checkpoint、不新增模型前向或更新。7项合成检查覆盖预登记判据独立等价、边界与坏输入、因子交互公式、全部6行冻结前真值关闭、成本口径、U标签不可解封及源域完成前不产生覆盖输出。本地与N607均7项PASS。真实输入训练不变；数据handles复用。

分析r01归档遗漏criteria.py参考依赖：远端5个分析文件编译通过，但检查导入失败，状态FAILED_TEST_DEPENDENCY_PACKAGING；原归档保留。r02补齐同提交内参考文件，单归档SHA匹配，远端检查7项PASS；没有修改或重启训练。Windows测试桩另已用Path.name修正路径分隔符，保留最初失败记录。这两项均不是模型训练失败。

零前向源域分析driver PID911017/source waiter PID911026已独立核实CWD/argv、CUDA为空、CPU线程2、等待态；源输出尚未创建。六行完整source结束后自动解析全部4类逐step日志、完整原生epoch记录、完整stdout、物理样本序列和成本，独立复算冻结的source判据；随后按原命名信道流复算3个As行的TX×RX×day×scene×传播状态覆盖，U序列不解码TX。此处不读取正式成绩。分析归档路径、准确命令和输出见analysis_delivery.json及原launch记录；最终正式分析须在源域判定固定并记录后进行。

日志口径：继承的concat/total_L、concat/L_forwards=4、concat/As_packets=64仅描述原四前向分支，其中As-off的64是raw控制槽。真实六前向、真实As-off=0和新增形态槽分别以reference_steps、shape_steps、observer/result及native encoder counters为准；训练总L独立复算为.5commonRD+.5mixedRD+原SupCon+.25shapeRD。native train/loss_tx_labeled仅第一原CE，不代表完整L目标；13个L-only轮可与native train/loss核对，U启用后不把native总loss冒充纯L。该差异属于历史字段语义，不是训练目标错误，不热改健康任务。


## 首两行实际闭合与完整源日志复算

B00和B10_WHITE完成各自scratch20+own10/E30，均为6660更新、L网络包5114880、原L观察852480、原U观察963900，实际覆盖全部L6300/U56700。全部4类step流、30个原生epoch数值记录、完整stdout和样本序列已复算；检查的活动损失有限，无非有限loss/grad跳步，epoch数值字段无非有限值，300次nan文本为未启用测试的占位打印。L目标分解与native监督总loss最大误差分别不超过2.69e-7/4.18e-7；此为执行核实，不是科学有效性结论。

每行实际原生student_train NN6078780、student_eval NN918004、teacher_eval NN963900、teacher_train NN0，加冻结源读出12600NN，source合计7973284NN。原生训练/验证两段墙钟B00=2146.190秒，WHITE=2153.091秒；外部冻结源读出墙钟未独立记录，不伪造全流程耗时。

两行均已自动完成原clean+6practical，逐view168000样本，全部7view固定后才独立truth-last；默认头无单独拟合，所有参数冻结、预测侧未读truth。每行实际正式预测identityNN1176002（其中1176000正式样本，另2个既有检查前向），预测耗时48.089/47.099秒。仅核对completion与独立边界，尚未读取正式成绩；其余4行继续原任务，整矩阵和性能分析未完成。全部源域判定固定前成绩继续关闭。详见first_completed_audit.json。


## 三个无As控制完成后的实际形态核对

B10_BG亦完成E30及全部7view冻结预测/独立truth-last；完整source日志复算通过，6660updates、LNN5114880、sourceNN7973284，实际Ag852480包、As0，source原生训练/验证墙钟2203.393秒。其正式prediction identityNN1176002，耗时48.988秒；尚未读取正式成绩。

实际6660step形态记录而非默认配置：WHITE与BG的relative_noise_RMS均为0.03162277489900589，记录范围无变化；每包delta功率在训练调用内与目标新增功率逐次allclose检查，实际out-x也核对，未做信号后RMS归一化。各包最大复采样点噪声能量/全包噪声能量的均值，WHITE=0.02392547，BG=0.20748355。二者形态不同、总新增噪声功率一致；这不是总SNR或身份保持/泛化有效性的证明。三组同RDrop对复用同一IQ，新增边界样本0，每个Ag或raw槽1704960NN。此阶段核实见noise_shape_actual.json，正式效应仍等全部源域判据固定后统一分析。


## 正式成绩打开前固定的源域结论

独立复算与原source_matrix_decision的全部10项条件一致，4项通过、6项失败，source_qualified=false。固定BG候选关闭，不扩seed，不按正式成绩修改形态/功率/概率/权重/seed或预算。所有6行已登记正式评价和报告仍完成；整体联合目标未完成。此结论在agent读取任何正式成绩前记录，详见source_decision.json。

全V27000 clean（%）：B00 98.359259，B10_WHITE 98.322222，B10_BG 98.351852，B01 98.255556，B11_WHITE 98.277778，B11_BG 98.222222。BG独立相对B00=-0.007407pp；BG联合相对B00=-0.137037pp、相对B01=-0.033333pp，均未达到预登记+.2pp。BG联合相对B01的2draw×mid/low四格为[-.111111,+.055556,+.222222,-.111111]pp，0/4达到+1pp；相对WHITE为[-.833333,+.055556,+.111111,+.611111]pp，仅1/4达到+.5pp。RX3与RX8的联合clean相对B00分别-.407407/-.370370pp，超过.25pp保持约定。

通过项：实际As的独立与含Ag时的适应增益均在4个source urban格达到+3pp；联合BG相对WHITE的clean保持通过，两draw源low_suburban保持通过。源V来自已见RX，只支持源域诊断，不等于目标跨RX证明；2draw不是2个模型seed。独立判定生成时完整180epoch/39960step复算已执行，完整源图与零NN状态覆盖产物仍按实际完成状态报告。

本次从头E30没有共同的已训练父checkpoint；保持率参照明确为匹配B00 FINAL30的固定clean正确集合，而非共同E20父模型。既有“固定父集合”的概括在本矩阵中按该具体口径解释，不伪造共同父权重。
