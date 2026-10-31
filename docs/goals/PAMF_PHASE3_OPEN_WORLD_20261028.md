# PAMF Phase3：跨接收机星地无标签开放世界识别

## 已有方法优先选择（2026-10-31，用户最新指定）

用户要求“先选取前面性能好的方法”。当前研究主基线为 DURA v1.0＋原 PAMF 默认身份头＋PCD 无标签原型适配条件密度比 `PCD_predicted_ratio`（body-prototype r02/code4273e88，训练复用 r01/a33612e）；原 source-only `predicted_ratio`（class-density r01/4af0668）、同预算 SOURCE 和 `pooled_ratio` 保留对照。选择依据为旧 RX24-1 同一开发集三类 DURA 的已完成比较，后续 RX3-2 的 PCD 验证与联合身份变体均未达标，不能称已确认普遍收益；具体失败指标保留在下方时间记录和各报告。方法和三条件数据/指标见 `configs/pamf_phase3_selected_baseline_20261031.json`、`docs/PAMF_PHASE3_METHOD_SELECTION_20261031.md`。

此次基线选择本身不新增训练、预测、评分、私有 truth 读取或阈值拟合，不改各原 run 的 NO_PROMOTION，不宣称独立确认或达到指标。native-energy 实现草稿保留，不自动启动。后续已授权的独立实验仍各自预登记并保留实际完整结果；固定 PCD 的全簇已知关联与匿名注册本行已完成且 NO_PROMOTION，不能替换所选基线；具体失败指标见本行结果及下方时间记录。下方完整 Phase3 目标及三类 DURA 验收保持 ACTIVE。

## 当前执行范围（2026-10-31，用户最新指定）

后续新候选默认同时完成三类测试：源域已知接收机、跨接收机 clean、跨接收机星地，全部接入同一冻结 DURA v1.0→PAMF。clean 输入也先经过 DURA，必须重新计算原 F0 在相同 DURA(clean) 输出上的闭集基线，不能挪用历史绕过恢复器的 clean 成绩。clean 仅为冻结测试对照，Phase3 训练、无标签适应、发现和注册仍使用星地样本。原 C_K≥90%、同条件原闭集减 A_K≤2pp 保持；跨 RX clean 的 R_U≥95%、跨 RX 星地的 R_U≥90%，主报逐 TX 宏 A_K/R_U/H-score。三类结果分别验收，源侧未通过不取消已登记的跨 RX 测试，旧实验不追溯补测或改判。

用户随后要求先选前面性能较好的方法，核实是否接入DURA，未接入则补测并比较变化。当前优先固定 P1 signal r03 的已知支持与密度比两个方案：它们源侧 R_U46.888889%/60.888889%本来就含DURA，C_K和原F0损失满足原known12；旧跨RX clean直接输入没有DURA，也不是相同开放网络，不能冒充本次配对。已完成独立dura-best run，对同一个模型、统计和两套原阈值执行了source/跨RXclean/跨RXstar各raw与DURA配对，三个DURA路径为主，raw为接入前后诊断；每路径原F0闭集也用该路径输入计算，不重新训练或调阈值。预测类条件密度已另行完成三类DURA测试，按用户最新要求作为重点候选；原pooled density ratio保留冻结对照。局部协方差已完成并保留负结果；联合身份与开放表征已完成并保留未达标结果；当前固定密度拒识边界源训练保留原两个候选且不回写原F0/G/r03，具体矩阵见最新预登记。完整跨RX星地、零标签适应、未知发现和未来匿名注册目标ACTIVE。

## 历史研发路线（2026-10-30，用户要求再优化 PAMF）

用户明确要求对 PAMF 再优化设计。后续主线转为源端任务式开放表征学习：保留三视图 SIG-TIME 与原默认身份头，在可微中层、PA/SIG 特征处增加开放表征及有界身份残差，以轮换注册集合和跨 RX/day 支持／训练查询学习共享成员关系。暂称 PAMF-OW，当前是设计版本，尚未实现或取得新性能，不替换 Phase1/2 固定方法。三行 S0/S1/M 分别区分源监督编码器、固定集合联合成员学习及主候选任务式成员学习；拟单模型种子、每行同预算2220增量更新，真实实施前另按最小流程登记，不能将设计当成已启动实验。

设计与机读矩阵为 `docs/PAMF_PHASE3_OPEN_REDESIGN_20261030.md`、`configs/pamf_phase3_open_redesign_20261030.json`。此前 uniform OE、VICReg、OVA、metric Gram、reciprocal 和二元图负结果保留，不重复换名或用控制替主。目标端仍推进零 TX 标签域适应、自主发现及未来匿名持续注册；源旧类留出只是已见祖先上的训练代理，不声称严格未见。当前固定 Q 的 practical_high 缺必需未知类，保留诊断用途，不能认证六景全通过；完整确认另需预登记覆盖充分的合法独立记录，不事后重抽或降低门槛。下方 v3 双90与原F0损失2pp、未来AN80等完整验收不变，正式 RX24-18 未访问，goal ACTIVE。

## 当前主报与验收（2026-10-29 指定，2026-10-30 同步确认）

Phase3 主场景为跨接收机星地观测。已知类正确接收率 A_K 相对同输入、同路径原冻结 PAMF 闭集宏准确率下降不超过 2 个百分点；已知类宏接收率 C_K≥90%，未知类宏拒识率 R_U≥90%，总体及预登记主场景均检查。C_K 的 90% 覆盖此前 Agent 设置的 95%（FRR≤5%）；当前对应 FRR≤10%。后续再收紧 2pp 损失预算，由用户指定。

主表固定报告逐 TX 宏平均 A_K、R_U、H-score，其中 H-score=2 A_K R_U/(A_K+R_U)，先各自宏平均再求调和平均；接收但身份错误不计 A_K。C_K、同输入闭集准确率和 closed−open 损失并列作验收依据，宏 AUROC/OSCR 辅助诊断。H-score 不另设统一硬门槛，不替代上述同时成立的约束。详细定义和计算边界见 `docs/PAMF_PHASE3_METRICS_20261029.md`，机读要求为 `configs/pamf_phase3_targets_20261029.json` v3。

后续继续零目标 TX 标签适应、未知发现和未来匿名注册；原未来新类宏识别≥80%、注册前后旧类损失≤2pp、后续星地未知拒识≥90%等未修改项继续保留。clean 跨接收机拒识95%仅为 Phase1 地面参考，后续模型新消费的 IQ 全部叠加星地。历史 run 的预登记、阈值、结果和判定不追改。本文下方旧目标、旧资源／方法矩阵和完成记录按原时间保留，不覆盖本节最新口径。完整目标仍 ACTIVE。

2026-10-30 根据用户“同步更新目标”再次核对：本节、下方完整目标 prompt、指标定义文档及 v3 机读配置一致，作为后续新研究的执行和验收口径。应用目标卡仍显示较早的“星地拒识95%／已知误拒≤5%”；当前目标工具仅支持状态变更，不能改写活动目标正文，因此该旧卡不作为数值验收依据，也不能通过虚报完成来替换。完整 Phase3 尚未达成，保持 ACTIVE。

## 当前完整目标 prompt（2026-10-31 同步，指标及三类 DURA 测试沿用用户最新指定）

在现有三视图 PAMF/Tri-SIG-TIME SSL 基础上，实现并验证 Phase3 跨接收机星地开放世界 RFFI，依次完成未知拒识、零目标 TX 标签域适应、未知设备发现和匿名持续注册。Phase3 的训练、源端校准、无标签适应、发现和注册均使用已固定 G10 星地观测，目标接收机与源接收机互斥。后续候选同时完成源域已知接收机、跨接收机 clean 和跨接收机星地三类冻结测试，全部先接入同一 DURA v1.0，再进入 PAMF；clean 仅为冻结测试例外，不进入上述学习环节。主报逐 TX 宏平均已知类正确接收率 A_K、未知类拒识率 R_U 和 H-score=2A_KR_U/(A_K+R_U)。星地 R_U≥90%，跨接收机 clean R_U≥95%；同时要求 C_K≥90%，A_K 相对同输入、同 DURA 输出的原冻结 PAMF 闭集宏准确率下降不超过 2 个百分点。总体及预登记主场景分别验收，缺少必需类别覆盖不能认证。C_K 与 R_U 在达标基础上尽量提高，H-score 不替代各单项约束，2pp 预算后续再按用户指定收紧。source 结果不通过也完成已预登记的跨接收机测试；历史 run 的矩阵、阈值和判定不追改。

无标签适应须在同一独立未来测试上保住拒识、接收和身份保护并获得可报告的旧类正确接收正收益。匿名注册须在与发现池物理互斥的未来观测上，新设备宏平均正确识别率≥80%，旧类较同 DA 状态注册前下降≤2个百分点，已注册类宏接收率≥90%，后续仍未注册设备的星地跨接收机宏拒识率≥90%。所有旧类与匿名新类共同竞争；真实未知 TX、known/unknown 角色、类数、配额与 U→TX 评分映射均不进入预测器。U 仅按无标签算法适应、发现和建簇，Q 不用于拟合或回收训练；匿名身份映射只由独立 scorer 在 U 拟合并冻结，不能在 Q 重拟合。注册失败的新设备保留在未来识别分母中，不以建簇样本回测代替未来验证。

先做单模型 seed 的小规模可证伪研究，合法源侧冻结候选、阈值与晋级规则后才访问预登记目标；通过科学阈值后扩展3模型 seed/未见 RX 确认。全部预登记预测固定后独立 truth-last 评分，报告逐 RX/场景/TX、宏主指标、配对收益、覆盖和统计限制；目标测试不用于调参、选模、选择性重跑或回收历史 query。保留 Phase1/2 与历史 Phase3 的数据、权重、输出、阈值和原判定，checkpoint 祖先必须匹配数据契约。代码配置以 Git 提交推送和远端 OID 读回固定，正式训练默认 N607，每 GPU 最多新增两个本任务训练/预测进程、启动前核实剩余显存，不停止、修改或重启其他用户/owner 工作，不覆盖已有输出。完整范围达成并有对应实测证据前，目标保持 ACTIVE；科学失败不宣称完成，真正缺少合法输入/权限时说明具体缺项。

## 最新源端共同可行点单调扩展（2026-10-30）

`20261030-phase3-star-common-seeded-risk-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，code `119210ba86fb8b569f6cdeb887d314c1458a2007`，唯一主joint::common_initialized_budget为NO_PROMOTION。固定两个r02 F/aux/rawmax；精确复用common与zero两个旧策略，只改变新预算从common可行点起步且六阈值逐项只增，42组宏C_K90／原F0总身份损伤2pp保持。

主宏A_K90.922222%、C_K98.262963%、R_U28.825397%、H43.773215%，原F0同P1闭集90.981481%到open损0.059259pp。相对自身common的R_U24.920635%，多拒对123、丢失拒对0（+3.904762pp），已知正确少22、接受少54；H+4.646486pp。θ单调和逐ID接收子集由独立scorer核对；OE域宏代理56.668574%到57.253128%，仅31新增OE分数层。该结果排除了上一acceptall贪心退步的一部分初始化影响，但绝对拒识仍未达标，也不是全局最优证据。

control的common初始化R_U32.317460%相对原11.682540%改善，OE域宏36.647997%到47.275538%、810新增层；控制仍不能替主。六运行点known42及原结构guard通过、unknown总体／五RX／六景共12组全部未90%，没有目标晋级或扩seed。完整方法／分组／轨迹／配对结果见 `docs/PAMF_PHASE3_STAR_COMMON_SEEDED_RISK_RESULTS_20261030.md`。

仅重用同G10 P1公共knownV27000/OE6134/novel3150预测和原r02模型身份；无新真实IQ读取或前向、无F/aux/G/optimizer更新。每arm仅2条实际checkpoint合成G10 P0 noquery smoke，总F4，G0；42584原物理记录不增加。源仍为已见五RX与历史三novel，targetU/Q/formal读取0，A_N/H_OW=N/A。25tests/42subtests、4AST/runtime与唯一P0/P1通过，单archive比较/远端compile通过。

driver3106658、source3106706/CUDA4和3106707/CUDA2有独立live PID/CWD/argv/env读回；前四次未出现owned nvidia显存样本，末次已全部终态，不伪造实际GPU占用或峰值，不为遥测重跑。所有source和独立CPUscore完整、live为空，foreign未改。完整跨新RX星地拒识、零目标标签适应、未知发现与未来匿名注册保持ACTIVE；source局部收益不能替代最终验收。

后续评分路线的只读核查已完成：现有100D／106D描述和MD/RMD主要基于六类原型、均值和协方差，没有完整source L/OE bank的k近邻距离或局部支持比；novel内跨RX1NN只是评分端结构诊断。originalF0同G10 P1的L6300/OE6134单位fused160可从r02 teacher复用，novel3150的originalF0单位特征可从r02 baseline_fused_unit复用；当前指定完成目录中缺少V27000的同类单位特征。下一独立研究可检验局部支持分数并先补该合法源特征缺口，不能把该只读可行性结论写成新方法已实现／有效或目标已达成。

## 最新源端阈值风险分配比较（2026-10-30）

`20261030-phase3-star-oe-risk-budget-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，code `585e10b07be1f6821edea08b4853582b92512aff`。冻结上述原r02两个F/aux及rawmax，比较精确复用的common_tail与新oe_risk_budget，唯一主joint::oe_risk_budget为NO_PROMOTION。新算法从接受全部开始，用source OE15域宏拒识收益贪心分配六预测类阈值，全部42组原F0身份保护2pp和C_K90约束保持；没有用novel调阈值或修改神经模型。

主预算阈值宏A_K90.900000%、R_U20.285714%、H33.169215%、C_K98.274074%，原同P1 F0闭集90.981481%到open损0.081481pp。相同模型、rawscore和全部ID上的common为A_K91.003704%、R_U24.920635%、H39.126729%；新预算净少正确接收28、净少拒146未知。OE域宏代理也由56.668574%降至53.122567%，不能把主失败解释成“训练代理改善但对novel迁移失败”。贪心从自身acceptall收益0升高，不保证超过原有common可行点，更不保证全局最优。

control的预算R_U20.253968%对原11.682540%有局部收益，OE域宏36.647997%到40.116098%，但旧正确接收净少246；控制不替主。四运行点known42均过、unknown12均未90%，结构guard因模型／特征不变仍通过，A_N/H_OW=N/A。全部非score公共字段和common margin逐条精确复现父r02，分数校准差不冒充表征提升。完整分组、trace和配对差见 `docs/PAMF_PHASE3_STAR_OE_RISK_BUDGET_RESULTS_20261030.md`。

新消费仅既有星地P1 OE6134/arm与2合成G10 P0 noquery smoke/arm，共12272 F前向，F/aux/G更新0、G新前向0；V27000/novel3150复用公共预测、不新读IQ，不增加42584原物理样本。source仍为已见五RX和历史三novel，targetU/Q/formal读取0。21tests/34subtests、四AST/runtime和唯一独立P0/P1通过，单归档比较/远端编译通过。已捕获driver3090943和source3090989/3090990正确CWD/argv/CUDA4/2环境；第二次观察已全终态，未捕获独立live CUDA显存样本，不伪造峰值或为遥测重跑，foreign未改。

本结果只否定当前从acceptall出发的贪心路径相对主common的收益，尚未证明独立阈值族没有更好的可行点。后续若检验common初始化的单调扩展，应另立固定科学比较，不能热改本run、回填较高分或宣称已验证。完整跨RX星地拒识、零目标标签适应、发现和未来匿名注册仍ACTIVE。

## 最新源端联合 OVA 表征研究（2026-10-30）

`20261029-phase3-star-joint-ova-s2026092701-r02` 已在 N607 完成原 F0 教师重导、两组各2220步训练、冻结预测及独立 truth-last 评分，VERIFIED/ANALYZED_SOURCE_ONLY，唯一主 `joint::ova_max` 为 NO_PROMOTION。code `5aa041122396d48c49b8cb3d5912649bdc3217e1`。两组独立从原三视图 F0 E50/11100初始化；原默认六类头联合更新，新增966参数binary辅助层只拒识。CE＋原F0 logits KD＋单位特征保真相同，唯一差别为OVA是否反传骨干。每组F2220/aux2220、F累计13320；两组共4440次F更新及4440次aux更新，不能将8880次optimizer调用写成F更新。

全部真实输入为既有G10 P1：L6300与OE6134仅训练／教师，V27000只校准三分数各六阈值，novel3150只在模型和18阈值冻结后预测。source五已见RX、三日、旧六类和历史开发三novel身份保持；OE仅outside-six成员关系／域，无TX／真实K。没有新增targetU/Q或formalRX24-18句柄。42584原物理记录重复曝光不增加独立样本，原U_s不新增消费，真实G新前向0。

主方法宏 A_K91.003704%、R_U24.920635%、H39.126729%、C_K98.462963%；原F0同P1闭集90.981481%，总损伤−0.022222pp（净多正确接收6/27000）。known42保护通过，但unknown总体及预登记各RX／场景仍未满足90%。匹配detach_control的max为A_K91.292593%、R_U11.682540%、H20.714309%、C_K99.144444%；joint净多拒417未知（+13.238095pp），已知正确净少78（−0.288889pp）。局部配对收益不等于绝对目标达成，也不能与上一不同输入和容量的冻结MLP读出作单因素因果比较。

源novel排除同RX的1NN为joint79.968254%、control79.206349%、原F0 78.793651%，结构guard通过。这只是已见source RX间的设备结构代理，不能写成未来匿名A_N或无标签注册。原r01是teacher完成、两train在任何smoke／更新前错误比较完整wrapper900722与部署encoder232993而退出的技术失败，无性能结果；一次计数对象修复、真实native graph RED→PASS回归和一次定点P1复核后，r02用新run/release/teacher重跑原矩阵，r01全部保留。

实际训练GPU2/4各一worker，已独立观察train PID3073601/3073602、eval3075149/3075217、正确CWD/命令/GPU和日志增长；终态所有job完成、live与ownedGPU为空，其他owner未改。活动部署参数500910（F232993＋aux966＋G266951），完整native训练容器900722单列，不冒充部署规模。详细数据、六决策、完整4440步及分组诊断见 `docs/PAMF_PHASE3_STAR_JOINT_OVA_RESULTS_20261029.md`。

本轮证明了匹配控制下开放梯度有局部拒识收益，且源设备结构没有下降；它同时表明保住身份和结构仍不足以得到可靠拒识边界。保持当前主方法及所有负结果，不改阈值、不换较高分控制、不扩种子或进入目标测试。完整跨RX星地拒识、零标签适应、发现及未来匿名注册目标继续ACTIVE，A_N/H_OW仍N/A。

## 最新源端类条件开放边界研究（2026-10-29）

`20261029-phase3-star-ova-evidence-s2026092701-r01` 已完成 N607 source-only拟合/校准/四决策预测/独立truth-last评分，VERIFIED/ANALYZED_SOURCE_ONLY，唯一主 `ova_max` 为NO_PROMOTION。原F0默认P1身份、G10 P0/P1 cache与DURA祖先保持；新class-conditional配对106维输入由每path八分支的指定类/最强其他类证据组成。只训练两个共享106→32→1 MLP（每个3457参数）各1000 CPU更新，共2000；六条件槽共享权重，不是六个独立身份头。所有身份仍是原P1 argmax，OVA只判old-presence，不替换原分类头。

source L6300真实旧六标签及OE6134成员关系/域用于拟合，V27000仅四组共24个predicted-class阈值，全部冻结后才读source novel3150；原Us不新增消费。全部新IQ均为既有G10星地，一条received共享P0/P1，无真实raw/信道重生，source42584物理记录已历史曝光；五source RX/day与训练相同，不是目标跨RX或新TX盲测。代码 `0655a5cb0975dc047c712a76630d3e0354da0e55`，23tests/55subtests和唯一独立P0/P1 PASS。

四决策 A_K/R_U/H（%）：membership_routed=90.4519/38.6349/54.1434，membership_max=90.4000/40.8254/56.2485，ova_routed=89.9259/58.9206/71.1940，主ova_max=90.0185/55.5873/68.7320。主C_K97.6889%，原同输入F0闭集90.9815%到open损0.9630pp，四方法的42已知保护均通过；主unknown总体＋五RX＋六scene共12组全部未达90%。源数据/条件与此前目标RX24-1四态不同，不能把该55.59%与目标76.39%作直接性能变化。

同max算子下，OVA相对membership净多拒465条未知（+14.7619pp），已知正确净少103（−0.3815pp），H+12.4835pp；两个reader的容量、初始化、更新预算和normalizer匹配。OVA的max相对routed救回25条已知正确，同时净少拒105条未知（−3.3333pp）；四决策各有独立source-known校准τ，比较包括校准后的完整运行点，不能把全部差异归因于无校准算子。主方法不换为较高H的ova_routed，所有控制保持原身份。

实际85170 F0前向（含2synthetic星地smoke）、真实G新前向0，原F0/defaulthead/G均不更新，两个新readout更新2000单列。driver3031346/source3031437独立live CUDA4、GPU95d925a1/388MB，source日志9782→11034→21611B；CPUscorer3032614完成，终态所有jobs0/live空。source进程83.421217s、score3.287446s、pipeline86.790642s，不是部署逐包时延。没有新增目标U/Q或formalRX24-18句柄/读取，A_N/H_OW=N/A，其他owner不变。完整结果见 `docs/PAMF_PHASE3_STAR_OVA_RESULTS_20261029.md`。

当前证据说明难负类目标有源端收益，但浅层开放读出仍不足以达到拒识要求。下一研究应在合法源侧检验开放表征学习，同时保持旧类和未注册设备细粒度结构，而不是继续在已暴露目标Q上改变阈值或簇数。完整跨RX星地拒识、无标签适应、匿名未来注册目标继续ACTIVE，没有已全部完成的证据。

## 最新跨 RX 无标签联合实验（2026-10-29）

`20261029-phase3-star-neighborhood-s2026092701-r01` 已在 N607 完成 source90聚合→全U无标签适应/发现→四态独立未来Q→CPU truth-last评分，状态 VERIFIED/ANALYZED_DEVELOPMENT，唯一主状态DA1_REG1为NO_PROMOTION。固定paired1000及六个预测类τ，新增源IQ只复用G10 L6300，U384/futureQ864均为RX24-1、source五RX互斥，原G10 MIX01与物理角色不变；真实TX/角色/K/配额不进入predictor。code `cb58217de28eb2cb5536d2392aef268b79006877`，24tests/34subtests通过且唯一独立P0/P1 PASS。

DA0_REG0：A_K65.9722%、R_U76.3889%、H70.7995%、C_K88.3681%，原F0同P1闭集72.9167%到open下降6.9444pp。DA1_REG0：A_K61.6319%、R_U79.1667%、H69.3074%、C_K86.6319%；旧类净少25条正确接收（−4.3403pp），未知净多拒8条，不能称适应成功。源prior的已知保护未能自动迁移到新RX，不能拿source42组通过替代目标约束。

全U内部silhouette选3个邻域簇（257/93/34），算法仅登记1个匿名簇；U-only独立评分查明该93簇含52旧类、41条到达类19-15，旧类污染55.9140%，另一个到达类19-16未发现且仍在未来分母。主DA1_REG1：A_K61.2847%、C_K96.3542%、后续未知R_L14.5833%、相应H23.5603%，A_N39.5833%、H_OW27.2351%；原F0损11.6319pp，计划注册类C87.8906%。注册对同DA旧A_K总体仅损0.3472pp，并不能抵消绝对旧类保护、A_N、R_L和注册类接收失败。REG0的R_U对象为全部三非旧TX，REG1主R_U为后续TX7-1，不能直接当同一未知分母比较。

源L重新前向12602 F0（含2合成smoke）、目标2500 F0/1250 G（含合成smoke），零新神经/默认头/读出/阈值优化。父读出实际保存并计算4个3233参数模型再取paired1000，不隐去额外计算。独立观测driver3000725与target3000900 GPU5/508MB，targetlog524→875B；源worker3000818只有terminal证据，scorer3001048完成。全部jobs完成、live为空，其他owner不改，formalRX24-18=0。完整分析见 `docs/PAMF_PHASE3_STAR_NEIGHBORHOOD_RESULTS_20261029.md`。

本轮保留负结果，不按目标分数重选K或阈值、不扩种子/正式RX；下一步需要在合法源侧检验能同时保留设备细粒度结构和跨RX可靠新颖性的方法，避免将混簇均值直接传播为身份或用宽半径吸收后续未知。完整Phase3仍ACTIVE，当前没有指标已全部完成的证据。

## 离开 Phase1 后全部使用星地观测（2026-10-29，用户最新澄清）

用户引用“评估 MILKR 的无标签适配”（thread `01a18197-50fe-7f93-bd50-7f98d3f59cff`），明确 Phase1 后接触的样本都叠加星地。已实际读取该任务，并核对 r09/r10 文档、六分量配置和已提交 `data.observe` 源码（`71f1428c8ac85695ebffb2efd791fe34061b325b`）。Phase3 新的 U、发现、注册、未来 Q，以及新消费的 source replay/辅助训练/校准 IQ 均为星地；Phase1 固定权重/聚合祖先保留，原始地面 IQ 仅离线生成 received，不供方法逐条参考。下文历史 clean/三景矩阵与完成记录保持原身份，以本节为后续范围。

主条件改为 MILKR G10：600km/25MHz/2.45GHz、residual/post_sync/noEQ、一次输入 RMS、输出不重归一化、receiver seed2027，六场景按覆盖面积概率抽样。综合 low/mid/high 为10–30/30–45/45–80度；指定单景 mid urban 仍20–45度。P0 是星地 received，P1 是同 received 经固定 DURA，两条都不是 clean。新 episode 重抽信道，方法/种子/P0P1共享该 episode 的固定 received；不把重复原记录计为更多物理样本，也不引入 Phase2 的标签/真实K或 query回收权限。

当前 Phase3 主验收以上方最新口径为准：跨RX星地总体及预登记主景未知宏拒识≥90%，已知宏接收≥90%、同输入closed到open损失≤2pp；适应须有旧类正收益，未来匿名新类macro≥80%、注册旧类损失≤2pp、后续未知拒识≥90%。原clean跨RX95%仅保留地面参考，不作后续Phase3新数据路由或星地成功证明。目标保持ACTIVE，完整无标签开放世界范围不缩减。

先构建新独立 `20261029-phase3-star-g10-pool-rx24-1-r01`：仅复用原RX24-1 U384/Q864物理角色，按MIX01原G10配置重叠加一次；旧received与成绩保留，不再新增clean实验。此数据作业不读模型、训练、预测、评分或正式RX24-18，后续方法另记run；未声明新G10性能已达标。细节见 `docs/PAMF_PHASE3_STAR_G10_INPUT_20261029.md`。

## 原启动摘要与历史目标

2026-10-28 启动，2026-10-29 曾按用户要求分开记录clean95%与星地90%。后续输入范围以上方“离开Phase1后全部使用星地观测”为准。默认N607，不停止或修改其他用户/owner任务；每张卡最多新增两个本任务训练或预测进程。本文同时承载目标prompt、设计和分步计划，不另设设计审批。

## 先前目标修订（2026-10-29，保留原指标解释）

用户明确“95%拒识正确率是指跨接收机的，同时叠加星地的跨接收机拒识正确率在90%以上，还有已知类接收率越高越好”。据此分开验收 clean 跨RX≥95% 与星地跨RX≥90%，后者各预登记主场景也≥90%；不再把星地95%作为当前硬门槛。已知类接收率定义为1−宏误拒率，在未知拒识约束下尽可能提高，保留宏误拒≤5%和同输入原F闭集到开放准确率损失≤2pp保护；错误接受为另一旧类不算成功识别。未来持续未知也按相应输入条件采用95%/90%。历史run的原预登记和NO_PROMOTION结论不追溯修改。机读口径见`configs/pamf_phase3_targets_20261029.json`。总目标仍为完整拒识、零目标标签适应、发现及未来匿名注册，未宣称完成。

## 原目标prompt（后续输入范围由首节覆盖）

在当前三视图PAMF/Tri-SIG-TIME SSL基础上，完成跨接收机开放世界RFFI：clean条件未知TX宏拒识≥95%，叠加practical residual星地后总体及每主场景≥90%；在这些约束下最大化已知类接收率，同时保留FRR≤5%及同输入原F闭集到开放损失≤2pp。依次实现冻结模型拒识、零目标TX标签适应、未知类别发现和匿名持续注册；适应须有旧类正收益，注册须在未来独立观测新类宏识别≥80%、旧类较注册前损失≤2pp、后续未知维持相应95%/90%拒识。目标端不提供TX标签、known/unknown角色、未知类别数或类别配额；允许的无标签发现数据必须与最终query物理互斥。真实身份命名/业务授权与匿名设备注册分开。预测先固定、独立truth-last评分；不根据目标成绩调阈值、选模或选择性重跑。保留Phase1/2原产物，先做单模型seed小规模可证伪实验，源侧冻结规则且达到科学门槛后扩展3个模型seed确认。指标未达时保留负结果，不擅自降低门槛或宣称完成。

## 原验收口径（历史，当前以首节最新要求为准）

|任务|量化目标|评价限制|
|---|---|---|
|未知拒识|按真实未知TX等权，clean跨RX≥95%；星地跨RX总体及各预登记主场景≥90%|unknown仅在scorer连接；不能用defer冒充成功|
|旧类接收与保护|最大化已注册类宏接收率；宏误拒≤5%；含误拒的正确率相对相同IQ/default-head闭集基线下降≤2个百分点|接收率与正确识别分开，同时报告逐类、RX、scene及95%区间|
|无标签适应|保持上述拒识/旧类约束，并报告同query旧类正收益|不适应、普通伪标签、可靠旧类筛选作明确控制；未证实则不得声称有效|
|匿名注册|未来独立观测新类宏平均识别≥80%；旧类相对注册前下降≤2个百分点|不在建簇样本上回测；真实TX映射只可评分端计算|
|持续开放|后续未注册设备clean跨RX拒识≥95%、星地跨RX拒识≥90%|注册后的新unknown集合独立；错误合并/拆分/重复注册分别报告|

目标针对点估计；95%统计区间另列，不将“95%拒识”写成“95%置信下界”。纯噪声/证据不足可暂缓，但全部已注册类暂缓/拒识均计错误。缺失已知或未知的组返回不适用，不能判成功。

## 数据与方法设计

默认仍为跨RX泛化：source RX与target RX不交。单星被动监听是首轮主场景，多节点协同不是前置条件。分别评价clean跨RX及固定practical residual/post_sync/noEQ/25MHz星地跨RX；每记录仍仅一份固定practical观测，known与unknown使用同一物理规则。clean为独立条件验收，不作为星地算法可读的逐条参考或监督。源训练保持当前合法旧6TX契约；严格未知TX必须未进入其监督/无标签/教师/恢复器祖先。曾无标签见过但未注册设备单列，不混称严格未见。

新Phase3独立划分source L/U/V、target无标签适应/发现池、future query及后续未注册设备。历史Phase1/2 query不转为训练/注册。首轮只使用冻结PAMF，不加载DURA，不更换或另训Phase1默认头。源类原型仅由L_s形成，V只用于冻结阈值。部署预测进程不读取source IQ/源loader、truth、角色或真实类别构成；源侧校准与truth侧评分分为独立命令。

首轮冻结比较MSP、maxlogit、margin、energy、合法L_s cosine中心距离；所有分数高值代表known，各自最终接受分数在同一source V上以固定99% retention校准一个全局阈值。初始公式/retention不得看target后改变；不按真实scene或TX路由阈值。所有拒识分数共享原default-head的闭集类别决策，隔离拒识贡献。融合/适配/注册属于后续独立候选，仅在合法source开发证据基础上固定并预登记。

## 分步与分工

1. 数据与基座核对：数据代理定位checkpoint全部祖先、旧/未知TX、RX/day/物理角色及未使用记录。主Agent决定最小可运行切片；只核对改变的数据事实，不重复验证既有capsule。
2. 拒识基线：主Agent实现source校准、truth-blind frozen predictor和独占run launcher。评分代理独立实现truth-last scorer，防止全拒绝、配额/角色泄漏和大样本二次内存。
3. 目标无标签适应：可靠已知候选与不确定/未知候选隔离；源锚点不可被target回写；先判决状态，再小adapter。每版独立run、同输入匹配比较。
4. 新类发现与注册：在独立无标签池建立候选簇，检查旧类污染、跨条件稳定与足够独立观测；以匿名ID及非参数代表注册，在future query验证。
5. 持续确认：新设备分轮到来，报告旧类保持、未来新类识别及后续unknown；达到预先门槛才扩展3seed，不提前铺满矩阵。
6. 一名独立审查者只做一次下次真实run所需P0/P1审查；主Agent整合、提交/push/OID读回、N607归档传输/编译/真实checkpoint无query smoke、启动后PID/CWD/argv/GPU/log增长核实及后续评分交付。

所有worker只拥有明确指定的新模块；共享工作区现有未提交/已暂存修改不回退、不混入提交。唯一launch owner为本聊天主Agent。

## 首轮实现计划

Tech stack：既有verified Python/PyTorch/NumPy；不安装新依赖。隔离在新包`experiments/pamf_phase3_open_world/`，不修改历史训练器。

- [x] `scores.py`：`knownness_scores(features, logits, prototypes)`及source-only quantile冻结；有限值、形状、类别置换一致性和组合阈值语义负测。
- [x] `data.py`：新单观测capsule的public received/opaque IDs与独立truth；验证source/target、discovery/query、场景物理互斥和无角色公开。输入复用现有合法构建器，禁止旧七视图loader改名冒充。
- [x] `runner.py`：源L原型/V阈值独立校准；严格加载三视图E50，批量导出无真值knownness/closed_prediction/accepted/prediction；全部文件写完才产生`ALL_PREDICTIONS_COMPLETE`。
- [x] `scoring.py`：ID一一对齐、closed baseline/known误拒/unknown拒识、逐TX/RX/scene、O(NlogN) AUROC/OSCR、完整预测后才连接truth；全拒绝/漏ID/重复ID/错shape/提前truth负测。
- [x] `launch.py`：不可覆盖root、每卡本run新增进程≤2、显存容量检查、单owner，不杀其他进程；固定技术停止条件。
- [x] `tests/test_pamf_phase3_*.py`：仅相关协议负测和算法行为测试；真实checkpoint no-query smoke为launcher第一步，PASS自动继续。
- [x] 按实际路径登记`20261028-phase3-pamf-open-set-s2026092701-r01`；实际矩阵在`experiment.json`维护，本文不伪造尚未核实的样本数/运行状态。

## 发布和停止

单一code/config commit；只stage相关文件，push当前upstream并比较远端OID。镜像保持相关新文件精确一致。N607一份不可覆盖release归档、一次本地/远端SHA比较、一次remote compile。每GPU本run最多两个训练/预测worker，其他用户进程不作为整卡禁用条件；启动前仍检查实际显存余量，优先较空闲卡，容量不足则排队。

仅停止经精确PID/CWD/cmdline/root验证属于本run的技术失败进程。协议越界、错误split/checkout、覆盖冲突、无法合法预测或两row相同pre-prediction异常属于技术停止；低拒识率或负收益只记录科学结果。可能已完成的远端动作先独立读回再决定是否重试。相同技术指纹经一次针对性修复仍复现则保留产物并请求必要信息。

## 当前状态

目标模式ACTIVE。首轮`20261028-phase3-pamf-open-set-s2026092701-r01`已在N607 GPU2完成真实模型smoke、source校准、1440条冻结预测与独立truth-last评分，产物状态VERIFIED/ANALYZED。五种简单拒识基线全部未达95%及旧类保护门槛；完整结果见`docs/PAMF_PHASE3_OPEN_SET_PILOT_20261028.md`。不按该target表调阈值/选方法，继续source-only类别/RX留出开发。新的目标无标签适应/发现池与后续query已经单独构建，历史query不转训练。其他用户进程保持，目标未完成。

Full eq下载已于2026-10-29由独立owner完成158PKL/8767867条全量验证，见docs/PAMF_PHASE2_FULL_EQ_VALIDATION_20261029.md；本Phase3复用该原始验证，不重扫20ZIP。RX `24-18`不在现有ManySig/ManyTx/ManyRx/SingleDay四子集RX列表内（四个header有界只读证据已保存），也与本轮Phase2原7RX隔离。新run`20261029-phase3-full-pool-rx24-r01`已完成VERIFIED/VALIDATED_ONCE：旧6加发现2类D1每TX每景32条无标签适应/发现（768条），D3上述8类加后续未知2类每TX每景64条query（1920条），总2688唯一物理接收记录、三景与两角色互斥。三景固定low suburban/mid urban/low urban压力分层，不当作实际被动监听概率。两public各仅3白名单文件，truth独立，零模型/预测/训练，正式Q未推理或评分；code commit e4ebaa2682d8c2e87182a8b8ce36ae5ea5eec4d9，详细数据句柄见`docs/PAMF_PHASE3_FULL_POOL_20261029.md`。source含D3地面数据，该切片不声明严格未来日期外推；不按Full index>=1000或容器前缀判断新记录，不改其他owner的下载/暂停Phase2任务。

source run`20261029-phase3-source-holdout-s2026092701-r01`已在N607 GPU2完成，VERIFIED/ANALYZED_SOURCE_ONLY。四类scratch232673参数、160维，TX0/1与RX8不进入训练，E20完整540步；source split392005、model seed2026092701。73项Phase3测试、一次本source候选独立P0/P1与真实metadata兼容检查通过。完整20epoch及87行stdout解析显示训练最终98.75%、已见RX V已知98.6528%、留出RX8为66.4722%；8规则全部未达95%推进，NO_PROMOTION，不扩三fold，详细见`docs/PAMF_PHASE3_SOURCE_HOLDOUT_RESULTS_20261029.md`。

物理分支run`20261029-phase3-source-branch-s2026092701-r01`也已VERIFIED/ANALYZED_SOURCE_ONLY，95项测试及一次新候选P0/P1通过；N607 GPU2零新增训练，一次导出30360条8分支，完整16规则均NO_PROMOTION。全V最高未知拒识3.378%、RX8最高7.278%，RX8最高AUROC0.7493，不从中挑正式方法。执行commit df34171b8dc9942ec236e4504cac61768d70997a，结果见`docs/PAMF_PHASE3_BRANCH_EVIDENCE_RESULTS_20261029.md`。无标签代理的合法输入和未来评分边界保持；下一步优先联合域偏移校正、unknown成簇和anonymous注册。正式无标签适应、未知分类与持续注册仍未完成，目标保持完整ACTIVE。

联合source run`20261029-phase3-source-joint-s2026092701-r01`已VERIFIED/ANALYZED_SOURCE_ONLY，执行commit6570e891982d72bc317c9219ead3e97221d93951。模型、数据、主Agent及review分工完成纯几何/源角色导出/四阶段流水线/独立truth-last评分；初始68项＋P1修复相关48项（23项未变）和独立4复核通过。仅D1R1可代表完整联合晋级，R0/D0控制不能替代注册/适应成功。真实2Lsmoke PASS，N607 GPU2只新增一条导出worker，零神经训练/原U_s/正式target读取；源代理U1050与未来V5400完整处理。D0/D1均首轮触及32组件上限、合格0，D1旧类开放准确率63.9167%较D0下降1.2222pp，后续未知拒识2.1111%，新类未来识别0，NO_PROMOTION；详细见`docs/PAMF_PHASE3_SOURCE_JOINT_RESULTS_20261029.md`。该失败不外推完整PAMF能力，不增cap/改阈值挽救旧run。

下一优先路线直接利用当前满意的完整六类PAMF，使用独立source开放开发数据，不再只依赖预算弱得多的4CE代理。元数据确认Full174类比四compact并集150多24类，其中7类在原source5RX×D1D2D3均≥70；按容量与字典序事前分为辅助暴露4类4200条、独立源新设备验证3类3150条，排除旧6和正式4unknown。科学边界已写入PROJECT_PROTOCOL，额外source已知/未知成员监督明确披露，原L/U/V和六类身份头不变；正式target仍零标签。数据run`20261029-phase3-source-oe-data-r01`已VERIFIED/VALIDATED_ONCE：7350唯一source记录、15成员、每TX/RX/day70、公开无辅助TX、17行完整日志无异常，原L/U/V/模型/target/channel均0，零GPU、不停止他人。6项测试/10subtests及一次独立P0/P1通过，执行commit795e4a9604f391cb53e19ba24c2d2e8d5f24188a，数据句柄见`docs/PAMF_PHASE3_SOURCE_OPEN_DATA_20261029.md`。

完整六类source拒识候选`20261029-phase3-source-open-readout-s2026092701-r01`已完成，详细在`docs/PAMF_PHASE3_SOURCE_OE_REJECTION_PLAN_20261029.md`及对应config。F与默认身份头冻结，8物理分支导出后由L拟合类置换不变的100维描述符；仅额外源成员监督训练101参数linear或1633参数MLP16，各200次CPU feature更新，known V仅99%阈值校准，辅助新设备3TX只在全部预测后独立评分。5静态控制不能推进OE候选，source门槛95%未知/5%误拒/2pp旧类保护含各源RX，双过时优先linear；这不是全Phase3验收。实际64项测试＋35个subtests与独立P0/P1通过，真实导出、200步读出训练及独立评分均已完成，科学NO_PROMOTION；实际结果见下节，正式RX24 U/Q仍未模型访问。

## 完整六类源OE拒识实际结果（2026-10-29）

`20261029-phase3-source-open-readout-s2026092701-r01`已VERIFIED/ANALYZED_SOURCE_ONLY，执行commit`76951634c0901175a6d78c88caffbf49fdce8d04`；4完整日志、真实2Lsmoke和四阶段产物齐全。当前完整6类F与身份头冻结，8分支完成40652条前向，两个源成员监督读出各完成200次CPU更新；原U_s／正式RX24 U/Q读取0，encoder／身份head更新0。linear_oe／mlp16_oe的3个辅助未知TX宏拒识68.4762%／76.6032%，known宏FRR均0.9963%，旧类开放宏准确率97.9148%／97.8852%（原closed98.4667%）。整体及全部sourceRX旧类保护通过，但unknown整体和RX3/4/6/8未达95%，仅RX1通过；eligible空、selected=null、NO_PROMOTION。详情见[源OE结果报告](E:/type10-7/docs/PAMF_PHASE3_SOURCE_OE_REJECTION_RESULTS_20261029.md)。这是source clean已见RX的新TX开发，非跨RX／星地／无标签适应／注册成功；本run不调参重跑、不扩seed、不访问正式池。后续仍需独立source候选，当前无新实验启动；总目标仍未完成。

## 下一独立源风险校准候选状态（2026-10-29）

source-known risk calibrated scalar 已由 `20261029-phase3-source-risk-calibration-s2026092701-r01` 真实完成VERIFIED/ANALYZED_SOURCE_ONLY，执行commit ba2b4ed63a03751b8934dd7385f16a05aa75bcf3。它复用冻结F／OE状态，只由known V标签校准总体及每RX宏FRR≤4%、原正确样本拒绝损伤≤1.5pp，所有源known约束通过；MLP未知宏拒识81.4603%（父76.6032%）、FRR1.3667%、known开放97.6185%、drop0.8481pp，未知整体和RX3/4/6/8仍未达95%，NO_PROMOTION。0新训练／Fforward／GPU，2L特征拒识checkpoint smoke PASS，正式target0；同knownV经验风险与同novel重复源开发，不是新盲测／跨RX／星地／适应或注册成功。见[风险校准结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_RISK_CALIBRATION_RESULTS_20261029.md)。不根据未知提高预算或重跑，Phase3总目标仍未完成。

## 下一跨分支类别对应候选（2026-10-29）

`20261029-phase3-source-consensus-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，执行commit951fbddea7cafb4deb5e0e18a4d404edd7d02b74。仅100D跨分支同类支持描述符改变，F／L-only几何和数据冻结；新linear／MLP各200CPU步，真实父checkpoint2L与新descriptor smoke PASS，新增Fforward／GPU／target／foreignstop0。linear未知宏拒识82.6984%、FRR1.6185%、drop0.6963pp；MLP未知85.7460%、FRR1.4444%、drop0.7815pp，较risk父方法81.4603%提升4.2857pp。known保护全过，unknown整体及RX3/4/6/8仍未达95%，NO_PROMOTION，eligible空、selected=null。同knownV经验风险／同novel重复source clean开发非新盲测，未证实接收机不变，也非跨RX／星地／适应或注册成功。见[consensus结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_CONSENSUS_RESULTS_20261029.md)。不调参重跑或访问正式池，总目标仍未完成。

## 源辅助设备多样性扩展（2026-10-29）

下一步先构建21台源辅助训练数据，保留3台novel和正式4unknown，原L/U/V与F保持。`20261029-phase3-source-oe-diverse-data-r01`已登记，builder5项测试/5subtests及一次独立P0/P1通过，待本机CPU真实构建。按原source5RX×3day每非空TXcell最多32、少量全取，预期6134唯一记录（原4台1920＋新增17台4214），不按IQ/score挑选。后续源读出拟按role内15域均匀权重，归一化保持旧6300/4200的known.6/OE.4先验；这是新的源监督与域配重候选，不宣称只改了样本数或已经有效，目标端标签和访问仍0。

后续若需要训练开放空间embedding，必须沿原`pamf_repl_view_shuffle.source.train`训练构造器重建三视图/固定排列/六类默认头后strict加载核实E50payload，不能对`frozen_inference.reader_from_payload`直接解冻：推理reader把Sinc._filters换为含no_grad的frozen_filters。合法增量训练需保留原CENTER_UNITS及标签权限、单列原11100步＋新增预算；当前只做只读可行性调查，没有开启该训练。

## 多设备源OE数据实际完成（2026-10-29）

`20261029-phase3-source-oe-diverse-data-r01` 已VERIFIED/VALIDATED_ONCE、ANALYZED，执行commit2f41fc2835798f583c54900ab4c791d762d0101c。6134条唯一source记录来自21台训练暴露设备，220非空／95空cell，保留novel19-15／19-16／7-1不转train；15个source容器读取、完整18行1467B日志及独立poststate齐全。额外source成员监督明确，原L/U/V／novel验证IQ／模型／channel／正式target／GPU调用0，不停止其他任务；尚无模型性能结果。见[数据结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_OE_DIVERSE_DATA_20261029.md)。

## 多设备源域均衡拒识实际结果（2026-10-29）

`20261029-phase3-source-diverse-readout-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，执行commit70c8346339f5d2f5bcbe14aa8863eca545fa167a，NO_PROMOTION。冻结完整六类F／身份头／L几何／100D consensus描述符，新21台OE6134同时采用role内15域均匀及normalizer .6/.4固定先验；linear／MLP各200CPU步。linear未知61.4286%、FRR1.5630%、drop0.5778pp；MLP未知75.0794%、FRR1.5185%、drop0.5852pp，较consensus85.7460%下降10.6667pp。known保护总体和每RX均过，unknown整体及RX3/4/6/8仍未达95%，eligible空、selected=null。真实2L IQ smoke与OE6134共6136新F前向，旧L/V/novel特征复用，GPU2只新增一导出worker；F/head/geometry更新、原U_s／正式target读取、foreignstop均0。34项测试＋3subtests及唯一P0/P1 PASS，4完整日志与400loss已归档。组合改动不能拆因，也不能断言更多设备有效或特征绝对上限；同knownV经验风险和同novel重复source开发，非跨RX／星地／适应／注册成功。见[结果报告](E:/type10-7/docs/PAMF_PHASE3_SOURCE_DIVERSE_READOUT_RESULTS_20261029.md)。不调阈值预算或重跑；该后续小预算source OE embedding微调＋同预算原生L监督增量对照已由下文r02完成，结果NO_PROMOTION。Phase3总目标仍未完成。

## 原生开放集特征增量候选（2026-10-29）

`20261029-phase3-source-embedding-increment-s2026092701-r01` 已登记并实现，实际发布后在优化前技术失败；同矩阵r02已完成，见下文。supervised_only保留原native KL/CRIC/As监督骨架，不是纯CE/PAMF-CE；uniform_oe加λ.1均匀六类KL，两行均从原F6 E50独立初始化，cold CENTER_UNITS恒lr1e-4、新增222步/累计11322，原头和骨干joint、U关闭。77项测试/52subtests及一次完整独立P0/P1 PASS、一次N607 preflight PASS；先原生重建并strictload避免冻结Sinc梯度，真实2L梯度smoke(和OE行2OE)无step且恢复state/RNG后继续。source已知保护独立对照原F6同输入closed，源门槛保持95%unknown/5%FRR/2pp total保护，各RX均验，额外closed损伤预算.5pp；仅uniform_oe::msp可晋级，两行完整才判。source21远端public数据复用，无新数据重传/重验；2行并发每卡≤2新增，不停他人，单行技术失败保留另一健康行。正式RX24仍不读。详细见`docs/PAMF_PHASE3_SOURCE_EMBEDDING_INCREMENT_PLAN_20261029.md`；总目标保持ACTIVE。

r01实际发布后两行在新增父兼容检查失败，0梯度smoke／优化／评估，未产生科学性能；原native祖先检查已过。实际source contract存split_seed392005、cache复用路径不同，新增校验错误要求config字段和路径相等。现仅定向修复该层，保留native E50与物理role_ids约束；2项／3subtests与一次本P1定向复核PASS，无raw重验／额外hash。r01已记FAILED并保留，r02同科学矩阵／222／种子／数据恢复，独立新root、一次release preflight PASS，现已实际完整训练、预测及评分，科学NO_PROMOTION。目标和95%门槛不降，当前仍无达标正式target证据。

## 原生特征增量 r02 实际结果（2026-10-29）

`20261029-phase3-source-embedding-increment-s2026092701-r02` 已VERIFIED/ANALYZED_SOURCE_ONLY，commit28d53ea9b38c19f642c59e31f178a3f65f40894d。两个独立原F6 E50各新增222步、累计11322，原native L监督／原头联合保持，uniform_oe额外λ.1均匀KL，原U_s关闭；真实原L2与OE2梯度和最终模型smoke均PASS。预指定MSP监督未知47.8095%、closed98.5074%、open97.9222%、FRR1.4815%、原F总损伤0.5444pp；OE未知59.9683%、closed98.4963%、open97.9593%、FRR1.5185%、总损伤0.5074pp，配对未知+12.1587pp。全部5分数总体/每RX编码器与开放损伤保护通过，unknown各RX仍未95，只有uniform_oe::msp可晋级且实际未过，NO_PROMOTION。每行170496 L包视图/28416 L采样/14208 As，OE另14208；GPU2监督/GPU1OE，owned在跑各1496MB观察转录有来源，最终均退出，foreignstop/target/V训练0。444步完整trace有随机As波动，不宣称收敛；r01原FAILED为0优化技术失败，无新性能，定向修复未改变科学矩阵。见[完整结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_EMBEDDING_INCREMENT_RESULTS_20261029.md)。同knownV经验风险及同novel3重复source开发，不是跨RX／星地／无标签适应或注册成功。下一独立固定E51＋原4OE4200／consensus100D组合已按下节完成；不改本轮MSP/95%门槛或重跑，总目标未完成。

## 固定 E51 与读出组合实际结果（2026-10-29）

`20261029-phase3-source-embedding-readout-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，执行commit806dec253b7556ee3d4af842bf9a2d8fd714009e。两编码器各自重导40652样本、L-only新几何＋原4OE4200读出，各linear/MLP200CPU步，历史222不变、新encoder/head0。MLP监督86.3492%、OE90.7619%，配对+4.4127pp，较原E50同novel参考85.7460%为+.6032/+5.0159pp。OE FRR1.5556%、open97.7407%、原F总损伤.7259pp；7方法known总体/每RX保护均过，两个MLP仅RX1未知达95%，故NO_PROMOTION。30tests/2subtests与一次P0/P1PASS、800loss/8logs/2Lsmoke、原始GPU2/1 PID和退出证据完整，foreign/target0。见[完整结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_EMBEDDING_READOUT_RESULTS_20261029.md)。仍是同novel重复source开发，目标未完成。

下一独立候选`20261029-phase3-source-baseline-risk-s2026092701-r01`已登记、当前实现中：冻结上述全部状态，无新F/head/readout更新，只用knownV和原Fbaseline直接校准原目标中的总open损伤≤2pp与FRR≤5%（总体/每RX）。此前自身损伤1.5pp/FRR4%的安全预留不追改；新候选仍须95%unknown/5%FRR/.5ppencoder/2pp总损伤全部通过，unknown不校准。正式RX24 U/Q仍不读，Phase3总目标ACTIVE。

## 原模型参照风险实际结果（2026-10-29）

`20261029-phase3-source-baseline-risk-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，执行commita18daf5080bca7e76c3837907dcfdefd97fbc5bd。两固定E51/读出只重校known-only阈值，F/head/readout新增0；MLP监督88.0635%／OE92.6667%，OE较父+1.9048pp，FRR2.0296%、open97.3852%、原F总损伤1.0815pp。总体及每RX known保护通过，监督RX6/OERX3总损伤2pp预算恰绑定，两个MLP仍仅RX1未知过95，故NO_PROMOTION。9tests/4subtests与一次P0/P1PASS，4完整logs、2Lreadout smoke、阈值最大合法及完成产物齐全；工作极短首次post即完成，未捕获liveCWD/GPU观察、不伪造/重跑。见[结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_BASELINE_RISK_RESULTS_20261029.md)。同novel重复source，不是跨RX/星地/DA/注册成功。

下一独立`20261029-phase3-source-embedding-budget-s2026092701-r01`已登记并实现中：同原F6 E50独立冷启动的监督/OE两行固定新增2220步（E51..60），累计13320，非E51挑选后续训；native L/CRIC/As、OE21 λ.1、Uoff保持，自己的8分支/newL几何＋原4OE读出200，knownV仍原F总损伤2pp/FRR5%。固定末步、无V/novel早停，95与旧类保护不降，每卡≤2新增、不停foreign；正式RX24U/Q仍未访问，Phase3总目标ACTIVE。

## 十轮源增量实际运行（2026-10-29）

`20261029-phase3-source-embedding-budget-s2026092701-r01`已VERIFIED_RUNNING，执行commit6ffccc4ac1bb80d2d47a77e08a8a288012fa1353，36tests/38subtests PASS、一次新候选P0/P1 PASS。N607一次preflight/单归档比较/compile后，真实原L2/OE2梯度smoke PASS，GPU1监督/GPU2 OE已独立核实PID/CWD/argv/CUDA与日志增长；均各一新增worker、不停foreign。固定两行各2220步，当前尚无终态性能；训练后自动source导出/读出/known-only校准/独立评分。正式RX24U/Q仍不访问。

状态工具在本次用户更新指令后读回总目标为`blocked`，未提供原因；该读回不等于科研指标完成。用户原Phase3研发授权继续适用，本轮已授权实验照常执行；没有将目标工具写成ACTIVE、没有修改其状态或新建重复目标。当前Phase3整体尚未完成。

## 十轮源增量实际结果（2026-10-29）

`20261029-phase3-source-embedding-budget-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY/NO_PROMOTION，执行commit6ffccc4ac1bb80d2d47a77e08a8a288012fa1353。两原E50独立分叉各2220步，累计13320；4440步trace、800loss、10logs、实际Sinc/原head梯度与own8branch/readout smoke齐全，五stage全部完成。MLP监督90.0635%、OE92.0635%，OE FRR1.7778%、open97.7222%、原F总损伤.7444pp；仅RX1未知达95，各eligible空。与E51已完成参考92.6667%相比，十轮低.6032pp，未证实加轮数进一步改善未知拒识。正式RX24U/Q仍未模型访问；其他用户保持、owned GPU1/2均已退出。见[完整结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_EMBEDDING_BUDGET_RESULTS_20261029.md)。

用户当前请求总结报告，正在汇总方法/训练验证测试角色/跨RX与未知设置；此时不启动新的科学候选。后续局部聚合几何、连续拒识证据到未知发现和future验证的机制仅记研究笔记，不冒充已实现。总目标工具读回blocked且未被改写，Phase3整体未完成；源best92.6667%是同RX clean开发成绩，不是正式跨RX星地95%。

## 最新跨接收机实证与下一步（2026-10-29）

用户最新指标已固定：clean跨RX未知拒识≥95%，叠加星地后总体及各主景≥90%；已知类接收尽可能高，保留FRR≤5%与原F同输入闭集到开放损伤≤2pp。历史各run原门槛/NO_PROMOTION不追改，完整目标的零标签适应、发现、未来匿名注册与持续未知保护仍保留。

RX24-1独立开发池已实际完成：U384（D1）/V864（D2），旧六类＋开发发现19-15/19-16、后续未知7-1；三景low suburban/mid urban/low urban。原source五RX、正式24-18与开发24-1互斥，原Full/compact元数据复用。star code26c3e79eb6d078fc492c6b885ae8831e4358fb6c，clean控制code959a1a68e76a7e2cfba6700c2a251da8330e59b9；两条件同1248原始物理记录/同序opaque ID，不计更多独立样本，模型端clean与star隔离。两个真实CPU构建有起始live PID/CWD/命令及日志增长、完成产物和退出证据。正式RX24-18 U768/Q1920仍未模型访问。

已完成两独立冻结迁移诊断。统一固定source E51 uniformOE/MLP/原knownV阈值，原F6同输入closed作保护基线，不根据开发成绩选endpoint/分数/调阈值；各自全部V864预测后独立truth-last评分，U0/sourceIQ0/逐条sourcebank0/formal0/更新0。P1/MLP为星地主行，P0和其余6分数仅诊断。

|条件/主MLP|未知拒识|已知接收|含拒识已知识别|原F同输入闭集|总损伤|
|---|---:|---:|---:|---:|---:|
|C0 clean跨RX|99.3056%|80.7292%|62.6736%|79.6875%|17.0139pp|
|P0 星地received|75.0000%|44.0972%|30.5556%|53.9931%|23.4375pp|
|P1 同received＋DURA v1|57.2917%|72.9167%|53.2986%|72.7431%|19.4444pp|

C0未知单项过95，但旧类FRR/总损伤失败；P0/P1均未过90且known保护失败，所有实际状态NO_PROMOTION。DURA改善known，却在当前source拒识器下多接受未知，不能仅报known收益当开放集成功。C0/P0/P1是配对物理记录的不同条件，不把sameRX source92.6667%与此新RX跨条件结果当同输入提升。完整报告分别见`docs/PAMF_PHASE3_DEVELOPMENT_TRANSFER_RESULTS_20261029.md`、`docs/PAMF_PHASE3_DEVELOPMENT_CLEAN_TRANSFER_RESULTS_20261029.md`。

实用景codea3b8bb24b6b9e357bf37678209cf0c21d666f7df，10tests/16subtests及一次P0/P1 PASS；clean生产算法未改，code/config1e1876db7d0f7b8f3724fe95acf3c8c8d5b64fef补3个C0 tests/8subtests及一次新C0配置边界P0/P1 PASS，未重复旧代码全审。两run各完成N607 preflight/单归档一次比较/compile、实际checkpoint合成2IQ no-query smoke后立即预测。前台任务很短，首次独立post已终态，未捕获运行期独立PID/CWD/GPU/log增长；实际CLI身份、完整日志/markers及已退出状态有保存，不冒充live证据/峰值/部署延迟，不为补观察重跑。GPU0实用景与GPU6 clean各只新增1worker，无foreign停止；后续GPU表其他任务保持。

下一研究重点直接转向合法U_dev上的无标签域偏移处理及身份保持的未知发现：先提高未见RX的已知接收/正确识别，同时守住95%/90%拒识。当前结果显示未见RX的已知样本被大量拒绝，支持域变化影响拒识的判断；单纯提高拒绝强度或只恢复known不足以达标。下一候选须另行固定算法/源校准/更新预算与D0/D1及注册控制，U只读公共无标签输入、不读真实构成，V不回收训练，正式RX24-18继续隔离。目标保持ACTIVE；当前没有成功无标签适应或未来匿名注册证据，不标完成。

## 首个真实无标签U适应结果（2026-10-29）

`20261029-phase3-development-alignment-s2026092701-r01`已VERIFIED/ANALYZED_DEVELOPMENT_ONLY，code`cdaeccaf201f8c242ac00400b5c499f4dc1e63a8`。当前F使用GN/LN，无可直接更新的BN统计；新方法为源锚点/初始可靠旧伪类约束的正交多分支特征adapter。源MSP/cos门槛＋head/fused近邻一致＋time/frequency/PA2票，≥8条/旧伪类且≥3类；不用MLP已接受或unknown比例/类数/配额。一次求R=polar(I+.5M.TP/K)，reference27固定0，std raw右乘后交原默认head重算logits/descriptor。它保留std空间欧氏距离/cos，不保留whitened距离，也不把latent坐标称真实PA恢复。原F/G/head/MLP/source几何/阈值不变，但确实拟合了U特定R状态，每condition234825个FP64系数/1878600B，不把0神经optimizer步数写成未适应。

C0/P1从同一source独立用U384（RX24-1 D1）拟合R并写冻结checkpoint后，再对D2 V864逐样本预测；D0只在R冻结后加载旧完成预测作匹配控制，不重复旧row；真值只独立scorer连接。两condition同1248原物理记录，不共享clean参考。两行均实际applied，C0合格core157，P1初选150/合格147；支持5个旧伪类0/2/3/4/5，不等于5个真实TX或已知纯度。U中的两个未注册设备已无标签见过，第三个later未进入U；这些开发TX此前source评过，不称source新TX盲测。正式RX24-18 U768/Q1920依然未模型访问。

|主MLP条件|旧open D0→D1|已知接收D0→D1|未知拒识D0→D1|D1原F总损伤|
|---|---:|---:|---:|---:|
|C0 clean跨RX|62.6736→63.1944%|80.7292→81.2500%|99.3056→98.9583%|16.4931pp|
|P1 星地+DURA跨RX|53.2986→55.0347%|72.9167→74.8264%|57.2917→57.2917%|17.7083pp|

旧open分别多3/576与10/576，+0.5208/+1.7361pp；都有正收益，但known FRR/原F2pp保护仍失败，P1总体/scene/later未知保护仍失败，均NO_PROMOTION，不扩seed、不按V改η或阈值。clean unknown单项仍过95，不代表整体成功。已完成真实无标签适应实验，但尚无通过验收的适应方法；匿名注册仍0，完整目标ACTIVE。

两行各真实checkpoint合成2IQ smoke先于U，所有模型与R在Q冻结；sourceIQ/逐源样本bank/formal/新阈值0。原pure+outer联合22tests/39subtests、launcher3tests PASS；唯一review的两个P1（storage同根与Torch2.1/NumPy2桥）已定点修复，新2回归＋1邻近3tests/14subtests PASS并一次定向复核PASS，未重复全审/数据验证。一次N607 preflight/单归档比较/compile PASS。真实driver/C0worker live PID/CWD/cmd和C0日志增长已捕获，模型运行时ownedCUDA采样未捕获，GPU3分配来自driver，不报告峰值或逐包时延；完成后owned均退出，foreignstop0。实际pipeline13.7489s包含模型载入/两条件U与Q/评分，不是部署延迟。完整报告见`docs/PAMF_PHASE3_DEVELOPMENT_ALIGNMENT_RESULTS_20261029.md`。

下一优先改进源训练的拒识分数/表征对星地域和冻结恢复后输入的覆盖。当前P1 D1主分数AUROC约71.6363%；本等量开发池若单global阈值同时已知接收≥95%且未知拒识≥90%，至少95%×90%=85.5%的known/unknown配对应正确排序，因此仅移动当前阈值无法满足两项。此为当前单分数/本数据的必要条件，不推导所有模型不可能，也不扫描V寻找oracle阈值。新独立候选可让合法source L与既有辅助OE通过同规则星地/G输入来训练拒识器、源known-only校准后再验证；不把开发V回收训练，不更换source基座/默认恢复器或擅改数据权限，不读取正式Q。未知身份聚类与未来匿名注册≥80%的完整目标继续保留。

## 源端星地读出匹配结果（2026-10-29）

`20261029-phase3-source-practical-readout-s2026092701-r01`已VERIFIED/ANALYZED_DEVELOPMENT_ONLY，代码/config为`ca60bdf20090f664fbcf27682138d63cfd2751dc`。原E51/F0/G/六类默认头/clean-L几何全部冻结。本次复用source5RX、三日的L6300/OE4共4200/V27000/novel3共3150，40650物理记录，每条clean＋三practical→DURA，共162600计算view，不新增独立记录。新channel seed2026102931；各角色同scene/opaqueID规则，不读OE TX或novel truth生成。源端成员监督额外存在，不称全流程无标签。

calibration_only精确复用原linear/MLP和mean/std；augmented_readout在L25200/OE16800视图上各固定200CPU AdamW步，拟合101/1633参数及训练mean/std。两arm同四view known V108000校准，整体/5RX/4view/20RX×view共30组FRR5%/原F同输入closed−open2pp，ties整组，一个global τ/方法。源未知不校准；两arm状态/阈值全部冻结后才source novel预测，8组预测固定后独立truth-last。两arm均known校准可行，但source未知拒识不达标。主候选augMLP事前固定，不按source或target切换。

开发仅复用RX24-1、D2的Q864（576known/288unknown）配对C0/P1，两条件共864原始物理记录，targetU0/正式RX24-18读取0。两个current arm共享本次F/G前向和六类head；历史original D0及原F baseline精确复用。两条件历史/本次closed差异均0。source负结果仍按预登记完成全部开发预测和独立评分，不重跑、不调目标阈值。

|条件/主MLP状态|未知拒识|已知接收|已知open正确率|原F closed−open损伤|AUROC|
|---|---:|---:|---:|---:|---:|
|clean original|99.3056%|80.7292%|62.6736%|17.0139pp|97.8709%|
|clean calibration_only|96.1806%|86.8056%|67.8819%|11.8056pp|97.8709%|
|clean augmented_readout|96.8750%|90.4514%|71.5278%|8.1597pp|98.2898%|
|星地+DURA original|57.2917%|72.9167%|53.2986%|19.4444pp|70.9262%|
|星地+DURA calibration_only|34.7222%|89.4097%|66.3194%|6.4236pp|70.9262%|
|星地+DURA augmented_readout|38.8889%|93.0556%|68.5764%|4.1667pp|86.1075%|

clean相对原状态多56/576已知接收、多51/576已知正确开放识别；未知多7/288被接受，仍单项超过95%。星地多116/576已知接收、多88/576正确开放识别，但未知多53/288被接受，拒识单项远低90%。相对同校准control，augMLP的星地AUROC增加15.1813pp，known接收+3.6458pp/open+2.2569pp/unknown拒识+4.1667pp；确有分数排序收益，尚非验收成功。

aug星地低郊区/中城市/低城市未知拒识分别47.9167%/35.4167%/33.3333%，known接收94.7917%/91.6667%/92.7083%；三景均失败。中/低城市AUROC84.3967%/82.1994%仍低于同时95%接收/90%拒识所需85.5%的必要下界，故仅挪动当前单分数阈值仍无法满足逐景保护。总体AUROC86.1075%超过必要下界也不证明存在合格阈值，不扫描开发oracle阈值。

两条件均NO_PROMOTION，不扩seed或访问正式池。仅源读出/校准不能消除域变化下的身份特征和未知分离问题；下一优先候选应在合法source训练角色上进行星地/G输入的编码器与原默认头联合增量训练，保留同预算source控制与clean身份保护，争取同时改善known正确识别和unknown排序。不回收V_dev训练、不读其真值拟合、不改变原默认F/G身份，不把更高接收率当未知拒识进展。未知发现/未来匿名注册仍未完成，完整目标ACTIVE。

9stage全部COMPLETE、实际checkpoint合成2IQ四view smoke在source IQ前通过，C0/P1也各在Q前smoke。聚焦29tests/24subtests PASS；唯一独立审查两个API P1已修复并定向确认。一次N607 preflight/单归档比较/远端compile VERIFIED。捕获实际driver2624334/export2624355的PID/CWD/命令、GPU3可见性、owned CUDA占用与日志增长；900MB为一次采样不是峰值。source导出437.2247s、两个readout拟合stage6.0401s，完整stage计时见post_state，不当作逐包部署时延。完成后本run进程退出，其他owner未停止/改动。完整报告为`docs/PAMF_PHASE3_SOURCE_PRACTICAL_READOUT_RESULTS_20261029.md`，原始评分/训练损失/后态在本run local_artifacts。

## 匹配预算的源端星地编码器联合增量结果（2026-10-29）

`20261029-phase3-source-practical-encoder-s2026092701-r01`已VERIFIED/ANALYZED_DEVELOPMENT_ONLY，实际code/config `b578c074ba68b8ff9aa539209b3410f1f103ba3d`。同一E51父状态各独立222步，累计11544，fresh AdamW CENTER_UNITS lr1e-4/wd1e-4/物理clip5；原六类默认head和Sinc都实际联合更新，未另训身份head。两arm保留native原六路监督/crossRX0.1/As和U-off，额外L128 CE权重0.68：clean_increment用clean，practical_increment用同记录随机三景固定source practical/G。OE21的64条物理记录以clean＋对应第二view共128平均uniformKL乘0.1；成员监督存在，未知TX不读取。control仍含native旧As，不称全无星地训练。

共享训练cache仅L6300/OE21共6134，12434物理/49736计算view，源U/V/novel/target0。源channel2931/receiver2912与另一个独立scene采样流2932按record选择，不按类别；primary额外L三景9568/9582/9266，共28416，额外OE4839/4663/4706，共14208。两arm各L native170496＋extra28416＝198912 packet，OE两view28416，222步全部完成。源端freeze后各用自身新clean-L几何、L/OE4四view读出各200CPU、known V四view30组校准，再novel预测/独立score；每arm40650物理/162600计算view。targetU0/正式RX24-18读取0，源/开发科学失败均不改已冻结配方。

两arm自动各完成source和C0/P1完整预测/独立truth-last。C0/P1仍同RX24-1 D2 Q864（known576/unknown288），两arm/条件不增加独立物理样本；父parent_E51精确复用上一run augmented_readout，不复用更早original读出或重新运行父模型。source与query阶段模型/聚合量冻结。以下MLP均预登记主方法，主arm一直practical_increment：

|条件/状态|未知拒识|known接收|known闭集正确|known开放正确|原F closed−open损伤|AUROC|
|---|---:|---:|---:|---:|---:|---:|
|C0 parent_E51|96.8750%|90.4514%|78.2986%|71.5278%|8.1597pp|98.2898%|
|C0 clean_increment|97.5694%|90.2778%|75.6944%|70.1389%|9.5486pp|97.5272%|
|C0 practical_increment|98.6111%|85.7639%|67.5347%|57.8125%|21.8750pp|97.1861%|
|P1 parent_E51|38.8889%|93.0556%|71.8750%|68.5764%|4.1667pp|86.1075%|
|P1 clean_increment|37.1528%|94.2708%|71.8750%|69.0972%|3.6458pp|87.2962%|
|P1 practical_increment|50.0000%|89.0625%|69.9653%|62.8472%|9.8958pp|85.2515%|

原F同输入闭集C0为79.6875%、P1为72.7431%。primary相对control，P1多拒未知37/288（+12.8472pp）却少接收known30/576、少开放正确36/576（−6.25pp），AUROC反降2.0448pp；C0闭集少47/576（−8.1597pp）、开放少71/576（−12.3264pp）。相对parent，primary的P1未知多拒32/288，但开放少33/576；不能只取拒识单项作正结果。两arm均NO_PROMOTION，不将control换成主候选、不扩seed或读正式池。

primary星地低郊区/中城市/低城市拒识60.4167%/46.8750%/42.7083%，known接收90.1042%/89.5833%/87.5000%，逐景均失败。P1整体AUROC85.2515%低于同时95%接收/90%拒识所需85.5%的必要下界；更直接地，primary C0/P1闭集本身已分别低于原F允许的77.6875%/70.7431%最低开放准确率，而开放准确率不可能高于自身闭集，所以移动阈值无法挽回身份保护。此为当前分数/模型/开发池的数学界，不推断所有方法无解，也不扫描oracle阈值。

源端known V上primary clean闭集98.3481%、lowurban闭集85.6704%，control为98.5519%/84.9519%；源域较小变化不能代替新RX的身份保持。source未知主目标仍失败，完整源/开发及逐RX/TX/scene结果见`docs/PAMF_PHASE3_SOURCE_PRACTICAL_ENCODER_RESULTS_20261029.md`。该结果不支持继续无约束地改动完整身份函数；下一优先验证保留父模型clean路径的受约束星地特征映射/身份保持，再接合法U的无标签域适应，降低附加TX监督依赖。新方案仍另行冻结目标/矩阵，不回收开发Q训练；未知发现和未来匿名注册≥80%继续保留，完整目标ACTIVE。

共享cache＋两row共15stage全部COMPLETE，native真实checkpoint梯度smoke及source/dev合成smoke通过，head/Sinc更新有实际状态比较，所有源/目标标签边界与冻结检查通过。32项唯一tests/68subtests覆盖：dev旧7/24及新增真实目录回归/邻近3/10分别保存；唯一独立review的一个parent/source/fit目录P1已修并一次定向PASS，不重复全审。一次preflight/单归档比较/compile VERIFIED；捕获driver2673776、cache2673864、两rowdrivers2675837/2675840、训练2675845/2675846和source2678276/2677698等实际PID/CWD/cmdline/CUDA及log增长/ownedGPU采样。训练分别GPU7/4、一次采样各1540MB，不是峰值；完成后owned退出、foreign0。源训练实计101.4671/86.8424s，整pipeline712.9620s含缓存/并行/加载/源开发评测，非因果加速或逐包时延证据。

## 2026-10-29 冻结身份的无标签适应与匿名注册实测

`20261029-phase3-unlabeled-registration-s2026092701-r01` 已完成全部 5 jobs 与独立 U/Q truth-last 评分，两个条件均 NO_PROMOTION；总目标 **ACTIVE**。身份 F0/E50、辅助 E51、G 与 source augmented 读出全冻结，无 target 标签输入或新神经更新。真实 U384 先适应/聚类/冻结，再预测未来 Q864；C0/P1 共用 1248 个物理 ID，正式 RX24-18 未访问。

| 条件 | 状态 | 旧类接收% | 旧类 open% | F0 损失 pp | REG0 未知拒识% | 新类未来 macro% | H% | 注册类 FRR% | 后续拒识% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | DA0_REG0 | 90.4514 | 71.7014 | 7.9861 | 96.8750 | N/A | N/A | N/A | 93.7500 |
| C0 | DA1_REG0 | 90.4514 | 71.1806 | 8.5069 | 97.5694 | N/A | N/A | N/A | 93.7500 |
| C0 | DA0_REG1 | 90.6250 | 71.7014 | 7.9861 | N/A | 75.5208 | 73.5616 | 12.7604 | 93.7500 |
| C0 | DA1_REG1 | 90.6250 | 71.1806 | 8.5069 | N/A | 75.5208 | 73.2865 | 12.8906 | 93.7500 |
| P1 | DA0_REG0 | 92.7083 | 69.0972 | 3.6458 | 40.9722 | N/A | N/A | N/A | 37.5000 |
| P1 | DA1_REG0 | 93.2292 | 69.2708 | 3.4722 | 36.8056 | N/A | N/A | N/A | 35.4167 |
| P1 | DA0_REG1 | 93.7500 | 69.0972 | 3.6458 | N/A | 15.1042 | 24.7895 | 11.5885 | 36.4583 |
| P1 | DA1_REG1 | 94.0972 | 69.2708 | 3.4722 | N/A | 14.5833 | 24.0942 | 10.1562 | 34.3750 |

C0/P1 均实际应用 bias；DA1_REG0 相对 DA0_REG0 的旧 open 为 −3/576（−0.5208 pp）/+1/576（+0.1736 pp），适应总目标未通过。C0 注册两簇，DA1 一簇混入12条旧类；P1仅一混合簇（old32/new35），两真实到达类合并，未单独发现者仍在未来 macro 分母。REG1 没有增加旧类正确数，两个条件新类 macro 均低于80%，注册 FRR、later拒识及旧类绝对保护均失败。

原 F0 同输入 closed C0 79.6875%、P1 72.7431%；主状态旧 open 为71.1806%/69.2708%，总损8.5069/3.4722 pp。C0 REG0拒识97.5694%仅单项过95%；P1为36.8056%。所有 source known-only 校准30组/七方法均合法，不等于目标保证。匿名ID的U真值映射只用于独立评分，不在Q上拟合；REG0新类/H为N/A，错误匿名接受不算拒识。

全pipeline 27.578251s、source统计 10.867493s，非逐包时延。实际双F＋G＋两diagnostic读出共734671参数（主MLP-only734570），未计入非梯度几何/锚点/proto等状态。独立只捕获启动driver PID/CWD/cmdline，未抓到live GPUworker/显存峰值；终态live/owned为空、foreignstop0，不为补采重跑。详情与全部分组索引见 `docs/PAMF_PHASE3_UNLABELED_REGISTRATION_RESULTS_20261029.md`。

该开发闭环完成不构成总Phase3完成；维持clean95%/star90%、FRR5%、同输入原F0open损失2pp、未来新类macro80%与旧类保留全部目标。

下一科学假设是：冻结旧身份函数仍不足以使当前星地拒识几何可分；应区分候选集的旧类污染、两到达类合并及跨日期原型失配，再检验有约束的表征/拒识证据能否同时改善未来新类识别与后续拒识。该假设尚未实施，不回收本轮 Q 调参。

## 统一G10星地开发输入已就绪

`20261029-phase3-star-g10-pool-rx24-1-r01`已VERIFIED/VALIDATED_ONCE，code/config `3f24f8016f5586f096bbaa8d83f221b397dbbeb8`。保持RX24-1原U384/Q864物理角色，1248原记录按MILKR实际G10代码生成一份新MIX01星地观测；原clean/三景压力数据保留，主输入改为星地P0及同received的DURA P1。12tests/47subtests及唯一P0/P1 PASS，真实CPU数据作业与完整日志/产物已独立核对；模型/GPU/训练/预测/评分/正式RX24-18读取0。后续新source replay/calibration也必须先星地化；本数据run未给出新性能，完整目标ACTIVE。详见`docs/PAMF_PHASE3_STAR_G10_INPUT_20261029.md`。

下一实际动作先建立与Phase1原L/V物理契约一致的G10 source统计/校准输入，再从Phase1原F0/default6头和冻结DURA进行新主条件的拒识/无标签适应/注册配对验证。不能把旧Phase3 E51的clean辅助训练或clean几何改标成满足新输入范围；若另作历史控制须完整披露祖先。源原始IQ仅builder可读，需从原始记录在一次RMS之前生成星地，避免把已归一化clean缓存当原始数据再归一化。此前讨论的relative/domain几何方案尚未实施/登记，须在正确星地输入上独立固定方法和控制；本轮没有新的神经训练或性能候选。

## 2026-10-29 G10 星地单 F0 相对距离四态结果

`20261029-phase3-star-relative-s2026092701-r01` 完成5jobs与独立U/Q truth-last评分；两 path 均 NO_PROMOTION，总目标 **ACTIVE**。新source只消费原L6300/V27000的G10观测，唯一F0＋G全冻结、神经0更新，无E51/OE/clean模型输入；σ只来自L，V只known-only42组τ校准。

| path | 状态 | 旧接收% | 旧 open% | F0 损 pp | REG0 拒识% | 未来新 macro% | H% | 注册 FRR% | later 拒识% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | DA0_REG0 | 99.1319 | 56.7708 | 0.3472 | 8.6806 | N/A | N/A | N/A | 13.5417 |
| P0 | DA1_REG0 | 99.1319 | 56.7708 | 0.3472 | 7.6389 | N/A | N/A | N/A | 14.5833 |
| P0 | DA0_REG1 | 99.1319 | 56.7708 | 0.3472 | N/A | 0.0000 | 0.0000 | 2.2135 | 13.5417 |
| P0 | DA1_REG1 | 99.1319 | 56.7708 | 0.3472 | N/A | 0.0000 | 0.0000 | 1.6927 | 14.5833 |
| P1 | DA0_REG0 | 97.2222 | 71.0069 | 1.9097 | 3.1250 | N/A | N/A | N/A | 4.1667 |
| P1 | DA1_REG0 | 97.5694 | 71.1806 | 1.7361 | 2.7778 | N/A | N/A | N/A | 4.1667 |
| P1 | DA0_REG1 | 97.2222 | 71.0069 | 1.9097 | N/A | 0.0000 | 0.0000 | 2.7344 | 4.1667 |
| P1 | DA1_REG1 | 97.5694 | 71.1806 | 1.7361 | N/A | 0.0000 | 0.0000 | 2.3438 | 4.1667 |

P0：REG0 RMD拒识7.6389%、旧接收99.1319%、open56.7708%；注册后新macro0.0000%、注册FRR1.6927%、later拒识14.5833%、F0总损0.3472pp。adaptation=False、registration=False；scene缺类用N/A/coverage，不填0或认证。
P1：REG0 RMD拒识2.7778%、旧接收97.5694%、open71.1806%；注册后新macro0.0000%、注册FRR2.3438%、later拒识4.1667%、F0总损1.7361pp。adaptation=False、registration=False；scene缺类用N/A/coverage，不填0或认证。

两path同一RX24-1 U384/Q864、同1248物理G10记录，跨日但非source未见日期；正式RX24-18未读。F0+G499944 NN参数，额外metric统计另计；source33300物理、66600F/33300G计算曝光。独立GPU采样[(2818487, 404), (2818487, 798)]是采样非峰值；终态owned/live为空与foreignstop0以observed_state核实，pipeline119.498955s非逐包时延。全部方法/scene/TX/簇组成与计数边界见 [完整结果](../PAMF_PHASE3_STAR_RELATIVE_RESULTS_20261029.md)。

两path全局旧类接受/同输入F0损失单项通过，但未知拒识和未来新类80%均失败；P1 high urban/low urban 的F0损伤4.1667/3.5755pp超限，high suburban缺19-15而不能完整认证。四态注册K全0，P0/P1新类184/188条错接为旧类，故低注册FRR不代表成功。DA1 RMD AUROC .774185/.652507低于.855必要界；P1 RMD较MD排序低7.7498pp，不能仅平移τ解决。

下一假设（未实施）：只在源侧区分RX/day扰动和身份方向，检验能否改善拒识排序及候选密度支持；当前没有可评估的已注册原型，不归因于已证实的跨日半径失败。保留全部科学目标，不用Q调参或提前完成总Phase3。

## 2026-10-29 综合指标与下一源端证伪

用户要求先明确联合性能指标，再继续研发。定义固定在 `docs/PAMF_PHASE3_METRICS_20261029.md`：身份正确且接收为主性能，接收率/FRR为保护；注册前未知N∪L与注册后到达新类N/后续未知L分别统计。H_OS及宏加权OSCR辅助比较，项目H_OW仅描述且不替代各硬约束。所有原数值目标保持；未知真值、K和评分仍不进入预测。

下一固定源端证伪为 `20261029-phase3-star-nuisance-s2026092701-r01`，唯一主候选P1/RX×day/RMD，control/RX-only与MD作消融。新source heldout3150沿原物理mapping生成G10，无目标访问；source已知V只校准阈值。方法/推进规则见 `docs/PAMF_PHASE3_STAR_NUISANCE_PLAN_20261029.md`，source通过不构成跨RX或完整Phase3完成。

## 2026-10-29 源端 RX/day 硬投影证伪结果

`20261029-phase3-star-nuisance-s2026092701-r01` 两jobs完成；主 `rxday::RMD::P1`：**NO_PROMOTION**，总目标 **ACTIVE**。

| 候选 | A_K% | R_U% | H_OS% | macroOSCR% | C_K% | F0损pp | 源gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control::MD::P0 | 75.1370 | 2.7937 | 5.3870 | 58.6502 | 99.0519 | 0.6741 | False |
| control::MD::P1 | 90.7037 | 0.8889 | 1.7605 | 77.2587 | 99.3296 | 0.2778 | False |
| control::RMD::P0 | 75.3778 | 4.0000 | 7.5969 | 62.5673 | 99.0852 | 0.4333 | False |
| control::RMD::P1 | 90.4519 | 6.4444 | 12.0317 | 78.1555 | 98.8370 | 0.5296 | False |
| rx::MD::P0 | 75.2815 | 4.6667 | 8.7885 | 61.4871 | 99.0815 | 0.5296 | False |
| rx::MD::P1 | 90.5963 | 1.9048 | 3.7311 | 77.6677 | 99.1037 | 0.3852 | False |
| rx::RMD::P0 | 75.2481 | 5.1429 | 9.6277 | 62.5699 | 98.7296 | 0.5630 | False |
| rx::RMD::P1 | 90.6667 | 2.0952 | 4.0958 | 77.1439 | 99.2111 | 0.3148 | False |
| rxday::MD::P0 | 75.4593 | 3.9365 | 7.4827 | 62.8788 | 99.2444 | 0.3519 | False |
| rxday::MD::P1 | 90.6333 | 2.1270 | 4.1564 | 77.9211 | 99.1963 | 0.3481 | False |
| rxday::RMD::P0 | 75.3593 | 3.8413 | 7.3099 | 62.4770 | 98.9741 | 0.4519 | False |
| rxday::RMD::P1 | 90.6296 | 2.3492 | 4.5797 | 77.2395 | 99.1741 | 0.3519 | False |

只L6300拟合投影/Gaussian/score scale，V27000额外known标签只作42组τ校准，novel3150固定预测后独立评分；target输入/U_s IQ/OE训练IQ/E51/NN更新均0。主candidate known42保护=True、unknown必需12组90%=False。
33300known为同physicalID/同episode的确定性重建；novel为旧3150物理记录的新G10观测，已有源开发暴露、同sourceRX，不能当新物理或DG确认。单F0＋G499944 NN参数，72900F/36450G计算，聚合状态另计。
主对control的ΔA_K/ΔR_U/ΔH_OS/ΔOSCR/ΔAUROC=0.1778/-4.0952/-7.4520/-0.9160/-1.0847pp；不按单项换主。资源samples=[(2849941, 798), (2849941, 832)]非peak，终态owned/live0、foreignstop0。
主分数AUROC82.7789%低于同时95%接收与90%拒识所需85.5%必要界；known保护已过，问题是源开放集分离，而非闭集head漂移。主P1五branch硬删除rank12/15/9/5/3虽保留preunit类间span，reunit后fused类均值仍变化31.29%；该机制未通过源端证伪，不推进target。
完整逐RX/scene/TX、42组保护、实际投影rank/能量与类间损伤、计数及权限边界见 `docs/PAMF_PHASE3_STAR_NUISANCE_RESULTS_20261029.md`。无target验证/DA/注册结果，不宣布Phase3完成；后续假设未实施。

## 2026-10-29 G10 源端表征学习结果

`20261029-phase3-star-representation-s2026092701-r01` cache＋三arm train/eval/score均完成；固定主 `oe_vic/P1/energy`：**NO_PROMOTION**，总目标 **ACTIVE**。

| arm/P1 energy | C_K% | A_K% | R_U% | H_OS% | macroOSCR% | F0损pp | arm源gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anchor | 97.4778 | 90.4852 | 17.0159 | 28.6450 | 79.3009 | 0.4963 | False |
| oe | 98.2074 | 90.8222 | 22.1905 | 35.6666 | 84.2900 | 0.1593 | False |
| oe_vic | 98.4481 | 90.7259 | 25.3968 | 39.6847 | 83.3727 | 0.2556 | False |

三组从原F0 E50/11100独立2220 IID更新至13320，L两view CE＋同输入原F0 KD；B加OE uniformKL，C再加OE raw-fused VICReg。L6300标签＋OE6134 outside-six额外成员监督，V27000只42组τ，novel3150只冻结后源评分；target/U_s0。
baseline保留cache同输入原F0，而非新encoder自身closed；完整训练6660步、Sinc/head梯度与buffer不变、曝光、七方法/两path/分组/逐TX/源1NN及资源边界见 `docs/PAMF_PHASE3_STAR_REPRESENTATION_RESULTS_20261029.md`。
本轮按事前C_K≥95%/R_U≥90%/原F0总损≤2pp判定，不追改；后续用户新C_K≥90%口径另见targets/metrics，未重算本轮τ。C相对B运行点H升4.0182pp，但A_K降0.0963pp、macroOSCR降0.9173pp、源1NN降13.2063pp，不能据H或接收率单项推进。
源1NN为已有五source RX间heldout TX结构诊断，不是A_N；A_N/H_OW=N/A，无DA/发现/注册。source成功也不意味着完整Phase3达成，控制不能替换C，下一假设未实施。

## G10 冻结 F0 多分支证据拒识器实际完成（2026-10-29）

`20261029-phase3-star-consensus-s2026092701-r01` 已 VERIFIED/ANALYZED_SOURCE_ONLY，代码 `652091c76546716d4ef687f771854ef0560dbcb6`。按最新 v3 实际执行双90%/原F0损失2pp，不改写上一轮阈值。原 F0 E50/11100与六类默认头完全冻结，复用现有G10 P0/P1缓存；L6300/OE6134拟合100D多分支consensus与域等权binary linear/MLP，各path各200步，总800CPU读出更新，F/head/G更新0。knownV27000仅42组阈值，novel3150仅两path全部模型与阈值冻结后的独立truth-last源评估。

唯一主 P1/mlp16_oe 的宏 A_K=90.1407%、R_U=44.9206%、H-score=59.9606%，C_K=97.9741%；原F0闭集90.9815%，总损失0.8407pp，宏AUROC92.4137%、宏OSCR84.9425%。42组known保护通过，unknown总体/5源RX/6scene未达90%，NO_PROMOTION，不访问target或以控制替换主候选。完整2×7方法、逐TX/RX/scene、800步读出loss、模型状态与成本见 `docs/PAMF_PHASE3_STAR_CONSENSUS_RESULTS_20261029.md`。

在本轮同F0、同输入的配对P1 energy控制上，MLP的R_U提高34.6667pp（未知净多拒1092条），H提高41.5472pp，A_K仅少1条正确接收（−0.0037pp）；这支持多分支拒识证据比该单标量更有效，但未知绝对拒识仍未达90%。不能以这项源收益替代跨RX或无标签未来注册验证。

一次独立P0/P1审查PASS，33项相关检查/35subtests PASS，单archive传输比较/compile/runtime/原F0实际checkpoint无query合成G10 smoke通过。独立首次观察确认driver2904272 live及CWD/argv；source predictor2904371（pipeline指派GPU2）和CPU scorer2904466在第二次观察前已完整退出，未捕获GPU子进程的独立实时占用，不伪造GPU显存或日志增长采样，不为证明监控而重跑。既有源物理记录/同episode复用不算独立新数据；源未知为历史暴露三设备、同源RX而非未见目标RX。正式与开发target读取0，无DA/发现/注册，A_N/H_OW为N/A，完整Phase3目标仍ACTIVE。

## G10 配对证据与匹配预算实际完成（2026-10-29）

`20261029-phase3-star-paired-evidence-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，实际code `bcc873cd9f9273133d74b5bc591c947b37ce16e1`，预登记 `af0369e6ca8fea4bc60ffa6ca4954acbff2853b0`。原F0 E50/defaulthead冻结，身份固定P1；P0/P1各100D consensus拼200D，两MLP均3233名义参数、相同初始权重，P1-only的后100维在拟合/normalizer/预测前置零。各1000 CPU更新，同轨迹事前保存200/1000快照，总实际2000步，不把快照加成2400步。

唯一主paired1000的宏 A_K/R_U/H为90.3593/43.7143/58.9227%，C_K98.4778%，原F0同输入P1闭集90.9815%，总损0.6222pp；宏AUROC94.2525%、宏OSCR86.6136%。42组known保护过、unknown总体/5源RX/6scene全未90，NO_PROMOTION。同预算paired−P1-only正确接收净+40、未知拒对净−6，A_K+0.1481pp、R_U−0.1905pp、H−0.1411pp；相对energy净正确+58、未知拒对+1054，但不能据此替代未知绝对目标。完整五方法、2×2效应、源分组/逐TX/两轨迹2000loss与聚合状态见 `docs/PAMF_PHASE3_STAR_PAIRED_RESULTS_20261029.md`。

本轮主global τ受RX6/low urban源交叉组的已知保护约束，所有方法下一候选τ均不合法；这是source-V经验保护，排序更高不等于运行点通过。该固定输入扩展和预算比较没有解决未知拒识，不能继续仅凭加特征/训练loss下降宣称改进。

一次独立P0/P1 PASS，30项相关检查/56子测试通过，原F0实际合成星地无query smoke/单archive比较/compile/runtime均VERIFIED。独立观察driver2925797/source2925886 live CWD/argv、CUDA2及GPU UUID4fb...离散388MB；日志11674→21345B，source/score已完整结束，owned/live终态空、foreignstop0。源角色继续L6300/OE6134/V27000/novel3150、同源五RX/三日与历史暴露novel；无新真实信道/G/raw/独立物理记录、无target/DA/发现/注册。A_N/H_OW=N/A，正式RX24-18未访问，完整目标仍ACTIVE。

## G10 预测类别条件风险阈值实际完成（2026-10-29）

`20261029-phase3-star-class-risk-s2026092701-r01` 已VERIFIED/ANALYZED_SOURCE_ONLY，实际code `bb507e2b6f4d66bd808a485d70d64d25c3556da3`、预登记 `a23d46b3e54b35db25f5c90f8e8f5b3458b92929`。冻结父paired1000原始分数/P1身份，只knownV27000以完整tie-prefix Fraction共同尾率校准六个预测类别阈值；查询路由只看模型预测c，不看真实TX/RX/scene。global对照逐值重现父τ和接收集合。无新神经、geometry、normalizer训练、IQ/F/G前向或物理记录；原F0/G和父L6300/OE6134的2000步读出祖先成本保留。

唯一主predicted_class的宏A_K/R_U/H=90.0593/59.6508/71.7668%，C_K97.8407%，原F0同P1闭集90.9815%，损失0.9222pp；margin宏AUROC96.0749%、OSCR88.5425%。42known保护过、unknown整体/5源RX/6scene共12组均未90%，NO_PROMOTION。相对global未知净多拒502、known净少正确接收81，R_U/H分别+15.9365/+12.8441pp、A_K−0.3000pp；三novel拒识净变+199/+342/−39，不能把宏提升说成每个新类都改善或未来注册成功。

最大q=97/4473，下一q=92/4241在源RX6/low urban使原F0损失达2.0316pp而不合法。六阈值仅需48B FP64聚合，无CDF/逐行source bank；它们是同一源V的经验风险校准，无conformal或跨域覆盖保证。详细数据/每预测桶与真实TX区分/42组/边界/配对和成本在 `docs/PAMF_PHASE3_STAR_CLASS_RISK_RESULTS_20261029.md`。

24项相关检查/45子测试及一次独立P0/P1审查PASS，单archive/compile/runtime与真实父readout checkpoint的2条synthetic-feature无query smoke均VERIFIED；该smoke不是新F0/G或星地IQ测试。CPU driver2955649/source2955662独立live CWD/argv、CUDA空，日志485→607B，CPUscore2955731完成，终态live空，GPU新增/foreignstop0。源known/novel同五sourceRX/三日、novel3历史开发暴露，不能替代未见RX target；本轮无target/DA/发现/注册，A_N/H_OW=N/A，正式RX24-18未访问，完整目标ACTIVE。

## 2026-10-30 冻结 F0 局部邻域支持度实际完成

`20261030-phase3-star-local-support-s2026092701-r01` 已 VERIFIED / ANALYZED_SOURCE_ONLY，实际代码 `84cdc012f7d9641b283c407f6048bfe4e6e21cf6`、预登记 `9c248810e16fece047b15bcadca32e15e4c31a90`。固定 k=8，比较 pooled_known、pooled_relative、rx_consensus_relative；唯一主方法为最后一项。原 F0 E50/default6head 完全冻结，已知身份由其原默认头给出；支持库仅使用原 F0 teacher 的 L6300/OE6134 单位 fused160，未使用 OE 真 TX 或 K。查询不按真实 RX、场景或标签路由。

| 固定方法 | A_K% | R_U% | H-score% | C_K% | 原 F0 损失 pp |
| --- | --- | --- | --- | --- | --- |
| pooled_known | 90.625926 | 24.095238 | 38.068883 | 99.159259 | 0.355556 |
| pooled_relative | 90.470370 | 27.015873 | 41.607187 | 98.414815 | 0.511111 |
| **rx_consensus_relative（主）** | **90.607407** | **13.111111** | **22.907458** | **98.925926** | **0.374074** |

三种方法均通过 known42 保护，均未通过 unknown 总体/五源 RX/六景共12组90%要求，主候选 **NO_PROMOTION**。主方法相对 pooled_relative 多正确接收37条、少拒识未知438条，R_U 降13.904762pp、H 降18.699729pp；不能因控制较好而事后换主。主方法三个未知 TX 的拒识为168/1050、25/1050、220/1050，分别16.000000%、2.380952%、20.952381%。该结果未支持按源 RX 距离取中位数的共识机制，不调 k、重新选阈值或扩大种子。

全部真实输入仍为同 G10 MIX01 的既有 P1；L/OE bank 在 V27000 前固定，三分数共18个预测类别阈值由源 V 校准后，才对 novel3150 预测，完整 prediction 后独立 truth-last。fresh 原 F0 与缓存同 P1 闭集均24565/27000（90.981481%），逐条身份变化0，基线未重设。源五 RX/三日和三个历史暴露 novel 仍非新 RX / 新 TX 盲测；target U/Q、正式 RX24-18 读取0。

实际新增 F30152（V27000+novel3150+合成 G10 P0 smoke2），G前向0、NN/optimizer/G更新0。部署 NN 参数 F232993+G266951=499944，无辅助头；bank FP32特征7957760B、FP64工作特征15915520B，另需 RX/IDs/距离工作区。生产 CPU 距离读出30150行39.840640s，独立公开分数复核36.003145s，二者均非 B1 端到端时延。pipeline94.166825s。

20项聚焦测试/46子用例和一次独立P0/P1通过；单归档传输比较/remote compile/runtime/原F0实际checkpoint合成星地smoke均 VERIFIED。独立观察 driver3140372、source3140756 在正确 CWD/argv、CUDA2，实时显存388MB为采样值，日志1716→7337→10481B；评分3141520，终态 live/owned GPU 均空，foreign干预0。详细逐RX/scene/TX、校准边界及配对见 `docs/PAMF_PHASE3_STAR_LOCAL_SUPPORT_RESULTS_20261030.md`。

本轮只完成源端拒识候选证伪，无目标适应、发现或未来匿名注册新结果，A_N/H_OW=N/A；用户 v3 指标与完整 Phase3 目标继续 **ACTIVE**。后续先依据已有实现收窄拒识表征和条件支持的缺口，另定机制与独立 run，不将此轮失败阈值转为目标反馈。

## 2026-10-30 预测身份条件局部支持实际完成

`20261030-phase3-star-conditional-support-s2026092701-r01` 已 VERIFIED / ANALYZED_SOURCE_ONLY，code `0e5cfae40fb7f91825f695fbf4a1ac4fdaa6f0fe`、prereg `1bec291758fb8b0415fa204aa34de76d37fc827f`。在同原 F0/default6head、同 G10 P1 单位 fused160、k8/eps1e-6 下，仅新增 source L 真实旧类分组；query 用原默认头的预测类选对应 L 子bank，OE 仍全bank，身份不改成邻居标签。匹配 pooled_relative 的 raw/六τ/margin/接收集合逐值复现父实验，未重新拟合控制。

| 固定方法 | A_K% | R_U% | H-score% | C_K% | 原 F0 损失 pp |
| --- | --- | --- | --- | --- | --- |
| pooled_relative（原控制） | 90.470370 | 27.015873 | 41.607187 | 98.414815 | 0.511111 |
| **predicted_identity_relative（主）** | **90.485185** | **25.111111** | **39.312393** | **98.381481** | **0.496296** |

两方法 known42 保护通过，unknown 总体/五 RX/六景12组90%均失败，主候选 **NO_PROMOTION**。主相对控制正确接收获得21/丢失17（净+4）、未知拒对获得52/丢失112（净−60）、known接受净−9；A_K +0.014815pp、R_U −1.904762pp、H −2.294793pp。宏 AUROC 91.564951%、OSCR 84.866768%略高于控制的91.231017%/84.367286%，不能替代已冻结运行点的拒识要求。三未知类拒识312/1050、190/1050、289/1050，相对控制分别少22/4/34条；本轮没有支持条件邻域带来运行点收益。

全部42584原物理记录和特征复用：L6300/OE6134 bank、V27000、novel3150，同已见五源RX、2021-03-08/15/23和历史曝光三novel。新增读取L的合法y元数据6300条，只用于source bank划分；不是解封U_s或OE的真TX/K。原F0同P1闭集仍24565/27000=90.981481%，fresh字段为父特征/头预测复用，不是本轮真实IQ新前向。bank在V前冻结，仅主六τ由V校准，控制六τ复用，全部12τ/cal/known写全后才打开novel特征；完整预测后独立truth-last评分。

协议明确部署端不能回读逐样本source特征，因此**完整bank仅为source机制诊断，不能直接带入目标部署**。本轮显式 source_bank_is_deployment_state=false、source_free_export_required=true，eligible_primary/target_eligible在科学PASS情形也保持false；通过只允许另记聚合锚点/参数化的source-free导出研究。这是既有权限边界，不新增未来实现或本轮发布门槛；上轮fullbank结果同样不能当可部署状态。源端负结果未授权任何target访问。

本轮真实IQ/F/G前向0、NN/optimizer/G更新0，额外F2仅actualcheckpoint synthetic G10 P0无query smoke；父F30152及历史P1恢复成本保留祖先。NN参数499944无aux，bankFP32特征7957760B/工作FP64特征15915520B，另需L_y50400B及RX/IDs/索引/workspace。CPU生产score30150行30.578845s、独立公有证据复核31.914745s、pipeline72.418846s，均非B1端到端时延或部署bank收益。

19项集成测试/57子用例PASS6.72s，一次独立P0/P1 PASS（定向7/21与主suite重叠不加计）；单archiveSHA/remotecompile/runtime及首步原F0实际checkpoint smoke VERIFIED。driver3155703/source3156415（正确CWD/argv/CUDA2、350MiB实时采样非峰）、score3156821已独立核实，source日志564→653B，终态live/ownedGPU空、foreign干预0。完整分组/校准边界/成本与数据权限见 `docs/PAMF_PHASE3_STAR_CONDITIONAL_SUPPORT_RESULTS_20261030.md`。

target U/Q、正式RX24-18新读取0，无新目标适应、发现或未来匿名注册，A_N/H_OW=N/A，完整 Phase3 **ACTIVE**。两项固定原F0邻域改动均未达拒识要求，后续不沿此结果扫k或重校准；优先研究可只导出参数/聚合状态的独立拒识表征，并保留未知结构。这是下一待证伪机制，不是已验证收益或对全目标的缩减。

下一机制的具体研究范围（尚未运行）：继续冻结同一原F0及默认身份头，在合法同G10 P1的L/OE特征上单独学习拒识度量；检验旧类跨source RX紧凑、OE与旧类分离，以及OE相互几何关系保持的联合目标。目标端仅允许读取冻结的度量网络参数和类别聚合状态，不读取逐样本source bank；保留原F0映射的匹配控制，V仍只校准阈值、novel只冻结后评分。新网络、固定损失/预算和唯一主分数在独立run预登记后才运行，不按本轮某个novel TX设专属规则。该方向同时服务拒识与后续匿名身份结构，不替代跨未见RX、零标签适应和独立未来注册的最终验证。

## 2026-10-30 独立拒识度量与未知结构保持实际完成

上述待执行机制现已完成：`20261030-phase3-star-metric-embedding-s2026092701-r01`，训练/预测code `efb82a4177e9b59533b7e71408ded6b63fc8733a`、预登记 `d2638af643fac0f788a41b6901c8d85ee3065e4c`。三臂identity（0步）、compact_oe（1000步）、compact_oe_structure（1000步）均保留原F0 E50/default6head与DURA；仅独立20704参数的160→64→160 residual-unit网络g训练。两个学习臂相同CPU初始化和1000/1000逐步相同采样：known跨sourceRX对比+OE排斥，唯一差别为主臂增加OE Gram结构保持。主分数在任何真实训练/novel读取前固定为预测类直接Mahalanobis距离的负值，RMD保留控制。

| 固定臂/分数 | A_K% | R_U% | H-score% | C_K% | 原 F0 损失 pp |
| --- | --- | --- | --- | --- | --- |
| identity / minimum_relative | 90.400000 | 12.317460 | 21.680801 | 98.681481 | 0.581481 |
| identity / predicted_identity_mahalanobis | 90.451852 | 10.539683 | 18.879480 | 98.651852 | 0.529630 |
| compact_oe / minimum_relative | 90.444444 | 18.317460 | 30.464941 | 98.900000 | 0.537037 |
| compact_oe / predicted_identity_mahalanobis | 90.492593 | 14.634921 | 25.195153 | 98.892593 | 0.488889 |
| compact_oe_structure / minimum_relative | 90.433333 | 16.920635 | 28.507366 | 99.048148 | 0.548148 |
| **compact_oe_structure / predicted_identity_mahalanobis（主）** | **90.433333** | **12.476190** | **21.927290** | **98.870370** | **0.548148** |

六运行点均known42保护通过、unknown整体/五源RX/六景12组90%未通过，固定主 **NO_PROMOTION**。主MD相对同预算无结构臂正确接收获得41/丢失57（净−16）、未知拒对获得151/丢失219（净−68），A_K−0.059259pp、R_U−2.158730pp、H−3.267863pp；相对identity净正确−5、未知拒对+61。不能按较高的控制或RMD替换主。原F0同P1闭集90.981481%保持，所有身份仍原默认head，拒识g没有另训/替换身份头。

完整2000步/10份stdout已解析。两训练臂各L120000/OE128000曝光、实际unique6300/6134；全部loss/gradient有限，采样逐步一致。末100步均值：无结构total4.133593、known3.734641、OE排斥0.039895、结构诊断0.276159；主total5.056047、known3.800412、OE排斥0.067109、结构0.058455。结构项降低了该训练诊断，但没有改善本轮拒识。source novel heldRX1NN为identity2482/3150=78.793651%、无结构2474/3150=78.539683%、主2493/3150=79.142857%；它是源设备结构诊断，不是A_N或未来匿名注册成功，也不新增晋级门槛。

数据仍同G10 MIX01 P1的L6300/OE6134/V27000/novel3150、已见sourceRX1/3/4/6/8与2021-03-08/15/23、三个历史曝光novel；非跨未见RX或新TX盲测。L旧TX/RX及OE成员监督用于g，V仅12τ校准，模型/四聚合冻结先V、部署包/τ冻结先novel。原Us/targetUQ/正式RX24-18新读取0，OE真TX/K不进训练，F/head/G更新0。真实新IQ/F/G0，原三臂F6仅合成G10P0 actualcheckpoint烟测；g生产forward合605350（含两个训练臂248000各、三臂L聚合+V/novel），父F30152与历史P1恢复成本保留祖先。

部署包确为四文件：g四FP32 tensor（20704参数/82816B payload）、四FP64类/背景聚合（418560B）、12τ（96B）及元数据；无source逐样本ID/标签/feature bank。F/G499944+g20704=520648神经参数。这证明本轮产出可无源样本读出的参数/聚合状态，不证明拒识有效或目标泛化成功；批量g/聚合score和独立复核耗时均非B1整机时延。

原pipeline因compactOE独立CPU FP32重建7/27000条略超原容差而为 **FAILED_PRESERVED**，模型/三臂prediction完整、identity/main评分已完成。已保留失败日志，只用新 `20261030-phase3-star-metric-embedding-score-recovery-s2026092701-r01` 恢复该control评分，code `4211e8981e85fa26e0ce354267043ecb2116530b`、预登记 `e08de80e520d07fb5fee2156a147ecb652bb46fa`。同一FP32权重无损提升到CPU FP64做独立参考，原unit容差1e-5/2e-6、raw/margin1e-10不变，原模型/预测/τ及两个已完成score不动；全部公有验证后才truth-last。实际重建maxdiff known1.1743978e-6/novel9.5367432e-7，actual g的2 synthetic smoke PASS，恢复新g reference30152、F/G0/GPU0。三臂科学矩阵因该独立恢复而完整，不改写旧pipeline为成功。

原21tests/47subPASS13.07s及一次P0/P1；恢复仅受影响8tests/23subPASS10.50s及一次问题针对性复核（不合计成单suite）。两个code/config提交/独占run、各一次preflight/archive比较/compile和实际PID/CWD/argv/log证据均VERIFIED；原driver3178144、identity3178462/CUDA2与compactOE3178464/CUDA3实时350/358MiB，main3179142/CUDA2采样360MiB均非峰；原81.408632s pipeline仍失败保留，恢复3193705 CPU PID/log0→1647B/终态live0，foreign干预0。完整逐组/配对/训练及恢复证据见 `docs/PAMF_PHASE3_STAR_METRIC_EMBEDDING_RESULTS_20261030.md`。

无新增跨未见RX测试、无标签目标适应或未来匿名注册结果，A_N/H_OW=N/A，完整 Phase3 **ACTIVE**。下一步优先检查训练排斥目标（prototype cosine）与当前Gaussian距离读出的对应关系，在同一固定主g上另记小规模、冻结模型的匹配读出证伪；不得据本轮较好control更换g、调训练预算或改known保护/未知门槛。该后续尚未运行，不是已经确认的收益。

## 固定主度量的余弦读出实际完成（2026-10-30）

Run `20261030-phase3-star-metric-cosine-s2026092701-r01`，VERIFIED / ANALYZED_SOURCE_ONLY。唯一主 maximum_cosine 的 known42 保护通过、unknown 总体/五源 RX/六场景12组90%未通过，NO_PROMOTION。主、控制与旧MD的身份均为同一原F0默认头，旧MD只复用父公开margin及原阈值，不是新部署包支持的第三分数。

| 固定分数 | 宏 A_K | 宏 C_K | 宏 R_U | H-score | 原 F0 closed−A_K |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 MD 外部参考 | 90.433333% | 98.870370% | 12.476190% | 21.927290% | 0.548148 pp |
| predicted_cosine | 90.385185% | 98.259259% | 31.746032% | 46.988330% | 0.596296 pp |
| maximum_cosine | 90.425926% | 98.348148% | 27.365079% | 42.015307% | 0.555556 pp |

主较旧MD正确接收99得/101失（净−2），未知拒对709得/240失（净+469），RU+14.888889pp、H+20.088017pp；改用训练变量的读出有所改善，但远低于90%。不能用较高的 predicted_cosine 控制替换预先固定的主方法。原F0同输入P1闭集90.981481%，本轮主损失0.555556pp，AUROC80.671049%、OSCR74.911718%。

固定 structured main g 的四FP32 tensor与L-only原型祖先不变，无新模型/optimizer更新、不读L/OE逐样本特征、不新增真实IQ/F/G前向。祖先仍L6300/OE6134/V27000/novel3150共42584原物理记录，五已见sourceRX/三日/G10MIX01P1，历史novel三TX各1050。V只校准两法12τ，全部模型/原型/τ/cal/known冻结后才novel，公有CPU64重建后独立单次truthlast。无targetUQ/正式RX24-18访问，无新目标适应/发现/未来注册；AN/HOW=N/A、fullgoalACTIVE。

实际gcheckpoint两syntheticunit smokePASS。生产新g2仅smoke、真实g/F/G0；CPU余弦读出30150行0.112731s，独立CPU64 g参考30150及读出重建1.525449s，均非部署B1端到端时延。已验证父g/Lmeans复制，unit最大差known6.92904e-7/novel5.32717e-7，原rtol1e-5/atol2e-6不变，FP64cosine/margin容差1e-12。部署exact4files，同g20704参数82816B、原型7680B、12τ96B，source逐样本行0，F/G/g神经参数合520648，祖先训练/缓存成本保留。

执行code abb2490048d7fcceffbffe814d15f880b79522f4、预登记b67a2ed4a6e13e383df2e38dbe6b1a012c75b647，11tests35subPASS5.19s/2AST/实际runtime与一次直接P0P1 PASS。一次preflight/archiveSHA22c06cd3cf83129cdefbc3a9720b92974b807e8714b8eafc930ab94ba2fdb476（363files/data0）比较/compile/runtime VERIFIED。首次独立观察pipeline3222125 live且CWD/argv/CUDA空正确；短时predict3222222和score3222229只由pipeline jobs记录，未抓到独立实时占用，不重跑补遥测。终态live空，predict991B/score1247B/pipeline528B完整日志，全部prediction/cal/deploy/score complete。pipeline9.822261s，predict5.178522s、score4.641403s含进程加载/验证，非readout逐包时延。CPU-only新增GPU0、foreign干预0。

完整逐RX/场景/TX、配对、成本与数据权限见 docs/PAMF_PHASE3_STAR_METRIC_COSINE_RESULTS_20261030.md。本轮结束固定g换读出的证伪，不沿historical novel真值扫阈值/选择控制，下一独立方法先检查开放表征训练和无标签适应的实际作用。当前星地跨RX CK/RU≥90%、AK距原同输入F0最多2pp、未来匿名注册目标保持。

## 2026-10-30 默认头开放暴露与相对关系保持已固定待发布

新run `20261030-phase3-star-relation-oe-s2026092701-r01`，执行code `19691a2cad35dcef6dd1ae42973bfba0e14ea705`；两原F0 native/defaulthead独立2220更新，CE/KD+OE关系Gram均有，主relation_oe额外uniform默认头暴露1、control0，无逐向量锚定/aux/增广/U。uniformOE历史已做，本轮只检验关系保持条件下开放暴露的配对作用，不预断旧权重5主导梯度。源knownV只按用户overall+六主scene7组CK90/F0loss2pp选单globalentropyτ；42源组全报、余35仅诊断、不改旧run。实际26tests41subPASS7.57s/10AST/runtime和一次P0P1通过，一次N607preflight源/teacher/G10Q句柄及显存已核实。

两row固定source结束后各一次RX24-1 G10Q864跨RX开发验证，source RU90不再作为这个新机制的entrygate。模型/τ先novel和Q冻结，全部预测完独立score；targetU384保持未读、Q不拟合、正式RX24-18不读、未知TX/K/配额隐藏。当前只DA0_REG0，成功仅下一独立UDA资格，不扩seed或完成目标，AN/HOW=N/A。此前cosine负结果已独立归档Git56df8c3fc8ef0526644c039f7e3a47f66d985e0a；完整v3星地跨RX目标保持ACTIVE。方法/数据/预算/停止见`docs/PAMF_PHASE3_STAR_RELATION_OE_PLAN_20261030.md`及本run登记；此处尚不声称已启动或取得性能。

## 2026-10-30 关系保持开放暴露的跨RX实际结果

Run `20261030-phase3-star-relation-oe-s2026092701-r01`，VERIFIED / ANALYZED_DEVELOPMENT，两臂train2220/sourceV校准/sourceNV预测/RX24-1Q864预测/独立CPU双truth评分完整。唯一主relation_oe::entropy_global为NO_PROMOTION；sourceknown7保护通过，但source与target未知拒识未达90%，target旧类保护和CK90也未达。source RU未通过确实未阻止预登记target验证，不改旧run阈值或将source成功当跨RX成功。

| 臂 / 数据 | 原F0闭集 | candidate闭集 | 宏 A_K | 宏 R_U | H-score | 宏 C_K | 原F0 closed−A_K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relation_control / source | 90.981481% | 91.577778% | 90.103704% | 28.666667% | 43.495239% | 96.251852% | 0.877778 pp |
| relation_control / target | 72.916667% | 72.048611% | 67.013889% | 35.763889% | 46.638044% | 88.368056% | 5.902778 pp |
| relation_oe / source | 90.981481% | 91.544444% | 89.848148% | 57.841270% | 70.376484% | 96.137037% | 1.133333 pp |
| relation_oe / target | 72.916667% | 68.576389% | 55.555556% | 60.416667% | 57.884232% | 76.909722% | 17.361111 pp |

主target known正确320/576、接受443/576、unknown拒对174/288；控制386/576、509/576、103/288。同Q IDs完全对齐，主净正确−66、known接受−66、unknown拒对+71；分别AK−11.458333pp、CK−11.458333pp、RU+24.652778pp，H+11.246187pp，综合分提高不能掩盖known损伤。主candidate闭集395/576比原F0 420/576已少25条（4.340278pp），超过2pp，即便全部接受也不能满足旧身份保护；另外拒掉75条原candidate正确（13.020833pp），总损100条/17.361111pp。不把两种损失归成单纯阈值问题。

源主较控制正确接收346得/415失净−69，unknown拒对1027得/108失净+919；全部IDs对齐。新校准允许总体及六scene7组，42源切片完整诊断、RX/RXscene余35不阻断，无common-tail约束。两个新全局阈值独立由knownV冻结，不读novel/target分布。该作用与此前42组/不同模型实验没有匹配对照，不把跨run变化单独归因于放松校准。

完整4440条步日志/两臂stdout已读取，所有loss/梯度finite、每臂2220个批次逐步完全一致、L/OE各284160重复曝光且unique6300/6134。每臂单Foptimizer2220、Sinc/默认头梯度与参数变化全程记录、持久buffer冻结；无aux/U/EMA/PL/G更新。末222均值control CE0.158770/KD0.046404/Gram0.000553/OE诊断2.386768；主CE0.179459/KD0.108160/Gram0.003672/OE0.077297。OE目标得到优化，但source到target仍有明显拒识与身份差距，不用训练loss宣布泛化。

原F0 E50/default6头独立起点、teacher r02/5aa0411原F0 OE unit、同G10P1与冻结DURA祖先保持；source L6300/OE6134/V27000/NV3150、已见5RX/三日/historicalnovel，target为源外RX24-1/2021-03-15/Q864、历史开发暴露不盲测。targetU384新读0、Q拟合0/未知TXK配额输入0、正式RX24-18读0，AN/HOW=N/A，无新DA/匿名REG，完整v3目标ACTIVE。两臂神经active F+G499944、无aux，900722训练wrapper含667729未用域参数另报，单globalτ8B、不部署逐样本sourcebank。

实际F输入总1206708（训练1136640真实、两臂evalcandidate62028真实+8合成、原F0参考8028真实、train4合成）；G1732=target1728真实+4合成。42584source+864target=43448既有物理记录，不将IID重复或同received重用当更多独立样本；源Gcache/teacher导出/Phase1训练成本保留。进程批量时间与GPU采样不代表B1端到端时延。

code19691a2cad35dcef6dd1ae42973bfba0e14ea705/prereg198e3f7795b26a1331ad7cd6dfdc73c6f44eb1b6，26tests41subPASS7.57s/10AST/runtime/一次独立直接P0P1，单preflight/archive03a4503de06209a7188447c9d75ef54949ed571bd1a2360f5644e53844f2b056（375files/data0）比较/compile、actualcheckpoint syntheticG10梯度烟测及targetF/G smoke均PASS。首次release helper在本地package.json尚未生成时退出，发生于任何SCP之前；已等待同一打包进程完成，只传输同一个归档，无重打包/远端重复release或实验重启。

driver3240533、row3240623/24及train3240632/CUDA4、3240631/CUDA2独立live正确CWD/argv已核实；GPU各706MiB为离散样本非峰值，train日志control759→1551→2326B/main670→1333→2007→2184B增长，eval/score短进程未独立捕获live，仅jobs/stdout，未为补遥测重跑。终态pipeline ANALYZED_DEVELOPMENT/live/ownedGPU空、所有8个complete齐、foreign干预0；pipeline248.125857s含并行/加载/预测/评分，不宣称因果加速。逐TX/RX/scene/CI、完整训练、source1NN诊断和成本见docs/PAMF_PHASE3_STAR_RELATION_OE_RESULTS_20261030.md。

本轮结束，保留失败科学结果，不换控制、不调targetτ/训练、不扩正式或seed。下一科学问题是源端开放暴露造成的跨RX身份损伤及目标域拒识错配，需要目标U的受约束适应和独立未来Q证伪；当前没有据此宣称已获得有效UDA或无标签注册。

覆盖补充：本小规模targetQ的practical_high中，scorer未知类6计数0、类7为4、类8为2，必需未知覆盖不完整，该景R_U/H为N/A，不能认证通过；总体三未知各96条覆盖完整。保留原固定观测和全部预测，不据缺项重抽信道或补挑样本。

## 2026-10-30 U互近邻适应与两阶段注册已固定待发布

新 `20261030-phase3-star-reciprocal-adapter-s2026092701-r01`，code `d3d8f5680fd72dc7ded272ffd1ded3143e490244`。回到原F0/default6head与G冻结，source聚合/pairedMLP/6τ直接复用，不继续训练源分类器。仅合法无标签U384学10432参数g200步（空mask0且精确旁路），控制selfPL、主reciprocal trusted MNN扩pseudo+allU Gram保结构；oldgate之后才匿名ball，图连通分量自主K/半分稳定/min16/boundary≤.05。两臂四态、所有U状态先Q冻结、Q864逐批只推理，独立score先U四映射再Qtruth；目标标签/K/配额0、source逐行读取0、formal0。最新v3绝对目标保持，DA0弱基线只是对照不作额外不可能entry gate，fullgoalACTIVE。

集成48tests39subPASS11.10s、producer4定向PASS5.15s（含1新增，非重复整组）、10AST/实际runtime与一次独立P0P1通过。一次N607preflight源/G10句柄和实际显存已核对，未读UQ/IQ/truth值；后续单GPU发布、自动完整预测和独立评分。当前尚未启动或取得新性能，详见`docs/PAMF_PHASE3_STAR_RECIPROCAL_ADAPTER_PLAN_20261030.md`和本run登记；此前relationOE负结果不改标、不承接其失败增量。

## 2026-10-30 U互近邻四态与匿名注册实际结果

Run `20261030-phase3-star-reciprocal-adapter-s2026092701-r01`，VERIFIED / ANALYZED_DEVELOPMENT。两臂×四态完整预测和独立评分，唯一主reciprocal::DA1_REG1仍NO_PROMOTION。目标端trueTX/K/配额输入0、source逐行feature/IQ新读取0、F0/G/defaulthead/readout/6τ更新0；各10432参数g只由U384学习200步，所有U状态先Q864冻结。四个scorer匿名map全部用Utruth固定后才读Qtruth，模型未接触映射或truth。完整v3目标ACTIVE/finalfalse，正式RX24-18未访问。

| 臂/状态 | 旧类宏 A_O（A_K） | 旧类 C_O | 当前 R_U | H-score | 未来 A_N | H_OW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| self_pl/DA0_REG0 | 65.972222% | 88.368056% | 76.388889% | 70.799458% | N/A | N/A |
| self_pl/DA1_REG0 | 69.618056% | 93.402778% | 60.763889% | 64.890331% | N/A | N/A |
| self_pl/DA0_REG1 | 65.972222% | 91.493056% | 65.625000% | 65.798153% | 4.687500% | 12.308736% |
| self_pl/DA1_REG1 | 69.618056% | 96.006944% | 25.000000% | 36.788991% | 13.541667% | 23.399066% |
| reciprocal/DA0_REG0 | 65.972222% | 88.368056% | 76.388889% | 70.799458% | N/A | N/A |
| reciprocal/DA1_REG0 | 68.923611% | 93.229167% | 60.763889% | 64.587052% | N/A | N/A |
| reciprocal/DA0_REG1 | 65.972222% | 91.493056% | 65.625000% | 65.798153% | 4.687500% | 12.308736% |
| reciprocal/DA1_REG1 | 68.923611% | 93.229167% | 27.083333% | 38.886377% | 11.458333% | 21.628696% |

REG0 R_U分母为全部三未知6/7/8；REG1主R_U为U未出现的plannedlater8，不能直接以两态H/RU差表示注册影响。A_N分母始终两个真实U到达类，漏发现记0；REG0 AN/HOW N/A。REG1另报实际未注册集合和intended旧六+两到达类接收率，不能把old C_O93.23%当全应注册类C_K已达标：主intended C=78.385417%仍低于90%。

主DA1REG0较同Q DA0REG0：known正确380→397净+17，A_O+2.951389pp；接受509→537净+28，C_O+4.861111pp；unknown拒对220→175净−45，R_U−15.625pp。原F0同P1闭集420/576=72.916667%，主开放397/576=68.923611%，仍损3.993056pp>2。适应有旧类开放收益，但不满足未知保护或完整联合条件；控制DA1旧正确401/576，比主多4，不能替换固定主。

主DA1只有一个35条匿名cluster，Utruth显示全为到达类7、old污染0；另一到达类6未发现。独立futureQ类7只22/96正确、类6为0/96，所以AN22/192=11.458333%，不能报U纯度100%为注册成功。原DA0匿名cluster16条含14old+2new6；控制DA1 K3包含34纯new7、16纯old3及原16混合cluster，weightedpurity93.939394%同样不代表未来AN13.541667%达标。

两阶段oldgate已在实际Q实现同DA REG旧A_O损失精确0；主later拒对在DA1REG前后均26/96=27.083333%，本轮REG没有再损伤该量。later的大幅损失发生在适应：DA0 REG0为64/96=66.666667%，DA1为26/96。注册仍漏设备、未来识别率低，不能仅凭oldretain通过宣布开放系统有效。

trusted136条中128old+8unknown，old伪标签正确103/128=80.468750%；主扩15条均old，13正确2错误，且0条改变原argmax。主selected151中143old+8unknown、old标签正确116/143=81.118881%。这是全部预测冻结后的Utruth诊断，未回算法。扩展虽增加较干净的known样本，却没有纠正trusted错误，当前训练不证明真实身份学习改善。

完整400条训练步和全部3stdout已读取，每臂200fullU384=76800曝光，共153600；伪标签CE曝光control27200/main30200。全部loss/gradientfinite、损失分解一致，真实parameters_changed8832/10432，各max preNorm residual .1000000164为FP32舍入量，.1常量未改。control首末total .0153068006→.0008124282，main .0246448591→.0011712685；F0/head/G参数与buffer和sourcegeometry/readout/τ不变，训练loss下降不能替代joint目标。

source祖先仍原三视图SSL E50/11100/392005 L6300/Us56700/V27000及sourceOE成员监督。当前实际新IQ只有G10旧U384/Mar8和futureQ864/Mar15，同RX24-1、source五RX互斥、P0received/P1冻结DURA、同MIX01/receiver2027/scene2026103001/channel2026104001，无clean新路线/重抽物理/信道，不把1248既有记录当独立新数据或新盲测。targetQ高郊区缺某必需unknown类，按原覆盖规则N/A且不认证，不重抽样凑指标。

### 完成后的身份与路由分解（post-hoc，非预登记候选）

在全部预测/主评分固定后，另一个独立CPU同row分析只重算固定数组：实际m1+原τ[currentclass]还原readout raw，再用原F0预测类的原τ索引；也检查保持m1但只恢复原F0身份。无新模型/阈值拟合、NNforward、Utruth读取或匿名重匹配，主predictions/verdict完全不动。该诊断不作新盲测、晋级或重判本主结果。

主candidate闭集413/576=71.701389%，较原F0净−7（3得10失）；开放误拒再损16/576=2.777778pp，总3.993056=闭集1.215278+正确误拒2.777778pp。只恢复原F0身份、保留当前gate，AK403/576=69.965278%、C93.229167%、RU60.763889%、H65.040763%，原F0损2.951389pp仍失败。再恢复原class路由为AK69.791667/C93.055556/RU60.763889/H64.965647%，仍失败。控制相应samegateAK70.138889%、origrouteAK69.791667/RU60.416667，也未过。

按同一顺序，原身份下gate改变带来known正确380→403净+23，再更换适应后身份403→397净−6，最终+17；不是闭集身份性能改善。条件τ路由变化也不能单独修复unknownR下降，不能据诊断直接换成更高数字版本。该证据将下一科学问题从单纯head/routing错误收束到target域knownness漂移及对未知设备的保护。

### 成本、发布与可观测证据

F0 full forward2500=2496real+4synthetic，G1250=1248real+2synthetic；g训练153600+终态诊断768+UQ推理2496=156864行，额外原head特征前向153600+2496+2=156098。source四个3233快照都存/每evidence都算四个，real14976MLP行、smoke8，sourceL/OE/旧Gcache/Phase1祖先成本不抹去。神经实际同时双g533740、单g523308；geometry/90聚合/原型/radii/部署封装另计，无source逐行bank、无B1时延证据。

code d3d8f5680fd72dc7ded272ffd1ded3143e490244/prereg544a313006685fa0f45f82a8a06e44923af30b4c；48tests39subPASS11.10s，后producer4定向PASS5.15s含1新增（非另一完整suite），10AST/runtime与唯一独立直接P0P1 PASS。一次preflight/archiveSHA1638a485996cc10480a733f3a4bdf7f008e6770fd166b955dba1b7d23a7e4d08（387files/data0）比较/compile VERIFIED。实际F0/G syntheticG10 smoke先U PASS，head inputgrad4.5401769、head logits roundtrip差0，realrows0读/updates0。

仅driver3267295独立live抓到正确CWD/argv/CUDA空；predict3267384由pipeline分配GPU6、score3267483 CPU，仅jobs/stdout，未抓子进程瞬时独立GPU显存或live log增长，不为遥测重跑。终态live/ownedGPU空、foreign干预0、predict/score/driver日志1052/1574/21B（driver0→21已观察）；三个complete齐。pipeline23.247853s、predictjob18.405200s/score4.763123s，predict内部13.545977s，含加载/数学检查、非B1或因果并行提速。

完整逐类/场景/CI、400步训练窗口、K与污染、配对计数和posthoc表见docs/PAMF_PHASE3_STAR_RECIPROCAL_ADAPTER_RESULTS_20261030.md。当前不扩正式/seed、不更换控制、不调整旧阈值；后续独立机制需同时解决knownness的未知保护和新设备跨日泛化，不能把本轮oldgate结构改进或U纯cluster缩减成整个Phase3完成。

## 2026-10-30 冻结身份的二元图适应已固定待发布

新`20261030-phase3-star-binary-graph-s2026092701-r01`，code`bb9f5823f53bdbba6dd3e65f9f58dfe70bf86cdb`，独立于失败reciprocal；原F0/G/defaulthead/sourceprior完全冻结，仅U384一次FP64二元graph solve+匿名registry，合法U memory部署明确；futureQ864逐行只推理，四态全固定后独立score先U两maps后Qtruth。26 passed, 632 warnings, 24 subtests passed in 7.51s、8AST/实际runtime和唯一freshP0P1 PASS，单N607preflight PASS，当前尚未新启动/取得性能。详见本run与`docs/PAMF_PHASE3_STAR_BINARY_GRAPH_PLAN_20261030.md`；v3完整门槛不减、正式RX未读、goalACTIVE。

## 2026-10-30 二元图与独立评分恢复实际结果

原producer code`bb9f5823f53bdbba6dd3e65f9f58dfe70bf86cdb`四态预测、U图/registry、真实F0/G合成smoke完整；其pipeline仍FAILED_PRESERVED。首个独立CPU在任何truth前因FP64 exactgate误拒，unit/ID/sigma逐值同，fmaxabs7.77e-16/residualdelta1.776e-15/jointmax4.44e-16、全部预测/closed/接受符号改变0。一次定向修复abs1e-12/rtol0只用于浮点重建，原数值/模型/τ/数据/四态pred全部保持。

新独立CPU恢复`20261030-phase3-star-binary-graph-score-recovery-s2026092701-r01` code`ca05fae1fbdd9b0174e8fb0804ffc2645c5516c5`已VERIFIED/ANALYZED_DEVELOPMENT；公有gate→U两个匿名映射全冻→Qtruth，唯一主DA1_REG1=NO_PROMOTION。scores保留原producer身份，recovery另记执行版本；不把原failedpipeline改称成功。source新IQ/逐行feature0、目标标签/K/配额输入0、Qfit0、formal0、新NN参数/更新0。

| 状态 | 宏A_K | 旧类C_K | 当前R_U | H-score | 未来A_N | H_OW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DA0_REG0 | 65.972222% | 88.368056% | 76.388889% | 70.799458% | N/A | N/A |
| DA1_REG0 | 68.229167% | 90.798611% | 57.291667% | 62.283887% | N/A | N/A |
| DA0_REG1 | 65.972222% | 91.493056% | 65.625000% | 65.798153% | 4.687500% | 12.308736% |
| DA1_REG1 | 68.229167% | 93.750000% | 15.625000% | 25.427019% | 19.270833% | 22.979986% |

REG0 R_U为所有未来非旧6/7/8，REG1主R_U为U未出现later8，不能直接以两态H/RU差宣称注册影响。主intended旧六+全部两到达类接收82.031250%<90，不能以old-only C_K93.75%冒充全注册接收达标。原F0同P1闭集420/576=72.916667%，两DA闭集逐值保持；主开放393/576=68.229167%，纯误拒损27/576=4.6875pp>2，无身份漂移。

DA1REG0相对DA0REG0：旧正确380→393净+13（21得8失），接受509→523净+14（25得11失），未知拒对220→165净−55（2得57失）。旧C_K总体90.798611%通过单项，但R_U57.291667%和原F0保护均失败。later拒对64/96→16/96，REG后15/96；同DA REG旧正确保留393→393损0，但旧接收523→540多17不代表身份改善。未来新类主仅37/192=19.270833%，远低80；U簇数DA0=1/DA1=2由算法自主而非真K输入。

二元U种子正136=128old+8unknown，成员精度94.117647%；负27=17old+10unknown，成员精度仅37.037037%，其中旧class4有12。这些是全部预测/独立评分后truth-side诊断，未回算法。低MSP并不能当可靠unknown标签，冻结identity和精确求解本身不足以保护未知。U图1197边，energy128.742233776→37.923038253、残差1.24345e-14、f在[-1,1]，一次solve数值成功不等于科学成功。

数据沿原source L6300/Us56700/V27000/OE6134监督祖先、五sourceRX与目标RX互斥；本轮当前U384Mar8/Q864Mar15均RX24-1、物理互斥、原G10 MIX01与同一固定received。所有postPhase1新IQ均星地，P0received/P1冻结DURA，旧source原权重祖先保留，不读source逐行bank。已暴露devRX/设备/Q，不是新盲测/未见日期或正式RX。high suburban缺必需unknown6，N/A/覆盖false，不重抽样凑指标。

实际F0 full2500=2496real+4smoke、G1250=1248real+2smoke、额外原head smoke2；四个3233读出每次都存/算，real4992/synthetic8MLP行，神经驻留512876无新adapter。合法U memory数值251904B+IDs59904B=311808B，npz312784B/json488B；新CPU恢复仅公有图重构验证和独立评分，无部署新拟合/原预测变动。源训练/恢复器/聚合缓存祖先成本保留。无B1/吞吐/峰值显存或提速证据。

完整4stdout已读取：原predict931B/driver17B、旧score2804B包含已记录技术traceback，恢复score837B无异常。原predictjob9.218162537s/internal5.381406529s、原失败pipeline12.263982534s，CPU恢复score0.501980066s，不能拼成连续端到端或B1。原driver3286708独立live CWD/argv核实，predict3286801/GPU4与旧score3286873只jobs/log支持，未抓liveGPU；恢复CPU3410600独立live正确CWD/argv/CUDAempty，终态live/GPU空、foreign干预0。

初始26tests/24subPASS7.51s和一次freshP0P1；直接数值修复3受影响PASS5.28s/真实公有gatePASS、CPU恢复执行3PASS4.36s与唯一问题定点复审PASS，不累计成同一次32test全suite。原397file/data0 archive d0244e8fdab1199d1fe4da1193108e731e861aaf3e099d884449c2798d72cc1c；恢复399file/data0 archive8d1417794f1a48914a2faf147b52435c7a6682b329287713dca6d24e1963798e，各一次SHA/compile/runtime核实，无重复训练/预测或新数据。

详细四态/24scene/主逐TX与CI、U组件纯度/匿名映射、数据权限、全部成本和失败恢复证据见`docs/PAMF_PHASE3_STAR_BINARY_GRAPH_RESULTS_20261030.md`。本轮不晋级、调τ/seed/K/复制控制，不扩大正式/多seed；全Phase3 goalACTIVE/finalfalse。下一独立科学问题是可靠unknown种子及跨日类别覆盖，不能把低置信度旧样本直接定义未知。


## 2026-10-30 PAMF-OW 动态注册集合已开始源训练

Run `20261030-phase3-star-open-episodic-s2026092701-r02`，code `1029bc4f4ab86022c75ae0f3332b5dfafc1fe909`。已实现同次t2/PA/SIG开放64、有界残差、共享关系及三臂源集合训练，完整原生Sinc/默认head联合更新。S0 detach、S1 full6、M rotate5，主M固定；2220步/每arm，原F0 E50/11100独立起点。真实checkpoint无query梯度烟测/直接P0P1与一次兼容修复定向复核通过，N607正式source已运行，S0/S1首步和增长实证齐，M排队。r01仅NumPy2/Torch2.1 metadata转换失败、0update/0target/0truth，产物保留，恢复不改科学设置。完整矩阵/数据/指标/停止见`docs/PAMF_PHASE3_STAR_OPEN_EPISODIC_PLAN_20261030.md`及本run登记。当前无新性能结论，自动完成每row预测与独立评分，targetU/正式RX24-18未访问；UDA/匿名注册和targets v3目标保持ACTIVE。


## 2026-10-30 PAMF-OW 三行源集合学习实际结果

VERIFIED 3x2220 source updates/all source+crossRX predictions/independent scores; primary M AK=60.590278%, CK=80.902778%, RU=64.583333%, H=62.523116%, originalF0loss=12.326389 pp; NO_PROMOTION; UDA/REG incomplete, fullgoalACTIVE.

源L6300/OE6134/V27000/NV3150/跨RX24-1 futureQ864，全部G10P1；原F0同输入基线与既有checkpoint祖先不变。M固定不重排；完整训练/参数/成本/逐scene/TX/缺覆盖及失败r01保留见`docs/PAMF_PHASE3_STAR_OPEN_EPISODIC_RESULTS_20261030.md`。


## 2026-10-30 冻结未知身份结构与融合边界诊断

Run `20261030-phase3-star-open-feature-diagnostic-s2026092701-r01`，VERIFIED frozen source-novel3150/sevenrepr/all4NNchoices before sole source truth. F0 fused crossRXday76.603175%, M fused76.825397%, M open76.984127%, S1 open77.587302%. M open RXvariance16.286785% vs F0 2.887514%, not wholesale collapse/no causalproof. Target/G/updates0, AN/REG N/A, fullgoalACTIVE.

实际源码reference仅最终加性贡献，未条件校正t2/PA/SIG；open64虽然保留GN/PA-LN/SIG-LN，未完整继承时频/reference/PA增益组合。不能把低TX占比当因果或把未启用canonical模块算当前能力。下一独立科学候选检查原融合证据及明确参考条件化，保持旧负路线/主M/数据权限，不按诊断换控制。详见结果报告。


## 2026-10-30 完整贡献与参考条件残差已经实施

Run `20261030-phase3-star-reference-open-s2026092701-r01`，code13d063b98699b1aab9e8e05f3583a65299104515，已实现同次完整四贡献640与小容量reference-conditioned residual；C0constant/C1real maskedr29，同结构/参数/采样/2220预算，soleprimaryC1。两行均从原TriSIGTIME SSL F0E50/11100独立开始，原defaulthead、NoExplicitCFO27、M五类任务/同P1 KD/fullL/共享梯度保护保留；nativeF0不变，旧M未继续。actualcheckpoint smoke/首步训练与PID/CWD/argv/GPU/loggrowth实证齐。详见`docs/PAMF_PHASE3_REFERENCE_OPEN_DESIGN_20261030.md`和本run预登记。两source训练进行中，自动冻结Lproto/Vcal/NV+crossRXQ864/独立CPUscore；U/formal0读，完整v3/无标签适应/匿名注册目标ACTIVE，当前无新科学成功声明。


## 2026-10-30 完整贡献与参考条件残差实际结果

VERIFIED C0/C1 2x2220 source updates, all source/crossRX-G10P1 predictions and independent truth-last scores; primary C1 AK=61.4583%, CK=84.2014%, RU=51.0417%, H=55.7677%, F0closed-loss=11.4583 个百分点; NO_PROMOTION; DA/REG not performed, fullgoal ACTIVE.

详见 `docs/PAMF_PHASE3_REFERENCE_OPEN_RESULTS_20261030.md`。


## 2026-10-30 源端匿名跨RX正对实际结果

VERIFIED source L6300/OE6134 G10 P0export/P1reuse; P1 candidate1358/1674 sameTX, PAIR22/25, no feasible L98%/25group rule; all final edges0; NO_RELIABLE_SOURCE_NEIGHBORS, no training/target/truth/G updates, fullgoalACTIVE.

详见 `docs/PAMF_PHASE3_SOURCE_NEIGHBOR_RESULTS_20261030.md`。


## 2026-10-30 同记录星地视图结构学习已实施

VERIFIED RUNNING: actual originalF0/noquery native+open+physical smoke PASS, both first updates/loggrowth, independent live trainPID/CWD/argv/GPU confirmed; same-record P0P1 source only, foreign0.

Run `20261030-phase3-star-physical-open-s2026092701-r01`，code31fa172946aeaecd2d3f1e784aee7366b7b5bee1；P_L与主P_LOE各2220，原F0独立起点、C0常量结构315106不变；原fullL/KD/M成员保留，新增known跨RXday对比与L同recordP0P1 raw64正则，两组仅OE正则0/.1有别，1280Frows/step。真checkpoint烟测和livePID/GPU/log已核实，自动fullprediction/truthlast。详见`docs/PAMF_PHASE3_PHYSICAL_OPEN_PLAN_20261030.md`及run登记。目标适应/匿名注册仍未完成，当前无新科学成功声明。


## 2026-10-30 同记录星地视图学习实际结果

VERIFIED 2x2220/all source+crossRXG10P1 predictions/independent truth-last; fixedprimaryP_LOE AK=69.2708%, CK=95.8333%, RU=20.1389%, H=31.2055%, originalF0loss=3.6458pp; NO_PROMOTION; sourceonly increment, DA/REG incomplete/fullgoalACTIVE.

详见 `docs/PAMF_PHASE3_PHYSICAL_OPEN_RESULTS_20261030.md`。


## 2026-10-30 物理配对后身份结构诊断完成

Run `20261030-phase3-star-physical-geometry-s2026092701-r01`，code `8aeac9583dc6becf4bcfd7ccebea06a1ee6176a8`，VERIFIED／ANALYZED_SOURCE_DIAGNOSTIC。source novel3150 固定 G10P1、五表征／20邻居选择后独立truth，新增6300真实F＋4合成、F0特征3150复用、G/更新/target0。F0 fused同时跨RX/day1NN76.603175%；P_L fused76.158730/open68.634921%；主P_LOE fused75.523810/open67.301587%。主open64 RX方差比.178240/TX.085569，对比F0.028875/.421118；非零秩但身份结构变弱，单纯增秩／同记录一致性不够，不能把问题只归为阈值。DIAGNOSTIC_ONLY不晋级；没有新AK/RU或未来AN成绩，原v3/无标签适应/发现/匿名注册目标保持ACTIVE。详见`docs/PAMF_PHASE3_PHYSICAL_GEOMETRY_RESULTS_20261030.md`。


## 2026-10-30 身份几何保护型再优化已设计并实现组件

当前跨RX星地physical-open主版本未达标，最新source3150诊断显示open64同时跨RX/day身份检索67.301587%低于原fused16076.603175%，且RX组间方差比例更高。下一轮固定为protected160：原Tri-SIG-TIME SSL F0/defaulthead完全冻结，身份仍输出原同P1 logits；以同pass四项贡献生成零初始化、相对L2幅度≤0.1的160维残差，最终拒识/后续发现用unit160；relation输入差/积320维。PAMF当前方法与原phase1/2模型不改名或替换。

组件代码 `62efcb32292710911d92549cf5274fad10768d5f` 已实现model/loss和合成集成验证；新增模块构造参数66177，与原F0的active299170待实际checkpoint首步smoke确认。model7＋18subtests PASS13.34s；loss8 PASS4.63s、dtype相关1定点复核PASS3.80s（不增加独立覆盖）；集成R0/R1两项PASS7.12s。只用合成接口/缓存tensor，尚无实际checkpoint smoke、新训练、目标读取或性能提升结果。

下一执行目标：完成source-only cache/train/predict/score launcher及聚焦协议负测，从合法原F0独立建立R0残差关闭、唯一主R1有界残差两组；同初始化/采样/2220步，冻结原身份、学习relation和主adapter，源L可见六类、OE只outside-six成员、无OE TX/K/Us/V/NV/target优化。L192/OE64软邻域T.1、权重.1，跨RX且跨day公共候选格等权；2/3/4/5旧注册子集循环，完整六类OE成员loss。具体固定定义见`configs/pamf_phase3_protected_geometry_20261030.json`和`docs/PAMF_PHASE3_PROTECTED_GEOMETRY_DESIGN_20261030.md`。

先执行两组完整源冻结预测及独立truthlast，V只既定风险校准、NV只独立评分；主R1必须通过既定known风险组与source RU90才另预登记跨RX星地开发评估，不从控制换主、不按NV调权/阈值、不访问历史Q训练或扩正式RX/多seed。失败保留科学负结果并继续研究，不以组件测试或检索率宣称全Phase3达标。v3 CK90/RU90、AK距原同输入闭集≤2pp，以及无标签适应、发现和未来匿名注册目标保持ACTIVE。


## 2026-10-30 protected160 源侧三组已实际运行

VERIFIED RUNNING at stored live snapshot: cache actualF0 noquery smoke/pass+sourceP1 features complete; three exact trainPID/CWD/argv/CUDA/GPU and actual positive step/loggrowth. NativeF0 frozen, sourceonly, foreign0; no newperformance result.

Run `20261030-phase3-star-protected-open-s2026092701-r01`，code7dd90671968ddf58aab5a5b302b4d56371c8b9c9。N0无OE、R0固定原unit160、主R1有界unit160，各2220 adapter步，原F0仍11100/native0；共享cache真实12434F＋2合成、三actual训练PID/GPU/梯度烟测及step已验证。逐行自动V风险校准/NV预测/先几何邻居再唯一source truth，只有主R1 source42known+RU90可进入下一次跨RX预登记，source不是target成功。没有新target Q/DA/REG/AN成绩，目标ACTIVE。


## 2026-10-30 protected160 三组源实验完成，不晋级

Run `20261030-phase3-star-protected-open-s2026092701-r01`、code7dd90671968ddf58aab5a5b302b4d56371c8b9c9：N0/R0/主R1各2220、总6660新parts步骤，原native0；全部训练/预测/独立CPU评分完成，完整trace与14stdout188行/全部随机索引重放核对。sourceL6300/OE6134/V27000/NV3150固定G10P1，原F0闭集90.981481%，主R1 AK90.596296/CK98.977778/RU9.269841/H16.818780%，源42known过、unknown失败、NO_PROMOTION，无目标新访问或多seed扩展。

源novel内部同时跨RX/day1NN由F0 76.603175→R1 76.571429%，TX/RX方差比近似保持；关系argmaxknownV90.744444%，不能归为未学会旧身份匹配。正确known条件AUROC .852538351低于AK损伤2pp+RU90必要界.880215754（RU95界.929116629），仅排除当前冻结margin统一阈值平移，不能推广所有新类条件规则不可能；未扫描/选择新阈值。必要性证明与实际秩、OSCR/B一致，sourceonly非target成功。

下一安全动作固定为原F0与主R1的源侧联合邻域诊断：利用已有sourceL6300特征缓存及当前novel3150同次特征，先固定无TX/K参与的旧known与novel联合跨RX/day近邻，再独立只读评分；不将novel或历史query转训练，不以novel内部几何代替known/unknown分离。先区分未知内部可分却贴旧类和分数未利用邻域，查重既有local-support/background研究后才提出新学习机制，避免重复试分数或预先认定pair监督缺失。任何新候选仍独立预登记、唯一主模型、源校准、完整预测/独立truthlast。全跨RX星地/零目标标签适应/发现/未来匿名注册目标保持ACTIVE，AN/HOW均N/A。

完整结果见`docs/PAMF_PHASE3_PROTECTED_OPEN_RESULTS_20261030.md`；真实F102884＋合成14、G0、训练adapter曝光1562880，单seed小矩阵未达标不伪装完整Phase3。


## 2026-10-30 旧六类与未知三类联合诊断完成

Run `20261030-phase3-star-joint-geometry-s2026092701-r01`、code3023c26b4bb2b2f74148303362bd615d854d18ff，VERIFIED/ANALYZED_SOURCE_DIAGNOSTIC。L6300+NV3150同G10P1，主R1同时跨RX/day未知同TX2040/3150=64.761905%、落旧类629/3150=19.968254%；仅NV参考同TX76.571429%，加旧类净失372条。原F0未知同TX64.285714%、落旧类20.476190%，R1未知结构只小幅改善；不能以近邻率替代拒识RU、未来AN或未见RX验收。全部16条件/混淆/分组/完整日志分析，实际checkpoint parts-only smoke2，L adapter6300、NV逐值复用3150、新F/G/更新/target均0。

下一候选主线：冻结原F0默认身份，拟合源L类条件多中心支持分布与只含outside-six监督的OE背景密度比，输出聚合中心/二阶统计/权重，不部署逐行sourcebank；匹配known-only分布控制，唯一主densityratio，区别于已失败的8NN/条件8NN、max关系或二元图。具体容量/拟合/风险公式在独立预登记固定，不由novel真值选；V只固定风险校准，先单seed源反证、合法预测全冻结后独立评分，known保护+RU90通过才另启动跨RX星地。该新候选尚未实现/运行，不宣称收益。完整零目标TX标签适应、未知发现及未来匿名注册继续，v3 CK90/RU90、AK损≤2pp、AN80/REG旧损≤2pp保持ACTIVE；详见`docs/PAMF_PHASE3_JOINT_GEOMETRY_RESULTS_20261030.md`。


## 2026-10-30 多中心密度源实验完成，不晋级

Run `20261030-phase3-star-density-s2026092701-r01`、codeb8163a03e8d3d299127c7914272a2d866f92c89a，VERIFIED/ANALYZED_SOURCE_ONLY、NO_PROMOTION。原F0/defaulthead冻结，L6300拟合90域Gaussian modes/共享收缩白化，主OE6134无TX/K拟合32背景mode/固定25Lloyd；V27000只known42校准、NV3150全两policy预测后独立truth。主AK90.374074/CK98.325926/RU28.095238/H42.864791%，原F0损失0.607407pp，AUROC91.705905%；控制AK90.285185/CK98.433333/RU21.015873/H34.095309%。背景净增未知拒对223、正确known净增24，两组known42过，未知失败。sourceRX已见/novel历史曝光，target0，真实F/G/NN更新0、合成F2；完整25轨迹/42组风险/所有分组/日志和独立数组读回齐全。

下一候选增加新的表征信息：保留原fused160/defaulthead身份，检验原F0融合前四贡献的逐分支单位化拼接，和当前fused160密度作同族统计对照；不根据novel真值挑分支/权重或调本轮τ。先在实际冻结构造器验证合法feature导出/维度与成本，再独立固定单seed源矩阵/执行；当前尚未实现或证明收益。源通过才另登记跨未见RX星地，原U/Q隔离、无标签适应/发现/未来匿名注册及完整目标仍ACTIVE，AN/HOW未执行。完整数据/方法/分组/成本见`docs/PAMF_PHASE3_DENSITY_RESULTS_20261030.md`。


## 2026-10-30 四贡献640密度完成，不晋级

Run `20261030-phase3-star-contribution-density-s2026092701-r01`、codeedc28ad00baace5e4b9567fbf3391154bf6c34b4，VERIFIED/ANALYZED_SOURCE_ONLY、NO_PROMOTION。主640 AK90.440741/CK98.470370/RU16.158730/H27.418664%，原F0损失0.540741pp；固定160控制原样复用AK90.374074/CK98.325926/RU28.095238/H42.864791%。主已知正确净+18，未知拒对净−376；19-16收益被另两TX退步抵消，不能按TX或分支择优。sourceL6300/OE6134/V27000/NV3150同G10P1，sourceRX已见、novel历史曝光；真实F30150＋合成2、G/NN更新/target0，全部25轨迹/42风险/分组/日志/独立control数组读回齐全。640与160不仅信息不同，统计维度/方差收缩/资源也不同，不归因为某单分支。

下一步单独学习开放embedding：原F0/defaulthead身份冻结，open以原unit160的零残差起点、仍输出160维；源L跨RX/day度量监督，主另加OE原F0软邻域保持，不读OE TX/K/硬伪边，不再用max成员关系头。拟两组同结构/初始化/固定预算的L-only embedding控制和L＋OE软结构主；两者后续统计背景均含OE，不能称整体无OE控制。新网络/损失/预算/随机流须独立最小预登记，尚未实现或验证；不以当前novel标签调参。通过源known保护＋整体RU90早筛才另登记跨RX星地，完整无标签适应/发现/未来匿名注册和更高拒识目标持续ACTIVE，AN/HOW未执行。详见`docs/PAMF_PHASE3_CONTRIBUTION_DENSITY_RESULTS_20261030.md`。


## 2026-10-30 独立度量残差两臂完成，不晋级

Run `20261030-phase3-star-metric-residual-s2026092701-r01`、code310588cbd8c79bc4c687939ab49202ea422e8a2c，VERIFIED/ANALYZED_SOURCE_ONLY、NO_PROMOTION。M_L/M_LOE各2220新aux102688参数，F0/defaulthead frozen/native0，全部4440 finite/同L indices/完整metadata重放/梯度路径通过。主AK90.418519/CK98.411111/RU23.174603/H36.893313%，原F0损失0.562963pp；M_L RU23.682540%，固定F0 RU28.095238%。known42皆过、unknown失败；主vsM_L正确净+8/未知拒对净−16，主vsF0正确+12/未知拒对−155。postfit背景两者含OE，M_L仅embedding无OE。sourceRX已见/novel历史曝光，所有数据仍原G10P1，target0/AN/HOW未执行；真实F00/合成2/G0，新open真实feature行1079728/合成16/native更新0，pipeline199.695379s非B1。完整25×2/42×2/分组/log/controlarrays/独立读回齐全。

下一候选补足L–OE跨集合约束：将已授权outside-six成员事实转为L–OE有效负关系，不假定OE彼此异设备、不解封OE TX/K，保留OE软内部结构的有/无匹配。现有L–L＋OE–OE几何不显式锚定两组相对位置，独立旋转表征层可保组内余弦，属新假设结构理由而非本轮失败因果结论。另预登记绝对known支持密度与known/OE比读出对照，避免默认OE代表所有未见设备；唯一主与具体预算/规则在新run固定，不按本轮novel truth调参/改阈值。尚未实现/验证，完整跨未见RX星地/高拒识/无标签DA/发现/未来匿名注册继续ACTIVE。详见`docs/PAMF_PHASE3_METRIC_RESIDUAL_RESULTS_20261030.md`。


### 2026-10-30 预登记L–OE分离与known-support主读出

独立新run `20261030-phase3-star-crossset-metric-s2026092701-r01`，code `fb016284e6c1f4715da4c690e2333efce89103b5`。两arm S_SEP/S_SEP_TOPO 同L192/OE64/102688参数/2220步，从原F0零残差重新开始；Lanchor在L+OE候选上显式学习outside-six合法负关系，不构造OE–OE硬关系。唯一主S_SEP_TOPO::learned_known_support，密度比仅诊断。V仅known42固定校准、NV仅truthlast；sourceL/OE/V/NV与G10P1不变、target0。主早筛通过才另登记跨未见RX星地，完整v3/更高拒识及无标签DA/发现/匿名注册继续ACTIVE；未取得本轮性能。设计见 `docs/PAMF_PHASE3_CROSSSET_METRIC_PLAN_20261030.md`。


### 2026-10-30 L–OE分离两臂实际完成

Run `20261030-phase3-star-crossset-metric-s2026092701-r01` / code `fb016284e6c1f4715da4c690e2333efce89103b5`，两arm各2220、三列预测/独立truthlast完成，执行VERIFIED、NO_PROMOTION。唯一主S_SEP_TOPO::learned_known_support A_K90.496296%、C_K98.685185%、R_U23.015873%、H36.698290%；主arm ratio诊断R_U32.952381%，不换主或晋级。sourceRX已见/novel历史曝光，target0；后续研究优先信号级开放分支和浅中层特征，尚无实现/性能声明。完整v3跨RX星地/更高拒识与无标签DA/发现/未来REG继续ACTIVE，结果见 `docs/PAMF_PHASE3_CROSSSET_METRIC_RESULTS_20261030.md`。


### 2026-10-30 预登记独立信号级开放分支

Run `20261030-phase3-star-signal-open-s2026092701-r01` / actualcode `eec44ce2f1d587d9e46ed4860fe6e204f5621155`，单E_SIGNAL，新headless信号232033+MLP102688=334721，原F0/default6身份路径固定。相同crossset+OEsoft结构、L192/OE64/3017/3018/2220及known-support主读出，将梯度推进浅中层，Sinc用CENTER_UNITS物理策略；冻结前端控制精确复用原S_SEP_TOPO，不重训。合法既有G10/P1 IQ L6300/OE6134训练，V27000仅固定known42/τ、NV3150仅truthlast，无新信道/G/二次RMS/P0/clean值/target。复制特征祖先11100+新增2220=13320、MLP2220、原识别器native0分别披露。主source早筛通过才另登记跨未见RX；完整v3与更高拒识/无标签DA/发现/未来匿名REG继续ACTIVE，当前无新性能。设计见 `docs/PAMF_PHASE3_SIGNAL_OPEN_PLAN_20261030.md`。


### 2026-10-30 signal-open r01导出技术失败

Run `20261030-phase3-star-signal-open-s2026092701-r01` / code `eec44ce2f1d587d9e46ed4860fe6e204f5621155` 执行技术失败已VERIFIED，尚未读训练IQ或更新，NO_PERFORMANCE_RESULT。真实SequenceReference被旧x3-only exporter类型guard拒绝；只读类/结构诊断确认原PAMF三视图/时间重排/232993/Sinc一致，父权重/数据合法性未变。原root/错误日志保留、进程全退出、foreign0；下一r02只修该导出接口，经回归/定点复审后使用新root，不重调科学配方。完整目标ACTIVE。


### 2026-10-30 独立信号级开放分支r02定点恢复

Run `20261030-phase3-star-signal-open-s2026092701-r02` / actualcode `b1f99b77a81fa4fd82f7d519a7db4f1aa46d7b15`，单E_SIGNAL，新headless信号232033+MLP102688=334721，原F0/default6身份路径固定。相同crossset+OEsoft结构、L192/OE64/3017/3018/2220及known-support主读出，将梯度推进浅中层，Sinc用CENTER_UNITS物理策略；冻结前端控制精确复用原S_SEP_TOPO，不重训。合法既有G10/P1 IQ L6300/OE6134训练，V27000仅固定known42/τ、NV3150仅truthlast，无新信道/G/二次RMS/P0/clean值/target。复制特征祖先11100+新增2220=13320、MLP2220、原识别器native0分别披露。主source早筛通过才另登记跨未见RX；完整v3与更高拒识/无标签DA/发现/未来匿名REG继续ACTIVE，当前无新性能。设计见 `docs/PAMF_PHASE3_SIGNAL_OPEN_PLAN_20261030.md`。


### 2026-10-30 signal-open r02合成前向依赖失败

Run `20261030-phase3-star-signal-open-s2026092701-r02` / code `b1f99b77a81fa4fd82f7d519a7db4f1aa46d7b15` 导出后合成probe失败已VERIFIED，未读真实IQ/训练/评分。原forward return_aux仍计算被丢弃的classify logits，删W引发AttributeError；原components的joint不依赖W。与r01类型guard ValueError不同指纹，r03将只旁路独立copy内部被丢弃的logits，保证原features/梯度、分类路径和科学目标不变。root/日志保留、live/GPU空、foreign0，无性能结果，完整目标ACTIVE。


### 2026-10-30 独立信号级开放分支r03前向依赖恢复

Run `20261030-phase3-star-signal-open-s2026092701-r03` / actualcode `dc01a402582f93e452dbaa3ef59a9fbb44151818`，单E_SIGNAL，新headless信号232033+MLP102688=334721，原F0/default6身份路径固定。相同crossset+OEsoft结构、L192/OE64/3017/3018/2220及known-support主读出，将梯度推进浅中层，Sinc用CENTER_UNITS物理策略；冻结前端控制精确复用原S_SEP_TOPO，不重训。合法既有G10/P1 IQ L6300/OE6134训练，V27000仅固定known42/τ、NV3150仅truthlast，无新信道/G/二次RMS/P0/clean值/target。复制特征祖先11100+新增2220=13320、MLP2220、原识别器native0分别披露。主source早筛通过才另登记跨未见RX；完整v3与更高拒识/无标签DA/发现/未来匿名REG继续ACTIVE，当前无新性能。设计见 `docs/PAMF_PHASE3_SIGNAL_OPEN_PLAN_20261030.md`。


### 2026-10-30 signal-open r03完整源侧结果

Run `20261030-phase3-star-signal-open-s2026092701-r03` / code `dc01a402582f93e452dbaa3ef59a9fbb44151818`，2220signal+projection更新、四列预测和独立truthlast完成，VERIFIED/NO_PROMOTION。唯一主A_K90.388889/C_K98.388889/R_U31.301587/H46.500199%，原F0损0.592593pp，known42通过；比匹配frozenknown未知+261（+8.285714pp）、正确−29/H+9.801909pp，三未知净+167/−14/+108，不是一致改善。四列均未RU90；原recognizer native0，复制feature祖先11100+2220=13320，sourceRXseen/NV历史曝光/target0。r01/r02技术失败不并为科学结果，smoke成本缺artifact记UNKNOWN。后续只用V检查局部42保护与总体2pp预算的差距，再独立固定校准对照；不扫NV或降低用户总体/主景门槛。完整跨RX星地/零标签DA/发现/未来REG目标ACTIVE，详见 `docs/PAMF_PHASE3_SIGNAL_OPEN_RESULTS_20261030.md`。


### 2026-10-30 固定表征known12风险校准预登记

Run `20261030-phase3-star-macro12-risk-s2026092701-r01` / code `26fbecff1df7e5476bcc5f459d8367c6fa49fd36`，固定父signal-open r03原分数/身份/known8，唯一主signal_known12；2×2冻结前端/信号×known42/known12。新hard12保overall+5RX+6scene的同macroC_K90/原F0损失2pp与覆盖，只把30cross切片列为完整诊断，不声称known42通过。两新六τ由V27000 exactFraction common-tail独立确定，全部τ固定后才novel3150，独立truthlast。NN/几何/归一化/真实IQ/F/G/signal前向新0，仅实际父CP synthetic_signal2；无target或未来REG结果。源NV历史曝光，完整用户总体/主场景与跨RX星地/零标签DA/发现/未来REG指标不变，goalACTIVE；当前未启动/无性能。设计见 `docs/PAMF_PHASE3_MACRO12_RISK_PLAN_20261030.md`。


### 2026-10-30 固定分数known12校准实际完成

Run `20261030-phase3-star-macro12-risk-s2026092701-r01` / code `26fbecff1df7e5476bcc5f459d8367c6fa49fd36`，四路预测/独立truthlast完成，VERIFIED/NO_PROMOTION。唯一主signal_known12 A_K89.948148/C_K97.522222/R_U46.888889/H61.643672%，原F0损1.033333pp，hard12通过但known42失败（5cross损伤>2pp）；同signal42未知+491（+15.587302pp）、正确−119/接受−234，旧控制完全复用。sourceRX3整体已用2pp预算，下一q损2.018519pp；总体尚余.966667pp不等于可继续提高同q。真实IQ/NN/几何新0、CPsynthetic2，父成本保留、sourceNV历史曝光/target0。后续拟独立检验P1身份与P0开放信号路径的匹配训练，不声称DURA损坏或已改善；完整跨RX星地/零标签DA/发现/未来REG目标ACTIVE，详见 `docs/PAMF_PHASE3_MACRO12_RISK_RESULTS_20261030.md`。


### 2026-10-31 P0开放分支与P1身份分工预登记

Run `20261031-phase3-star-raw-open-s2026092701-r01` / code `617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb`。固定原F0/P1 default6身份，新增从原F0初始化的P0 headless信号分支，原F0 P0 OEteacher重新计算6134，模型导出先于teacher冻结。与P1r03同结构／2220步／全部采样索引／目标；四列P1known12、P1ratio12、P0known12、P0ratio12，唯一主P0known12。两路径均星地，P0不等于clean；原P1闭集为统一保护基线。新NN／统计先freeze，V27000仅hard12风险，所有τ后novel3150，最后独立truthlast；不访问目标、Us或未来query。sourceNV历史曝光，只是早筛，主hard12＋整体RU90通过才另登记跨未见RX测试。原用户v3总体／主景CK90、RU90、AK较同输入原闭集损失≤2pp，以及零目标TX标签DA／发现／未来匿名REG均保持，goalACTIVE。当前未启动／无性能。


### 2026-10-31 P0开放路径匹配训练完整结果

Run `20261031-phase3-star-raw-open-s2026092701-r01` / actualcode `617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb`，训练/四列预测/独立truthlast COMPLETE，执行VERIFIED，科学 `NO_PROMOTION`。唯一主P0known12 A_K90.048148/C_K98.503704/R_U3.936508/H7.543258%，原P1闭集损0.933333pp；相较P1known12 RU-42.952381pp、AK0.100000pp。全部2220匹配索引/finite/Sincstep/clip核实；原F0 P0teacher6134、signal训练568320＋评估42584新增前向，原身份native0。两路径都星地，source5RX已见/NV历史曝光，V同时用于known阈值风险，不能当新RX盲证据。目标访问0，完整v3跨RX星地/零标签DA/发现/未来匿名REG仍ACTIVE，详见 `docs/PAMF_PHASE3_RAW_OPEN_RESULTS_20261031.md`。


### 2026-10-31 较好既有方法DURA前后六路径预登记

Run `20261031-phase3-dura-best-s2026092701-r01` / actualcode `9fc23a4cb78111aa96cc72fde0d35c036a876e5d`。按用户优先选择原P1 signal known/ratio两既有较好方案，唯一主ratio；原sourceDURA RU46.888889/60.888889已经含G。本轮同模型/统计/两原τ冻结，source/clean/star各raw与DURA配对，clean也G，每路径F0闭集同输入。新F0/signal各33606、G1728真实行，sourceDURA精确复用；无新NN/fit/τcal/L/OE/Us/U/formal。六pathpred后3truth独立评分，比较两侧各自closed下的正确接收/未知拒对净收益，三DURA为主、raw仅诊断。指标clean95/star90/CK90/损2pp和主景覆盖保持，历史NV/devQ曝光。prepared条件密度暂不运行，完整零标签DA/发现/未来匿名REG目标ACTIVE，当前无新性能。


### 2026-10-31 较好既有方法DURA前后完整结果

Run `20261031-phase3-dura-best-s2026092701-r01` / actualcode `9fc23a4cb78111aa96cc72fde0d35c036a876e5d`，同一P1 signal r03、统计和两套旧τ的2method×6path预测/3truth独立评分全部完成，执行VERIFIED、科学NO_PROMOTION。主ratio源raw→DURA A_K25.333333→89.770370、C_K25.911111→97.107407、R_U97.650794→60.888889、H40.229909→72.561330%；clean raw→DURA A_K68.229167→68.923611、C_K85.243056→85.416667、R_U100→99.652778、H81.114551→81.487441%；star raw→DURA A_K10.243056→63.020833、C_K12.326389→84.895833、R_U97.569444→76.388889、H18.539766→69.063927%。原F0同DURA闭集clean80.034722/star72.916667，损11.111111/9.895833pp仍超2；CK90未达，starRU90未达、主景缺TX保持N/A。DURA恢复星地已知接收，raw高RU同时大量误拒known，不能当综合成功。旧source较好成绩本来含G且精确复用，不虚报新恢复收益；prepared条件密度未启动。新F0/signal各33606/G1728真实行、各synth2，NN/fit/retau/U/formal0。完整零标签DA/发现/未来匿名REG仍ACTIVE，详见 `docs/PAMF_PHASE3_DURA_BEST_RESULTS_20261031.md`。

当前证据把下一步重点指向跨RX已知误拒与星地未知分离的联合改进。后续只从合法无标签U学习域适应/未知结构，不回收Q、不用Q真值调阈值，不把source或clean单项RU通过当全系统成功；三类DURA主测试及同路径F0保护继续。新机制需另行固定输入/目标/对照/预算后实施，完整目标不缩小。


### 2026-10-31 固定较好密度比的无标签混合适应预登记

Run `20261031-phase3-mixture-uda-s2026092701-r01`，actualcode `d72a7b7c301fec2ff3736e4221dfe5047ab78efc`。沿用此前固定较好的P1signal known90/OE32 density_ratio为主，known_support为控制；新DA0/DA_K/DA_M六列唯一DA_M_ratio。仅合法starU384经同DURA→signal，源15域公共偏移SVD定义rank14子空间/λ/半径，固定20轮软责任MAP-EM，known-only与背景混合0.5/0.5控制，不读真TX/K/quota或cleanU。身份/密度/原两套六τ不更新，两个adapter冻结后再读既有sourceDURA/cleanDURA/starDURA预测；sourceRXdelta0，target两条件同starUdelta。六列三条件全部完成后独立truth-last，无Utruth/Q统计/重拟合。新G/F0/signal各384，source/Q新前向0，目标统计fit2×20明确计费。主需starAK正增且clean95/star90/CK90/相对F0损2pp和场景覆盖全部满足，历史开发曝光/缺景保持，正式RX未访问。无匿名REG，本轮结果尚无，完整Phase3目标ACTIVE。


### 2026-10-31 无标签混合平移的完整负结果与保留基线

Run `20261031-phase3-mixture-uda-s2026092701-r01` / actualcode `d72a7b7c301fec2ff3736e4221dfe5047ab78efc`，六列×三DURA条件全部预测/独立评分 COMPLETE，执行VERIFIED，科学NO_PROMOTION。唯一主DA_M_ratio源AK89.770370/CK97.107407/RU60.888889/H72.561330%，clean68.576389/85.069444/99.652778/81.244267%，star63.020833/84.895833/75.694444/68.778682%。对DA0_ratio，star正确接收0增/0失、未知拒对0增/2失，AK0pp/RU−0.694444pp；clean正确接收0增/2失、未知不变，AK−0.347222pp。已知模式控制DA_K_ratio的starAK63.541667但RU73.611111/H68.206751%，不是成功或替换主。两目标统计拟合各20轮，objective均改善且delta范数1.412971/0.915601小于源r2.370508、rank14，但无任务收益，不能以拟合目标改善宣称有效。新G/F0/signal各384真实＋各2synthetic；网络/源密度/τ更新0、Q/sourceIQ新读0、Utruth0，保留原F0同路径基线和完整源状态。实际predict8.230080s/CPUscore7.996030s，短GPU阶段未捕获live显存，UNKNOWN而不为遥测重跑；独立CPU_PID/CWD/argv/CUDAempty/日志增长及终态无所属进程已核。clean相对F0损11.458333pp、star9.895833pp，CK90/星地RU90及场景覆盖未满足。保留此前源端固定的DURA＋P1signal＋density_ratio DA0作为后续基线，本轮adapter不晋级、不回流Q调τ。完整跨RX/零标签发现/未来匿名REG仍ACTIVE，正式RX未访问，详见 `docs/PAMF_PHASE3_MIXTURE_UDA_RESULTS_20261031.md`。


### 2026-10-31 预测身份条件密度与三类DURA测试预登记

Run `20261031-phase3-class-density-s2026092701-r01` / actualcode `4af0668331682636ee212077b38cd912857a1b31`。严格冻结原P1 signalr03和known90/OE32，只比较全类/预测类15mode条件密度×known/ratio，唯一主predicted_ratio。全部NN/密度在V前冻结、四τ在任何NV/Q前冻结；源27000/3150两旧列精确控制，crossRX clean864与star864都接入同一DURA，并重新计算同DURA输出F0基线。三个条件共享源星地τ，不按目标校准；source未过也全测。主报宏AK/RU/H，cleanRU95/starRU90、CK90/损2pp/覆盖，缺主景类则不认证。历史source/devQ曝光，sourceL/OE/Us/U/正式RX24-18新读0，NN/密度/normalizer拟合0；源V已知标签用于两套新六阈值校准，另两套精确复用。完整跨RX星地、零标签DA/发现/未来匿名REG仍ACTIVE，当前无性能。


### 2026-10-31 预测身份条件密度完整三条件结果

Run `20261031-phase3-class-density-s2026092701-r01`，actualcode `4af0668331682636ee212077b38cd912857a1b31`；四列×三DURA条件全部预测/独立truth-last COMPLETE，执行VERIFIED、科学NO_PROMOTION。唯一主predicted_ratio：source AK/CK/RU/H 89.744444/96.759259/61.936508/73.291437%，原F0损伤1.237037pp；clean_DURA AK/CK/RU/H 70.833333/87.326389/100.000000/82.926829%，原F0损伤9.201389pp；star_DURA AK/CK/RU/H 64.062500/85.590278/77.083333/69.972325%，原F0损伤8.854167pp。相对pooled_ratio，clean正确接收净+11、未知拒对净+1；star正确接收8增/2失/净+6、接收10增/6失/净+4、未知拒对4增/2失/净+2，即AK+1.041667pp、RU+0.694444pp、H+0.908398pp。source正确接收净−7、未知拒对净+33，known12保护保持但不能冒充42全过。两个新六阈值仅源V27000已知标签校准，两旧精确复用，NN/密度/normalizer拟合0；原F0/G/P1冻结，synthetic各2实际CP smoke PASS，新Q G/F0/signal各1728、sourceIQ/L/OE/Us/目标U/正式RX新读0。predict/CPUscore实际批处理{"predict": 29.048555, "score": 9.49642}s；所属GPU7/PID1021309/344MiB独立离散样本、CPU1022185 CUDAempty及终态空均核，344MiB非峰值；foreign操作0。跨RX clean虽RU100%，CK87.326389%、损9.201389pp仍失败；starCK85.590278%、RU77.083333%、损8.854167pp失败，high未知缺19-15保留N/A。结果为局部改善，不替换此前冻结基线或认定零标签DA/匿名注册完成；历史NV/devQ曝光、单RX/seed和原MIX01观测保持，完整Phase3目标ACTIVE。详见 `docs/PAMF_PHASE3_CLASS_DENSITY_RESULTS_20261031.md`。


### 2026-10-31 类别相关源域偏移的无标签适应预登记

Run `20261031-phase3-class-shift-s2026092701-r01` / actualcode `74e0a3de37d0f3673375804fb09b06c4f624745b`。固定原bestP1signal pooled-known/ratio与两旧tau，六列DA0/SHARED/CLASS各known/ratio，唯一CLASS_ratio。源known90的6×15类别域变化不先平均，拼15×960/sqrt6 SVD得rank14/r5.263391 RMS源球，单a产生六delta、OE用其均值；固定0.5known/0.5OE/zero init/20MAP-EM及60bisect受约束M步。只用旧mix合法cachedstarU384 signal特征，不用U真TX/K/quota/F0logits/cleanU，新prior先freeze、CLASS先fitfreeze再任何source/Q行。DA0best精确、SHARED旧DA_M精确复用；source不同RXdelta0，targetclean/star同starU状态。全部三DURA×六列pred后独立3truth，无Utruth/EMin score/Q统计/retau。实际新realIQ/G/F0/signal0、synth各2，targetstat新CLASS20/继承SHARED20分列，source模型/密度/τ保持。主需starAK正增及cleanRU95/starRU90/CK90/F0损2pp/场景覆盖；历史开发曝光和high缺类不放宽，formalRX未访问。无匿名REG，完整Phase3目标ACTIVE，当前未启动无新性能。


### 2026-10-31 类别相关偏移的完整无标签适应结果

Run `20261031-phase3-class-shift-s2026092701-r01` / actualcode `74e0a3de37d0f3673375804fb09b06c4f624745b`，六列×三DURA条件全部预测/独立truth-last COMPLETE，执行VERIFIED、科学NO_PROMOTION。唯一主CLASS_ratio：source_DURA AK/CK/RU/H 89.770370/97.107407/60.888889/72.561330%，原F0损伤1.211111pp；clean_DURA AK/CK/RU/H 70.312500/86.805556/99.652778/82.450204%，原F0损伤9.722222pp；star_DURA AK/CK/RU/H 64.062500/86.111111/71.180556/67.434211%，原F0损伤8.854167pp。对DA0，clean正确接收/接受净+8、未知拒对不变；star正确6增/0失、接受7增/0失，未知拒对0增/15失，AK+1.041667pp/RU−5.208333pp/H−1.629716pp。对SHARED star正确净+6、未知拒对净−13；source六列重复原两列，原F0所有身份不变。U-only新CLASS20步目标181695.052814→182029.344876，RMS2.044845小于源r5.263391、maxclass3.058136/背景.887298，约束乘子全0；末步仍增约.000135，不声称完全收敛或追加迭代。新source类域prior rank14，target移动mode统计实际拟合20步，SHARED20和旧U384导出为继承成本；本轮cachedU384，真实IQ/G/F0/signal前向0、synthetic真实三CP各2PASS，NN/source密度/τ更新0。批处理predict/CPUscore {"predict": 7.573466, "score": 6.886858}s；独立GPU7/PID1043324/374MiB离散及日志增长/终态空已核，CPUscorer仅pipeline PID1043469/gpuNone及代码CUDAempty配置，未捕获CPUlive，不虚报独立CPU遥测。foreign操作0。cleanRU单项过95但CK86.805556/损9.722222pp失败；starRU71.180556/CK86.111111/损8.854167pp失败，high主景缺未知TX不认证。无目标真TX/K/配额或Q统计反馈，无未来匿名REG，正式RX未访问；原冻结基线保持，完整Phase3目标ACTIVE。详见 `docs/PAMF_PHASE3_CLASS_SHIFT_RESULTS_20261031.md`。


### 2026-10-31 源重放与目标实例对比的深层适应预登记

Run `20261031-phase3-signal-ssl-s2026092701-r01` / actualcode `72b0b60d6c445e907b1bd05061cc3dbcd36132af`。独立复制原r03完整334721信号分支为SOURCE/TARGET_SSL，各freshCENTER_UNITS444步、sourceL192/OE64索引逐步匹配；TARGET增加U64同记录独立phase/gain两view的NT-Xent1/temp0.1，SOURCE U0，只有SOURCE全完成后才真实开starU384并G恢复一次。原F0/default6、DURA、r03本体冻结；新学生Sinc/body/projection可训练，BN/Dropout eval/buffers保持。SOURCE纯源/TARGET显式U祖先，未知是未注册且可无标签曝光，不称严格未见。全部NNfreeze后两sourceL/OE density90/OE32＋四sourceV hard12tau，DA0原11/两tau保持；全部六tau后NV/两个Q，六列×源DURA/cleanDURA/starDURA全预测后独立3truth，无Utruth/trueK/quota/Q统计或评分回流。唯一SSL_ratio须starAK同时胜DA0/SOURCE且cleanRU95/starRU90/CK90/同G原F0损2pp/主景coverage。新增训练888更新/源曝光227328/targetUview56832，预测新source/querysignal90352、G/F01728，trainG384，旧F011100/r032220/G祖先成本保持。历史sourceNV/Q曝光和high缺类保留，formalRX及未来query未访问；不做匿名REG，完整Phase3目标ACTIVE，当前未启动无性能。


### 2026-10-31 深层目标实例对比的完整匹配实验结果

Run `20261031-phase3-signal-ssl-s2026092701-r01` / actualcode `72b0b60d6c445e907b1bd05061cc3dbcd36132af`：两个独立学生各444更新、六方法×三DURA条件的预测与独立truth-last均完成，执行VERIFIED、科学NO_PROMOTION。source_DURA AK/CK/RU/H 89.496296/96.640741/69.936508/78.516570%；clean_DURA AK/CK/RU/H 58.506944/74.826389/96.527778/72.855232%；star_DURA AK/CK/RU/H 55.208333/74.131944/82.291667/66.082702%。相对原DA0_ratio，星地正确接收净−45、未知拒对净+17，AK−7.812500pp/RU+5.902778pp/H−2.981225pp；相对同预算SOURCE_ratio，正确接收净−34、未知拒对净+27，AK−5.902778pp/RU+9.375000pp/H−0.411253pp。未知拒识上升不能抵消已知正确接收损失。原F0/G/r03冻结、SOURCE U0、TARGET U384且适配祖先明确；模块梯度仅前两步有日志，不推广到全部444步。正式RX未访问，匿名注册未完成，完整目标ACTIVE。详见 `docs/PAMF_PHASE3_SIGNAL_SSL_RESULTS_20261031.md`。


### 2026-10-31 按用户要求选取既有较好方法

本轮选取两种已完成方法：原P1 signal r03＋pooled known/OE density_ratio作为冻结对照（`20261031-phase3-dura-best-s2026092701-r01`）；同一冻结表征的predicted_ratio作为重点候选（`20261031-phase3-class-density-s2026092701-r01`）。后者按原F0预测身份选取15个源RX/day模式，背景32模式保持，只使用原源V确定自身六阈值；身份头与DURA不改。所有三类测试均使用DURA。前者clean/star AK/RU/H为68.923611/99.652778/81.487441%与63.020833/76.388889/69.063927%；后者为70.833333/100.000000/82.926829%与64.062500/77.083333/69.972325%。优先保留已知正确接收与未知拒识共同改善的方法，不按RU单项选择SSL、CLASS偏移或SHARED偏移。

此次只筛选历史开发证据并归档已完成实验，不新增训练、预测、评分、阈值拟合或正式RX访问。不修改任何历史run主方法/结论，不将重点候选视为科学晋级或独立盲测确认；原冻结对照继续保留。两者仍未达到CK≥90%、星地RU≥90%和相对同DURA闭集AK损失≤2pp的联合要求，目标ACTIVE。


### 2026-10-31 既有较好方法的局部协方差预登记

Run `20261031-phase3-local-cov-s2026092701-r01` / actualcode `4d1d645dd13de875a7f6adf32508a9732ffd3f47`，固定原pooled/predicted ratio两控制，新known90各模式仅以原L6300P1估完整协方差C=70/230*S+160/230*pooledI，原mean/whitener/OE32/F0/G保持。新local pooled及唯一主local predicted，各Vonly six-tau，旧两tau精确，全部freeze后sourceNV/cleanQ/starQ缓存score、独立三truth-last。补sourceLsignal6300真实前向，F0/G/Q新真实0；cov90/tau12实际拟合，非全零成本；不读U/正式RX。检验方向相关形状，不再训练失败SSL学生或对历史Q调参；全部三DURA测试自动完成，主gate保持clean95/star90、CK90/损2pp/场景覆盖及starAK/RU不低于predicted。source/NV/Q历史曝光及high缺类保留，不称匿名REG或独立新盲测，完整目标ACTIVE，当前未启动无性能。方案见 `docs/PAMF_PHASE3_LOCAL_COV_PLAN_20261031.md`。


### 2026-10-31 源模式局部协方差完整结果

Run `20261031-phase3-local-cov-s2026092701-r01` / actualcode `4d1d645dd13de875a7f6adf32508a9732ffd3f47`，sourceL6300冻结signal前向、90局部cov和两新six-tau、四列×三DURA预测及独立三truth-last均COMPLETE，执行VERIFIED、科学NO_PROMOTION。唯一主：source AK/CK/RU/H 89.844444/96.907407/63.777778/74.599611%；clean_DURA AK/CK/RU/H 61.111111/77.777778/100.000000/75.862069%；star_DURA AK/CK/RU/H 60.763889/82.986111/71.180556/65.561038%。对选定predicted_ratio，source正确接收净+27、未知拒对净+58；clean正确净−56、未知不变；star正确8增/27失/净−19，未知拒对7增/24失/净−17，AK−3.298611pp/RU−5.902778pp/H−4.411287pp。source局部改善未迁移到新RX，不替换原重点候选predicted_ratio或冻结pooled_ratio对照。

原90means/whitener/背景OE32、F0/G/r03和所有身份保持；重算mean最大绝对误差2.442491e−14、trace1.110223e−16、floor0，无数值技术失败，不重跑/扫参数。新NN/原密度/背景/normalizer0，新cov90和Vonly tau12实际fit；真实signal6300/F0G0、synthetic各2、另存约18MB局部矩阵，旧训练/源缓存/三DURA query前向成本继承。predict21.183924s/CPUscore19.138905s；driver1104163/CPU1105453的live/CWD/argv/CUDAempty及最终空已核，predict1104252/GPU4仅pipeline/日志记录、未捕获独立predictorlive或显存样本，不虚报GPU遥测。foreign0。历史sourceNV/devQ、单RX/seed、high未知缺19-15维持N/A，U/正式RX未读，无匿名REG/AN/HOW。完整跨RX星地拒识、零标签DA/发现/未来注册目标ACTIVE。详见 `docs/PAMF_PHASE3_LOCAL_COV_RESULTS_20261031.md`。


### 2026-10-31 原身份与开放特征联合密度预登记

Run `20261031-phase3-joint-feature-s2026092701-r01` / actualcode `b7f81eca9a2a0bb55afc361acd50f3fbd739b68a`。保留原pooled/predicted ratio两精确控制，唯一主joint_predicted_ratio；同记录原F0 u160/r03 z160单位拼320，新sourceL6300/OE6134 known90/OE32/全局cov白化和源V两组六tau。不同于localcov/四贡献640/P0P1融合，不更新原F0/G/r03或另训身份头。sourceu缓存对齐、sourcez12434新冻结前向；sourceV/NV双特征复用，Q新同G/F0u1728并核同logits/closed、z原样复用。新stats freeze先V/tau，全部四tau后NV/cleanQ/starQ，三DURA全部pred后CPU三truth-last。统计新fit1/tau12，NN0，目标U/TX/K/quota/正式RX/REG0。clean95/star90/CK90/损2pp/必需场景覆盖及starAK/RU不低于原predicted保持，sourceRU仅诊断；历史NV/Q曝光、单RX/seed/high缺类NA不变，完整目标ACTIVE，当前未启动无性能。方案见 `docs/PAMF_PHASE3_JOINT_FEATURE_PLAN_20261031.md`。


### 2026-10-31 身份与开放联合密度完整结果

Run `20261031-phase3-joint-feature-s2026092701-r01` / actualcode `b7f81eca9a2a0bb55afc361acd50f3fbd739b68a`：sourceL/OE12434冻结signal前向、joint320 sourceknown90/OE32/normalizer与两组六tau、四列×三DURA预测及独立三truth-last均COMPLETE，执行VERIFIED、科学NO_PROMOTION。唯一主：source AK/CK/RU/H 89.722222/96.785185/66.666667/76.494967%；clean_DURA AK/CK/RU/H 67.708333/84.201389/100.000000/80.745342%；star_DURA AK/CK/RU/H 63.368056/84.722222/78.472222/70.115939%。对选定predicted_ratio，source正确42增/48失/净−6、未知拒对223增/74失/净+149；clean正确3增/21失/净−18，未知不变；star正确5增/9失/净−4、未知拒对11增/7失/净+4，AK−0.694444pp/RU+1.388889pp/H+0.143614pp。H微增和RU局部收益不能代替AK不退步及绝对联合目标；原predicted重点候选、pooled冻结对照保持，不按Q重调权重/阈值/改主。

原F0/defaulthead/G/r03均冻结、旧160state/tau/cal/原始身份逐值复用。newjoint320密度/normalizer1及Vonly tau12实际fit，不能称全部统计0；源F0u缓存对齐同IQ/IDs/metadata/G祖先，newz12434，Q新同G/F0u1728核相同logits/closed、z复用，synthetic各2；原训练与sourceF0cache/父Qsignal成本继承，不增加物理样本。predict24.475319s/CPUscore12.593357s；独立driver1124895/predict1124984的CWD/argv/CUDA4/GPU_UUID95d925a1与444MiB采样及log549→695B、最终空已核，444MiB非峰值；CPU1125157仅pipeline/gpuNone与CUDAempty代码，未捕获CPUlive，不虚报遥测。foreign0。登记锁冲突先核已写row事件，锁自然释放后去重补齐global，无删除锁/重发run/干预其他owner。sourceknown12仅源开发保护，cleanCK84.201389/损12.326389pp与starCK84.722222/RU78.472222/损9.548611pp仍失败，high未知缺19-15保持NA。单RX/seed/sourceNV及devQ历史曝光、U/正式RX未读、无DA/发现/匿名REG，AN/HOW=N/A，完整目标ACTIVE。详见 `docs/PAMF_PHASE3_JOINT_FEATURE_RESULTS_20261031.md`。


### 2026-10-31 固定密度拒识边界源训练预登记

Run `20261031-phase3-density-boundary-s2026092701-r01` / actualcode `e3194bc6ed46ce1580967f036a35fcb138e1c906`，两独立r03 source-only副本REPLAY/BOUNDARY各2220/freshCENTER_UNITS，L192/OE64索引匹配；共同crossset+OEtopo+r03点anchor，仅主加固定conditionalGMMratio/160 BCE(权1)。原F0/G/r03/旧11Gaussian和pointteacher保持，F0缓存u经原头的预测c路由，非L真y；无新head/U/clean/V训练。只parentheader/models先actualCPsmoke和两gradientprobe，referencez与head-only各12434一次，学生真实1136640/新NN4440，总feature15540/MLP4440、F0仍11100。两神经freeze后sourceV四新6tau+E0旧两tau exact，六套tau后NV/cleanQ/starQ，六列×三DURA完整pred再独立三truth。GMM/normalizer新fit0，eval两学生source60300+Q3456=63756signal，QF0/G1728、E0z/score缓存exact。gate保持sourceknown12、clean95/star90/CK90/损2pp/主景覆盖，starAK/RU均不低于E0与REPLAY；sourceRU仅诊断，不取消Q。sourceNV/Q历史、单seed/RX/high缺类NA、无U/正式RX/匿名REG均保留，完整目标ACTIVE，当前未启动无性能。方案见 `docs/PAMF_PHASE3_DENSITY_BOUNDARY_PLAN_20261031.md`。


### 2026-10-31 固定密度拒识边界配对训练完整结果

Run `20261031-phase3-density-boundary-s2026092701-r01` / actualcode `e3194bc6ed46ce1580967f036a35fcb138e1c906`：REPLAY/BOUNDARY各2220完整更新、六读出×三DURA预测及独立三truth-last全部COMPLETE，执行VERIFIED、科学NO_PROMOTION。唯一主：source_DURA AK/CK/RU/H 89.611111/96.696296/62.761905/73.820998%；clean_DURA AK/CK/RU/H 74.652778/91.666667/97.222222/84.455668%；star_DURA AK/CK/RU/H 62.673611/84.027778/78.125000/69.551480%。对同预算REPLAY_pred，star正确13增/16失/净−3、未知拒对49增/8失/净+41，RU+14.236111pp/H+6.011710pp但AK−0.520833pp；对旧E0_pred，star正确14增/22失/净−8、未知12增/9失/净+3，AK−1.388889pp/RU+1.041667pp/H−0.420845pp。clean对E0正确净+22、未知拒对净−8，CK91.666667/RU97.222222单项通过，但同G闭集80.034722−AK仍损5.381944pp，超过2pp；starCK84.027778/RU78.125/损10.243056pp失败。保留原E0_pred重点候选及E0_pooled冻结对照，不因相对续训控制的拒识收益或clean局部改善替换基线。

完整4440步由独立scorer核连续/配对索引/有限，两个arm均实际覆盖L6300/OE6134；sourcepointteacher、原F0/G/r03/GMM及学生buffers保持，BCE梯度实际到完整encoder而非只MLP。REPLAY总loss .356811643→.226494700，BOUNDARY .731422193→.473986185，末BCE .212238527对REPLAY诊断.352863794；训练loss降低不代表跨RX达标，模块梯度仅首2步记录。原GMM/normalizer新fit0、Vonly四新6tau实际fit、两旧exact；sourceAUC主较E0下降5.282118pp（固定margin排序诊断，不以RU微增宣称整体分离提升）。新sourceNN4440/feature15540/MLP4440，学生真实1136640+parentreference12434 signal，F0head-only12434与sourcebackbone/G0分列；eval63756 signal/F0G1728/E0real0，synthetic另列，旧训练/cache成本继承，teacherbank不部署。train462.670735s/predict26.080005s/CPUscore16.561103s；独立driver1155253/train1155342 CWD/argv/CUDA4/GPU_UUID95d925a1 758MiB采样与log增长/末态空已核；predict1161117与CPU1161391仅pipeline GPU4/gpuNone及代码CUDAempty，未捕获其独立live遥测，不为补采样重跑，foreign0。sourceNV/devQ历史、单RX/seed/high缺19-15继续NA，U/正式RX未读、无目标DA/发现/匿名REG，AN/HOW=N/A，完整目标ACTIVE。完整目标prompt另同步已授权三DURA测试的clean例外，矩阵/实际代码/既有判定不变。详见 `docs/PAMF_PHASE3_DENSITY_BOUNDARY_RESULTS_20261031.md`。


### 2026-10-31 源接收机日期扰动积分预登记

Run `20261031-phase3-nuisance-marginal-s2026092701-r01` / actualcode `93fc96a25b929c534e9b67570d4f2b38bc054c3e`，唯一主marginal_predicted_ratio，原pooled/predicted两精确控制＋newmarginalpooled，共四列×三DURA。源90means经类内中心化、六类平均得到15domain offsets，固定C=o.T@o/15/rank≤14，Gaussian解析积分cov=vI+C，known/OE均同C、原中心/v/weight/whitener不改，rank/scale不搜索。actual三CP synthetic2smoke后deriveprior/freeze，再V27000两新6tau＋两旧exact，全部tau后NV3150/clean864/star864缓存score，全预测才独立三truth；新真实IQ/NN/F0Gsignal/U/正式RX0，非零sourceaggregateprior和V阈值成本另列。gate保持clean95/star90/CK90/同DURA闭集损2pp/主景coverage、源known12与主starAK/RU不低于oldpredicted，source失败不取消Q。单seed/RX/历史NV及devQ曝光/high缺19-15保持NA，无DA/发现/匿名REG，完整目标ACTIVE，当前无新性能。见 `docs/PAMF_PHASE3_NUISANCE_MARGINAL_PLAN_20261031.md`。


### 2026-10-31 源接收机日期扰动积分完整结果

Run `20261031-phase3-nuisance-marginal-s2026092701-r01` / actualcode `93fc96a25b929c534e9b67570d4f2b38bc054c3e`：四读出×三DURA预测及独立三truth-last均COMPLETE，执行VERIFIED、科学NO_PROMOTION。唯一主：source AK/CK/RU/H 89.762963/96.796296/61.079365/72.693983%；clean_DURA AK/CK/RU/H 71.354167/87.847222/100.000000/83.282675%；star_DURA AK/CK/RU/H 64.062500/85.763889/76.041667/69.539963%。对原predicted，star正确接收0增/0失、接受1增/0失但新增为误分类、未知拒对1增/4失/净−3，AK0pp/RU−1.041667pp/H−0.432362pp；clean正确接收4增/1失/净+3、未知不变，AK+0.520833pp/RU100%，仍CK87.847222%/闭集损8.680556pp未过。starCK85.763889/RU76.041667/闭集损8.854167pp未过；sourceknown12通过仅诊断，源RU61.079365%。新marginal pooled的局部AK收益不替换唯一主，不按Q重调rank/尺度/τ；原predicted重点候选和pooled冻结对照保持。

新增一个源aggregate prior实际rank14（18204B），known90means的15domain共享C解析积分给known/OE cov=vI+C，原中心/权重/标量v/白化保持，旧Gaussian364340B和normalizer不重拟合。Vonly两新6τ共12标量拟合、两旧exact，全部24τ192B；单主部署只需自身6τ48B。原F0/G/r03参数及默认身份不更新，新真实IQ/F0Gsignal/L/OE/Us/U读取与NN更新均0，actual三CP synthetic2eachPASS；旧训练、source星地/cache/原Q前向成本继承，单NN834665。predict25.616272s/CPUscore14.970574s/pipe40.674194s为完整进程耗时；独立driver1180081/producer1180175 CWD/fullargv/CUDA7/GPU_UUID9c4eb614的406MiB离散样本（非峰值）、日志512→674B与末态空已核，CPU1180467仅pipeline/gpuNone及代码CUDAempty证据，未捕获独立CPUlive，不补跑，foreign0。sourceNV/devQ历史曝光、单RX/seed/日期重合/high缺19-15保持N/A，正式RX未访问，无目标DA/发现/匿名REG，AN/HOW=N/A，完整Phase3目标ACTIVE。详见 `docs/PAMF_PHASE3_NUISANCE_MARGINAL_RESULTS_20261031.md`。


### 2026-10-31 当前开放表征无标签匿名注册预登记

Run `20261031-phase3-signal-register-s2026092701-r01` / actualcode `e48f3b357eccb12e778c1eb348776e5c6b457922`，唯一主predicted_REG1，原pooled/predicted REG0精确＋对应REG1，共四列×三DURA。actual3CPsmoke→sourceL6300原P1/r03源7th-other95%eps冻→starU384同DURA F0/r03→完整U DBSCAN/min8/簇16且predicted拒识strictmajority，任意K含0→匿名Gaussian全部成员mean/v shrinkn/(n+160) towardsourcepool，不改原GMM→anonfreeze→sourceV两新globalτ/old两6τexact→NV及clean/star缓存Q全pred→独立CPU Utruth+sourceNV+两Qtruth。U-only Hungarian不回流，所有真实Uarrivals进未来AN分母，unmatched匿名接受计错；REG0RUallnonold/REG1later分母分开并报同later配对。最新v3 ANoverall80/oldrel和abs2pp/old及registeredCK90/laterclean95star90/scene覆盖，sourceknown12保护；不新增sceneAN80，high缺类保持NA。NN0/signal6684/GF384/synthetic2each，targetREG真实U-derived，DA未做，不宣称全系统source-only/持续多次注册或Phase3完成。历史NV/Q/单RXseed/日期重合/正式RX未访问保持，完整目标ACTIVE，当前无新性能。见 `docs/PAMF_PHASE3_SIGNAL_REGISTER_PLAN_20261031.md`。


### 2026-10-31 当前开放表征无标签匿名注册完整结果

Run `20261031-phase3-signal-register-s2026092701-r01` / actualcode `e48f3b357eccb12e778c1eb348776e5c6b457922`，四方法×三DURA预测及独立U/sourceNV/两Q四truth评分均COMPLETE，执行VERIFIED、科学NO_PROMOTION。源L6300第7other95%eps=17.987383312；完整U384 DBSCAN形成305和24两簇，predicted拒识75/305=24.590164%、4/24=16.666667%，均未strictmajority；noise55中54拒识、U总拒识133，不能把拒识候选等同真unknown。资格匿名K0，实际仍6旧+0匿名，U-only映射空，全部原REG0/REG1同base逐条预测完全相同，两个真实U到达设备的未来AN/H_KN/H_OW全0，不称无标签注册成功或持续注册完成。

主predicted_REG1 source/clean/star oldAK/CK分别89.744444/96.759259%、70.833333/87.326389%、64.062500/85.590278%，同DURA原F0闭集损1.237037/9.201389/8.854167pp；实际未注册all3拒识仍61.936508/100/77.083333%，对应旧R0完全不变。REG1预登记later-only RU71.523810/100/79.166667%及H79.604812/82.926829/70.818182%分母只有U未见7-1，不可当相对all3旧RU/H提高；同later配对增失均0。预期注册集合old6+全部真arrivals宏CK83.283730/65.494792/70.182292%含未发现新类，并非实际已注册8类；主AN0和原绝对旧保护/星地拒识门槛均未过。sourceknown12只源V经验校准保护，high缺19-15的AN/预期注册CK/HKN/HOW保NA，完整六景不认证，不改主/阈值/聚类参数重跑。

原F0/defaulthead/G/r03/GMM/旧two6tau保持；新NN0，真实sourceLsignal6300+U384=6684、GF各384，synthetic三CP各2，sourceεfit1。新Gaussian0、匿名空schema160B/source_radius156B/原density364340B；K0两套V校准只产生0/0占位、一候选，无有效匿名阈值学习，oldτ96B+zeros16B=112B，原834665NN及训练/cache成本继承，source/U行bank不部署。predict25.931523s/CPUscore11.398478s/pipe37.415431s为完整进程；独立driver1206800/producer1206891 CWD/fullargv/CUDA4/GPU_UUID95d925a1的510MiB离散样本（非peak）及log642B到终态796B已核，CPU1207091仅pipeline/gpuNone与代码CUDAempty证据、未独立live捕获，不补遥测重跑；终态所属进程/GPU空、foreign0。sourceNV/Q历史曝光、single seed/RX/日期源重合、正式RX未读、无targetDA/持续多session注册均保留；原predicted重点候选和pooled冻结对照保持，完整Phase3目标ACTIVE。详见 `docs/PAMF_PHASE3_SIGNAL_REGISTER_RESULTS_20261031.md`。


### 2026-10-31 层次密度发现与匿名注册预登记

Run `20261031-phase3-hier-register-s2026092701-r01` / actualcode `d6f3a19fe06252c2e4b9ed199939896b4504ab62`，唯一主predicted_REG1，原两base REG0精确/对应REG1共四列×三DURA；HDBSCAN固定minsize16/minsamples8/eom/brute/alpha1/eps0/allowSingleFalse/copyTrue/njobs1/freeK，不sourceε切割、不未知真K/配额。strictmajority/size16/fullmembersGaussian/no1K/同sourceVknown12校准与共同竞争不改。actual三CP synthetic2→e48原sourceaggregate和U四公开特征cache（ignoreoldlabels/results）→新anonfreeze→V两globalτ+旧two6τexact→全部4x3pred→CPU U/sourceNV/两Qtruth；U-only映射、所有真实Uarrivals入AN、分母/缺类口径不变。newNN/realGFsignal/sourceL/sourceεfit0，旧source6300+U384前向/模型/GMM/Q缓存成本继承；当前targetU算法消费384真实缓存特征、anon U-derived，非source-only总状态。v3 AN80/oldrel和abs2pp/old及预期registeredCK90/laterclean95star90/scenecoverage，sourceknown12保护；仅新增竞争不能修复旧AK/laterRU，AN单项不宣称全目标成功，后续无标签适应/联合拒识/持续多session仍保留。历史NV/Q/单RXseed/source日期重合/high缺19-15NA/正式RX未访问保持，完整目标ACTIVE，当前无新性能。见 `docs/PAMF_PHASE3_HIER_REGISTER_PLAN_20261031.md`。


### 2026-10-31 层次密度匿名注册结果与现有优良基线保留

Run `20261031-phase3-hier-register-s2026092701-r01` / actualcode `d6f3a19fe06252c2e4b9ed199939896b4504ab62`，四方法×三DURA预测和独立U/sourceNV/cleanQ/starQ四truth评分全部COMPLETE，执行VERIFIED、科学NO_PROMOTION，完整Phase3目标ACTIVE。按用户“先选取前面性能好的方法”，继续固定原source-only r03＋预测类条件密度`predicted_ratio`为重点候选，`pooled_ratio`为冻结对照；新注册候选未替代基线。原跨RXclean/star baseline宏AK/CK/RU/H分别70.833333/87.326389/100/82.926829%、64.062500/85.590278/77.083333/69.972325%，均接冻结DURA；clean未知拒识单项通过不能掩盖星地拒识、CK及same-DURA原F0闭集2pp保护未过。

仅将发现算子改为固定HDBSCAN(minsize16/minsamples8/eom/brute)，不设真实K/配额。U384原公开特征缓存重用；raw簇190/21/28/27/31，noise87，strictmajority/size16仅21条簇合格。独立U评分确认该簇全部来自19-15，purity100%、旧类污染0、anon0→label6；未发现另一真实到达设备19-16，不能把一簇纯度写成完整发现或未来识别率。实际old6+anon1，预期注册集合old6+全部arrivals2中的未发现类仍入指标分母。无targetDA、无持续多session完成。

主predicted_REG1 source/clean/star oldAK为89.703704/70.833333/64.062500%，oldCK96.948148/87.500000/87.152778%，later-only RU50.571429/66.666667/60.416667%，H64.679240/68.686869/62.186193%，完整两arrival宏AN6.142857/47.916667/33.854167%；同DURA原F0闭集损1.277778/9.201389/8.854167pp。19-15 futureclean92/96=95.833333%、star65/96=67.708333%，19-16均0，故不得用单类数替代宏AN。sourceknown12通过仅经验校准；目标全门槛仍失败，high缺19-15的AN/预期注册CK/HKN/HOW保持NA，不认证完整六景。

主REG1对原REG0同ID：clean/star旧正确均gain0/loss0；接收分别新增1/9条且均错。19-15新增正确92/65；同一later7-1拒对clean96→64(-32)、star76→58(-18)，即100→66.666667%、79.166667→60.416667%；source旧正确损11、later拒对损220。REG0主RU77.083333%分母含全部三非旧类，REG1later主60.416667%只含7-1，二者不能当同集合配对；实际仍未注册19-16与7-1的star宏拒识67.187500%另列。仅固定旧margin上新增竞争的结构不能提高旧正确或同later拒识，当前部分注册收益不能替代原综合目标。

原F0/defaulthead、signalr03、G、sourceGaussian/normalizer、两旧六τ保持；newNN/真实GFsignal/sourceLreads/sourceεnewfit均0，U算法消费384个缓存公开特征。原sourceε仅祖先记录不用作聚类，原sourceL6300+U384前向和F0/r03/G训练、Q缓存成本继承，actualCP synthetic各2先smokePASS。新增anonGaussian1/1484B、旧density364340B、祖先radius156B；两globalτ=-280.6977305402709/-281.16204591272106，old12τ96B+new2τ16B=112B，NN834665参数；逐行source/U bank不部署。

source/sourceNV、U/Q角色、旧六及三非旧TX、RX24-1/index26、U03-08/Q03-15、G10单固定received和三DURA路径保持；sourceNV/Q历史曝光、单seed/RX/source日期重合、正式RX24-18未读。predict47.769382s/CPUscore12.666512s/pipeline60.521157s为进程时间非B1。独立driver1234212/producer1234312 CWD/fullargv/CUDA7及GPU_UUID9c4eb614的406MiB单样本已核（非peak），log525→675B；CPU1235017仅pipeline/gpuNone与代码CUDAempty证据、无独立live捕获，未补遥测重跑。终态owned进程/GPU空、foreign0。10unique+45sub已通过/一次P0P1PASS；653files/data0单archiveSHA/compile/runtimeVERIFIED。详见`docs/PAMF_PHASE3_HIER_REGISTER_RESULTS_20261031.md`。


### 2026-10-31 冻结 GMM 类别竞争拒识预登记

Run `20261031-phase3-class-competition-s2026092701-r01` / actualcode `b30db356419342dd15ecf811e444916aa1e5c4df`，四列pooled_ratio/predicted_ratio/class_competition/hybrid_competition，唯一主最后。原sourceF0/defaulthead/G/r03/known90/OE32/白化及两old6tau保持；每类15等质量ℓ_j，b_c=logmeanother，诊断ℓ_c−b_c、主ℓ_c−logaddexp(ℓOE,b_c)+log2，背景0.5/0.5/T1固定，不把旧OVA等竞争泛称首次。原F0 c不变、不densityargmax/另训身份头。actual3CP synthetic2首步→公式与exactdensity冻→源V27000两new6tau/旧two6tau exact→全τ冻→NV3150/clean864/star864三DURA全四列pred→CPU3truth-last。newNN/density/normalizer/真实GFsignal/源IQ/U0，原source训练/cache和统计成本继承；Vonly新12标量实际fit。gate保留sourceknown12、clean95/star90/CK90/同DURA闭集损2pp、主景coverage及主starAK/RU不低于原predicted；source失败不取消Q。历史NV/Q/单RXseed/source日期重合/high未知缺19-15 NA、正式RX未读不变，无targetDA/发现/注册，AN/HOW=N/A；原重点候选和对照保持，完整目标ACTIVE，当前无新性能。见`docs/PAMF_PHASE3_CLASS_COMPETITION_PLAN_20261031.md`。


### 2026-10-31 冻结 GMM 类别竞争拒识完整结果

Run `20261031-phase3-class-competition-s2026092701-r01` / actualcode `b30db356419342dd15ecf811e444916aa1e5c4df`，四列×三DURA预测及独立三truth评分COMPLETE，执行VERIFIED、科学NO_PROMOTION。唯一主hybrid_competition source/clean/star宏AK/CK/RU/H分别89.688889/96.514815/61.301587/72.826729%、68.576389/85.069444/100/81.359423%、62.673611/83.506944/76.736111/68.995607%；sameDURA原F0闭集损1.292593/11.458333/10.243056pp。源known12通过不代替跨RX综合门槛，星地CK/RU/原F0保护及必需场景仍失败。

对原predicted_ratio：source旧正确18增33失/net−15、unknown拒对33增53失/net−20；clean旧正确0增13失/net−13、unknown拒对不变；star旧正确0增8失/net−8、接收0增12失/net−12、unknown拒对1增2失/net−1，AK−1.388889pp/RU−0.347222pp/H−0.976718pp，不能替换原重点候选。仅known竞争诊断starAK64.583333/CK86.979167/RU22.569444/H33.449535%，局部旧AK微增不能掩盖未知大量误接收，更不能把类间自信等同开放集已知证据；保持原predicted_ratio重点候选与pooled_ratio冻结对照，不据Q重调0.5/T1、换主或选择性重跑。

原F0默认c、G/r03、known90/OE32/normalizer及两old6τ精确保持；主密度背景0.5OE+0.5otherknown、诊断无OE，只新公式而无新NN/Gaussian/normalizer统计参数。sourceL/OE/Us/U/真实GFsignal新读及前向0，actual三CP synthetic2each首步PASS；V27000额外knownlabels新校准two6τ=12标量96B，旧12exact，总24τ192B，原density364340B/NN834665、source训练及cache成本继承，无逐行source/U bank部署。U/正式RX24-18不访问，无DA/发现/匿名REG，AN/HOW=N/A；历史sourceNV/devQ、单seed/RX/source日期重合/high缺19-15之RU/H NA保持，完整目标ACTIVE。

predict17.873421s/CPUscore15.735417s/pipeline33.693221s为完整进程非B1。独立driver1253032与CPUscorer1253692的CWD/fullargv/CUDAempty、predict日志0→674B/score47B/driver21B及终态live/owned空已核；producer1253122/GPU4只有pipeline配置/产物和实际smoke证据，未独立捕获predictorlive或显存样本，不虚报GPU/peak、不补遥测重跑，foreign0。9unique+30sub/一次P0P1PASS，661files/data0单archiveSHA/compile/runtimeVERIFIED。详见`docs/PAMF_PHASE3_CLASS_COMPETITION_RESULTS_20261031.md`。


### 2026-10-31 全局正对角仿射无标签适应预登记

Run `20261031-phase3-diagonal-uda-s2026092701-r01` / actualcode `73f460c2e0c89b192e52af2e1b6ef01a576ca686`，六列原pooled/predicted、SHIFTpooled/predicted、DIAGpooled/predicted，唯一主最后。原F0/defaulthead/G/r03/known90/OE32/白化/two6tau不改，sourceidentity、新RX两Q用同starUstate；h=a*w+b正a/no postunit，SHIFTb160 vsDIAGa/b320各identity-start20MAPEM，mix0.5/0.5、Jacobian及lambda160固定，非旧rank14或类平移/源cov积分。actual3CPsynthetic2→原sourcefreeze→旧d72适应前r03 publicU4fields384→两新20fit/freeze→sourceV/NV及clean/starQ6x3pred→独立CPU3truth。U真实TX/K/配额/logits/Q不fit，旧adapter/prior/results不继承，无newtau/NN/GMM/normalizer/真实GFsignal/sourceIQ，U新统计fit40步和320/160参数单列；original12tau96B重复六列288B。gate sourceknown12/clean95star90/CK90/原F0闭集损2pp/主景coverage及starAKstrictgain/RUnotbelowoldpred；不以U似然收益替代future。sourceNV/Q历史曝/单seedRX/source日重合/high缺19-15NA/正式RX未读保持，无发现/匿名REG，AN/HOW N/A，全目标ACTIVE，当前无新性能。见`docs/PAMF_PHASE3_DIAGONAL_UDA_PLAN_20261031.md`。


### 2026-10-31 正对角无标签适应完整结果

Run `20261031-phase3-diagonal-uda-s2026092701-r01` / actualcode `73f460c2e0c89b192e52af2e1b6ef01a576ca686`，两新20步适配、六列×三DURA预测及独立三truth评分COMPLETE，执行VERIFIED、科学NO_PROMOTION，完整目标ACTIVE。主diagonal_predicted_ratio的source同原predicted89.744444/96.759259/61.936508/H73.291437%；cleanAK73.611111/CK90.625000/RU98.263889/H84.169473%、同G闭集损6.423611pp，CK/RU单项过线但2pp保护失败；starAK66.666667/CK89.236111/RU61.805556/H64.144144%、同G闭集损6.25pp，未达绝对联合目标及拒识不退步。source旧RX不应用目标adapter，原两列按六列重复，不能称源域也获得适应收益。

对原predicted，clean旧正确16增0失、接受19增0失（新增3条错身份）、未知拒对0增5失；star旧正确15增0失、接受21增0失（新增6条错身份）、未知拒对0增44失。star AK+2.604167pp/RU−15.277778pp/H−5.828181pp；这是恢复部分原F0正确样本的接收，不是身份头纠错，更不等于拒识受保护的UDA成功。只平移SHIFTpred starAK仍64.0625%、RU77.430556%，不可按其单项比主高或cleanH微增而换主/调lambda/迭代/混合权重。原predicted重点候选及pooled冻结对照保持。

完整42条EM记录有限非下降：同初始目标181695.05281440573，SHIFT终182980.2905443705/offsetnorm1.9081906559501747；DIAG终186962.9954438197/正逆尺度0.6649160885532033至1.3917106154662513/offsetnorm1.882684410540968，knownposterior0.5075638714→0.5859596224不是U真实known比例。U似然提高不能替代future联合性能；Utruth始终0，不据其猜簇/设备构成。lambda160固定identityprior/Jacobian在fit中保留，ratio中抵消，no postunit；旧source Gaussian/whitener/F0/defaulthead/r03/G和two6tau全不变。新U统计40步、SHIFT160/DIAG320有效参数；adapter数组各2720/2716B（数值各2560B），两old6tau12唯一96B、六列存储重复288B/单主48B，NN834665/density364340B及训练/cache成本继承。newNN/Gaussian/normalizer/tau/newrealGFsignal/源L-OE-Us IQ均0，actual3CPsynthetic2each首步PASS，target状态U-derived非sourceonly，source/U行bank不部署。

输入仅原d72适应前r03四公开U384特征，旧adapter/prior/results不继承；U RX24-1/03-08与future同RX03-15/Q864及source物理互斥，三测试均DURA、clean仅冻结测试。历史sourceNV/Q、单seed/RX/source日期重合/high缺19-15宏RU/H NA/正式RX24-18未读保持；无发现/匿名REG，AN/HOW N/A，全目标ACTIVE。predict9.470567s/CPUscore4.272151s/pipeline13.853247s是进程时间非B1。独立driver1268991/CPU1269163 live-CWD-fullargv-CUDAempty及log0→predict627B/score42B/driver21B、终态owned空已核；predictor1269086/GPU4仅pipeline/实际smoke/产物证据，未独立捕获predictorlive/显存，不虚报GPU峰值或补遥测重跑，foreign0。9unique+30sub unionPASS（两fixture定向修复保留首失败）、AST7/mirror8/唯一P0P1PASS；669files/data0单archiveSHA/compile/runtimeVERIFIED。详见`docs/PAMF_PHASE3_DIAGONAL_UDA_RESULTS_20261031.md`。


### 2026-10-31 固定未知背景的选择性适应预登记

Run `20261031-phase3-selective-uda-s2026092701-r01` / actualcode `0f1fda4c8c543394a938bd75dcb5cadbc29c6491`，六列原pooled/predicted、oldSHAREDpooled/predicted、newSELECTIVEpooled/predicted，唯一主最后。原F0/defaulthead/G/r03/known90/OE32/白化/two6tau保持；pU=.5pKnown(a*w+b)prod(a)+.5pOE(w)，每样本同时评估两分支，不oracleknownunknown路由。knownsoftrespons/Neff Mstep、knownJac不取消，λ160/newidentitystart20固定；旧73f DIAG320/20只matchedcontrol不初始化，fixedOE是否保护RU待验证。actual3CPsynthetic2→source/control聚合及Ucacheheader→sourcefreeze→原sourceonly r03 U4features384→new20fit/freeze→V/NV/clean/starQ6x3pred→CPU3truth/Utruth0，旧Q/score不fit。sourceidentity/newRX两Q同starU，newNN/GMM/normalizer/τ/真实GFsignal/sourceIQ0，newtargetstats20/320、旧shared20及上轮full40继承；old12tau96B重复288B/单主48B。gate sourceknown12/clean95star90/CK90/同G闭集损2pp/主景coverage/starAKstrictgain和RU不低于oldpred保持；sourceNV/Q历史曝/单RXseed/source日重合/high缺19-15NA/正式RX未读，无发现REG，AN/HOW N/A，完整goalACTIVE，当前无新性能。见`docs/PAMF_PHASE3_SELECTIVE_UDA_PLAN_20261031.md`。


### 2026-10-31 选择性已知密度适应完整结果

Run `20261031-phase3-selective-uda-s2026092701-r01` / actualcode `0f1fda4c8c543394a938bd75dcb5cadbc29c6491`，新SELECTIVE20步、旧SHARED20步控制复用、六列×三DURA预测与独立三truth评分COMPLETE，执行VERIFIED、科学NO_PROMOTION，全目标ACTIVE。主source原pred不变89.744444/96.759259/RU61.936508/H73.291437%；cleanAK74.826389/CK91.666667/RU98.263889/H84.958347%、同G原F0闭集损5.208333pp；starAK66.319444/CK88.888889/RU63.194444/H64.719243%、闭集损6.597222pp。cleanCK/RU单项过线不能替代2pp保护；star已知保护、CK、RU、场景与拒识不退步仍未过。

对原pred：clean正确23增0失、接受25增0失（另2条错身份）、unknown拒对0增5失；star正确13增0失、接受19增0失（另6条错身份）、unknown拒对1增41失/net−40，AK+2.256944pp/RU−13.888889pp/H−5.253081pp。对同320参数20步SHARED：clean正确8增1失/net+7、unknown4增4失/net0；star正确2增4失/net−2、unknown19增15失/net+4，即RU+1.388889pp但AK−0.347222pp，不能因比弱控制略高或cleanH升而换主/晋级。保持原predicted重点候选与pooled对照，不据Q调λ/组权重/迭代或选择性重跑。

pU=.5pKnown(a*w+b)prod(a)+.5pOE(w)，每U/Q同时算两密度，无trueknownunknown路由。OE输入不变仍不能保护未知拒识；这不是已解决UDA，更不能把旧正确接收恢复称F0身份纠错。仅knownsoftrespons/Neff参与M步、knownJac不取消/no postunit/λ160固定、新identitystart20；旧targetSHARED仅控制不初始化，原F0/defaulthead/G/r03/known90/OE32/normalizer/two6tau不改。新21条trace有限非下降，目标181695.05281440576→183843.67566550476，逆尺度0.6481322740055696至1.3438724975871188、offsetnorm1.8977789113737333、knownpost0.5075638714→0.6516592685；后者不是U真实known比例/设备构成，Utruth0。新targetstats20/320参数，SELECTIVE数组2740B、继承SHARED2716B（各数值2560B），source NN834665/density364340B、原12唯一τ96B/重复288B/单主48B及训练/cache/旧runfull40成本保留；newNN/GMM/normalizer/τ/真实GFsignal/sourceIQ0，actual3CPsynthetic2each首步PASS，source/U行bank不部署，整个targetUderivedstate非sourceonly。

原r03适应前U384/cache沿73f←d72，U RX24-1/03-08与samefutureQ864/03-15/source物理互斥；三测试均DURA、clean仅冻结测试。sourceNV/devQ历史曝、单seed/RX/source日重合/high缺19-15宏RU/H NA/正式RX24-18未读保持，无发现/匿名REG，AN/HOW N/A，全目标ACTIVE。predict10.401474s/CPUscore5.277220s/pipeline15.790489s为进程非B1；独立driver1290055/predictor1290081 CWD/fullargv/CUDA1与GPU_UUIDa2d039ff的256/406MiB两离散采样及log492→522→636B已核（非peak），CPU1290333仅pipeline/gpuNone/代码CUDAempty与完成产物、未捕获独立live，不补遥测重跑；终态owned空foreign0。9unique29subPASS/AST7mirror8/唯一P0P1PASS，677files/data0单archiveSHA/compile/runtimeVERIFIED。详见`docs/PAMF_PHASE3_SELECTIVE_UDA_RESULTS_20261031.md`。


### 2026-10-31 多模态共同方差密度比预登记

Run `20261031-phase3-tied-density-s2026092701-r01` / actualcode `9a80e2bef7eee4d8c2ec4d4725e0ca552552443b`。沿用户选定predicted_ratio基线做单候选假设检验：known90/OE32评分统一为原pooled_L_variance，中心/权重/白化及原11arrays冻结，query共有二次项/体积/Jacobian抵消；不称任意域偏移不变，也不称首次相对距离。固定四列oldpooled/oldpredicted/tiedpooled/tiedpredicted，唯一主最后。actual3CPsynthetic2→scorepolicy冻→V27000两new6tau+两oldexact→全τ冻→NV3150/clean864/star864三DURA全pred→独立CPU3truth。新NN/variancefit/densityfit/realGFsignal/sourceIQ/U0；12newtau实际fit、旧12复用192B，祖先成本继承。sourceknown12/clean95star90/CK90/原F0同G损2pp/主景coverage及starAK/RU不低于oldpred固定，source失败仍Q。历史NV/Q/单seedRX/source日期重合/high缺19-15宏RU/H NA/正式RX未读不变；无DA/发现/REG，AN/HOW N/A。原所选基线和延后的native-energy状态保持，完整目标ACTIVE；当前无新性能。见`docs/PAMF_PHASE3_TIED_DENSITY_PLAN_20261031.md`。


### 2026-10-31 多模态共同方差密度比完整结果

Run `20261031-phase3-tied-density-s2026092701-r01` / actualcode `9a80e2bef7eee4d8c2ec4d4725e0ca552552443b`，四列×三DURA预测和独立三truth评分COMPLETE，执行VERIFIED、科学NO_PROMOTION。唯一主tied_predicted_ratio source/clean/star宏AK/CK/RU/H分别89.625926/96.388889/63.301587/74.198073%、72.048611/88.888889/98.611111/83.262688%、63.888889/83.680556/76.736111/69.725652%；原F0同G闭集损1.355556/7.986111/9.027778pp。源known12和cleanRU单项不替代跨RX综合门槛；星地AK/RU相对所选pred均下降，不能晋级。

对原pred同ID：source正确90增122失/net−32、unknown拒对254增211失/net+43；clean正确18增11失/net+7、unknown0增4失/net−4；star正确10增11失/net−1、unknown18增19失/net−1。starAK−0.173611pp/RU−0.347222pp/H−0.246673pp，cleanAK+1.215278pp/RU−1.388889pp/H+0.335859pp。原F0身份完全不变，均是接收集合差异；不按Q调s或重选运行点，原predicted_ratio主基线及pooled冻结对照保持。

固定共同s=原pooled_L_variance0.54650927979218，两侧query共有二次项/体积/Jacobian抵消；known90/OE32中心/权重/白化11arrays逐值不变，新评分分布明确不同。newNN/variancefit/densityfit/normalizer/realF0Gsignal/sourceIQ/U0；V27000实际新2×6τ，旧12精确复用，总24/192B，祖先NN834665/density364340B及训练缓存成本继承。真实3CPsynthetic2each首步PASS，不部署source/U行bank。该单候选未支持预期星地收益，不推广为所有共同方差或域消除方法无效。

三DURA路径、sourceL6300/隐藏Us56700/OE6134/V27000/NV3150、source五RX三日、目标RX24-1/03-15同864物理Q（old576/unknown288）保持；sourceV校准非独立test，NV/Q历史曝光、单seed/RX/source日期重合/high缺未知19-15宏RU/H NA、正式RX24-18未读。无DA/发现/匿名REG，AN/HOW=N/A，完整目标ACTIVE。独立driver1323759/predictor1323782 CWD/fullargv/CUDA4/GPU_UUID95d925a1及10/406MiB离散样本（非peak）、log487→629B已核；CPU1324080仅pipeline/gpuNone/代码CUDAempty和产物，未独立捕获live；终态owned空foreign0。10unique34subPASS/workerAST6+launcherimport/一次P0P1PASS；688files/data0单archiveSHA/compile/runtimeVERIFIED，详细报告`docs/PAMF_PHASE3_TIED_DENSITY_RESULTS_20261031.md`。


### 2026-10-31 非平衡传输混合先验适应预登记

Run `20261031-phase3-uot-prior-s2026092701-r01` / actualcode `c0b7cc09a808e832d4d64a18080239e72e910250`。沿所选predicted_ratio检验UOT目标结构先验适应，非完整UniOT复现。source122模式到starU384，costwhitedist²/(160variance)，a=.5known+.5OE/buniform不是U真比例；eps.05/rho1/固定200，knownmass跨六类汇为shared15RXday、OE32分别归一，均.5source+.5learned；新target47权重/45DOF，原F0/defaulthead/G/r03/mean/variance/白化/density11/two6tau不改。actual3CPsynthetic2→sourcefreeze→d72原sourceonlyUcache只featuresfit1→adapterfreeze→sourceidentity/pairedcleanstar sameadapter全4x3pred→独立CPU3truth/Utruth0，source失败仍Q。newNN/sourceGMM/tau/realGFsignal/sourceIQ0，200targetstat与祖先成本分列；source12uniqueτ96B/重复24值192B。sourceknown12/clean95star90/CK90/同G原F0损2pp/主景coverage及starAKstrictgain/RU不低于oldpred固定。历史NV/Q/单seedRX/source日期重合/high缺19-15宏RUHNA/正式RX未读不变，无未知发现或匿名REG，AN/HOW N/A；原baseline和延后native-energy状态保持，完整目标ACTIVE，当前无新性能。见`docs/PAMF_PHASE3_UOT_PRIOR_PLAN_20261031.md`。


### 2026-10-31 非平衡传输混合先验适应完整结果

Run `20261031-phase3-uot-prior-s2026092701-r01` / actualcode `c0b7cc09a808e832d4d64a18080239e72e910250`，四列×三DURA预测和独立三truth评分COMPLETE，执行VERIFIED、科学NO_PROMOTION。主UOT_predicted_ratio宏AK/CK/RU/H：source: 89.744444/96.759259/61.936508/73.291437%；clean_DURA: 70.659722/87.152778/100.000000/82.807731%；star_DURA: 64.062500/85.590278/77.430556/70.115031%；同DURA原F0闭集损1.237037/9.375000/8.854167pp。source与原pred全ID不变；clean正确0增1失/net−1、unknown拒对0；star known接收及正确接收全0变化、unknown拒对1增0失/net+1。starRU+0.347222pp/H+0.142706pp但没有预登记starAKstrictgain，绝对门槛和coverage仍未过，不替换所选predicted_ratio主基线或pooled对照。

source122→starU384一次200步logUOT（eps.05/rho1）完结，KKT2.1826436214e−9/scaling2.2930457533e−9；sourceknownmass跨class汇15域/OE32分别归一后各.5source+.5learned，47权重45自由度/数值376B/含schema数组504B。means/vars/白化/原11arrays/NN/two6tau均不改；sourceprior适配只用于新RX，两Q同starUstate，entireadapter target派生非source-only。known transport fraction.6356480不是U真实设备比例。新增NN/sourceGMMgeometry/τ/realGFsignal/sourceIQ0但新targetpriorfit1/200，祖先NN834665/density364340B/训练缓存成本继承；12uniqueτ96B重复24值192B/单主48B，actual3CPsynthetic2各首步PASS。

仅由U派生先验与源聚合得到的全输入分数变化界为[-0.913254410,0.702078218]；known域权重相对uniform仅0.9494521至1.0394007，当前先验适配没有改变表征域偏移，不能因1条未知改善宣布UDA完成，也不据Q调eps/rho/blend或扩种子。sourceL6300/Us56700/OE6134/V27000/NV3150、source5RX3day、U RX24-1/03-08与future同RX03-15的864物理Q、三DURA/clean只测试保持；NV/Q历史曝光、单seedRX/source日期重合/high缺19-15宏RU/HNA、正式RX未读不变。无未知发现/匿名REG，AN/HOW=N/A，完整目标ACTIVE。

独立driver1341003/predict1341026 live/CWD/fullargv/CUDA1/UUIDa2d039ff及两次406MiB离散样本（非peak）、日志472→497→599B已核；CPU仅pipeline/gpuNone、代码CUDAempty及产物，未独立live，终态owned空foreign0。11unique33sub/workerAST6+launchimport/唯一P0P1PASS，697files/data0单archiveSHA/compile/runtimeVERIFIED。详见`docs/PAMF_PHASE3_UOT_PRIOR_RESULTS_20261031.md`。


### 2026-10-31 目标原型开放投影适配预登记

Run `20261031-phase3-target-prototype-s2026092701-r01` / actualcode `35d41680b2039452562baa76d407e7d0ef0dacdb`。两独立r03 SOURCE/TARGET_PCD各444 freshAdamW1e-4/wd1e-4/普通clip5，仅projection102688更新、encoder232033/Sinc/F0head/G冻结；source原crossset/OE＋KD10，source192/64逐步matched，SOURCE先完才开U。TARGET384星地U经原G一次，父encoder缓存weak和两固定phasegain view；K16×160aux prototype seed2026103113/25sphericalLloyd，balancedSinkhorn20/eps.05/T.1 currentweakq detach，strong自身q/global及固定父NNq/local，target权重.1，Ubatch64seed2026103112/aug3114；不是完整SwAV/UniOT，不估计真实K或注册。actual3CPsynthetic→两NN训练冻→各ownsourceGMM冻→V4new6tau/old2exact冻→NV3150/clean864/star864六列三DURA全pred→CPU3truth/Utruth0，source失败仍Q。唯一PCD_predicted_ratio；sourceknown12/clean95star90/CK90/原F0同G损2pp/coverage及starAK高于E0与SOURCEpred且RU不低于两control固定。新888projectionupdates/aux2560，encoder原13320不加444，projection2220到2664；sourceparentcache12434/Uparent1152/G384、两newGMM/24newtau和祖先成本单列，整TARGET含U祖先非sourceonly，prototype/训练bank不部署。历史NV/Q/单seedRX/source日期重合/high缺19-15宏RUHNA/正式RX未读保持，U已无标签接触设备不冒称严格未见；AN/HOW N/A，原所选baseline及native-energy延后状态保持，完整目标ACTIVE。见`docs/PAMF_PHASE3_TARGET_PROTOTYPE_PLAN_20261031.md`。


### 2026-10-31 目标原型开放投影适配完整结果

Run `20261031-phase3-target-prototype-s2026092701-r01` / actualcode `35d41680b2039452562baa76d407e7d0ef0dacdb`，SOURCE/TARGET_PCD各444神经更新、六列×三DURA预测和独立三truth评分全部COMPLETE，执行VERIFIED、科学NO_PROMOTION。主PCD_predicted_ratio宏AK/CK/RU/H：source_DURA: 89.740741/96.814815/64.571429/75.103446%；clean_DURA: 70.312500/86.805556/100.000000/82.568807%；star_DURA: 64.930556/86.631944/76.041667/70.048235%；同DURA原F0闭集损1.240741/9.722222/7.986111pp。对E0pred：source正确28增29失/net−1/未知134增51失/net+83，clean正确5增8失/net−3/未知0，star正确8增3失/net+5、接收net+6/未知2增5失/net−3。对SOURCEpred：star正确6增10失/net−4、未知13增1失/net+12；主AK高于E0但低于SOURCE，RU高于SOURCE但低于E0，绝对联合门槛和coverage仍失败，不以H微增或较弱control晋级。原所选predicted_ratio/pooled保持。

仅projection102688各更新444，encoder232033/Sinc/F0/defaulthead/G/原r03/buffers均不变；SOURCE先于U完成并source-only，TARGET三viewU384学习16×160aux原型和projection，原型不是真实设备数且不部署。source原crossset/OE+KD10、target.1(global+local)、源192L/64OE索引匹配，父sourcecache12434/U三view1152总13586、G384/F0real0，学生projection总454656行。encoderfeature累13320不加444，projection2220→2664，原F0/头/G/native/Sinc更新0。两new源GMM/4new六τ、旧2组六τexact，共36τ288B；TARGET派生密度含U祖先不冒称纯源，无V/NV/Q神经训练。

原数据sourceL6300/Us56700/OE6134/V27000(extra sourcecal)/NV3150、source5RX3day、starU24-1/03-08与future同RX03-15同864物理Q（old576/nonold288）保持；两臂实际NN训练unique各L6300/OE5913，密度拟合仍全OE6134。全部三DURA，clean仅测试，U无标签接触设备不冒称严格未见。历史NV/Q/单seedRX/source日期重合/high缺19-15宏RU/HNA/正式RX未读不变；无匿名REG，AN/HOW=N/A，完整目标ACTIVE。全部888步与18点/分组/成本详见`docs/PAMF_PHASE3_TARGET_PROTOTYPE_RESULTS_20261031.md`，不以训练loss下降代替测试成功。

实际train79.466468s/predict39.882487s/CPUscore13.978042s/pipeline133.502019s是过程时间非B1。独立driver1375698/train1375719/predict1377133 live-CWD-fullargv-CUDA1/UUIDa2d039ff，20/406MiB训练初段和446MiB预测离散样本非peak，日志train517→1485/pred769→1655B；CPU1378094未独立live，终态owned空foreign0。23测试例25子项/workerAST8/2CLI+launcherimport/一次P0P1PASS，708files/data0单archiveSHA/compile/runtimeVERIFIED。Git登记时HEAD并发推进已在启动前无破坏恢复，不改实际code35d身份/不重写别人历史，实际三阶段无技术失败。


### 2026-10-31 冻结Sinc的目标原型骨干适配预登记

Run `20261031-phase3-body-prototype-s2026092701-r01` / code `a33612e978948da25bd733bcb9edba3ac9d74e9a`，主PCD_predicted_ratio，SOURCE/TARGET各444，从原r03独立初始化；只固定Sinc48/buffers，body231985＋projection102688更新。源原crossset/OE＋KD10，source indices matched；同上轮target.1 global/local/K16/全部seeds，SOURCE先完成才开U384。全部三DURA/六列预测后独立truthlast；原4af基线保持，目标/门槛不降低。真实学生IQ454656、父12818、G384；完整方案与数据/祖先/指标/停止规则见`docs/PAMF_PHASE3_BODY_PROTOTYPE_PLAN_20261031.md`。无匿名REG或formalRX，完整目标ACTIVE。


### 2026-10-31 原型骨干技术失败与冻结评估恢复

r01在a336下两臂444训练全部完成，但首次eval加载把6个nonpersistent缓存误判为缺失，FAILED_PRESERVED/无性能结论，未读Q/truth。实际权重persistent/Sinc差异0；一次定点RED→GREEN修复后独立r02 `20261031-phase3-body-prototype-s2026092701-r02` / evalcode `4273e88392abe0717c1bd94fca1370d19c67cbb3` 精确复用旧两CP，训练仍a336/r01，不重训或改科学配方。r02只predict→CPUscore/新optimizer0，继承888训练成本，原六列三DURA/τ/数据/主读出/门槛不变；原基线保留，完整goalACTIVE，匿名注册未完成。见`docs/PAMF_PHASE3_BODY_PROTOTYPE_RECOVERY_20261031.md`。


### 2026-10-31 原型骨干恢复后完整结果

Run `20261031-phase3-body-prototype-s2026092701-r02` / evalcode `4273e88392abe0717c1bd94fca1370d19c67cbb3`，精确复用r01/a336两臂444训练，r02新optimizer0；首次加载故障一次定点修复后实际smoke、六列三DURA全预测和独立3truth评分均VERIFIED完成。主PCD_predicted_ratio宏AK/CK/RU/H：source_DURA: 89.740741/96.729630/65.523810/75.743693%; F0loss1.240741pp；clean_DURA: 73.090278/89.756944/99.305556/84.204711%; F0loss6.944444pp；star_DURA: 65.277778/87.152778/78.125000/71.125908%; F0loss7.638889pp。star对E0pred正确+7/未知+3，对同预算SOURCEpred正确+5/未知+8，严格AKgain两对照和RU均不降通过，sourceknown12通过；但绝对联合CK90/starRU90/同G原F0损2pp/coverage失败，科学NO_PROMOTION，原所选baseline保持，不能以相对改善宣布完整UDA成功。clean对E0正确+13/未知−2，源正确−1/未知+113，源V校准复用非known盲测。

Sinc48/所有buffers、F0/defaulthead/G/r03保持，body231985+projection102688各444×2实训练且actualchanged；训练signal父12818/学生454656、eval90352，总557826/G2112/F01728，祖先训练成本和r01失败IO分列。仅U384无标签学习、V/NV/Q不神经训练，全部τ冻后全pred再3truth，Utruth0；TARGET权重和密度含U祖先，不冒称sourceonly。历史Q/单RX单seed/日期重合/high缺19-15宏RUHNA/正式RX24-18未访问不变；无匿名REG、AN/HOWNA，完整goalACTIVE。

driver1415050/predict1415080及CPU1415842独立live/CWD/fullargv（predictCUDA1、CPUempty），预测GPU304/392MiB为离散非peak，log509→1597B；终态owned空/foreign0。完整source888曲线/18点/两配对/场景/实际成本与r01失败到r02修复证据见`docs/PAMF_PHASE3_BODY_PROTOTYPE_RESULTS_20261031.md`。另只读metadata发现非正式R_dev3-2早03-01/未来03-15+23具有容量，但尚未完成packet角色排除/建池，独立MIX场景覆盖不保证；不将它作为本轮测试证据。


### 2026-10-31 公共类传输对齐预登记

Run `20261031-phase3-common-transport-s2026092701-r01` / actualcode `4038105cb7f7c2defe75eb7fbfa825e7f932b40f`，只新训source-r03 TARGET_CCD444，SOURCE/PCD(a336/r01)与六列冻结readouts(4273/r02)精确复用，不从U适配权重初始化。原source+PCD配方/Sinc48/buffers/F0/G冻结保持，前100步同PCD，第101起额外.1CCD：源L父z六固定mean、reg.01/rho.5/gamma.7/mu.7/log400、adaptive填充/六beta及统计均值筛选，不读目标TX/K/配额；fillseed2026103120，其余源/U/aug/proto seeds保持并核全索引。神经冻→源GMM1冻→V12新τ/旧36exact全冻→八列三DURA全pred→独立CPU3truth/Utruth0，唯一CCD_predicted_ratio，starAK须高于E0/SOURCE/PCD且RU均不降并满足绝对门槛/coverage。新NN444/父12818+学生340992/eval44312，总398122signal/G2112/F01728，旧888等历史成本另列。原基线保持，不是完整UniOT/未知发现/匿名REG；formalRX未读/历史开发曝光/单RXseed/high缺类保持，goalACTIVE。见`docs/PAMF_PHASE3_COMMON_TRANSPORT_PLAN_20261031.md`。


### 2026-10-31 公共类传输对齐完整结果

Run `20261031-phase3-common-transport-s2026092701-r01` / actualcode `4038105cb7f7c2defe75eb7fbfa825e7f932b40f`，新TARGET_CCD444＋旧SOURCE/PCD control精确复用，八列三DURA全预测/独立3truth均VERIFIED完成、Utruth0。主CCD_predicted_ratio宏AK/CK/RU/H：source_DURA: 89.755556/96.785185/65.904762/76.002910%; F0loss1.225926pp；clean_DURA: 69.965278/86.631944/99.652778/82.210992%; F0loss10.069444pp；star_DURA: 64.583333/86.458333/77.430556/70.425835%; F0loss8.333333pp。对PCD：source正确+4/未知+12，clean正确−18/未知+1，star正确−4/未知−2（AK/CK/RU各−.694444pp、H−.700073pp）；比E0/SOURCE少量增益不替代胜PCD/绝对保护，sourceknown12通过而target保护和coverage失败，NO_PROMOTION/不扩seed，原基线和PCD比较保留，完整goalACTIVE。

全444 L/OE/U索引exact，前100标量最大total差1.516342e−4（跨GPU是可能非已证解释），101..444实344 CCD，各400UOT且有限；伪hist只分配到0/3/5，不当真类构成/纯度。新父signal12818/学生340992、eval44312，共398122/G2112/F01728；新增444、旧888继承、1GMM/12新τ+36旧τ。Sinc48/buffers/原F0默认头/G不变，无匿名REG/AN-HOW NA/正式RX未访问。完整训练/24点/三配对/分组/成本/发布前P1修复及Win206本地传输修复见`docs/PAMF_PHASE3_COMMON_TRANSPORT_RESULTS_20261031.md`。

source-only聚合几何诊断说明六质心并非近共线、gamma.7源L阳性99.460317%/OE8.493642%，源域区分不能证明目标纯度。未来3-2 metadata审计在排除原known/OE/NV/formal共34TX后列5-5/10-7/2-19/7-11四未来未知候选，早期也有记录须从U整类排除；尚未完成packet历史排除/新池构建，不进当前实验，不宣称独立盲测。

## 2026-10-31：已有方法选择更新

按用户“先选取前面性能好的方法”，结合已经完成的三类 DURA 比较，将 `PCD_predicted_ratio` 选作下一步研究起点，原 source-only `predicted_ratio`、同预算 SOURCE 和 pooled 读出保留对照。所选 PCD 的跨接收机 clean A_K/C_K/R_U/H 为 73.090278%/89.756944%/99.305556%/84.204711%，星地为 65.277778%/87.152778%/78.125000%/71.125908%；星地四项均高于 E0/SOURCE 及后续 CCD。指标直接取自 body-prototype r02 冻结评分，训练仍为原 r01 的 444 步，不新增实验或调阈值。

这是依据历史开发结果更新研究起点，原实验的 NO_PROMOTION 和未达绝对目标不追改。所选 PCD 有目标 U384 的神经适配祖先，不能当作纯源权重；其他新 RX 的研究需从原 source-only r03 独立初始化。保留单种子、单目标 RX、历史曝光、逐景缺类和未完成发现/匿名注册的限制。clean R_U≥95%、星地 R_U≥90%、C_K≥90%、同 DURA 原 F0 闭集减 A_K≤2 个百分点不降低，完整目标 ACTIVE。方法与证据见 `docs/PAMF_PHASE3_METHOD_SELECTION_20261031.md` 和 `configs/pamf_phase3_selected_baseline_20261031.json`。


## 2026-10-31：新开发接收机固定 PCD 验证预登记

Run `20261031-phase3-fresh-development-rx3-2-s2026092701-r01` / actualcode `f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7`。本机 CPU 已完成 RX3-2 的早日星地 U576 和双未来日期 Q10238，共10814个唯一 canonical 记录；future clean/star同ID同顺序，U无clean，后来未知整类未进入U。接收机已核实排除source/compact/Phase2/旧开发/正式角色，但全项目packet历史未用仍UNKNOWN。PCD从原source-only r03独立训练444，source-only SOURCE444仅复用对照；六读出三DURA预测全冻后CPU三truth/Utruth0。仍按cleanRU95/starRU90/CK90/原F0同G损2pp及覆盖、相对两控制收益验收，低分不省略测试，无发现或匿名REG，完整目标ACTIVE。矩阵、角色、种子、成本与技术停止规则见`docs/PAMF_PHASE3_FRESH_DEVELOPMENT_PLAN_20261031.md`；公共句柄与精确命令见本run登记。


## 2026-10-31：新开发接收机固定 PCD 验证完整结果

Run `20261031-phase3-fresh-development-rx3-2-s2026092701-r01` / actualcode `f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7`，数据池与新PCD444、六读出三DURA/独立三truth均 VERIFIED 完成。主PCD宏AK/CK/RU/H：source_DURA: 89.733333/96.748148/64.285714/74.907247%; 同G闭集损1.248148pp；clean_DURA: 31.661804/43.761391/61.979647/41.912794%; 同G闭集损16.905211pp；star_DURA: 27.951722/48.582412/49.307698/35.678111%; 同G闭集损13.769336pp。star相对E0/SOURCE宏AK+1.511126/+1.200385pp，但RU−3.350826/−2.962281pp、未知拒对−190/−168；绝对保护和拒识均失败，NO_PROMOTION，不扩seed。所选PCD仅研究起点，不能称已在新RX确认普遍收益，旧RX24-1负结果和既定门槛保持。

RX3-2早U576/两未来日Q10238，旧4508/未知5730，后来7-11整类不进U；10814唯一canonical、同Qclean/star配对、cleanU0，03-23缺20-19/8-20时AK/CK/H=N/A而RU独立可报。合并六景覆盖完整仍未通过性能门槛。原F0同G闭集clean48.567015%/star41.721059%，该测试组合身份识别本身也弱；这支持下一阶段同时关注跨RX身份适应和拒识表征，但不是单因素原因证明，不能据Q调阈值或挑简单数据。sourceV仍非known盲测，全局packet历史UNKNOWN，无发现/REG、AN/HOW NA、formal0，完整目标ACTIVE。全444步/18点/分组/成本见新结果文档。


## 2026-10-31：共享身份与拒识联合适应预登记

Run `20261031-phase3-joint-identity-s2026092701-r01` / actualcode `f540589c0478bed77a9fb4e664481c80ffe7eecc`。新SOURCE_JOINT/TARGET_JOINT各444，从原纯源r03 body/projection与原F0默认W960相同初值独立训练；固定Sinc/buffers/G及原模型，源旧metric/OE/anchor10增加CE1，target另原PCD.1，SOURCE先完成再开U576。四模型八读出三DURA，E0/OLD_PCD精确复用，两个新sourcefit与四new6tau24＋旧24共48tau384B全冻后独立三truth/Utruth0。新学生native身份与各自密度/阈值路由相配，保护基准仍原F0同G闭集。既有RX3-2未来Q10238已曝光，不新建数据或回收Q；03-23缺旧类仍N/A。新训练888，单主部署602632，科学门槛和正式RX边界不变，无发现/匿名REG，完整goalACTIVE。见`docs/PAMF_PHASE3_JOINT_IDENTITY_PLAN_20261031.md`。


## 2026-10-31：共享身份与拒识联合适应完整结果

Run `20261031-phase3-joint-identity-s2026092701-r01` / actualcode `f540589c0478bed77a9fb4e664481c80ffe7eecc`，真实2×444联合训练、8×3DURA预测和独立三truth评分均VERIFIED完成。主TARGET_JOINT宏AK/CK/RU/H：source_DURA: 89.718519/95.888889/67.777778/77.219871%; 原F0闭集90.981481, 学生闭集91.581481, 原F0损1.262963pp；clean_DURA: 21.147817/34.325475/63.883173/31.776407%; 原F0闭集48.567015, 学生闭集42.228981, 原F0损27.419198pp；star_DURA: 21.357750/40.113129/52.561046/30.373484%; 原F0闭集41.721059, 学生闭集38.639796, 原F0损20.363309pp。star较SOURCE_JOINT AK+0.664492pp/RU−5.012763pp，较OLD_PCD AK−6.593973pp，绝对保护/拒识与相对晋级条件失败，NO_PROMOTION、不扩seed，完整目标ACTIVE。

两学生body/head/projection联合训练、Sinc/buffers/原F0G冻结；SOURCE先完成再开U576，源L/OE全444索引匹配，默认头960原生复制无替换/另训，各模型ownclosed路由/τ，原F0同G闭集独立保护。新888、真实signal679034/G21052/F020476、2源GMM/24新τ+24旧τ、单主602632参数，不部署原型/行bank。初始r03body与F0W的完整分类并不等原F0，无初始组合Q读出，因此不能唯一归因CE；后续先诊断源端初始化和身份保真，不依据Q重挑参数或简单数据。V校准非blind、已曝光Q/缺类日N/A/global historyUNKNOWN、formal0/REG0保持。完整888步/24点/分组见新结果文档。


## 2026-10-31：身份初始化源端诊断预登记

Run `20261031-phase3-identity-source-diagnostic-s2026092701-r01` / actualcode `602a57b1fbb61c05c6fd7ac780dc7656a93238f8`。固定F0/未训练r03+F0W初值INIT/纯源SOURCE_JOINT444三个冻结模型，仅读源G10 P1 L6300/OE6134，V/NV/U/Q/private truth0，不加载TARGET适配权重，无优化/新head或adapter拟合/阈值。诊断原生身份和raw/z/W漂移、source TX-RX-day几何及OE无TX置信度，L缺scene保留N/A。真实F012434/signal24868/G0，合成smoke另计。此为已训练源角色诊断，不是目标或blind测试，不更改旧候选/指标，完整goalACTIVE。见`docs/PAMF_PHASE3_IDENTITY_SOURCE_DIAGNOSTIC_PLAN_20261031.md`。


## 2026-10-31：身份初始化源端诊断完整结果

Run `20261031-phase3-identity-source-diagnostic-s2026092701-r01` / actualcode `602a57b1fbb61c05c6fd7ac780dc7656a93238f8`，冻结 source-only F0/INIT(r03+原W)/SOURCE_JOINT444、L6300/OE6134全量诊断VERIFIED完成。F0: macroL=91.777778%, CE=0.293837；INIT: macroL=96.746032%, CE=0.110546；SOURCE: macroL=98.000000%, CE=0.066901；INIT对F0正确净+313、SOURCE对INIT净+79，INITW逐值相同且其余state等r03。源端不支持gross头接错/语义置换作为唯一解释；未测目标初值，不宣称已经定位目标退化或推广原因。

rawcos INIT/F0均值.925550、SOURCE/INIT .999043，z SOURCE/INIT .998893；SOURCE W相对L2 .072261。五种source质心几何及全TX/RX/day/OE无TX置信度见诊断报告，源L/OE本为训练角色，scene无metadata保持N/A。新F012434/signal24868/G0、opt/fit/τ0，V/NV/Us/U/Q/truth/TARGETCP0，全模型/buffer不变；无目标性能结论，完整目标ACTIVE。后续核查源覆盖与Phase1身份表征保真，避免继续把修复明显head拼接故障当作既定方向；新候选仍需源端固定选择及全三DURA评测，不从Q调参。


## 2026-10-31：PCD 全簇匿名注册预登记

Run `20261031-phase3-pcd-cluster-registry-s2026092701-r01` / actualcode `16e6f04e7475827e7c8e24b398fb029a39a8138c`。复用 fresh PCD444/DURA/F0及GMM/τ，sourceL6300新源中心支持域、全U576固定HDBSCAN形成旧类关联/匿名支持原型；注册先于V/NV/Q，预测四读出三DURA后独立四truth。无神经更新/新GMM或阈值，signal6876/F0576/G0，clean不学习、Q不拟合；AN完整三到达类、later9独立拒识、sourceNV独立命名空间。所有保护与原95/90/90/2pp门槛不降低，03-23缺old类N/A；目标ACTIVE，无晋级承诺。见`docs/PAMF_PHASE3_PCD_CLUSTER_REGISTRY_PLAN_20261031.md`。


## 2026-10-31：PCD 全簇匿名注册完整结果

Run `20261031-phase3-pcd-cluster-registry-s2026092701-r01` / actualcode `16e6f04e7475827e7c8e24b398fb029a39a8138c`，完整4×3 DURA预测及独立四truth评分 VERIFIED。主predicted_REG1：source: AK/CK/RU/H=33.437037/92.818519/69.301587/45.109417%；AN=N/A；原F0损57.544444pp；clean_DURA: AK/CK/RU/H=7.414701/33.905864/68.865248/13.387928%；AN=0.000000%；原F0损41.152314pp；star_DURA: AK/CK/RU/H=4.221721/38.168624/54.042553/7.831647%；AN=0.000000%；原F0损37.499337pp。全U HDBSCAN3簇378/25/26、147噪声，两个匿名槽分别混合旧212+新166及纯旧26，严格多数U-only匹配为空，未来三到达类AN全0，不能称发现成功。source旧AK由89.733333降33.437037%、clean31.661804降7.414701%、star27.951722降4.221721%；主源/旧保护、注册及原REG0门槛皆失败，NO_PROMOTION、不扩seed，所选PCD基线保留。REG1主RU是仅未来9类，不能与REG0四未知直接作收益；同later1410 clean66.666667→68.865248%、star56.028369→54.042553%，actual仍未注册6..9另报。新NN/GMM/τ更新0，signal6876/F0576/G0，合成4/2/2另列；predict9.193640s/score4.929865s非B1，foreign0。源V仍校准非独立、Q已曝光、03-23缺oldN/A、正式24-18未读，完整goalACTIVE。见`docs/PAMF_PHASE3_PCD_CLUSTER_REGISTRY_RESULTS_20261031.md`。


## 2026-10-31：源端簇几何排障预登记

Run `20261031-phase3-source-cluster-geometry-s2026092701-r01` / actualcode `074a84caa44387e5ad8472ab8f217c92150421fd`。只读sourceL6300 G10 P1，冻结F0/r03/freshPCD四坐标，五RX原型留出×早日420/后日840共20stat任务，other4RX5040支持，原HDBSCAN16/8固定。全20pred先于评价，early-only严格多数匹配；模型已见六TX五RX及L记录，不称未知/未见RX验证，AN/RU/HOW=N/A。PCD有U576祖先但本行U/Q/V/NV/OE等新读0。真实F06300/signal12600/G0，无NN/GMM/τ更新，统计新拟合20组，筛选仅future研究资格、不自动访问target，原PCD与完整目标ACTIVE保持。见`docs/PAMF_PHASE3_SOURCE_CLUSTER_GEOMETRY_PLAN_20261031.md`。


## 2026-10-31：源端簇几何排障完整结果

Run `20261031-phase3-source-cluster-geometry-s2026092701-r01` / actualcode `074a84caa44387e5ad8472ab8f217c92150421fd`，sourceL6300三模型同P1前向及20组原型留出/早日HDB/后日预测评价 VERIFIED 完整完成。F0_unit 后日宏正确接收五RX均值56.785714%（范围27.142857%至71.190476%），source-screen=False；r03_unit 后日宏正确接收五RX均值81.285714%（范围68.333333%至85.595238%），source-screen=False；PCD_unit 后日宏正确接收五RX均值82.928571%（范围66.785714%至89.523810%），source-screen=False；PCD_white 后日宏正确接收五RX均值56.095238%（范围16.071429%至80.476190%），source-screen=False。只作模型已见六TX五RX及训练L记录的统计排障，不能当真实unknown或未见RX性能。PCD仍有U576祖先，本行targetU/Q/private等值0读，AN/RU/HOW均N/A；不更换PCD、不自动扩seed或target。新F06300/signal12600/G0、NN/GMM/旧τ更新0，统计支持20/HDB20，合成另列、foreign0，完整goalACTIVE。见`docs/PAMF_PHASE3_SOURCE_CLUSTER_GEOMETRY_RESULTS_20261031.md`。
