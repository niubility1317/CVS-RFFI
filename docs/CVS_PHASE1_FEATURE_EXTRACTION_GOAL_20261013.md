# Phase1 84% 目标：特征提取与历史旧门控基线

日期：2026-10-13。目标保持 active，现有完整结果未达到目标域 clean 三主种子均值 84%。

用户最新明确要求：提升网络特征提取能力，保留现有分类头；对比基线使用历史“旧伪标签 batch_neighbor 门控＋EMA＋cos”完整配方，不能以纯 CE 替代。用户已经从项目两种“旧门控”定义中明确选择历史配方，此含义不再需要重复确认。此前纯 CE E50 保留为独立历史结果，不追溯改写成该基线。

后续 epoch 由 Agent 自主选择，原则每模型不超过 50。当前固定四组×三种子：OLD_GATE_COS50、TIME_MS50、PA_MS50、JOINT_MS50；主方案 JOINT_MS50，种子 2026092701/02/03。每模型从零 50 轮、222 步/轮、11100 次更新。历史基线原 130＋70 的阶段比例缩为 33 轮有标签阶段＋17 轮伪标签阶段；这属于预先固定的 50 轮版本，不冒称相同 E200 训练成本。所有结构对照采用同一日程和原门控、EMA、增强及优化参数。

实际历史来源代码为 experiments/cvs_legacy_no_sat 与 cvs_phase1_stack。已核对原配方：EMA=0.999、batch_neighbor 时间门控、窗口 2、最低置信度 0.8、强视图一致，rx/day quantile=0.86，lambda_u=0.16、lambda_ent=0.01、label_smoothing=0.01，L batch128/U batch256，AdamW lr0.0002、weight_decay0.0001、clip5、cosine 最低 0.000001。训练不使用卫星拼接或卫星一致性。U_s TX 继续隐藏，不改既有 L_s/U_s/V 的物理角色；冻结最后第 50 轮，逐 row 自动 clean 预测与独立 truth-last 评分。原用户 NoExplicitCFO 身份输入约束继续保留，对全部对照一致应用，不删内部逐包补偿。

特征结构改造仅位于 t1/t2/t3 与 pa_b1/pa_b2/pa_b3。时序块包含 kernel/dilation=(3,1)/(5,1)/(5,2) 三个深度卷积分支；PA 块包含 kernel3/5/7，逐层原 dilation1/2/4。每块增加可学习 1×1 残差映射，保留原 norm、activation、pool、dropout 与输出通道，最终保持 160 维和原 ResidualPhysicalHead cosine6×160/scale30/融合接口。classifier 参数仍按原训练损失正常学习，结构、功能和初始化不改。

本地已验证真实四种模型：classifier 与对应种子的原基线初值及相同 embedding 分类输出逐项一致，所有新卷积分支实际执行且收到有限非零 CE 梯度，特征输出发生变化，全部训练参数纳入优化器，strict state round-trip 与 epoch51 负例通过。可训练参数为 177025 / 237697 / 291585 / 352257；性能优先，真实时间及额外成本另列，尚未声称改造有效。

此前未登记、未发布的 cosine 梯度诊断与自身学生 SSL 特征草案仅保存在工作树 .codex_tmp，按当前用户方向暂缓；没有远端进程、权重或目标访问。新历史配方四组的训练、预测、独立评分和队列已实现并完成本地闭环检查及预登记。独立审查发现并修正了 15 域训练权重与 5 域预测构造不一致的问题；定向复审 PASS，已发布并核实启动。

按最小 CVS 流程完成剩余闭环后发布；不要求额外设计审批、hash/receipt 或重复审查。目标数据不得用于配方、参数或权重选择，不因目标低分停止或选择性重跑。健康旧任务不停止、不重启、不热改，每 GPU 最多 3 个总进程，包括其他 owner。

用户再次明确（2026-10-13）：每次优化探索的实现和对应实验结果都上传 Git。每轮代码、配置、预登记与逐 seed/配对/失败缺项的结果报告提交并推送，独立核对 remote branch OID 与 HEAD；checkpoint、原始预测、完整日志和数据留在原实验目录，用报告路径定位。当前全流程本地闭环与依赖检查已通过，旧配方四组已预登记；首种子的四组均完成 11100 次更新、冻结且各自末 222 次训练 L_s CE 均值 ≤0.5 后，自动扩展任意组的另两 seed，无 V/target 门槛。原生 15 域模型至预测模型的严格加载、四组首种子门槛与分类头不变检查均通过，独立定向 P1 复审 PASS，已发布并核实启动。

当前运行：`20261013-phase1-feature-multiscale-oldgate-cos-e50-clean-manysig-m12-r01`，release `cvs_feature_multiscale50_20261013_r01`，执行 commit `5fa536f9ca2d90e7d30d547d58503a41b612d57f`。首种子四组已到第 39/41/41/40 轮并均进入 pseudo 阶段，实际 EMA 更新与无标签损失执行、U_s 真值隐藏、无非有限跳步 VERIFIED，无失败记录。首种子四组全部 source-only 通过后队列自动扩展，逐行自动 clean prediction 和独立评分；完整结果报告与实际种子数随后提交推送。目标仍 active。

后续扩展已核实：首种子四组完整闭环，终轮源训练 CE 0.288426/0.249913/0.196179/0.193749 全部通过既定条件，剩余 8 行已自动启动并到第 4 至 6 轮，无失败。当前实际完整种子数仅 1，尚无三种子完整均值；未读取目标分数参与扩展。仓库保留 `first_seed_source_summary.json`，全矩阵结果完成后仍须审计和上传。

2026-10-13 完整更新：历史门控E50多尺度12模型全部完成与完整审计 VERIFIED，clean均值 OLD/TIME/PA/JOINT=80.526786%/81.690476%/79.003175%/81.289484%，主目标84%未达。全部seed和负结果已保存于该run final_results.md/result_summary.json/CSV，须推送核对。旧run实际代码5fa536f9c，不追溯修改。

用户新六路线说明已存 docs/design_reports/phase1_feature_six_routes_20261013.md。后续改为归一化信息恢复与分支竞争首轮，BASE/同参数普通残差/NORM/COMP/NORM_COMP；NORM仅t1浅层，恢复当前样本GN前相对mu/std，alpha初值.05有界.2；两种旁路均新增1225 identity参数。COMP p=.1、5epoch warmup后L_s学生随机撤去4种实际证据路径之一，处理频域rho/PA统计侧路及独立reference_projection，U/EMA生成方式和推理全部路径保持。当前只有本地原型和无query检查PASS，冻结骨干普通cos refit与source/RX诊断、实际全流程矩阵及Phase2同support旧新类比较仍待实现/验证，不能称机制有效。四条后续路线分阶段保留，编码率先做几何诊断。新参数在读取现多尺度目标值前已固定，不依据该目标结果调整。

2026-10-13 新路线正式闭环已登记：20261013-phase1-restitution-competition-oldgate-e50-clean-manysig-m15-r01。五条件BASE_REF/ORDINARY/RESTORE/COMP/RESTORE_COMP固定，12新增scratch50＋3同seed历史基线复用。L_s-only冻结普通cos10轮及源RX/部分路径诊断均已实现，成本独立披露；source-only首seed完整屏障、不劣界限及双readout truth-last checks PASS，唯一独立P0/P1 review PASS。下一步提交发布与真实checkpoint smoke，不把代码检查称为识别提升；Phase2仍未验证，84%目标保持active。

## 2026-10-14 当前用户附件覆盖

下一项优先为时间覆盖×分支内SiLU四条件，默认分类头joint训练，禁止另训head。独立从零scratch50/33标签+17旧门控，保留PA/freq/reference/NoExplicitCFO，四新arm参数/对应权重/CPU与CUDA随机流匹配。旧TIME仅历史参照和冻结source诊断，不混入新匹配统计；Sinc量纲修复等保持独立后续机制。主证据复核：[phase1_feature_evidence_review_20261014](../analysis/phase1_feature_evidence_review_20261014.md)。当前唯一新run 20261014-phase1-time-coverage-nonlinearity-oldgate-e50-clean-manysig-m15-r01；原sequence健康计划不变。

2026-10-14完整结果：sequence24row已ANALYZED，默认co-primary RFF81.5742%、SIG_TIME83.0393%（对基线＋2.5125pp/3正），84%尚未达；完整日志、分组、纠正/引入错误、真实参数更新与pureexport证据回收。CUDA初始化对基线的混杂保留，未热改历史run。另一分支也发布同构TIME，初始化offset62103与本run48213、source门槛/诊断预算不同；保护两份健康不可变run、不混合6seed、不再加第三套，只继续本run闭环。详见主证据报告。

2026-10-14 TIME15row完整ANALYZED：四条件均值81.2161/81.3268/81.4810/81.8115%，预定LONG_SILU距84%2.1885pp。紧凑SiLU相对linear＋0.2649pp/3正；SiLU下长覆盖＋0.3306pp/3正，线性长覆盖仅1正。12个真实final-model导出W删改不改z、featureparams/buffers诊断不变；60阶段/180RXday反馈、270CM、252万决定/600新增ep全部核验。原头收益与迁移结论分开，当前禁止新headfit。完整source/seed/group/误差/成本与剩余容量/旁路归因统一主报告，不按目标重跑或额外扫核。

源门控已完成r02的三seed只读L代理：参数/buffers/EMA/headfit更新0，完整18900次观察/56700前向、1260分组阶段及25batch/seed复算VERIFIED。confidence94.2804%、结构邻居上限14.2011%、最终接受13.2011%，2495接受标签无观测错误；仅已见L代理，不解封/推断U精度，不据此改gate。r01fresh-native-import技术失败保存、一次针对性修复后r02完成。实际源day为native1/2/3，旧0/1/2登记描述纠正，不改变固定角色物理ID。另一分支Sinc坐标0f81373c健康38/37轮，未完成3seed；本对话不重复发布另一unit试验。主报告保持全部正负结果，84%未达到。


## 2026-10-14 Sinc 完整回收与单尺度残差控制

Sinc坐标9row全部完成，6新scratch50/300模型轮次/66600更新、3历史参考成本单列；1512000固定决定和162CM独立复算、全量新step及450epoch核对VERIFIED，worker与dispatcher自然终止。RAW_HZ80.5212%、CENTER_25K80.7637%，差＋0.2425个百分点/3正；CENTER实际24核均改变，但固定L720条件恢复初始Sinc的准确率不变，只确认作用进入z及小幅默认头收益。距84%3.2363个百分点，不追加单位扫描或单独训练分类头。

用户完整目标中的相近参数量单尺度残差控制已独立发布：20261014-phase1-single-basis-residual-oldgate-e50-clean-manysig-m6-r01，执行a6d75a259949d66a7f8da80319ba904c1fc6d72a。3新scratch50＋3固定原基线，完成COMPACT_LINEAR3seed作源健康和最终配对历史控制，无继承。时间参数81744对81672、卷积MAC仅多0.1568%，最大局部覆盖相同，单时间滤波基＋skip核秩上界2对4；优化参数化/初始函数差异披露。最新首seed第38轮，真实进程和日志增长VERIFIED，尚未完整三seed；固定50轮LCE/V总体/最差RX源门槛后自动放行后两seed，逐row冻结后自动测试/独立评分。源门槛不读测试分数，不新增评分完成门槛，不改健康任务。完整正负结果集中主报告，84%仍是未达成的研究目标，当前不新增第三个核心假设。


2026-10-14 单尺度残差已完整ANALYZED：3新scratch50/33300更新、6登记row/1008000决定与3外部COMPACT历史控制/504000决定完成；1512000固定决定及162CM独立重算、全部3模型全50轮/11100步日志、179个真实初始非时间state一致、真实全部t参数张量更新、preGN核秩2及W无关pureexport验证。SINGLE81.2000%±1.5039个百分点，对原基线＋0.6732个百分点/2正，对COMPACT−0.0161个百分点/2正；距84%2.8000个百分点。参数和MAC相近的单尺度残差达到相近均值，不能将TIME平均收益全归于多个时间滤波基，也不证明不同函数族等价。完整seed/group/误差/source曲线/15阶段/765组反馈/成本与限制集中主报告和time_basis_*表。无另训分类头、无复测/继承COMPACT、无目标反馈重跑。该矩阵已闭合，84%研究目标仍未达；后续保持最多两个基于合法源证据的独立动作，不再盲目扫核/初始化/分类头。


2026-10-14 用户再次明确，本聊天主线为训练机制和损失创新。独立候选为身份保护接收机切向平坦化；未叠加 TIME/Sinc/MixStyle，保留默认头、旧门控EMA和同源scratch50/33+17。首种子 BASE_REF/ZERO/RAW_MATCHED/IDSAFE 四行已发布，执行 840be47b7d6b43dbb00ac2cf3864a4025fd41ca8。数学负例、真实CPU/CUDA模型接线、唯一P0/P1审查、真实checkpoint noquery与发布后PID/CWD/argv/GPU/log增长VERIFIED。每新模型2498辅助调用/449640额外L前向包次，0新参数/0换头；只固定首seed，不将单seed当84%三seed目标完成。训练后自动clean预测/独立评分；源健康决策与目标汇总分离，未按目标反馈改参数。机制、公式、同能量控制、预算和风险在本run report.md。另一owner现有任务不重复派发或干预。

## 2026-10-14 最新训练机制主线

按用户最新材料推进“跨 RX 个体参照关系＋CVS 增量证据响应”。所有合法 U 学习连续个体关系，不新增六类投票/置信判定链。真实 joint160 辅助目标不经 W，只对辅助重组的 PA/ref 融合增益 stopgrad，原默认头与增益保持 CE 联合训练。固定 PAIR/ownEMA/33L+17U/50epoch、物理 L/U/V、NoExplicitCFO与原entropy，不叠加新结构/提前U/梯度重加权。

候选run `20261014-phase1-evidence-relation-pair-oldgate-e50-clean-manysig-m12-r01`：ZERO/全源关系/跨RX关系/跨RX＋响应，各3预定义seed。先seed01 A/B/C源健康与真实学习信号标准，再D，完整firstseed源条件通过后才扩02/03；每row固定final50后自动clean预测、独立truth-last评分。方法温度0.2、削弱0.5、R/E系数1固定，不用目标反馈修改。关系蒸馏已有先例；只有D稳定优于C才支持CVS证据响应的贡献。

此前切向机制firstseed源健康条件已在target汇总前通过，用户要求收益路线不落下，原样保留ZERO/RAW/IDSAFE剩余seed02/03确认。有效运行 `20261014-phase1-identity-tangent-oldgate-e50-clean-manysig-s0203-m8-r02`；R01归档遗漏模块导致未提交preflight失败，原目录保留。不是按target选RAW或改投影。firstseed结果BASE80.6685%、ZERO80.7399%、RAW81.4405%、IDSAFE81.0298%，完整三seed证据尚未返回。

可信纠错源代理R02完整三seed共41新包全正确，但固定数量门槛不通过；不放开B/D或从V反调阈值。U真实精度仍未知。84%固定3seed目标未达成，本目标保持active；新方法实际结果尚无，不能用接线PASS或源代理代替泛化成功。


当前关系机制及切向固定确认均已发布/独立启动回读VERIFIED。另保留完整3seed的COMPACT_LINEAR、LONG_SILU和Sinc居中坐标正收益路线，待预定CROSS_RESP通过源标准后做逐项匹配组合，不按target择父或混合全部结构。当前结果未达84%，目标继续active。

切向路线完整3seed已收尾：ZERO80.6038%、RAW80.5579%、IDSAFE80.3077%，原首seedRAW正差未确认稳定收益。保留负结果，不将该路线当作已确认正收益继续叠加。COMPACT_LINEAR/LONG_SILU/Sinc的独立既有证据保留；关系/响应矩阵按原源规则推进，目标active、84%未达。

关系首seed完整A/B/C/D＝81.0488/79.3310/79.5982/79.7000%；D对C有小增量但整体低于A，不宣称有效或达84%。原source-only规则已在target汇总前通过，remaining8rows启动；保持冻结参数/预算，不从本结果反调。


同包源方向诊断已完整 VERIFIED：四 firstseed frozen models/720 seen-L probes/9360额外前向、0拟合及0 U/target访问。强视图 R+E 与同包默认头CE实际参数余弦仅 .0590 至 .1061，签名教师分支响应不保证改善身份间隔；下一机制须检验有效身份学习方向，不能靠增大关系loss或更多门控。当前8row仍按原固定预算自然完成，positive COMPACT/LONG_SILU/Sinc路线保留。详见主报告和源诊断run。84%目标active。


关系12row三seed完整ANALYZED/VERIFIED：ZERO80.6899%、ALL80.1006%、CROSS80.0782%、RESP80.1050%；RESP对CROSS＋.0268pp/2正、对ZERO−.5849pp/1正，没有确认稳定机制收益。600ep/133200更新/2016000决定/216CM和全量source日志已完成。原源决策触发的COMPACT/LONG×ZERO/RESP匹配组合另立run，保持原R/E参数，Sinc独立保留。研究目标84%仍active，本矩阵闭合，不能用接线或小增量缩小目标。


已落实预定正收益结构组合：`20261014-phase1-relation-temporal-factorial-oldgate-e50-clean-manysig-m12-r01`，执行`0af34c5345cba3488a5e4a5d429b2a26447fd09d`。COMPACT_LINEAR/LONG_SILU×ZERO/原CROSS_RESP，12new scratch50、先4seed01源资格后8row，τ/α/权重/PAIR/旧gate/ownEMA/entropy/strong均沿用。真实四200tensor初值与CPU RNG相同，native安装入口一致、default6×160和auxW/gainNone/浅层gradient负例通过；GPU实际4worker第15至18轮，preflight/archive/compile/PID-CWD-argv-GPU-log增长VERIFIED。原关系三seed未有效的结果独立保留，此组合尚无性能结论；Sinc正收益路线独立、不堆入本矩阵。84%研究目标active。

## 2026-10-14 用户最新报告：有效身份证据的训练机制

最新附件 c718649d 调整下一阶段优先级。关系＋响应完整三种子无稳定增益，不能继续把其关系拟合当成有效身份信号。先用相同 L/U 曝光与实际更新预算比较 U 训练日历：TAIL34 至 50、MID17 至 33、SPREAD17/19/…/49，共17个完整U轮与33标签专用轮，L始终参与全部50轮。原默认架构/分类头联合训练、PAIR/旧gate/ownEMA/cos/strong/entropy保持，不叠加R/E、TIME或Sinc。run `20261014-phase1-U-calendar-pair-oldgate-e50-clean-manysig-m9-r01`，预定主条件MID；首seed3源健康后其余6row，9new scratch50/99900实际更新。当前仅本地实现和验证，不声称性能提升。

每模型在第1/6/12/17个U轮前使用当时ownEMA、原完整V27000做无拟合当前教师accuracy/ECE/coverage/precision代理，V标签只用于下游计数，状态与训练随机流保护，不能用最终教师回填早期阶段，也不冒充U真实精度。在对应首batch用同真实AdamW状态的两个临时分支测量 `.16pseudoCE+.01entropy` 对更新的增量，各自clip5并纳入moments/weightdecay；真实模型只更新一次，总临时分支读回匹配实际更新。模型间L1420800/U963900曝光相等，额外V216000前向/模型和8临时步骤单列，无额外实际更新。三组191初始state及CPU随机状态、native安装入口、日历负例和本地真实模型优化回调已PASS。冻结逐行自动clean预测与独立truth-last，原数据/目标反馈禁令保持。

下一独立创新主机制为同TX、同day、跨RX的PA实际增量证据重组，用detached既有W和辅助增益计算有标签CE，保留PA特征梯度，主CE正常联合训练默认头。原样、同RX重组、跨RX重组三控制先在L进行，不能把不同采集包解释为同波形硬件反事实；不新增U伪类判定链。有停止条件的语义间隔随后，分支更新约束仅在有用辅助信号出现后研究。已发布的时序×关系组合继续原计划，必须报告结构内机制差和交互；COMPACT/LONG_SILU/Sinc既有正收益独立保留。84%固定三seed目标仍active，任何接线、源诊断或发布状态都不是目标完成。

等U时序R01发生确定技术失败：原不可变archive缺少既有source_label_gate，MID/SPREAD第17轮当前V代理导入时同异常。两行各完整16轮，按预登记系统技术故障仅停止绑定的本run TAIL/dispatcher（TAIL完整28轮），至少60模型轮次与可能partial成本保留；无target预测/评分，不当作机制负结果。原archive隔离复现、实际publish PATHS修复后隔离导入/原gate负例PASS，只有一次对应P1定向复审。替代run `20261014-phase1-U-calendar-pair-oldgate-e50-clean-manysig-m9-r02` 新根、新scratch，日历/矩阵/seed/预算/源规则保持；无R01权重继承，不影响其他健康run。

PA证据重组原型已实现并验证，见 experiments/cvs_pa_recomposition/README.md。共同TX/day样本集合内各RX cell均衡选包，SAME cell内、CROSS cell间一对一循环配对，每包各作一次anchor/donor，两向端点曝光匹配；只重组真实PA增量。α=.2/λ=.1为固定原型，已有W与辅助增益stopgrad、内部features有梯度、主CE默认头正常joint训练；实际normalize eps1e−4与极小norm负例匹配。数学FP64与真实CVS FP32接线PASS，ZERO精确0，SAME/CROSS真实t1/PA/ref梯度有限非零，auxW/gainsNone，主CE头梯度非零，0新参数/0数据读取/0实际更新/0换头。只完成原型，尚无训练入口/发布/性能证据；不称混合特征新发明，不将不同包当硬件反事实，不扩展U或堆入新日历。本轮主线仍先完成等U时序和既有结构交互闭环，再正式验证该创新机制。84%研究目标保持active。


结构×关系12row已完整ANALYZED/VERIFIED：COMPACT_ZERO81.4137%、COMPACT_RESP80.6952%、LONG_ZERO81.4754%、LONG_RESP81.0359%。关系在COMPACT内−.7185pp/0正，在LONG内−.4395pp/1正；交互＋.2790pp只表示损害较小，不支持关系收益。600ep/133200更新/2016000决定/216CM、全量source日志和48梯度/48实际更新、0非有限/0Utruth/0headfit已核实，进程自然退出。旧COMPACT/LONG/Sinc正结果独立保留，不将结构组合高于旧模型当作机制成功。已固定的等U时序和PA真实身份重组继续，84%仍active。详见[结构交互完整报告](../automation_reports/CV-SincNet/20261014-phase1-relation-temporal-factorial-oldgate-e50-clean-manysig-m12-r01/report.md)。


PA重组已接入真实native已有L128前向，主LCE独立不变，EMA在boundcallbacks前copy，原U/teacher/oldgate/cos/tail33+17保持。ZERO/SAME/CROSS同cohort端点曝光，α=.2/λ=.1先于新target固定、5epochwarmup/every4共2498calls、0额外IQforward/0新param。三组191states/CPUrng/nativeinstall/EMA deepcopy/真实coreloss→callback→probe→backward→clip→AdamW接线PASS，ZEROauxgradient/update差0，activeaux/Lnorm约.1168且actual-shadowerror0。唯一P0/P1 review PASS，新run 20261014-phase1-PA-recomposition-pair-oldgate-e50-clean-manysig-m9-r01 已预登记但尚未发布；等U源阶段校验后执行，不从calendar/旧交互target选参数或parent。默认头与自动cleantruthlast闭环已实现，实际三seed性能/真实cohort/参数更新仍待训练，84%active。


PA正式训练已发布VERIFIED：20261014-phase1-PA-recomposition-pair-oldgate-e50-clean-manysig-m9-r01，执行76ab0b9944210f8aa3f09f051ea9694f55457cd6，3首seed真实第1112步auxiliary/AdamW/参数更新已核实，cohort约55%、active aux/Lnorm约.1168/.1038、auxW/gainsNone，t1/PA真实更新。CPU ZERO差0但真实GPU出现2.44e−8数值残差，梯度仍0，active增量约3至4e−4；如实保留。source/currentcheckpoint smoke/archive/compile/PID-CWD-argv-GPU-loggrowth PASS，0额外IQ前向/0新params/0头拟合，当前无3seed性能。等U首3 source-only资格及实际相同L/U/update预算已核实，后6运行；原结构交互完整负结果闭合。84%目标active。


等 U 时序 9 模型已完整 ANALYZED / VERIFIED：TAIL 80.5853%、MID 80.6206%、SPREAD 80.6403%。MID − TAIL +0.0353 个百分点、1/3 正；SPREAD − TAIL +0.0550 个百分点、2/3 正，未确认稳定收益，不继续按目标扫描日历。450 轮/99,900 更新/1,512,000 决定/162 混淆矩阵、全量源曲线、36 实际优化器点与36 当前 EMA 的 V 代理、0 非有限/0 U 真值/0 另训头已核实，进程自然退出。9 模型同90个合法L包的初始/最终特征见证均证明函数变化与 W 独立，不能证明时序贡献。原 R01 技术失败及成本保留。已固定 PA 重组继续，历史正收益独立保留；84% 仍 active。详见[等 U 时序完整报告](../automation_reports/CV-SincNet/20261014-phase1-U-calendar-pair-oldgate-e50-clean-manysig-m9-r02/report.md)。


PA 重组首种子源条件已 VERIFIED：ZERO/SAME_RX/CROSS_RX 的 final L CE .287650/.288041/.286805，源 V 97.8926%/97.8667%/97.9037%，共同队列覆盖率均为 54.8170%。每行 2,498 辅助调用与四个真实梯度/AdamW 点齐全；辅助 W/增益梯度为 None。预定源条件通过，后6个模型已自动启动。首3个冻结模型的同 L90 函数见证均确认特征变化与 W 独立，不能认定辅助收益。将对已保存源特征进行 margin=.1、同 day/同参照 RX 的1,800关系分析，0新增训练；不为制造梯度扫描 margin。完整 PA 三种子结果尚待返回，84%仍 active。详见[PA 重组报告](../automation_reports/CV-SincNet/20261014-phase1-PA-recomposition-pair-oldgate-e50-clean-manysig-m9-r01/report.md)。


首种子固定 margin=.1 的源 L90 诊断已完成：ZERO/SAME_RX/CROSS_RX 各1,800关系，未满足95/96/98，hinge均值 .009905/.009622/.009890。ZERO 的94.72%关系已满足，但有56个直接排序错误，RX3/day1的未满足率约10.28%/10.33%。54至56个包收到局部特征导数；这不是实际模型新损失梯度或泛化收益。0新增IQ前向/拟合/训练/Utruth/target；不增大 margin 制造信号。需要完整源学习空间核验与独立 SupCon 对照，暂不叠加 PA。原报告首次只读入口的导入顺序错误已修正，诊断输出 VERIFIED，健康训练未受影响。


PA 重组 9 模型已完整 ANALYZED / VERIFIED：ZERO 80.6710%、SAME 80.5823%、CROSS 80.6165%。CROSS − ZERO −0.0546 个百分点、1/3 正差；关闭固定 PA 矩阵，不延伸隐藏 U donor 硬类别。完整训练/独立默认头评分、全曲线、36 特征梯度及 AdamW 点、纠错/引错、9 个函数见证均已核实。完整三个 ZERO 的 L6300 几何有 20,497/378,000 条固定 margin=.1 关系未满足。新报告将主线转向受控证据候选集合监督，排序原型保留未发布；正收益结构独立保留，84% 仍 active。详见[PA 完整报告](../automation_reports/CV-SincNet/20261014-phase1-PA-recomposition-pair-oldgate-e50-clean-manysig-m9-r01/report.md)。


受控证据集合首seed完整V27000可行性已ANALYZED/VERIFIED，但科学条件不通过：原硬门控补集2570个样本只保留81.9066%真实TX，465个集合排除真类；S1有效1554中417个排除真类，多类有效集合192中70个排除真类。固定beta=.5/pi=.9版本关闭，不扫参数、不启动B0/B1/S0/S1，不追加seed。54,000源前向/4无查询smoke，0训练/头拟合/U/target访问；270项独立逐包计数核实。下一步严格可见周期残差源可预测性检查，历史正收益结构独立保留，84%仍active。详见[集合可行性报告](../automation_reports/CV-SincNet/20261014-phase1-evidence-set-sourceV-manysig-s2026092701-r01/report.md)。


完整 L6300 可见周期残差诊断已 ANALYZED / VERIFIED：均值预测相对参考 pooled MSE 下降35.5360%，但RX1/4/6分别反向7.1700%/.5490%/19.8136%，固定所有RX不反向条件失败，24维统一预测版本关闭。126,000固定关系gap .02650916，但同RX异TX cosine .31964857高于跨RX同TX .12242586；前1%包占59.0209%均值预测误差。6,300源IQ读取、0网络前向/更新/头拟合/U/V/target；135项及全部关系独立复算。下一独立假设为公共周期线性参考正交补观测＋可见重复波动停止预算，先源验证，不能冒充已有效训练。SIG_TIME等正路线按各自配方保留，84%目标active。详见[可见周期源报告](../automation_reports/CV-SincNet/20261014-phase1-visible-period-sourceL-manysig-m1-r01/report.md)。


周期参考正交补14维＋停止预算的完整L6300已ANALYZED/VERIFIED，但科学失败：mean相对zero误差增加89.9480%，所有5RX反向；固定跨RX身份gap−.00129858、3/5RX正，126000关系完整复算。zero在92.3016%包已满足预算，非零active统计不足以成为有效身份监督。6,300源IQ读取/0网络前向、更新、头拟合或U/V/target，6,300budget及24tail独立核实。关闭两条原始观测pretext，不扫参数；下一主线为SIG_TIME实际中层停止排序与实际AdamW跨RX增量credit，先源接线和更新诊断。历史正收益独立保留，84%仍active。详见[正交补源报告](../automation_reports/CV-SincNet/20261014-phase1-null-budget-sourceL-manysig-m1-r01/report.md)。


SIG_TIME首seedownfinal50真实源诊断已ANALYZED/VERIFIED：pre-gain160固定126000关系未满足6.5397%，joint160未满足11.1143%；辅助anchorORDER真实梯度非零且W/gainNone，原native188参数/72moment/LR1e−6严格重建。四L90×五RX40临时步的sameW跨RX贡献全部低于固定1e−6容差，0/20接受；不降容差、不称早中期无用。6300IQ读取/9540真实L前向＋2zero smoke，0新训练/另训头/U/V/target；252000关系和20决定独立复算。下一最多一个正式SIG_TIME训练矩阵，比较BASE/SupCon/停止ORDER/实际增量CREDIT，同预算default头joint与原U配方保持；源/性能结论分开，84%active。详见[中层增量源报告](../automation_reports/CV-SincNet/20261014-phase1-SIGTIME-crossRX-credit-sourceL-manysig-s2026092701-r01/report.md)。


SIG_TIME 匹配训练矩阵已发布并 VERIFIED：BASE、SUPCON、无门控 ORDER、实际增量 CREDIT 首种子均从零训练，原默认头与 33L+17U 配方保持；固定 50 轮，每 16 步执行一次中层辅助评价，前 5 轮预热。首种子源域健康、机制调用及实际梯度条件通过才扩展其余 8 行，每行冻结后自动 clean 预测和独立 truth-last 评分。四组的进程、CWD、命令、GPU 与日志增长已经独立核实；当前没有新的性能结果，84% 仍 active。详见[匹配训练报告](../automation_reports/CV-SincNet/20261014-phase1-SIGTIME-credit-oldgate-e50-clean-manysig-m12-r01/report.md)。


SIG_TIME CREDIT 的实际中期学习信号已 VERIFIED：固定 step 4,448 的辅助梯度范数 .01119972，全参数增量 2.80581e−5；实际时序分支与 t2 增量分别为 2.15263e−5、1.06615e−5，一次实际更新与临时预测最大差 7.45058e−9。CREDIT 第 23 轮的 250 次调用中 57 次选中梯度非零，75/1,250 个 RX 分项通过；四组已保存接受决定和计数独立复算，0 新 IQ/前向/更新/目标访问。当前检查点仍属 L 阶段，尚待 U 阶段固定点、源扩种子条件及 clean truth-last；不能据此宣称泛化收益，84% 仍 active。详见[训练中的源信号证据](../automation_reports/CV-SincNet/20261014-phase1-SIGTIME-credit-oldgate-e50-clean-manysig-m12-r01/source_signal_progress.json)。


SIG_TIME 训练机制首 seed 四组已完整 ANALYZED / VERIFIED：BASE 83.3625%、SupCon 83.3655%、ORDER 83.2613%、CREDIT 83.2726%；相对 BASE 净判对分别 +5、−170、−151，只有一个 seed，不确认稳定收益。CREDIT 60 次非零梯度调用全部属于 L 阶段，U 的 236 次调用全部停止；固定梯度点仅 1/4 非零，源条件失败，剩余 8 行未启动，不降低阈值或按目标重跑。200 轮/44,400 主更新/672,000 决定/72 混淆矩阵、2,496 调用与16实际点已核实，进程自然退出。浅层及中层参数实际增量存在；Sinc 频率在16点有梯度而更新为0，原因未确认，下一只读核查单位/精度，历史 SincCENTER 等正收益保留。84%仍 active。详见[完整首种子报告](../automation_reports/CV-SincNet/20261014-phase1-SIGTIME-credit-oldgate-e50-clean-manysig-m12-r01/report.md)。


2026-10-15：固定 CENTER_25K × 排除当前 L 源检查 `20261015-phase1-SIGTIME-center-heldcredit-sourceL-manysig-s2026092701-r01` 已完成。RAW/CENTER × SAME/HELD 四单元各 0/20 accepted，固定可用性条件未通过，不扩完整训练。CENTER 的 actual base Sinc kernel 改变 12 至 13／24，aux 最多再改变 3／24，但 held-L credit 最大仅3.45e−9（固定容差1e−6）。540L IQ／18004forwards／160临时steps，0实际训练／headfit／U/V/target。旧 contribution eval cache 的版本碰撞已复现并补记旧main归因限制，旧结果产物保留。已有固定 CENTER 和 SIG_TIME 正路线保留，不能把历史增益相加；下一轮检验实际早中期持续特征学习信号，末期 CE 微增量筛选不单独作为晋级依据。默认头 clean 三种子84%仍ACTIVE、未达成。
