# Phase1 84% 目标：特征提取与历史旧门控基线

日期：2026-10-15 更新。预登记旧 SIG_TIME＋RDrop E50 完整三种子 clean 均值已达到 84.340079%；机制优化目标保持 active，性能里程碑和创新机制证据分别记录。

用户最新明确要求：现在起只优化 SIG_TIME＋RDrop，重点是 RDrop 机制。固定既有 SIG_TIME 结构、RAW_HZ、普通增强、LS0.01、默认 cosine 头联合训练及历史“旧伪标签 batch_neighbor 门控＋EMA＋cos”配方，研究 RDrop 的一致性对象、梯度作用和启用方式；其他结构、特征增强、教师和优化器路线保留历史证据，暂不新增或叠加。此前纯 CE 和其他配方保留为独立历史结果，不追溯改写。

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


2026-10-15：跨RX feature sharpness源原型 `20261015-phase1-SIGTIME-crossRX-feature-sharpness-sourceL-manysig-s2026092701-r01` 4/4通过固定信号可用性条件。currentRX造detachedρ=.05特征方向，在otherRX／排除currentID的seenL上优化relu(pertCE-cleanCE)，两项均向17个部署参数传梯度，固定preW/defaulthead。CROSS最大RXgap6.325e−4至4.005e−3，梯度范数.001616至.007193，与GLOBAL相对差.441至3.532；actualAdamW aux增量改变2至3个Sinc核。540L IQ／14762forwards／8tempsteps，0realtrain/headfit/U/V/target。没有证明DG收益或原创性。保留SIG_TIME／固定CENTER既有正路线，已经准备新的E20匹配矩阵接口；BASE_CE／BASE_SSL／CLEAN_CE／GLOBAL_GAP／CROSS_GAP计划先首seed再固定source规则扩展，实际阶段与预算须预登记，不默认E50续训。84%默认头clean三种子仍active／未完成。

## 2026-10-15：源可用性通过后发布 E20 跨 RX 脆弱性矩阵

跨RX源原型4/4满足既定lossgap/实际梯度/滤波器条件，只证明信号可用，不证明识别收益。新增固定 `20261015-phase1-SIGTIME-center-crossRX-sharpness-e20-clean-manysig-m15-r01`：SIG_TIME＋CENTER25K、原默认头、scratch20/4440更新，BASE_CE、BASE_SSL、CLEAN_CE、GLOBAL_GAP、CROSS_GAP各三seed；先首seed5行源证据，满足固定健康与信号条件再扩其余10。BASE_CE20L/0U、U/EMA实际关闭；其余13L＋7U/原oldgateownEMA，全部普通L增强与星地训练关闭，U专用Gaussian保留。250scheduledauxcalls/四实际探针，默认头clean逐row自动预测＋独立truth-last。训练和规则不接受target反馈；原E20与历史正负结果保留，0E50训练权重继承。84%默认头三seed目标仍active。实际codecommit/命令/环境/GPU/停止条件/产物见新run预登记；源原型不是新盲测或原创性证明。


E20跨RX特征脆弱性 `20261015-phase1-SIGTIME-center-crossRX-sharpness-e20-clean-manysig-m15-r01` 已ANALYZED/VERIFIED：实际15行/3seed，CROSS_GAP=81.0853%，相对BASE_SSL=-0.0480百分点；源决定EXPERIMENT_COMPLETE。完整默认头truth-last/配对错误/源曲线见[结果报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-center-crossRX-sharpness-e20-clean-manysig-m15-r01/report.md)。84%三seed数值核验=False；该布尔值不替代机制对照或整体目标完成判定。未改参或选择性重跑，历史正负路线保留。


本版跨RX loss-gap闭环为负结果：CROSS_GAP81.0853%，相对BASE_SSL−0.0480百分点，净−242/504000；三seed未到84%。15模型全部20轮/4440更新/默认头truth-last，270CM与300源epoch完整。source V180真实initial/final表明17selected张量均变化，但身份间隔相对SSL两降一升；feature-only/W删除接口PASS只证明解耦结构。全局shadow容差受unusedHz参数放大，原PASS不证明紧精度。保留SIG_TIME/CENTER历史正路线，关闭本版loss-gap参数扫描；下一个最多两动作：同身份跨RX分支证据重组的source原型，以及真实feature更新采集。新主训练尚未发布，84%目标active。


2026-10-15：SIG_TIME＋CENTER25K 同TX/eq/day跨RX PA完整交换源原型已ANALYZED/VERIFIED，run `20261015-phase1-SIGTIME-PA-evidence-recomposition-sourceLV-s2026092701-r01`，执行`047dfc2f4e88d18c5ebe0f584f88f8fc48ccfd64`。own BASE_CE第一seed initial/final20×4L90×ZERO/SELF/SWAP_PA，固定gate0/8；初始梯度相对差.004381至.005784，末态1.053592至10.727818但V身份间隔最高3.59e−7、preWCE减少最高8.70e−7均低于预登记条件。真实4项joint含reference、auxW及3gainNone、2至6Sinc核辅助变化均核实；360L/180V/5404forwards/48tempAdamW，0realtrain/headfit/EMA/U/target，原状态不变，进程自然退出。关闭该固定完整交换变体，不扫描混合/权重/U donor，不证明中期或其他重组无效。新独立假设为源RX身份风险约束下的AdamW实际feature位移，仅设计/纯数学原型，真实源验证和主训练未发布；V不得作投影梯度。SIG_TIME/CENTER/COMPACT/LONG历史正路线独立保留，84%默认头clean三seed仍active。详见[源原型完整报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-PA-evidence-recomposition-sourceLV-s2026092701-r01/report.md)和[下一独立机制方案](../analysis/phase1_rx_update_projection_20261015.md)。


2026-10-15：RX actual-update projection源否证已ANALYZED/VERIFIED，run `20261015-phase1-SIGTIME-RX-update-projection-sourceL-s2026092701-r01`，执行`846a539adeeec9a0d76d41e4d89e14bd3ce0705e`。ownBASE_CE initial/final20×4L90、riskL180／独立heldL180，gate0/8；初始RAW全RX可行，末态RAW及梯度投影后的AdamW仍3RX正风险。直接位移投影保留约92%并改变16浅中层张量/真实joint，但held meanCE相对RAW升1.05e−6至1.30e−6；两者相对更新前都下降，不能说投影一概增加原CE。720L/10444forwards/16tempsteps、240perRXmetrics，0realtrain/headfit/EMA/U/V/target，真实参数舍入逐RX核实、状态不变、进程自然退出。关闭固定Rx-average投影，不放宽阈值或扩main。下一独立机制为可信身份下的主路径证据补偿，a/SIG定向stop-gradient辅助及停止概率目标，设计尚未实现/发布；先机会与活跃子集可靠性census，不用V梯度或U真值。历史正路线保留，84%默认头clean三seed仍active。详见[本轮源报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-RX-update-projection-sourceL-s2026092701-r01/report.md)和[下一机制](../analysis/phase1_evidence_compensation_20261015.md)。


2026-10-15：身份主路径证据补偿已实现定向stop loss，并完成原q全源冻结机会普查，run `20261015-phase1-SIGTIME-evidence-compensation-sourceVU-s2026092701-r01`，执行`ee303b067ebd614986bc5cd59e2d634935979394`。V27000中原q接受880、EVIDENCE活跃4、新增4，活跃precision=1.0；隐藏U56700接受934、活跃3、新增3，U真实精度/error均null。source gate NOT_PASSED，决策`CLOSE_FIXED_TRUSTED_U_COMPENSATION_ENTRY`。167404forwards、0actual/tempopt/headfit/EMA/target，默认头不改，CPsmoke/完整分组/状态不变/自然终态核实；不扫描beta/pi/gate/sampler。冻结evalBASECE-copy不是当时train-dropout/currentEMA，源proxy不能当U精度。历史正收益独立保留，84%默认头clean三seed仍active、未完成。见[完整普查报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-evidence-compensation-sourceVU-s2026092701-r01/report.md)。

下一独立设计固定主攻合法L的跨RX类间差方向一致：同RX真实类均值相减，再跨RX比较15类对/20有序RX对，共300项；真实fourtermjoint不改，辅助端点/gain detached而a/s可达，主CE原头joint保持；SELF差分/原CE匹配控制及同状态AdamW/独立heldL检验另行登记。设计未实现、无真实源run或训练收益，不能从本固定U入口科学失败直接扩main。见[下一机制设计](../analysis/phase1_crossrx_identity_difference_20261015.md)。


2026-10-15：跨RX类间差方向已实现及源诊断ANALYZED/VERIFIED，run `20261015-phase1-SIGTIME-crossRX-identity-difference-sourceL-s2026092701-r01`，执行`1c629a7f54fd69a0b1dc9751368f38165b21505a`。INITIAL/FINAL20×4L180与独立heldL180，gate0/8。末态103至129/300活跃、CROSS/SELFgrad相对差2.21至4.22、CROSS/CE真实F位移差1.86%至2.45%，14至16浅中层tensor/Sinc4至8核变化；heldraw身份margin相对两控制各4单元正，但增益1.90e−7至5.16e−7低于1e−5。初态heldCE相对CE四减/相对SELF三减一增弱正保留，不代替FINAL20晋级。900L/6124forwards/32tempCPUAdamW/120RXmetrics，0realtrain/headfit/U/V/target、原状态不变自然终态。关闭固定变体不扩main/不扫stop与coef。末态savedmoments LR1e−6不等于拟续训reset阶段，后续只做纯CE阶段状态/余量独立诊断，不据此复活CROSS。历史正路线保留、84%默认头clean3seed仍active。见[完整源报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-crossRX-identity-difference-sourceL-s2026092701-r01/report.md)。


2026-10-15：纯CE阶段状态源诊断ANALYZED/VERIFIED，run `20261015-phase1-SIGTIME-pureCE-restart-state-sourceL-s2026092701-r01`，执行`be3062a0941b1a197305ba73270d6a0f72f483b8`。sameBASE_CE final20/current4L180/heldL180，ENDsavedLR1e−6、KEEPsavedLR2e−4、RESETemptyLR2e−4，RESETgate0/4；RESETpreWCE全升.00220至.00914、raw关系margin三降一升，实际F位移.083左右/24核变化，不是无图无更新。KEEPpreWCE四降3.88e−6至1.03e−4而rawmargin四降，正负均保留，不替代RESET资格。3gain跟afterfeature/仅preW分析，71F/195041coords，900L/3062forwards/24tempCPUsteps/60RX/4paired，0realtrain/headfit/U/V/target，原state/payload/RNG不变/自然终态核实。关闭固定reset阶段资格，不扫LR/moment比例、不复活CROSS。下一独立历史feature动量RX对比过滤只设计、未实现发布；保留当前g和二阶尺度，W正常学习，需新source证据，不宣称创新/效果。历史正路线保留，84%默认头clean三seed仍active。见[本轮报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-pureCE-restart-state-sourceL-s2026092701-r01/report.md)和[下一机制](../analysis/phase1_receiver_contrast_momentum_20261015.md)。


2026-10-15：源RX对比只过滤历史feature动量实际源原型ANALYZED/VERIFIED，run `20261015-phase1-SIGTIME-RX-history-momentum-sourceL-s2026092701-r01`，执行`c91085d7f858b7ca24337294bcab409b6999e351`。samefinal20/samewholeCE/current4L180/held180，KEEP/M_ZERO_F/RX_M_FILTER同LR2e-4，v/step/W/当前g保持，3gain包含实际F71/195041coord、ACTUAL/PRE_W共用afterjoint且W正常联合梯度。固定source资格0/4；关闭此固定过滤机制，不扫rank/LR/强度/频率或替代控制晋级。900L/3062forwards/24tempCPUsteps/60RX/8paired，0realtrain/headfit/U/V/target，live/payload/RNG不变，真实CPsmoke与自然终态核实。单步已见源证据，不是DG或新clean分数，84%默认头三seed仍active。历史正路线分别保留。见[本轮报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-RX-history-momentum-sourceL-s2026092701-r01/report.md)。


2026-10-15：trueL跨RX双边身份匹配实际源原型ANALYZED/VERIFIED，run `20261015-phase1-SIGTIME-identity-transport-sourceL-s2026092701-r01`，执行`6b4f3fde6ed607b95f0502e81dd71c8e74ff5bbb`，32cells/24paired/160RX，INITIAL与FINAL20固定资格各0/4，关闭固定BI，不扫τ/π/weight/iterations/不替换控制晋级。早期cross匹配真实held质量四正但CE相对ZERO较差；FINAL20 CROSS_ROW控制在独立无容量质量对ZERO四增.001904至.002218、错误−2/−1/0/−1、ACTUAL CE四降3.11e-5至4.27e-5，保留探索正迹象，不称DG或正式candidatepass。下一独立source确认用新固定L ord2/3,19/20,37/38,54/55、held66/67及ZERO/SAME_ROW/CROSS_ROW，条件未实施发布；不能按target选父/直接controlfallback。真实actualjoint/F71/3gains/normalW、7564forwards/64tempsteps/900L，0realtrain/headfit/U/V/target，finite80近似说明，live状态保持/自然终态核实。历史正路线保留，84%三seedactive。见[结果](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-identity-transport-sourceL-s2026092701-r01/report.md)。


2026-10-15：跨RX真实ROW对应新900独立确认ANALYZED/VERIFIED，run `20261015-phase1-SIGTIME-identity-correspondence-confirm-sourceL-s2026092701-r01`，执行`1f65c70457bc2e85218a2512f2b1df8d1fb2ff2b`。12cells/8paired/60RX/3062forwards24temp，0actualtrain/headfit/EMA/U/V/target。相对ZERO及SAME_ROW true对应质量四正、错数四降，但ACTUAL/PRE_W CE四组均较差，准确率不降；固定gate0/4，关闭ROW确认、不松阈值/扫描/发布main。旧900上的CE四正保留但不跨子集复现。下一研究转向由原defaulthead检验的跨RX公共成分+身份残差转移，loss内L mean不部署/不fit/不U真值；尚未实施。历史正路线独立保留，84%三seedactive。见[报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-identity-correspondence-confirm-sourceL-s2026092701-r01/report.md)。


2026-10-15：默认头跨RX公共成分/身份残差转移ANALYZED/VERIFIED，run `20261015-phase1-SIGTIME-identity-decision-transfer-sourceL-s2026092701-r01`，执行`a2a4337a3034ed01c8f3066f3149c2a516e63b81`，12cells/8paired/60RX/3062forwards24temp，0actualtrain/headfit/EMA/U/V/target。current CROSS active12/17/18/12(1.667%至2.5%)不足36，heldtransfer相对ZERO两正两负且幅度<1e-4，原defaultCE两正两负，fixedgate0/4；关闭晚期mean候选不扫π/λ/LR/meanpool/partialstrength。不是空图，actualF71/18浅中层均差，但任务多数已解决。下一独立假设是异TX参照c!=y的RX变化迁移 z_r,y−z_r,c+z_s,c/defaulthead truey停止任务，同RX控制；尚未实现/发布，新source拟INITIAL+FINAL区分形成/修整，实际训练阶段与资格需新结果前固定。旧ROW/mean资格保持，单步confidence非完整DG；历史正路线保留，84%三seedactive。见[报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-identity-decision-transfer-sourceL-s2026092701-r01/report.md)。


2026-10-15：异TX参照RX差分迁移absoluteπ.9源原型ANALYZED/VERIFIED，run `20261015-phase1-SIGTIME-identity-contrast-transfer-sourceL-s2026092701-r01`，执行`ed5980ab9b5627c6aed80adf74a82365b8249c76`，24cells/16paired/120RX/6124forwards48temp/0realtraining headfit EMA U V target。INITIAL与FINAL固定各1/4不足2，关闭fixedcandidate不扫π/λ/LR/c/strength。INITIAL四CE正但Faux/maincos .999744至.999804近共线；FINAL active316至462/3600、distinct81至107/180，cos.407至.688，8transferCE比较全正但原CEvsZERO两正两负。保留新功能差分学习机会；下一独立relative-excess任务只修复virtual比实际destination同y更差的trueclass风险，factual门槛stopgrad、同参照/正常defaultW及actual3gains，尚未发布。旧资格不按新规则重判，历史正路线保留，84%active。见[报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-identity-contrast-transfer-sourceL-s2026092701-r01/report.md)。


2026-10-15：relative-excess identitytransfer sourceANALYZED/VERIFIED，run `20261015-phase1-SIGTIME-identity-excess-transfer-sourceL-s2026092701-r01`，执行`6d4f66c830cfb6ab471bf58e2cf45741ed08cc0c`。24cells/16paired/120RX、6124forwards48temp、60calls各FP64/FP32 864000解析判别，0realtraining headfit EMA U V target。INITIAL0/4，Faux/maincos.995631至.996684；对ZERO positiveexcess四降但rawvirtualCE/factualCE/原CE四差，体现detach门槛不保证共同encoder的参照性能。FINAL1/4，active2322至2398、distinct180/180、cos.392784至.694862、8heldtransferCE及positiveexcess比较全正；原CEvsZERO2正2负，其中2/54相对harm1.3018%/1.0193%。固定候选关闭，不扫λ/LR/容差/参照/strength。保留晚期独立方向/覆盖，未确认稳定identity或DG；本轮两个假设已证伪资格，无技术blocker，84%active。下一轮从已保存TX/RX条件风险解释参考delta内容，再提出新监督机制，不继续覆盖或loss缩放。历史正结构/CENTER25K路线保留、defaulthead/U真值隔离/预算不变。见[报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-identity-excess-transfer-sourceL-s2026092701-r01/report.md)。


2026-10-15：真实证据责任路由sourceANALYZED/VERIFIED，run `20261015-phase1-SIGTIME-evidence-responsibility-sourceL-s2026092701-r01`，执行`34b8309871822c2756169f39183133d1a4af839a`。INITIAL3/4通过、FINAL0/4，固定scratch20资格成立。相同loss/mask/720分母下ROUTED只反传负责component，W/3gain auxNone、mainCE正常joint；INITIAL auxF/maincos.9323至.9334，vsJOINTcos.9487至.9519，ordinal37超.95保持失败。早期heldACTUAL/PREW CE相对两control四降（PREWvsZERO.00117至.00262/vsJOINT.000695至.00202），late变化不足1e-5，未支持E20+20。24cell/16paired/120RX/96branch、6124forwards48temp、0realtraining/U/V/target。下一真实mainM12已开始实现，四armBASE_CE/BASE_SSL/JOINT/ROUTED scratch20，辅助前13epoch每16step180calls、后7轮保留oldhiddenU，日程在main结果前固定。不得停在sourcePASS、不按target改日程或seed扩展。84%active，历史正路线保留。见[源报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-evidence-responsibility-sourceL-s2026092701-r01/report.md)。


2026-10-15：source INITIAL3/4-qualified责任路由主训练正式预登记run `20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r01`，新scratch20四arm×3seed，前13轮每16step180auxcalls，后7轮保留oldU；先首seed4源health/signal再其余8，每row自动clean默认头预测/独立truth-last。实际入口检查发现历史sharpness20 M15 actual_native_args全部label_smoothing=.01而声明0；已独立只读核实并保留旧产物，新make_args显式0/新匹配CE/SSL不复用旧分数。sourceINITIAL untrained不受late训练smooth差异影响；late0/4边界保留。local实际CPU四arm188params/863730参数/200state逐项同初值，observer/protocolnegative/唯一新mainintegrationP0P1 PASS，尚未发布训练、无target访问。完整report/配置见[main预登记](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r01/report.md)，84%active。


2026-10-15：责任路由主训练 R01 在已启动三行 step16 的 actual-versus-autograd-shadow 诊断失败，0完整epoch/预测/评分；自然退出 VERIFIED，失败记录已Git交付e29570508197a6d33034fa5e279be2222388a3b0。CPU不复现，本地CUDA实际构造器L128×16仅Sinc坐标两张量proxy误差最大8.5309148e-7，重复autograd与backward rawgrad差4.5474906e-8，actualrawgrad/postclipgrad oracle188张量全部逐位0差。修复只将真实更新核对切至实际postclipgrad oracle，原两proxy保留近似诊断，原容差2e-7/2e-6不变；12temp/model取代8temp，4440真实步和全部科学日程不变。新不可变R02 `20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r02` 正式登记，同12scratch20/LS0/前13epoch180auxcall/后7oldU、源首seed4规则和自动clean truthlast。focused negatives、direct-fix regression与唯一targeted P0/P1 PASS；未宣称主训练或84%完成。见[R02报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r02/report.md)。


2026-10-15：真实证据责任路由 main `20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r02` ANALYZED/VERIFIED，执行 `3a5cff49091deef46cad5d787e5daea7b4c0dfb9`，12freshscratch20/53280实际更新/216指标/2016000默认头clean决定完整。ROUTED均值81.120635%±1.171185%，vs匹配实际LS0 BASE_SSL -0.054167pp（1/3正），vsJOINT +0.026190pp（2/3正）；84%目标False。三个seed内各200初值张量相同；全240epoch/53280step/callback/48probe日志完整解析，actualgradient oracle全0差、proxy近似边界明确。源180/180有效辅助和24Sinc学习不能替代DG证据；历史CENTER25K正路线及所有负结果保留。R01技术失败另计48partialupdates不继承；本main无追加/重排/target反馈。文献OGM/模态估值/CGGM已有先例，具体创新候选及范数匹配/零消融因果限制写入[完整报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r02/report.md)。

2026-10-15 补充：责任路由总体vsSSL净新增273错误，不支持当前配方晋级；次要最差RX均值+0.245833pp、3/3正须保留，最差day-0.180159pp、0/3正。局部收益不替代overall84%目标、不用于候选重排。下轮首先按源证据核对RX留一与梯度范数归因缺口，避免重新扫描本配方或把已关闭set/观测路线直接当新机制。


2026-10-15：RX-blocked 范数归因 `20261015-phase1-SIGTIME-RX-blocked-norm-attribution-sourceL-s2026092701-r01` ANALYZED/VERIFIED，执行 `3e5ee1ffc2d610c6fce3c8d153ed5b08492839d8`。80cell/20cohort/5RX，当前训练144包排除heldRX、评价36包，同cold scratch0/emptyAdamW，5942packet前向/160temp/0actualtrain headfit U V target。ROUTED相对ZERO/JOINT/JOINT_NORM的ACTUAL与PRE_W CE全部20/20改善；vs同辅助F范数JOINT_NORM均值−0.00114540/−0.00119855。错误2cohort改善18持平、5RX均4/4CE改善；仅180不同评价包被重复4组，非20独立重复。实际AdamW F增量.00789961 vs.00539780未匹配，不作纯方向因果归因；不是完成E20真正leaveRX，未设main资格、不重新晋级已负的M12配方。下一具体假设是合法源RX身份风险约束U的实际AdamW feature增量，先建全程排除heldRX的scratch源折，保留defaulthead/hiddenU/7-63-30，增量范数控制与源规则需先固定；未实施/发布。历史正路线/次要worstRX正收益与总体负结果并存保留，84%active。见[归因报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-RX-blocked-norm-attribution-sourceL-s2026092701-r01/report.md)。


2026-10-15：真正源RX1留出U增量机制 `20261015-phase1-SIGTIME-RX1-U-increment-sourcefold-s2026092701-r01` 已启动 RUNNING/VERIFIED，代码 `779e77620ea724154bfb2c899498578643234f00`。一个scratch20 RAW SSL sourcefold，L5040/U45360隐藏/V21600、所有训练/校准/probe/特征缓存入口排除RX1后才建模型/ownEMA；R02仅代码依赖不继承权重。固定13L+7U/222步/4440更新、U7轮396368观察；4probe2896/3344/3792/4432比较RAW、L_ONLY_F、同实际FP32位移范数SHRINK与分RX身份风险CONE，F71含3featuregains、类别W保持RAW默认联合更新，无另训头、0实际候选纠正。16项聚焦测试及原native过滤loader集成PASS，CPU/CUDA原188参数合成oracle逐位0差、唯一独立P0/P1 PASS；ordinaryGPU0容量/CPsmoke/PID/CWD/cmd/GPU/loggrowth已独立读回。资格需至少2/4在几何/覆盖条件下同时胜RAW、SHRINK及L_ONLY_F的两读出CE≥1e-5且错误不增，finalLCE≤.5；只允许完整sourcefold控制，不直接晋级main/target。文献GEM/PCGrad/CAGrad及更近Fix-A-Step先例已明确，不称全局首创；核心差异检验实际有状态AdamW中U增量、跨RX风险交集与feature-only范围。未完成训练或84%，goalactive，历史正负结果保留。见[报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-RX1-U-increment-sourcefold-s2026092701-r01/report.md)。


2026-10-15：RX1 U-increment `20261015-phase1-SIGTIME-RX1-U-increment-sourcefold-s2026092701-r01` 已确认 FAILED 技术终态；13完整epoch/2895成功更新全部日志解析、U真值仍隐藏，第2896步第一次探针因Torch2.1.0+cu121与NumPy2.2.5桥接数组TypeError，在实际step之前中断，0完整probe、资格NOT_EVALUATED。零sourceIQ/CP最小算术复现旧NumPy桥和旧投影FAIL，nativeNumPy与纯Torch FP64 SVD PASS。保留部分产物/计2895技术成本，拟r02仅移除桥接、同seed/20轮/13+7/四probe/三控制/阈值从零，不继承权重；尚未发布。此为技术故障，不计科学0/4或重排理由，84%active。见[故障证据](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-RX1-U-increment-sourcefold-s2026092701-r01/technical_failure_evidence.json)。


2026-10-15：真正源RX1留出U增量机制 `20261015-phase1-SIGTIME-RX1-U-increment-sourcefold-s2026092701-r02` 已启动 RUNNING/VERIFIED，代码 `56bd5feec7edefb8aba6fea595c0b701be0b2809`。一个scratch20 RAW SSL sourcefold，L5040/U45360隐藏/V21600、所有训练/校准/probe/特征缓存入口排除RX1后才建模型/ownEMA；R02仅代码依赖不继承权重。固定13L+7U/222步/4440更新、U7轮396368观察；4probe2896/3344/3792/4432比较RAW、L_ONLY_F、同实际FP32位移范数SHRINK与分RX身份风险CONE，F71含3featuregains、类别W保持RAW默认联合更新，无另训头、0实际候选纠正。17项聚焦测试（原8项未改行为测试复用）及原native过滤loader集成PASS，CPU/CUDA原188参数合成oracle逐位0差、唯一独立P0/P1 PASS；ordinaryGPU0容量/CPsmoke/PID/CWD/cmd/GPU/loggrowth已独立读回。资格需至少2/4在几何/覆盖条件下同时胜RAW、SHRINK及L_ONLY_F的两读出CE≥1e-5且错误不增，finalLCE≤.5；只允许完整sourcefold控制，不直接晋级main/target。文献GEM/PCGrad/CAGrad及更近Fix-A-Step先例已明确，不称全局首创；核心差异检验实际有状态AdamW中U增量、跨RX风险交集与feature-only范围。未完成训练或84%，goalactive，历史正负结果保留。r01技术失败2895更新单列，r02从零恢复科学配置不变，未继承部分状态。见[报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-RX1-U-increment-sourcefold-s2026092701-r02/report.md)。


2026-10-15：真实RX1源留出 `20261015-phase1-SIGTIME-RX1-U-increment-sourcefold-s2026092701-r02` 已ANALYZED，执行 `56bd5feec7edefb8aba6fea595c0b701be0b2809`；scratch20/4440连续更新、20完整epoch、四probe、最终冻结均独立确认，所有loss有限/Utruth0，RAW oracle0差，默认头联合训练。固定资格0/4，finalLCE0.02320659，完整source控制资格False；不直接main/target。关闭该固定投影U增量候选，不扫lambda/阈值/RX。失败r01的2895更新单列，总技术+恢复7335更新；零target，未达84%，goalactive。见[同row报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-RX1-U-increment-sourcefold-s2026092701-r02/report.md)。


2026-10-15：按用户新增报告转向SIG_TIME时间共享失活和同轨迹导出，run `20261015-phase1-SIGTIME-t2-locked-tailavg-sourcefold-s2026092701-m3-r01` RUNNING/VERIFIED，代码 `a4a6dd6fc79a8073113ac9a5385e75d858d325fc`，ordinary GPU0真实CPsmoke/PID/CWD/argv/GPU/loggrowth独立确认。ELEMENT/NONE/LOCKED仅改实际t2.drop，原p.1、t2签名取点和默认头joint不变；3×scratch20/13L+7U/4440步，总13320更新，真实RX1全程排除于训练/teacher/校准/缓存；L5040/U45360hidden/V21600。同步固定17-20尾四状态FP64平均all188params含W/gains/normalization与12常量buffer，FINAL/ownEMA/AVG独立读出，不回流训练、不跨seed/坐标。现有三seed缺whole tail，不单独为平均重跑基线；旧nativeSWA30未过资格保留。9项聚焦负例/实际CPU-CUDA模型/完整合成导出及唯一独立P0/P1 PASS。source主资格只看LOCKED_FINAL同时胜ELEMENT_FINAL/NONE_FINAL的错误且CE不升、三row健康LCE≤.5；AVG不能救援失败主机制，不扫p/窗口。R-Drop第三暂不叠加；旧投影0/4关闭、历史CENTER25K等正收益独立保留。零target，未达84%，goalactive。见[预登记及启动报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-t2-locked-tailavg-sourcefold-s2026092701-m3-r01/report.md)。


2026-10-15：SIG_TIME时间共享失活 `20261015-phase1-SIGTIME-t2-locked-tailavg-sourcefold-s2026092701-m3-r01` ANALYZED/VERIFIED，3×20epoch/13320更新/60epoch完整、9默认头导出、200state同初值及四tail独立FP64平均逐位核实；固定主source资格False。各条件FINAL/EMA/AVG、paired修正/破坏、F71变化和全部曲线保存在[同row报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-t2-locked-tailavg-sourcefold-s2026092701-m3-r01/report.md)。源RX1全程排除、L5040/U45360hidden/V21600、无新loss/headfit/target、1seed/1260独立held包，不能当正式三seed/84%。窗口和p未扫，历史正负结果保留；84%goalactive。


2026-10-15：既有全5源RX固定BASE_SSL ownEMA导出 `20261015-phase1-SIGTIME-ownEMA-export-clean-manysig-m3-r01` ANALYZED/SOURCE_FALSIFIED，真实EMA no-query smoke及终态VERIFIED；首seed V27000错误501→504，CE0.07992632806→0.07925969537，固定资格失败，0新增训练/头拟合/target/评分，其他seed按规则未触发。原父、heldRX1两条件EMA小正和三条件tail平均负结果全部保留，不扫描decay/窗口救援。短run未捕获独立live GPU成员，已如实披露。84%goalactive。见[同row报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-ownEMA-export-clean-manysig-m3-r01/report.md)。


2026-10-15：下一明确训练机制候选 `20261015-phase1-SIGTIME-clean-signature-sourcefold-s2026092702-m3-r01` RUNNING/VERIFIED，执行 `f695b43bb8ff2c8584871cf439315557ced718cf`、ordinary GPU0，PID/CWD/argv/GPU/log及真实scratch checkpoint no-query smoke独立核实。seed02真实RX1留出，ELEMENT/NONE/CLEAN_SIG各20/13L+7U/4440步，总13320；CLEAN_SIG主干保留t2.drop及原RNG，仅签名读取pool后drop前输入、梯度不断、0新loss/参数/IQ前向。实际CPU/CUDA200state同初值/863730params/eval joint160默认logits逐位相等/主干mask及RNG相等/三处联合梯度有限非零，7聚焦检查及唯一独立P0/P1 PASS。CLEAN_SIG须同时胜两个控制的错误且CE不升；NONE正路线另以胜ELEMENT独立确认，不救援失败主假设。无tail/EMA窗口扩展或R-Drop组合；原R-Drop局部兼容性正迹象保留第三优先。旧LOCKED/source01及EMA失败均已交付，84%goalactive。见[本轮预登记及启动](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-clean-signature-sourcefold-s2026092702-m3-r01/report.md)。


2026-10-15：新seed02签名噪声位置 `20261015-phase1-SIGTIME-clean-signature-sourcefold-s2026092702-m3-r01` ANALYZED/VERIFIED，3×20epoch/13320更新完整、所有Utruth0/默认头joint、200state同初值及1260 heldRX1各默认头读出/paired独立复算。CLEAN_SIG主资格True，NONE独立确认False。完整曲线/F71及W变化/发现与确认seed对照见[同row报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-clean-signature-sourcefold-s2026092702-m3-r01/report.md)。四RX权重不转正式全5RX；正式三seed须新固定配置，未读target，84%goalactive。


2026-10-15：源资格通过后正式矩阵 `20261015-phase1-SIGTIME-branch-noise-e20-clean-manysig-m6-r01` RUNNING/VERIFIED，执行 `0fe97397defa9c72cb238556f0facc8d56c55513`，ordinary GPU0所属PID/CWD/argv/GPU/log及真实scratch checkpoint no-query smoke核实。source02 CLEAN_SIG68.253968%，相对ELEMENT65.555556%净修正34（46修正/12新增）、CE-.08224494，通过同时胜ELEMENT/NONE；NONE63.015873%独立确认失败，保留source01+11.349206pp但不冒充稳定正收益。正式固定CLEAN_SIG主arm及NONE必要控制，各三seed从零20/13L+7U/4440步，总6模型120epoch/26640更新，全5RX L6300/U56700hidden/V27000，不继承四RX权重；原BASE_SSL三seed预测/评分复用且成本单列。首seed两arm源健康通过后逐row自动clean168000预测与独立新CPU truth-last，最终另新CPU聚合，0拟合头/新loss/参数/额外训练IQ前向。12聚焦协议/两矩阵阶段顺序、所有3seed/两mode200state同初值及feature-only=z/扰动W不改z实际检查、唯一独立P0/P1 PASS；不增加新source诊断。总新预测1008000，目标主arm3seed clean84%仍active，未给出未完成性能。见[正式预登记和启动报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-branch-noise-e20-clean-manysig-m6-r01/report.md)。


<!-- RDROP_PARENT_AVAILABILITY_20261015 -->
2026-10-15：回收既有 R-Drop 第三优先路线的来源条件。首模型 `20261015-phase1-source-dropout-consistency-manysig-s2026092701-r01` 的阳性仅为 final50、label smoothing .01 的源梯度兼容，尚无训练泛化收益。Git 中原 DRO 设计固定单 seed；ordinary 只读核对其指定 run `20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s01-m4-r01`，仅 seed01 的 SIG_TIME source/checkpoint/completion 存在，seed02/03 对应路径均不存在。可用性核对未读取 IQ、未反序列化 checkpoint、未训练；不代表已验证任何新父来源。证据见[指定父目录回读](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-branch-noise-e20-clean-manysig-m6-r01/next_source_parent_availability.json)。旧报告的原样02/03复核当前缺指定父，不能静默换父或补新 E50 训练。

当前最多两个优先动作：先完成已冻结六行噪声位置矩阵及完整默认头三seed分析；随后如推进 R-Drop，以已核实的固定 E20 BASE_SSL 父另登记零更新 source-only 诊断，明确 E20/LS0 与旧 E50/LS.01/strong-L 的差异。沿用同一 R-Drop 公式不等于复现旧源结果；新源诊断只回答兼容性，后续训练矩阵另按源证据固定。既有兼容性阳性、源留一负结果和当前正收益全部保留，不按目标微差切换主方案。


2026-10-15：正式SIG_TIME噪声位置 `20261015-phase1-SIGTIME-branch-noise-e20-clean-manysig-m6-r01` ANALYZED/VERIFIED，source6/6、clean已评分6/6。固定CLEAN_SIG均值81.309524%±0.940673pp，vsBASE_SSL+0.134722pp、2/3正；vsNONE-0.136111pp、1/3正。NONE81.445635%±0.853798pp，vsBASE_SSL+0.270833pp、3/3正，正收益保留而不切换主指标。完整日志、初始200state公平性、F71与默认W更新、既有truth-side CM与paired正确性独立复算，无新真实IQ/truth/推理/训练；另有48个固定合成packet特征前向，0头拟合/CP改写。SOURCE_FALSIFIED=False，84%数值检查=False；全目标仍未完成。结果及实际预算见[同row报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-branch-noise-e20-clean-manysig-m6-r01/report.md)。


<!-- RDROP_E20_SOURCE_COMPLETE_20261015 -->
2026-10-15：`20261015-phase1-SIGTIME-rdrop-source-e20-clean-manysig-m3-r01` ANALYZED/VERIFIED，固定原 BASE_SSL final20 三父 source 3/3 PASS；t1/t2/SIG 的 KL 导数各 5/5 非零且跨批平均与 paired CE 同向，准确数值见[同 row 报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-source-e20-clean-manysig-m3-r01/report.md)。900 个不同 L 跨三个父复用，8100 次源前向、120 导数向量，另 6 合成 smoke packet；0 新训练/头拟合/target，三份独立产物复算通过。源端兼容性允许另行登记单 seed 真正 held-RX 的 matched two-CE/RDrop 训练证伪，不等于训练增益或 84% 完成，不以 NONE 的 target 高分选父。全目标保持 active。


<!-- CROSSFIT_TEACHER_ROUTE_20261015 -->
2026-10-15：用户追加跨RX交叉拟合教师报告，作为独立监督来源路线。已核实 DomED 存在互补域蒸馏先例；研究价值需先以匹配 SAMPLE 与 EXCLUDE_RX1 的纠正−破坏证明。urban 对话已实现 E50/11100、Lteacher4464/Ldev720（RX1=144）的两教师方案但当时未发布，保留其唯一 owner，本侧不重复教师训练。只共享机制证据，不把该 E50/不同 L 契约当成本侧 E20 对照或直接继承权重。外层 heldRX 必须在全部教师及继承来源排除；不对齐独立教师 joint160，不叠加新旧冲突U标签。RDrop 与跨拟合教师保持独立；尚未扩五教师库或学生蒸馏。


<!-- RDROP_TRAINED_SOURCE_COMPLETE_20261015 -->
2026-10-15：`20261015-phase1-SIGTIME-rdrop-sourcefold-s2026092701-m2-r01` 完整source两组scratch20/8880更新已独立复算；TWO_CE RX1=69.285714%/CE1.492366164，RDROP=75.158730%/CE1.157227747，配对Δ+5.873016pp，修正104/新增30，固定资格=True。0 target/另训头，16合成packet特征边界通过；不把冻结源梯度相容等同训练效力。仅允许单独登记独立source seed确认。[同row报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-sourcefold-s2026092701-m2-r01/report.md)。全目标仍active。


<!-- RDROP_E20_E50_USER_BUDGET_20261015 -->
2026-10-15 用户新预算：明确要求完整E50三seed，再要求E20预算比较，覆盖本轮旧E20限制及尚未发布的source-seed02等待条件。`20261015-phase1-SIGTIME-rdrop-budget-e20e50-clean-manysig-m12-r01` 已登记并实际启动12个全五源RX从零模型：E20/E50 × TWO_CE/RDROP × 2026092701/02/03；13+7/33+17，420modelepoch/93240更新，默认头联合训练，逐行clean预测后独立CPU truth-last。主结果预登记E50_RDROP，E20_RDROP预算比较，均检查84%但不挑target-best；当前RUNNING而非完成。此前四RX源fold是科学依据，不是全源E20预算对照。[当前矩阵报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-budget-e20e50-clean-manysig-m12-r01/report.md)。原有正负路线保留，其他健康任务不动，全目标active。


<!-- RX_TEACHER_EXPOSURE_SOURCE20_FIXED_20261015 -->
2026-10-15 独立监督来源路线已固定并登记：`20261015-phase1-RX-teacher-exposure-source-e20-s2026092701-m2-r01`，两个teacher从零纯CE E20/20L/4440更新，SAMPLE_80/EXCLUDE_RX1各5040L、每TX/day280，fit/calV仅4RX21600，held RX1 V5400只在final20冻结后评价。默认头joint/LS0/native dropout，0U迭代/EMA/训练增强/另训头。固定原BASE_SSL E20 student+ownEMA只作sourceV nativegate proxy，不继承到教师；资格为更少source错、CE不高且拒绝区域净纠正>0及源健康。44异常负例+4资格否决边界、200初态和真实CE梯度PASS；唯一P0/P1 PASS；当前PLANNED未发布，不是已有教师互补结果。B方E50/4464L/720dev仍是其独立草稿，不代替本试。该参数在未读取本次RDrop目标分数时固定；RDrop全E20/E50矩阵不变。[本试预登记](../automation_reports/CV-SincNet/20261015-phase1-RX-teacher-exposure-source-e20-s2026092701-m2-r01/report.md)。所有正负路线保留，goal仍active。


<!-- RDROP_E20_E50_COMPLETE_20261015 -->
2026-10-15 `20261015-phase1-SIGTIME-rdrop-budget-e20e50-clean-manysig-m12-r01` 全12行完整ANALYZED，93240更新/420epoch/2016000决定/216CM及完整source日志独立核验，默认头joint/headfit0。E20_TWO_CE/E20_RDROP/E50_TWO_CE/E50_RDROP均值依次80.974603%/82.495437%/81.150397%/82.102976%；配对E20_RDROP minus E20_TWO_CE: +1.520833pp/3正; E50_RDROP minus E50_TWO_CE: +0.952579pp/3正; E50_TWO_CE minus E20_TWO_CE: +0.175794pp/2正; E50_RDROP minus E20_RDROP: -0.392460pp/1正。预登记主条件E50_RDROP的84%检查=False；E20比较检查=False。预算含日程与U轮次变化，既有RDrop不是全球新方法；不以目标结果挑主条件、重跑或自动完成全研究目标。[完整报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-budget-e20e50-clean-manysig-m12-r01/report.md)。goal仍active，下一机制按合法源证据推进。


<!-- RX_TEACHER20_SOURCE_COMPLETE_20261015 -->
2026-10-15 `20261015-phase1-RX-teacher-exposure-source-e20-s2026092701-m2-r01` source-only 两行 E20 完整 ANALYZED / VERIFIED，8,880 更新和 40 epoch；无 U/EMA/换头/target。固定 RX1 排除教师 60.685185% 对等量 SAMPLE_80 的 99.444444%，纠正3/引入2096，CE升高；固定 A0 拒绝区纠正3/引入1787，科学资格 FAIL，关闭此配方，不扩库/学生/选择性救跑。原已见 RX 拟合 V 反而 +0.222222pp，不能用作跨域互补替代证据。固定源 V A0拒绝不代表预测错（拒绝区 EMA 99.423248%），不是实际 U 精度。RDrop 两预算三seed正收益独立保留，84%目标仍 active。[完整报告](../automation_reports/CV-SincNet/20261015-phase1-RX-teacher-exposure-source-e20-s2026092701-m2-r01/report.md)。


<!-- ORIGINAL_SIGTIME_RDROP_USER_REQUEST_20261015 -->
用户最新明确要求旧 SIG_TIME叠加RDrop并发布验证。固定 `20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01`：旧RAW_HZ/use_aug=True/LS.01/defaultheadjoint/oldgateownEMA，E20/E50×ORIGINAL/TWO_CE/RDROP×固定3seed，共18scratch，主条件E50_RDROP；保留E20预算比较。近期LS0/cleanL/CENTER25K的82.1030%只证明其内KL增量，未超过历史83.0393%，新ORIGINAL为匹配对照。两次paired复用L context捕获的真实增强IQ，不使用native增强前x_l，不重抽增强。每预算首seed3源判据后放行其余6，逐row自动clean truthlast，不看target扩种子。当前PLANNED/CPU-CUDA198state及实际接线PASS，未发布/无新结果；残差交换原型未登记或发布，暂不叠加。[本试报告](../automation_reports/CV-SincNet/20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01/report.md)。84%目标仍active。


<!-- ORIGINAL_SIGTIME_RDROP_LAUNCHED_20261015 -->
2026-10-15：旧SIG_TIME叠加RDrop的 `20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01` 已发布并独立启动核实。执行 `b3ac442e2d33a5b315fb3251c30e0ad7083566bc`，18行预登记，首seed6行在GPU0至5运行，另12行按每预算固定source-only规则放行。原RAW_HZ、LS.01、普通增强及默认头联合训练保留，两次paired前向实际复用增强后同IQ。尚无最终测试收益；全目标保持active，旧正负结果与未发布残差交换原型保留。


<!-- ORIGINAL_SIGTIME_E20_SOURCE_EXPANDED_20261015 -->
2026-10-15 `20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01` E20首seed3行source资格PASS（源健康、KL和固定V容差，未读target指标），其余6行已自动启动并完成ownCP smoke/首epoch。首seed3行已完成504000固定clean决定和独立CPU truth-last评分；E50首seed3行仍在训练、后6行按其自身源资格待放行。当前12/18已启动、3/18已评分、9活跃；不解读部分target成绩，不改配方/预算/主条件，goal active。


<!-- ORIGINAL_SIGTIME_FIRST_E20_SOURCE_LOGS_20261015 -->
2026-10-15旧SIG_TIME×RDrop首seed E20三行全13320step/60epoch源日志核对PASS，真实损失重构max2.38419e−7；每条件396900次U重复观察，ORIGINAL/TWO_CE/RDROP接受6143/6203/5611，RDrop未增加接受量。该计数不代表U正确率，源V97.792593/97.966667/97.870370%不替代目标泛化。分析0目标指标/真值/IQ/CP读取、0参数或头更新；固定18行继续，研究目标仍active。


<!-- ORIGINAL_SIGTIME_E20_SCORED_SUMMARY_20261015 -->
2026-10-15 `20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01` E20全9行、3条件各3seed均已完成独立truth-last评分。既有产物汇总核对162CM/1512000决定及配对：ORIGINAL83.115675%±1.559519pp、TWO_CE82.486111%±1.428440pp、RDROP83.942857%±0.928437pp；RD−ORIGINAL+0.827183pp/2正，RD−TWOCE+1.456746pp/2正，TWOCE−ORIGINAL−0.629563pp/0正。84%检查False，不能四舍五入达标；本次汇总0新truth/IQ/CP读取或预测/训练，最终18行fresh独立复算尚待E50全部结束。E50_RDROP主条件和源资格固定不变，global goal active。[完整E20报告](../automation_reports/CV-SincNet/20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01/E20_scored_summary.md)。


<!-- ORIGINAL_SIGTIME_TECHNICAL_FAILURE_20261015 -->
2026-10-15旧SIG_TIME×RDrop原调度器Popen后即时/proc观测失败退出；独立核对14模型已启动、2E50健康（其中TWOCE seed02未被记录）、4行未启动，原12行预测完成/11评分。E50源资格也已PASS。保留所有产物和健康任务；不原地重启或重复训练。待本地复现修复、新不可变技术run只补四个未启动行，原预测缺评分另按原自动truth-last闭合，最终逻辑矩阵仍精确18。E20完整三seed汇总83.115675/82.486111/83.942857%，全目标active。


<!-- ORIGINAL_SIGTIME_TECHNICAL_RECOVERY_PREREG_20261015 -->
20261015-phase1-originalSIGTIME-rdrop-e50-technical-clean-manysig-m4-r02已登记四个原未启动E50行；配方不变、新不可变root。先留PID再有限观测，启动回归8/8与配置负例8/8PASS，待一次聚焦P1审查后发布。原14不重训，缺评分仅固定prediction后闭合。全目标active。


<!-- ORIGINAL_SIGTIME_TECHNICAL_RECOVERY_LIVE_20261015 -->
20261015-phase1-originalSIGTIME-rdrop-e50-technical-clean-manysig-m4-r02 四行全部 RUNNING/启动核验 VERIFIED，模型代码 b3ac442e2d33a5b315fb3251c30e0ad7083566bc，调度代码 33f6171a0fec6ac432df672d0c6f542b8d71434c。原14行已全部 clean/独立评分，未停止、重训或覆盖。逻辑18行固定矩阵保留，尚待新4行终点；全目标 active。


<!-- ORIGINAL_SIGTIME_RDROP_COMPLETE_20261015 -->
2026-10-15 `20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01` 实际18/18行已ANALYZED/VERIFIED，139860更新/630epoch，保留旧RAW_HZ/LS.01/普通增强/默认头joint配方。E20_ORIGINAL 83.115675% n=3；E20_TWO_CE 82.486111% n=3；E20_RDROP 83.942857% n=3；E50_ORIGINAL 82.922024% n=3；E50_TWO_CE 81.986111% n=3；E50_RDROP 84.340079% n=3。E20_RDROP minus E20_ORIGINAL +0.827183pp 正2/3；E20_RDROP minus E20_TWO_CE +1.456746pp 正2/3；E20_TWO_CE minus E20_ORIGINAL -0.629563pp 正0/3；E50_RDROP minus E50_ORIGINAL +1.418056pp 正3/3；E50_RDROP minus E50_TWO_CE +2.353968pp 正3/3；E50_TWO_CE minus E50_ORIGINAL -0.935913pp 正0/3；E50_ORIGINAL minus E20_ORIGINAL -0.193651pp 正1/3；E50_TWO_CE minus E20_TWO_CE -0.500000pp 正0/3；E50_RDROP minus E20_RDROP +0.397222pp 正2/3。E50_RDROP/E20_RDROP完整三seed84%检查分别True/False；不足三seed不冒充完整矩阵。固定源资格及缺项保留，无target反馈或选择性救跑。既有正负证据和未发布原型保留，RDrop不作为全球新方法；goal保持active。[完整报告](../automation_reports/CV-SincNet/20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01/report.md)。


<!-- ORIGINAL_SIGTIME_RDROP_84_MILESTONE_20261015 -->
预登记E50_RDROP完整3seed clean均值84.340079%±0.838539pp，匹配ORIGINAL82.922024%±1.363415pp；净+1.418056pp/3正，vsTWO_CE+2.353968pp/3正。固定全18行/630epoch/139860更新/3024000决定/324CM独立复算VERIFIED，三个seed内各198初值张量在六条件相等，所有部署joint160和默认W真实更新，headfit0。clean三seed均值84%性能里程碑已达，不能说每seed均84%或全球新机制。E50较E20_RDROP overall+.397222pp/2正，但worstRX−2.951389pp如实保留；不按target改主条件/日程/重跑。RDrop已知方法、机制创新要求仍需独立源域研究，因此总goal保持active。原14+新4、缺一个原launch receipt、故障和全部负结果保留。[最终报告](../automation_reports/CV-SincNet/20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01/report.md)。


<!-- SIGTIME_RDROP_DIFFERENCE_SOURCE_PREREG_20261015 -->
用户范围已收敛SIG_TIME＋RDrop。本试 20261015-phase1-SIGTIME-rdrop-difference-gradient-sourcefold-s2026092702-m4-r01 PLANNED：FULL_KL/FEATURE_KL/SHRINK/DIFFERENCE固定seed02各scratch20，旧RAW_HZ/原增强/LS.01/defaultheadjoint/oldU保持；仅KL梯度对象/共同分量改变。CPU/CUDA各51negative与5gatecases、198初值/原logits/前向目标/RNG逐位相同PASS。真RX1先全路径排除，最终1260源L和固定四控制资格，无target、无旧CP继承。diff投影不保证encoder中心或AdamW步长不变；文献先例与代价记录。本试尚未发布或真实训练。全goal active。[报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-difference-gradient-sourcefold-s2026092702-m4-r01/report.md)。


<!-- SIGTIME_RDROP_DIFFERENCE_SOURCE_RESULT_20261015 -->
SIG_TIME＋RDrop 差分 KL 源域四条件 E20 完成，run `20261015-phase1-SIGTIME-rdrop-difference-gradient-sourcefold-s2026092702-m4-r01`；完整 17760 步与 80 model-epoch、默认头、真实 RX1 留出和无 target 核验 VERIFIED。预登记资格 FAIL。DIFFERENCE / FULL_KL / FEATURE_KL / SHRINK 的 RX1 错误分别为 247 / 284 / 291 / 286（各1260）。关闭固定差分梯度算子，不扩种子或用目标评分救跑。当前仅优化 SIG_TIME＋RDrop，既有 E50 RDrop 84.340079% 三种子均值继续保留；总目标仍 ACTIVE。详见相同 run 的 report.md。


<!-- SIGTIME_RDROP_RX_EVIDENCE_PREREG_20261015 -->
固定20261015-phase1-SIGTIME-rdrop-RX-evidence-sourcefold-s2026092703-m4-r01四条件FULL/LOCAL/SAME_RX/CROSS_RX，下一源seed03各scratch20，RX1全路径先排除，默认头联合训练与旧配方不变。独立RDrop一致性对象假设：保留自包binary KL，在同身份跨RX间传输conditional混淆关系，停止关系质量权重梯度；同RX和跨RX有效配对质量匹配。无差分投影、父模型或target。CPU/CUDA57负例/5资格/198初态及原logits/CE/RNG、零真类raw logit直接关系梯度PASS。新四行尚未发布，科学资格见[同run报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-RX-evidence-sourcefold-s2026092703-m4-r01/report.md)。goal仍active。


<!-- SIGTIME_RDROP_RX_EVIDENCE_RESULT_20261015 -->
20261015-phase1-SIGTIME-rdrop-RX-evidence-sourcefold-s2026092703-m4-r01四组source-only E20已ANALYZED/VERIFIED，17760步/80epoch，默认头联合，RX1全路径留出。FULL/LOCAL/SAME_RX/CROSS_RX错误依次为335/335/355/364（各1260），固定资格FAIL；逐包CE转移分解、实际梯度/更新/特征导出和完整日志已核验，0target/另训头。关闭固定配方，不扩种子或用目标救跑。 仅继续SIG_TIME＋RDrop机制研究；既有E50三seed84.340079%里程碑保留，goalactive。[同run报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-RX-evidence-sourcefold-s2026092703-m4-r01/report.md)。


<!-- SIGTIME_RDROP_PAIRED_MASKS_PREREG_20261015 -->
仅SIG_TIME＋RDrop新假设：固定13处原生失活边际p，独立采样与不重复丢弃配对分别用双CE/原RDrop，2×2四行scratch20，固定seed01，默认头联合训练、RX1全路径先排除、旧U与其他配方不变。ANTI_RDROP必须比三对照少错且CE不升，错误交互项严格负；FAIL关闭固定配方。无历史CP/target。CPU/CUDA58负例与原生数值/RNG、掩码审计PASS。20261015-phase1-SIGTIME-rdrop-paired-masks-sourcefold-s2026092701-m4-r01尚未发布，见[预登记](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-paired-masks-sourcefold-s2026092701-m4-r01/report.md)，goal仍active。


<!-- SIGTIME_RDROP_PAIRED_MASKS_RESULT_20261015 -->
20261015-phase1-SIGTIME-rdrop-paired-masks-sourcefold-s2026092701-m4-r01四组source-only E20已ANALYZED/VERIFIED，17760步/80epoch，13处原边际p的成对失活实审通过，默认头联合、RX1全路径留出。INDEP_CE/ANTI_CE/INDEP_RDROP/ANTI_RDROP错误依次为284/319/192/187（各1260），错误交互项-40，固定资格PASS；0target/另训头。仅允许另行登记主实验。 历史E50三seed84.340079%保留，仅继续SIG_TIME＋RDrop机制研究，goalactive。[同run报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-paired-masks-sourcefold-s2026092701-m4-r01/report.md)。


<!-- SIGTIME_RDROP_PAIRED_MASKS_MAIN_PREREG_20261015 -->
源域固定条件PASS后登记20261015-phase1-SIGTIME-rdrop-paired-masks-e20e50-clean-manysig-m12-r01：ANTI_CE/ANTI_RDROP×E20/E50×3固定seed，12次从零、420epoch/93240更新；复用已实际匹配12个独立采样历史控制，不继承其或源折CP。完整5RX L6300/U56700hidden/V27000，默认头联合，逐row自动clean预测/独立truthlast。主判据E50均值≥84%、相对原RDrop均值正且至少2/3正、平均错误交互负；无目标反馈。尚未发布，goalactive，见[预登记](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-paired-masks-e20e50-clean-manysig-m12-r01/report.md)。


<!-- SIGTIME_RDROP_PAIRED_MASKS_MAIN_E20_RESULT_20261015 -->
20261015-phase1-SIGTIME-rdrop-paired-masks-e20e50-clean-manysig-m12-r01的6个新E20模型和6个历史控制完整三seed对比已VERIFIED。新ANTIRDROP均值83.592262%，原INDEPRDROP均值83.942857%，配对均值-0.350595pp，0/3正；原RDrop正收益保留。E50六行仍健康按固定50轮执行，主判据PENDING，不据E20结果改预算/参数/seed或重跑。完整日志、默认头/实际joint160和truthlast逐row审计齐备；goalactive。[E20结果](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-paired-masks-e20e50-clean-manysig-m12-r01/E20_summary.md)。


<!-- SIGTIME_RDROP_PAIRED_MASKS_MAIN_RESULT_20261015 -->
20261015-phase1-SIGTIME-rdrop-paired-masks-e20e50-clean-manysig-m12-r0112新+12旧控制完整ANALYZED/VERIFIED，新增420epoch/93240更新、24行4032000决定与432指标，默认头联合、无U真值/拟合头。E50新RDrop clean3seed均值84.295040%，相对旧RDrop-0.045040pp、2/3正；E20对应-0.350595pp、0/3正。预登记主条件FAIL。关闭此固定配方，不用目标反馈救跑或选预算，继续仅RDrop新假设。[完整报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-rdrop-paired-masks-e20e50-clean-manysig-m12-r01/report.md)。


<!-- SIGTIME_RDROP_MARGIN_ENVELOPE_PREREG_20261016 -->
仅 SIG_TIME＋RDrop：固定同包身份间隔包络、停梯度共享软目标；FULL/MEAN/同真类置信度/包络四臂 scratch E20、seed01、RX1 真排除、原 U/默认头联合训练不变。CPU/CUDA56负例与5资格否决通过，一次P0/P1 PASS。PASS仅固定RX3源确认，FAIL关闭，不据目标调参。20261016-phase1-SIGTIME-rdrop-margin-envelope-sourcefold-s2026092701-m4-r01 已发布，四行启动 VERIFIED，见[预登记](../automation_reports/CV-SincNet/20261016-phase1-SIGTIME-rdrop-margin-envelope-sourcefold-s2026092701-m4-r01/report.md)，goal active。


<!-- SIGTIME_RDROP_MARGIN_ENVELOPE_RESULT_20261016 -->
20261016-phase1-SIGTIME-rdrop-margin-envelope-sourcefold-s2026092701-m4-r01 四臂 source-only E20 已完成独立核验，80 model-epoch/17760步，默认头联合，RX1全路径排除。FULL/MEAN/CONFIDENCE/ENVELOPE 错误依次 195/264/270/286（各1260），固定资格 FAIL，关闭本配方。逐包CE转移与全日志、实际浅中层/SIG/W更新和导出边界均核验，0target/另训头。保留原版RDrop正收益及全部负结果，goalactive。[完整报告](../automation_reports/CV-SincNet/20261016-phase1-SIGTIME-rdrop-margin-envelope-sourcefold-s2026092701-m4-r01/report.md)。


包络闭合后的源证据：FULL/MEAN/CONFIDENCE/ENVELOPE 固定final20原生KL分别 0.035320/0.058906/0.068049/0.068834。后三者LCE/V略好、U接受观察更多，但heldRX1更差；不能仅靠更好V/更大置信度/接受量判定泛化。下一动作先做原版FULL同权重同L同mask的现有算子梯度诊断（尚未执行），不拟合头或救跑包络；仍仅SIG_TIME＋RDrop，goalactive。


<!-- RDROP_OPERATOR_PROBE_PREREG_20261016 -->
原版 FULL 固定源 final20 同权重同输入梯度诊断 20261016-phase1-SIGTIME-rdrop-operator-gradient-source-s2026092701-r01 已登记未发布：144fitL，CLEAN/NATIVE_AUG×4固定dropout draws，复用同一paired图比较4旧算子和CE，实际logits/joint160/t1/t2/SIG/defaultW梯度；0更新/头拟合/U/V/RX1/target IQ，父训练20轮单列。不是新训练候选，不救跑包络，不设科学晋级阈值；CPU9负例与真实梯度接线PASS。见[诊断方案](../automation_reports/CV-SincNet/20261016-phase1-SIGTIME-rdrop-operator-gradient-source-s2026092701-r01/report.md)，goalactive。


<!-- RDROP_OPERATOR_PROBE_RESULT_20261016 -->
20261016-phase1-SIGTIME-rdrop-operator-gradient-source-s2026092701-r01 固定原版 FULL 源 final20 诊断 ANALYZED/VERIFIED；144fitL、两上下文各4dropout抽样、2304真实身份前向包次、240记录、0更新/头拟合/U/V/RX1/target IQ。NATIVE_AUG中 MEAN 的 t1/t2/SIG 范数约为 FULL 的18.05%/18.50%/19.74%，包络约42.06%/38.95%/48.46%；同loss系数不等于同特征梯度，方向也改变，不能归因于单因素。全部state/buffer/gradslots/counter保持，独立核对终态/摘要数学，未重算梯度向量或捕获live过程。下一独立机制研究 FULL 未覆盖的特征变化，不扫参数救援包络。[完整报告](../automation_reports/CV-SincNet/20261016-phase1-SIGTIME-rdrop-operator-gradient-source-s2026092701-r01/report.md)。goalactive。


<!-- RDROP_CURVATURE_COMPLETION_PREREG_20261016 -->
仅SIG_TIME＋RDrop，原fullKL保持，按默认头centered normalized-W rank5的可见/余空间增加等迹特征稳定性；FULL/SPAN/ISOTROPIC/NULL，固定seed02各scratchE20，RX1全路径排除、默认头联合。CPU/CUDA53协议负例/7科学否决、198初态及实际梯度/RNG检查PASS。NULL须同时少错且CE不升，FAIL关闭，PASS仅另行固定RX3源确认。20261016-phase1-SIGTIME-rdrop-curvature-completion-sourcefold-s2026092702-m4-r01 RUNNING，四行启动VERIFIED，首轮真实几何/等迹信号通过，尚无最终源评价。[固定方案](../automation_reports/CV-SincNet/20261016-phase1-SIGTIME-rdrop-curvature-completion-sourcefold-s2026092702-m4-r01/report.md)。goalactive。


<!-- RDROP_CURVATURE_COMPLETION_RESULT_20261016 -->
20261016-phase1-SIGTIME-rdrop-curvature-completion-sourcefold-s2026092702-m4-r01 四臂source-only E20已ANALYZED/VERIFIED，80model-epoch/17760步、默认头联合、RX1全路径排除。FULL/SPAN/ISOTROPIC/NULL错误依次281/283/263/273（各1260），固定资格 FAIL，关闭本配方。完整训练目标/几何/等迹审计与源逐包预测、特征实际更新及导出边界已核验，0target/另训头。原版RDrop正收益及所有负结果保留。[完整报告](../automation_reports/CV-SincNet/20261016-phase1-SIGTIME-rdrop-curvature-completion-sourcefold-s2026092702-m4-r01/report.md)。goalactive。


同批全空间ISOTROPIC对照相对FULL修正45/新增27，净少错18，held RX1准确率+1.428571pp、CE−0.050462。保留为单seed源对照的未确认正迹象，不替换主NULL、不触发RX3或主目标矩阵。下一仅可做本批冻结joint160无拟合参数的源跨RX几何诊断；没有新训练或换头授权需求，原关闭规则不变。详见20261016-phase1-SIGTIME-rdrop-curvature-completion-sourcefold-s2026092702-m4-r01/report.md。


<!-- RDROP_FEATURE_GEOMETRY_PREREG_20261016 -->
20261016-phase1-SIGTIME-rdrop-feature-geometry-source-s2026092702-m4-r01 PLANNED：父curvature4源模型固定final20，只读actual joint160跨RX特征关系，144fitL+1260heldRX1L共同物化1404、4模型5616前向、8ownCPsynthetic smoke、0训练/拟合头/U/V/targetIQ。每模型3528000同day三元比较是相关组合而非独立样本，1modelseed/4conditions。CPU14负例及实际模型边界PASS。无晋级阈值，父NULL FAIL不变，不扩训练/目标。[固定方案](../automation_reports/CV-SincNet/20261016-phase1-SIGTIME-rdrop-feature-geometry-source-s2026092702-m4-r01/report.md)。goalactive。


<!-- RDROP_FEATURE_GEOMETRY_RESULT_20261016 -->
20261016-phase1-SIGTIME-rdrop-feature-geometry-source-s2026092702-m4-r01 ANALYZED/VERIFIED：1404shared源L、四冻结模型5616前向、8syntheticCPsmoke、0训练/拟合头/U/V/targetIQ。直接欧氏距离独立重算4份features、576anchor/344groups，计数全对齐。FULL/SPAN/ISOTROPIC/NULL关系满足率92.475709%/92.257795%/93.103373%/92.906293%；ISO相对FULL+0.627664pp，day3/3、fitRX2/4、TX4/6正，72cell31正30负11平。raw范数10.077678→19.100467，总体均值能量占比84.5333%→90.4530%，事后矩统计不改变既定关系或资格。支持同源折特征层正迹象而非仅W，但不等于独立训练/新TX/目标证明；NULL FAIL保持，原RDrop三seed正收益保留。下一独立训练假设需区分方向一致性与范数/公共分量变化并另固定公平源对照。[完整诊断](../automation_reports/CV-SincNet/20261016-phase1-SIGTIME-rdrop-feature-geometry-source-s2026092702-m4-r01/report.md)。goalactive。
