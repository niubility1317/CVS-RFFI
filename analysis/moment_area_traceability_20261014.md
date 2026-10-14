# 报告落实记录：统计形状与有序面积（2026-10-14）

依据用户附件 e2e8f861-e763-457f-821f-b9041c097e79/已粘贴的文本.txt，现按报告路线一先实施单因素描述子比较。已有 MixStyle 三种子结果已完整回收，更新为80.7246%，相对同路径控制无稳定收益；SIG_TIME统计组合已完成三种子闭环，组合82.4972%，相对三路SIG_TIME −0.6883个百分点，0/3正差；全部负结果已上传。此轮依据报告及实际算子结构，不依据当前首seed目标分数调参数或选父权重。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| R01 | 三·路线一 | 同6D投影、四窗，μ/方差/协方差 vs μ/方差/有序面积，27D同固定lift→28 | experiments/cvs_moment_area50/features.py | verified | local_check.json：公式、同lift、同初始参数PASS | 只替换15个交叉关系槽位 |
| R02 | 三·路线一 | 同输出112→160/norm/gain/参数量及初始化，加入真实joint160 | model.py/features.py | verified | 三真实native：联合grad、新参数数值更新、精确导出、W独立、teacher deepcopy PASS | 保持t2完整输出，暂不改Dropout取点 |
| R03 | 五·约束 | 默认cosine联合训练，禁止额外头/探针拟合 | source.py/predict.py | verified | 首seed3实际source closures均零拟合；feature-only导出/W独立PASS | 旧门控EMAcos/scratch50/33+17 |
| R04 | 五·顺序1 | MixStyle全部种子回收且不按负分改计划 | 已提交三种子MixStyle报告 | verified | f1d1e860e；1512000预测/162指标独立复算 | 历史RX/day探针违规如实披露 |
| R05 | 二·结构分析 | 区分面积、统计形状和常数时间坐标；不把顺序敏感视为泛化证据 | checks.py/report | verified | float64：cov置乱不变、area反转/内部顺序及时间项恒等式PASS | 性能只由真实冻结测试证明 |
| R06 | 三·路线三 | 独立比较t2与t2.pool取点 | experiments/cvs_read_position50及seeds | verified | 完整3seed/9row/66600steps/1512000pred/162metrics独立复算VERIFIED | PRE82.5599vsPOST82.6909，无稳定提升 |
| R07 | 三·路线二 | 连续4→2→1层级含三个边界增量和统一全局时间 | experiments/cvs_hierarchy50及seeds/source_hierarchy | verified | 完整3seed15row/133200新增steps/2520000pred/270metrics/12真实feature witness复算VERIFIED | HIER_ORDER82.8734%，vsSIG+.1784pp、2/3正差，84均值未达 |
| R08 | 三·路线四 | 可信门控内有界stopgrad难度权重及真实训练梯度 | experiments/cvs_source_weight、cvs_trusted_weight50及seeds | verified | 完整3seed9row/66600新steps/1512000pred/162metrics/11988真实梯度记录复算VERIFIED | weighted82.7030% vsuniform82.7514%，−.0484pp、1/3正差；84未达 |
| R09 | 五及目标 | 首seed同路径SIG_TIME与两个描述子、旧基线；源PASS才扩02/03；自动默认头truthlast、Git | design/source/dispatch/score/report | verified | 完整三seed12row／99900steps／2016000默认预测／216组指标复算VERIFIED，Git交付 | 不使用目标分数控制扩seed |

当前最高风险未完成项：新描述子能否在真实source训练中更新并保持合法默认预测路径；局部算子通过不等于识别提升。实施按路线一的公式严格对齐，其他路线按原报告顺序有意后置；无需新增报告审批、签名或数据重验。


已执行：verified Python -m experiments.cvs_moment_area50.checks、package_check.py、check_control_context.py，均 PASS。本地不读源或目标 IQ。实际训练/预测/评分尚未发布，不将这些局部检查当作性能结果。审查发现的 FEATURE_CONTROL 参数比较硬编码已修复，COV→AREA 定向回归 PASS。


实际发布 VERIFIED：run 20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s01-m4-r01，执行提交 512c10a267b5e654cfd2c983ab1debdeebc7f626；真实checkpoint无查询smoke、归档传输比较、remotecompile和三worker进程/CWD/命令/GPU/日志增长均PASS。R03/R09仍是implemented：完整训练收尾零拟合及预测/独立评分须等真实产物，不用RUNNING冒充verified。


首seed真实执行/独立复算已完成：200epoch记录、33300新增steps、672000固定默认头预测、72分组指标 VERIFIED；R03升级verified。R09源PASS→原样扩02/03已实际执行，完整三seed仍待六模型结束，维持implemented。现为5项verified、1项implemented、3项deferred、0项rejected/blocked。严格按路线一公式实现，无报告近似替换。最高剩余风险是三seed有序面积能否提供稳定性能收益，首seed实际为负；不依据此目标分数改参数或取消健康实验。

实际验证入口：python -m experiments.cvs_moment_area50.checks（既有一次局部PASS）；python -m experiments.cvs_moment_area50.audit（已固定产物复算，无新forward）；python -m experiments.cvs_moment_area_seeds.publish --output <delivery> --inspect（实际PID/CWD/argv/GPU/日志增长VERIFIED）。


R06已独立实施源诊断：20261014-phase1-source-dropout-descriptor-manysig-s01-m1-r01，执行b9144fee45b26342442e8a39bbe475bd19c03bd5，源feature area扰动比31.9940%，8mask不是模型seed；真实训练取点对照仍待落实，因此标implemented而非性能verified。现5verified／2implemented／2deferred，0rejected/blocked。


R06独立真实训练已发布：20261014-phase1-read-position-oldgate-cos-e50-clean-manysig-s01-m3-r01，实现087b3f61abb4444f05e7c22f4af6f631865188df，执行b4d55dba50c8601017604a2a0eb0d037f8d835e8。只移descriptor取点，主干Dropout.1/公式/参数保持；本地真实两模型gradient/数值更新/teacher/export/W独立/初态参数和eval函数PASS，实际resolved取点/零fit配置正确。完整训练结果仍pending，R06维持implemented；5verified/2implemented/2deferred、0blocked/rejected。既有R1六模型继续，不热改旧release。


R09完整3seed闭环 verified：9新/3历史，12row／450新增model-epochs／99900steps／2016000默认预测／216指标；所有9新模块数值更新/导出/W独立/零拟合真实证据完成。AREA82.0833%相对COV −0.5355pp、SIG −0.7097pp，均1/3正差；84未达。R09是执行及交付要求已验证，不代表性能目标满足。最新为6verified／1implemented（R06实际取点训练仍pending）／2deferred，0blocked/rejected；最高剩余风险是取点优化完整3seed性能，不能以源Dropout回放证明收益。路线一公式严格对齐，无近似替换。验证命令：python -m experiments.cvs_moment_area_seeds.audit（只读完整固定产物），剩余取点run尚未发布，未宣称远端执行；仅已做4实际模型context/源资格负例/两种子source汇总检查，不冒充远端执行。


R06首seed实际训练/默认评分完成：PRE83.3595% vsPOST83.7488%，−.3893pp；源初始function同、最终L2 2.131700、0额外fit、22200steps/504000pred/54metrics复算VERIFIED。02/03在target读取前依据SOURCE PASS原样发布4新模型，执行038ec31397797e4631a969d0da8167de2ba861c3，完整3seed未完成，R06维持implemented。6verified/1implemented/2deferred，0blocked/rejected；最高剩余风险是完整取点效果，不能把source噪声诊断当提升。实际验证python -m experiments.cvs_read_position50.audit、python -m experiments.cvs_read_position_seeds.publish --output <delivery> --inspect 均VERIFIED；源码按报告route3只移取点，主干Dropout不变。


R07源几何已完成：20261014-phase1-source-hierarchy-geometry-manysig-s01-m1-r01，实现13728d263、执行4cc15c4db，720实际源包/135统计/0训练或分类头拟合。continuous/direct64误差8.882e-16，native32误差7.153e-7；省略cross-area的feature/time相对RMS37.0827%/87.0831%。必须保留global-time-only局部控制与同容量大窗统计控制，不能把描述子差异写成性能提升。6verified/2implemented/1deferred，0blocked/rejected；最高待验证是实际层级训练是否提供稳定收益及取点完整三seed结果。理论按一阶＋二阶反对称截断严格实现，不是全阶signature。验证：python -m experiments.cvs_source_hierarchy.checks；publish --inspect；results中的saved_geometry_recount=VERIFIED。


R06完整3seed已闭环verified：PRE82.5599%±1.5656pp、POST82.6909%±1.6836pp，配对−.1310±.2239pp、1/3正差，净正确−660/504000。所有6新模型pool实际更新、初态L720function完全同、最终源z差异、精确导出/W独立和native零额外拟合核实。450epoch记录/66600新增steps/1512000固定pred/162指标复算VERIFIED。现7verified/1implemented（R07网络尚未训练）/1deferred（R08），0blocked/rejected；最高剩余风险是层级实际训练是否有收益，源几何非性能。验证：experiments.cvs_read_position_seeds.audit固定保存产物重算，未新增forward。


R07连续层级网络已实现并发布：20261014-phase1-hierarchy-oldgate-cos-e50-clean-manysig-s01-m5-r01，实现588209f4186e3056a310458237f4e478c6c15a64、执行7bdfdbf448b9c97ead5702f958bee293d646a09f；4新scratch50/11100+1历史oldgatecos，singleprojection/global32/3bridges/4→2→1，LOCAL_GTIME和同容量HIER_STAT控制。native4模型grad/update/defaulthead/teacher/export/W独立及连续direct梯度检查PASS，一次P0/P1 PASS；实际PID/CWD/argv/GPU5/6/4/7/日志增长VERIFIED、nativefittedprobes关闭。公式按报告一阶+二阶skewarea严格实现，统计/签名原始量级信息非完全匹配已披露。剩余02/03 20261014-phase1-hierarchy-oldgate-cos-e50-clean-manysig-s0203-m10-r01 已实现登记/8实际modelcontext及sourcegate negatives PASS，但源资格未完成、未发布。当前7verified/1implemented/1deferred、0blocked/rejected，最高剩余风险是HIER_ORDER完整3seed能否稳定强化SIG_TIME，不把RUNNING或源几何当性能。验证命令python -m experiments.cvs_hierarchy50.checks，native check_control_context.py，hierarchy50.publish --inspect；remaining check_context/source_negative只局部PASS。


R07首seed真实闭环：20261014-phase1-hierarchy-oldgate-cos-e50-clean-manysig-s01-m5-r01，4新+1历史，250epoch记录/44400steps/840000defaultpred/90指标复算VERIFIED；HIER_ORDER82.7732% vs同容量HIER_STAT83.9994%（−1.2262pp），vsSIG_TIME83.5899%（−.8167pp），不将83.9994四舍五入作84达标。全部4新增参数更新、导出/W独立/零拟合和匹配初态完成；L720候选−统计zL2 2.944889。source PASS后先提交 63f59df3933c6fcb778c76d9b1eaed3087f4213c 再发布原样02/03八新模型，实际8进程/配置/GPU/日志增长VERIFIED，target值随后读取。现7verified/1implemented/1deferred，0blocked/rejected；最高风险完整3seed层级效果仍未证实，source PASS非性能PASS。严格一阶+二阶skewarea/3bridge/globaltime实现；统计与签名rawscale/信息不同披露，不能从配对单独断言顺序因果。验证python -m experiments.cvs_hierarchy50.audit只读固定产物；hierarchy_seeds.publish --inspect确认活进程，不重复审查或测试。


R08源代理已实施完成：20261014-phase1-source-trusted-weight-manysig-s01-m1-r01，实现051de1a177819c0f8a9cfe5aba0fd09e7cefef6d、执行c84b17396bc0861b8856e867d17aab54206436c5；L6300/18900包前向/25batch/15分层/75实际t1/t2/t3导数记录保存复算VERIFIED。旧mask接受867包均正确，weighted precision未提高；同层λU/L由.244/.240/.228升至.357/.349/.320，cos .994至.996，不能冒充真实U或训练阶段/性能收益。0训练/optimizer/默认头或额外头拟合，student/EMA state保持。现7verified/2implemented/0deferred、0blocked/rejected。最高未完成项：R07完整三seed层级性能及R08实际33/34/50训练梯度/固定默认头结果。严格落实报告的stopgrad/bounds/accepted加权均值；源代理是明确范围缩小，不能标为完整训练路线已验证。验证 python -m experiments.cvs_source_weight.checks（一次PASS）、preflight真实CP2zeroIQ（PASS）、publish --inspect及source_weight.pt重读复算（VERIFIED）。


R07完整三seed闭环verified：HIER_ORDER82.8734%±1.9546pp，vsSIG+.1784±.9861pp（2/3）、vs同容量STAT+.3230±1.5336pp（2/3），距84均值1.1266pp。12新/3历史、600新增epochs133200steps＋150历史epochs；750源epoch/2520000固定defaultpred/270指标/21错误表保存复算VERIFIED，12实际pool5tensors全元素更新、导出差0/W独立/额外fit0；3匹配初态最终L720z差2.944889/3.629342/4.719027。现8verified/1implemented（R08只有源代理，实际训练待实现）/0deferred、0blocked/rejected。最高未完成项R08实际33/34/50骨干L/U梯度及固定默认头三seed性能，84目标仍未达；不是head-free新TX证明。层级公式严格一阶＋二阶skewarea/globaltime/3bridge；STAT rawscale/info/conditioning不完全匹配已披露。验证python -m experiments.cvs_hierarchy_seeds.audit复用固定产物，15行源和预测计数合并检查VERIFIED，无新增targetforward或重复审查。


R08真实训练已实现发布：20261014-phase1-trusted-weight-oldgate-cos-e50-clean-manysig-s01-m3-r01，实现23ea591bba17dd938d2b9a93f8b9d9dc5f02e97a、执行7708d4afcef141bb614e0c2171f2a9207c4469ff；原SIG_TIME特征/旧gate/defaultjointhead，acceptedCE stopgradraw.75至1.25/scale.1/mean-normalization，lambdaU.16/entropy.01/33＋17保持。native只添10行scoped hook，双真实模型正常backward/optimizer/defaultW/pool更新、初始parameters/z同、无observer RNG/state/.grad变化、33inactive与9配置负例及13compile PASS；一次直接P0/P1 PASS。实际CP无query smoke及两个GPU0/3进程/配置/日志增长VERIFIED，当前2epoch无failure；真实33/34/50共666batch/1998层批参数贡献、固定默认头成绩尚pending。现8verified/1implemented/0deferred、0blocked/rejected，最高未完成项R08完整真实训练和三seed收益；公式严格照报告，源proxy此前范围缩小已明确，未将其冒充真实训练。验证python -m experiments.cvs_trusted_weight50.checks、check_inactive.py、publish --inspect；未新增审批/签名/数据重验/重复审查。


R08首seed实际闭环：20261014-phase1-trusted-weight-oldgate-cos-e50-clean-manysig-s01-m3-r01，执行7708d4afcef141bb614e0c2171f2a9207c4469ff，weight83.5810%vsuniform83.7345%，−.1536pp，0/1。150完整epoch/22200新updates/504000defaultpred/54metrics/3996真实33/34/50层批grad保存复算VERIFIED，2新pool实际更新/export差0/W独立/默认fit0，候选-controlL720zL2=1.736686。源feature geometry额外2160源forward/0optimizer/无headfit，是非阻塞描述性证据，不能选模或作性能门槛。sourcePASS先于新target数值读取冻结fcc252f3938a21b415f5354b2a2e9db7315934bd，剩余02/03四新原样发布a8ff278fc8dd1b54b69279503c4cda9282795261、GPU1/2/4/6真实进程/日志增长VERIFIED，首seed不重训。现8verified/1implemented/0deferred，0blocked/rejected；最高待验证是R08完整三seed性能，梯度放大≠更强泛化。严格实现stopgrad/bounds/acceptedmean/原gate及entropy保持；源代理与真实梯度差异、源几何limitations全部披露。验证cvs_trusted_weight50.audit仅读完整保存产物和已有预测，3996梯度norm/cos及22200实际mask/loss逐步对齐VERIFIED；cvs_trusted_weight_seeds.publish --inspect核实4健康worker，未重复审查或热改健康release。


R08完整3seed9row已闭环verified：weighted82.7030%±1.6081pp、uniform82.7514%±1.6961pp，配对−.0484±.1039pp（1/3），净−244正确/504000。6新300epochs66600steps＋3历史150epochs；450完整epoch/1512000固定defaultpred/162组指标/11988真实33/34/50层批grad保存复算VERIFIED。六新pool5tensors全元素数值更新/export0/W独立/fit0，三初始source函数同、最终sourcezL2 1.736686/1.700653/1.804470。source几何9models/6480源forward/0optimizer/无classifierfit，ratio不是DG排序。现9verified/0implemented/0deferred、0blocked/rejected；verified指报告方法和矩阵落实，不是性能目标满足。最高未完成项为整体84%均值及更可迁移特征改善；不能以gradientamp/sourcegeometry/旧head得分冒充newTX或Phase2证据。公式严格照stopgrad/bounds/acceptedmean/原gate/entropy保留，源proxy与真实训练范围不同全部披露。验证cvs_trusted_weight_seeds.audit重读完整现有日志/grad/pred，7992新增梯度层批/44400实际loss/mask步及108指标复算VERIFIED，再与首3996/22200/54合并；cvs_trusted_weight50.geometry保存feature距离复算VERIFIED。未新增targetforward、头fit、候选重跑或额外审批/重复审查。


新报告31957f78独立浅中层有限身份监督已完整实现验证：完整浅中层有界排序3seed闭环：ORDER82.7127%±1.4787pp，SIG82.6067%±1.7438pp，SupCon82.5843%±1.7028pp，旧gatecos80.5268%±2.3776pp。ORDER−SIG+.1060±.2932pp/2 of3，修正2168引入1634净+534/504000；−SupCon+.1284pp2/3。12row9新450epochs99900steps＋3历史150epochs33300，600完整源epoch/2016000defaultpred/216metrics/5940实际层cohortgrad保存复算VERIFIED，0头拟合/新增targetforward。9fθ实际更新/export0/W独立；初态同，最终ORDER−SIGsourceL2 1.592485/1.398321/1.142613。更低SupCon源关系率和更强梯度没有均值收益，不据几何调参。实现26bfdf044caa1be3c966370248c3bc3372009943，剩余执行6d65f77a7659cb182bdb39852dc6c6b76001e81d，报告20261014-phase1-local-order-oldgate-cos-e50-clean-manysig-s0203-m8-r01/final_analysis.md；84未达成。 后续源签名诊断20261014-phase1-source-signature-order-manysig-s01-m1-r01全L6300/4视图126000关系各/840辅助＋210CE梯度，实际投影/浅中层有梯度、WNone/0fit；常量坐标平方范数均值17.6511%，去常量改变关系率不证明DG。未新增门槛、hash、头拟合、重跑或重复审查。


后续签名内监督最小消融（源诊断依据，目标未用于调参）：
|ID|来源|要求|目标文件|状态|验证|说明|
|S01|全L签名四视图诊断|直接SIG112与仅辅助去4常量SIG108身份排序，各自与零损失控制|experiments/cvs_signature_order50/design.py,observer.py|pending|待实际native接线|固定m.1/coeff.1/5warmup+10ramp，单路部署SIG112不变|
|S02|当前默认头约束|W正常联合训练、无辅助W依赖、真实fθ更新及W独立导出|checks.py,feature_check.py|pending|待native梯度/实际CPsmoke与训练|全L数据角色、旧gateEMAcos33+17、每模型scratch50不变|
|S03|源资格与自动收尾|首seed三新＋旧历史，源资格固定后仅原样扩预定义02/03；每row自动defaultclean/truthlast|score.py,dispatch.py,publish.py|pending|待一次P0/P1/归档/真实启动|不以目标分数或源几何选父权重、调参、重跑|

S01/S02本地实现与三实际native检查verified：8negative，actualt1/t2/SIGprojection更新、同初态warmup、1800显式公式、W辅助None/defaultjointupdate、mode/RNG保持PASS；唯一直接P0/P1 PASS。实现f6dfdb587c505b5720469d4a4a101b8df8b321f4；S03自动defaultclean/truthlast/源资格已实现，远端真实checkpoint/archive/compile/启动证据待发布，不把实现当训练结果。

S03发布路径verified：20261014-phase1-signature-order-oldgate-cos-e50-clean-manysig-s01-m4-r01，执行2658815a2fe5ef931f24558a6c6c61146433fd9c，realSIGCP noquery90/单归档一次SHA/remotecompile及3实际samePID/CWD/argv/GPU/loggrowthVERIFIED，首3/2/2epoch，继续固定50及自动defaultclean/truthlast。S01/S02实际训练更新/完整多seed性能仍implemented待证据，不以本地检查或启动替代结果，0deferred/blocked/rejected。


签名内监督健康首seed运行时，仅准备原样02/03条件扩展包装：不新增科学假设，不修改首seed已发布archive。新增wrapper复用相同Observer/模型/预算，检查first执行2658815a2及bothSOURCEPASS后才可发布；无目标数字选择。此步骤为implemented待验证，不代表已扩种或已有性能。


用户2026-10-14最新重心已纳入主报告：SIG_TIME网络足够，固定网络/默认头，研发泛化训练与半监督机制。当前signature内监督仅trainingloss改变，不改部署结构；后续不再层级/分支/阶数/位置扩展。半监督先核对现有sourcegate/PAIR实现与同SIG参照，再选独立学习机制，禁止将覆盖率或梯度幅值当作收益。


固定SIG_TIME半监督参照整合（用户最新重心＋报告31957f78第六节）：
|ID|来源|要求|目标文件|状态|验证|说明|
|U01|SIG_TIME＋PAIR统一起点建议|随机U vs原物理邻近PAIR，仅U顺序改变，网络/默认头/门控/EMA/总曝光固定|experiments/cvs_sigtime_pair50/design.py,observer.py|pending|待合成loader与真实SIG路径|已知方法整合参照，不冒充新算法/84核心机制|
|U02|当前梯度量不等于增量监督|实际33/34/50全部222批LCE、.16U_CE、.16U_CE+.01entropy真实三参数梯度，原loss不改|observer.py,checks.py|pending|待native图/无grad累积/RNG检查|mainL128 vs U256，不做同包/累计AdamW推断|
|U03|公平实验与默认收尾|首seed2新scratch50＋1旧历史；source资格冻结后再原样扩种，逐rowdefaultclean/truthlast|score.py,dispatch.py,publish.py|pending|待唯一P0/P1/实CPsmoke/真实发布|不使用Utruth/目标反馈，不叠加签名排序|

U01/U02本地verified：6negative、labelblind56700once/tail124、原random/RNG同、两actualSIG初态同、原Uloss对象identity不变、实际head与t1/t2/SIG投影更新、export差0/W独立，源码/remote模板compile PASS。唯一直接P0/P1 PASS，未重审既有PAIR算法；U03自动source资格/defaultclean/truthlast已实现，实际远端预算和性能待发布证据。

U03发布verified：20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s01-m3-r01，执行86c22cb76e5eff4a7b106cca769d198879ab22b4；realSIGCP noquery90/单归档一次SHA/compile及2actualsamePID/CWD/argv/GPU/loggrowth PASS，实际labelstage1至4无Ufetch，继续固定50/defaultclean/truthlast。U01/U02实际全程采样预算、U阶段梯度和性能仍implemented待证据，不把接线/覆盖当收益。

S01/S02首seed实际训练与默认测试verified：固定SIG_TIME签名内训练首seed闭环：RAW83.6363%、ZERO83.5815%（+.0548pp，修正588引入496净+92/168000）；DYNAMIC83.4476%（−.1339pp，修正573引入798净−225），DYNAMIC−RAW−.1887pp。4row/3新150model-epochs33300updates＋旧50/11100，200完整源epoch/672000defaultpred/72metrics/8250extraL90/1980实际层cohortgrad/7920向量保存复算VERIFIED，0头拟合。三fθ实际5pool张量全元素更新、export差0/W独立；初态函数同，源特征RAW/DYNAMIC−ZERO L2=1.310938/1.795317。更强gradient/去常量源几何不代表收益；仅n1、完整3seed84未达成。SOURCEPASS决定d56f9b79c9fea83629c525e362d473ff0fcafb1e先于新target数字回收，两候选原样02/03六新已发布同commit、samePID/CWD/argv/GPU/loggrowth VERIFIED（6/6/6/6/6/5），未择优、改参或重跑。报告20261014-phase1-signature-order-oldgate-cos-e50-clean-manysig-s01-m4-r01/final_analysis.md。 样本出现次数不冒充uniquecoverage，梯度不同L批不冒充累计AdamW；sourcequalified不等于DG证明。后续02/03性能仍implemented待证据，最新用户重心固定网络/研发训练与SSL保持。


固定SIG_TIME半监督参照健康训练时，仅准备同机制02/03条件扩展（无新科学假设、不改已发布archive）：源码资格固定finalLCE<=.5、PAIR整17U曝光接收>=.25、实际fθ变化/export/W独立及精确预算；首3row闭环SOURCEPASS后原样扩两臂，目标数字不参与。另实际第34轮两臂各56700U，RANDOM接受908/32emptybatch，PAIR51378/0empty；源loss/真实参数梯度已读回，尚不证明性能。


### 固定 SIG_TIME 半监督配对训练首种子

20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s01-m3-r01 已独立审计：PAIR83.8452%−RANDOM83.6738%=+0.1714个百分点，n=1；固定网络/默认联合头，U顺序为唯一机制差异；0额外前向/0头拟合；source资格提交 f5b19b97bb259de87737dda7a2366036604d12b1 后扩展02/03。完整分析与源证据见对应final_analysis.md/results.json；未完成三种子84%。


### 固定 SIG_TIME 签名身份排序：完整三种子

20261014-phase1-signature-order-oldgate-cos-e50-clean-manysig-s0203-m8-r01 ANALYZED：SIG82.6778%、RAW82.6952%、DYNAMIC82.6690%；配对 +0.0175/−0.0087个百分点，2/3和1/3为正；完整600源轮次/2016000默认决策/216指标、5940实际层cohort梯度均VERIFIED。5个SIG pool张量实际更新且export0/W独立，无另拟合头。未形成稳定DG收益，84%未完成。完整报告见对应final_analysis.md/results.json。


### 固定 SIG_TIME RANDOM/PAIR：完整三种子

20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s0203-m6-r01 ANALYZED：PAIR82.7667% vs RANDOM82.6645%，配对+0.1022±0.1220个百分点，2/3为正，净+515/504000；未形成可靠稳定收益，84%未完成。450源轮次/1512000默认决策/162指标/11988实际层批梯度独立复核；导出0差/W独立，单训头0。sourceonly三种子U覆盖91.5781%vs1.6146%，平均梯度更小，不能代替性能或伪标签准确率。后续只研发固定网络的训练/DG/SSL机制。


### 真实骨干梯度方差首种子准备

20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01 LOCAL_VERIFIED；code 9c981766a0727cf072755ddd46cd5b639bae89ce；固定SIG_TIME/defaultjointhead，真实t1/t2梯度，无extrahead/PAIR。source依据为此前L180真实方差诊断与一般Fishr思想；不是目标反馈选参或原论文逐项复现。局部9负例/真实2阶/训练hook/源资格窄回归和一次P0/P1 PASS，尚无新性能证据。


### 真实骨干梯度方差矩阵启动

20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01 RUNNING，execution 600afeeefe84e0d9f50f07c38ae0767bf1cabf07；三新实际首轮/13L180/2340额外包，预热有效R0、逐包参数梯度矩阵和生产端保存复算已核实。同PID/CWD/argv/GPU/loggrowth VERIFIED；完整独立审计与性能待终轮，无头拟合或Phase2声明。


### 固定SIG_TIME源域伪标签错误与实际参数梯度

20261014-phase1-source-sigtime-pseudo-gradient-manysig-m3-r01 ANALYZED，code d17f140c3528d8964ffedf464976532bc56e0760 / execution 3bc96d53291ca3b2f1eeb0c19236e236a8d25866。只读固定3个RANDOM final50/defaulthead+ownEMA，L实际t1/t2/SIGprojection同强图同接受分母的pseudo/true/correct/wrong梯度；V仅无梯度joint160描述。完整199800源包/786batch、1800向量独立保存复算VERIFIED。L接受错误0使错误梯度不可识别；V错误组特征一致性没有稳定更低，当前不支持简单特征阈值。0头/模型/优化器更新，U/query访问0；报告见对应source_pseudo_gradient_analysis.md/results.json。骨干方差训练保持健康继续，不把该诊断当DG收益或84%达成。


### 固定 SIG_TIME：教师学生特征一致性训练

20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01 RUNNING，code 373262b51f037f03e3238471b3aefb88e45fba5b，execution 6b23a5b033ee063dea97d6783d5634e220a1d93d。仅已有同U包joint160的EMAweak stopgrad/strong student单位特征损失，无额外模块/头/forward；原33+17旧门控+ownEMA+cos和UCE对象不变。实际网络t1/t2/SIG与原头数值更新、auxW梯度None/mainW梯度不变、导出/W独立及8负例PASS，一次P0/P1 PASS。两新worker GPU7/2实际PID/CWD/argv/loggrowth VERIFIED，仍前33轮监督阶段，实际U/最终性能待证据。源last5 featureloss5%降低+CE/边界/梯度决定扩固定02/03，目标不反馈。BF epoch17真实非零effective梯度已读回，非性能证明。


### 实际U特征梯度及固定02/03执行准备

FEATURE_MT原run34轮222批实际t1/t2/SIGprojection effective梯度候选全非零、ZERO全0，Waux梯度None检查及producer保存复算VERIFIED；36轮170100U/臂仅部分预算，目标/终轮独立审计待。BF02/03 20261015-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s0203-m8-r01 code b69d6ffbbd229284c4cfde35124db10bf871fe07 与FEATURE_MT02/03 20261015-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s0203-m6-r01 code 68b3143dbdfcde27c80997f970121a23e34e5f2e 已LOCAL_VERIFIED_NOT_LAUNCHED：配方未改、10/11 source负例PASS、同seed初态和args配对通过，无新假设/另训头。仅原首SOURCEPASS后发布，未根据目标排名/扫参或扩种子。


### 固定 SIG_TIME 均值教师首种子闭环与扩展技术恢复

20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01 ANALYZED/VERIFIED：FEATURE_MT 83.3619% vs SIG_TIME 83.6679%，−0.3060 个百分点，纠正544/引入1058，n=1。原baseline80.6685%架构不同，不能归因SSL。完整150源epoch、22200新step、504000默认决定/54指标、原生＋feature共7992层记录/23976向量独立重计；真实可部署joint160导出一致、W独立、193460 feature元素/默认960W元素更新，另训头0。源U discrepancy比.8214877且实际梯度生效，但未形成首seed泛化收益。源PASS扩展决定 ec8ea03f7810cb3eb1bcb8108cdb60b4053a59c9 先于目标数字读回提交，固定02/03原样继续。扩展R01缺失import在workers0/runroot不存在时FAILED，保留release；技术修复 729c01e878a0df790c1056d00d1b547f709a57be、针对P1复审PASS，R02 execution ab2c3127e14b48dea5bc2e064ce42c0be7cfdeeb 四新PID/CWD/argv/GPU/loggrowth VERIFIED，科学配方未变。完整三seed/84%仍待证据。


### 真实骨干梯度方差：首seed闭环与固定扩展

20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01 ANALYZED/VERIFIED：同计算SIG83.5333%、RAW83.7542%（+.2208pp）、TX_DAY_CENTERED83.6548%（+.1214pp），n=1、84%未达成。完整200epoch/33300新step/672000默认决定/72指标、1950L180/312真实参数梯度矩阵/468层cohort/1872向量独立重计；feature导出0差、W独立、实际SIG五tensor数值更新、头fit0。源PASS双候选原样02/03决定 3c052b2d36903e28694674a39e232a73acd8b6a6 已提交推送后才回收新目标数字，未凭RAW首seed优劣挑选。固定扩展 20261015-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s0203-m8-r01 已发布6新模型，原候选/系数/预算不变，完整三seed待证据；本轮无新增科学假设。


### 固定 SIG_TIME 均值教师：完整三seed负结果

20261015-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s0203-m6-r01 ANALYZED/VERIFIED：FEATURE_MT82.4359%±1.5874pp vs匹配SIG82.6806%±1.7906pp，配对−.2446±.2101pp、0/3正，纠正1660/引入2893净−1233。完整450源epoch/66600新step＋33300历史/1512000默认决定/162指标、23976层记录/71928向量均独立复核；6新真实feature导出0差/W独立、SIG五tensor均更新、单训头0。源Udiscrepancy各下降17.85/20.75/19.89%，不能替代DG收益。源决定 ec8ea03f7810cb3eb1bcb8108cdb60b4053a59c9 先于目标数字，R01失败/R02技术恢复产物保留，未按目标调参或筛seed。该冻结配方结论为负，84%未达。现只继续既定BF三seed，并计划同合法L强图CE/feature梯度源诊断（未实现/启动），不扫系数或另训分类器。


### 固定SIG_TIME同L强图CE/feature：六模型完整源诊断

20261015-phase1-source-feature-ce-gradient-manysig-m6-r01 ANALYZED/VERIFIED，code da1aafdde1931fcef67ecd509e079cfa371ec166 / execution 2fb6cf1ee4cdbf830f3921a19ef2346c4f1de185。两臂×三固定seed/各L6300，同强图trueCE(.01)与原feature(.1诊断权重)，四组共全批分母；CPU2threads122.91秒模型计时，头/optimizer/model更新0，V缓存物化但forward0，U/query0。独立150批/450层记录/4950向量复算，实际参数buffer不变。教师对学生错组18/18对象余弦正；教师错学生对16/18负但仅3至8个seen-L样本/模型，整体符号混合，不能解释实际U或目标下降因果。源诊断不是84%或新类能力。仅后续源证伪一个固定零置信差teacher>=student规则，尚未实现/启动，不按目标调阈值或另训头；BF6新仍健康24至27轮。

## 2026-10-15 固定置信度次序源证伪完成

20261015-phase1-source-teacher-confidence-manysig-m6-r01：代码e2eee613b9657d336657012eb33d060072a402d7、执行3003478354f9e9d3a45175f8140cae45f1f282eb；六个预定义50轮SIG/MT模型，L梯度/V仅no_grad，真实checkpoint+默认头/ownEMA不更新。固定FP32 teacherweak置信度>=studentstrong，ties kept，零阈值差且不扫描；V两个对错不一致组非空并6/6有益保留率严格高于有害，源资格PASS。399600forward包次/7200向量独立复算VERIFIED，头/optimizer/model/V梯度/Utruth/query0。整体梯度并非普遍改善，未证明实际U或目标增益；后续单一训练对照须另行实现/检查/预登记，不能将源资格当训练成功。详见automation_reports/CV-SincNet/20261015-phase1-source-teacher-confidence-manysig-m6-r01/source_teacher_confidence_analysis.md。

## 2026-10-15 置信度门控特征训练最小实现

20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s01-m3-r01：实际code325d113dff08206cb2f174e74d33e48f3e0b2ad5（主体cadd843d569e85b81b5e9e8bc14c73b0ab4e00ae），基于六模型TCF source资格PASS，只增加固定detached FP32 teacherweak>=studentstrong的joint160一致性mask/fullbatch分母。SIG_TIME/默认头/旧batch_neighbor+ownEMA+cos不改，2scratch50匹配首seed模型+历史baseline，原U CE/entropy完全保留，不堆叠Fishr，无新增前向/模块/头。local10负例/实际合成optimizer+featureexport等PASS；唯一P0P1发现发布依赖缺失，最小修复后定点PASS。source扩种子规则/逐rowclean168000+truthlast已预登记，未启动、无新训练结果。

发布后VERIFIED：20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s01-m3-r01执行704cf37536d3cb6b8764af2131547db4a9939480，实际SIG2588400/GPU3与CONF2588549/GPU6均完成第1轮，首步真实CPnoquery/单archiveSHA/remotecompile/PID-CWD-argv-GPU-loggrowth PASS，failure0。原baseline默认头自动clean链完成，新变体仍训练，feature从34轮启用，无新目标数值读回。源资格读取入口local_artifacts/confidence_feature50_delivery_r01/source_qualification_only.py，扩种子决定须先commit/push；不以启动或源PASS宣称性能提升。

## 2026-10-15 骨干Fishr完整三seed与门控SSL源决定

20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01 + 20261015-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s0203-m8-r01 全12行已ANALYZED/VERIFIED：RAW−SIG +0.0143±0.1918pp/1正，COND−SIG −0.1391±0.2340pp/1正，COND−RAW0正；216CM、2016000默认头决策、936参数梯度矩阵独立复算，9新模型450轮/99900更新、头拟合0。源penalty优化并无稳定DG增益，保留完整负结果，不扫权重。

20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s01-m3-r01 实际E34门控接线确认；完整预登记source规则PASS（retained比0.820886/allU比0.828006/coverage0.920871），在任何该run新目标数值读回前冻结 20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s0203-m6-r01 固定02/03同配方scratch50扩展（4新模型/200轮/44400updates），当前尚未实现或发布。该决定先commit/push，再回收首seed目标结果，目标分数不取消已确定扩展。


- 2026-10-15：CFM 源资格PASS后固定02/03扩展，source决定 3b0b38c708029570cd1d6327cd4db96e2752a6ad 在首目标数值前已推送核实。新增seed适配器 fe2d8287ef1f9c167c5b68ee395e8bfe01b08360 本地14负例/4native上下文与一次独立P0P1 PASS；默认头正常联合训练、原feature目标/系数/数据/预算不变。当前仅预登记，未运行，不混入旧模板结果。
