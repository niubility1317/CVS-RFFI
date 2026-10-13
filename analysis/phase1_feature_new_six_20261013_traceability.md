# Phase1 新六路线实现追踪（2026-10-13）

原说明：docs/design_reports/phase1_feature_six_routes_20261013.md。当前用户按目标84%、特征提取优先、历史旧伪标签门控＋EMA＋cos、每模型骨干不超过50轮、Git实现/结果上传的约束继续。

先执行归一化恢复与分支竞争。首轮增加精确同参数普通残差控制，形成BASE/RESIDUAL/NORM/COMP/NORM_COMP五条件；BASE固定引用当前旧门控E50同seed，不作为变体初始化。三个固定seed2026092701/02/03；先小同row科学检验，新增变体不根据现运行目标成绩改变。

已确认实际z为ReferenceCVS.features输出：core.features加tanh(response_gain)乘response_projection(response(x))。时域/频域先进入core.fuse；PA局部和频域统计增量还经原ResidualPhysicalHead组成feature。mask按实际投影贡献定义，不声称消除所有物理频率信息；参考响应作为明确第四路径，绝不漏掉。

冻结换头是共同诊断流程，不作为分类头创新。原头与普通cos重训分别报告；L6300拟合、V27000只读诊断，不能fit V或U真值。Source骨干50与诊断head每个10epoch分别披露。Phase2待匹配已有capsule/split；不会把source cache带入部署。

| ID | 来源 | 接受项 | 目标文件 | 状态 | 核验证据 | 说明 |
|---|---|---|---|---|---|---|
| G01 | 总则 | 保持历史旧门控EMAcos、三个seed、骨干每model最多50轮 | design.py/source.py | pending | 待固定E50配置 | 当前健康multiscale12行继续，不读取目标成绩决定新参数 |
| G02 | 总则/六 | 新增能力必须进入实际部署z160，不能仅存在辅助头 | features.py/checks.py | verified | 实际id_feat_joint差异与W独立性检查 | 原classifier结构与初始化不变 |  Local真实模型/noquery检查PASS，远端训练验证尚待。
| G03 | 总则/二 | U_s真值隐藏；额外屏蔽只作用L_s学生 | native L入口/features.py | pending | 明确out_l上下文、U/teacher不变负例 | 不是根据batch尺寸猜角色 |
| G04 | 总则 | NoExplicitCFO、内部补偿、L/U/V物理角色保持 | model/source | pending | 沿用既有已核实契约 | 原基线7/63/30 |
| G05 | 七 | 冻结骨干、普通cos换头收益是表征证据 | probe.py/predict.py | pending | L-only固定10epoch head，V只诊断，目标预测固定再评分 | 主骨干50轮与各诊断头10轮分开记成本 |
| G06 | 七 | 同Phase2 support下旧类与新类统一识别 | phase2 followup | deferred | 匹配VALIDATED_ONCE capsule/split与当前P1来源尚待定位 | G0先推进；未来阶段不构成当前额外发布gate |
| G07 | 总则 | 每轮实现、配置、负结果与完整结果推送Git | 报告/registry | pending | 独立remoteOID对比 | 原始数据/预测/权重/日志不入Git |
| N01 | 一.1 | 保留GN，在一个浅层位置研究被移除信息 | features.RelativeNormBlock | verified | 选t1，C72/G8；不删除GN | 不是恢复采集时已丢绝对幅度 |  Local真实模型/noquery检查PASS，远端训练验证尚待。
| N02 | 一.2 | q=[组mu/r,log组sigma/r]只读当前样本 | features.py | verified | 公式、有限梯度、整体增益数值诊断 | eps沿用GN；q不读TX/RX/概率 |  Local真实模型/noquery检查PASS，远端训练验证尚待。
| N03 | 一.2 | 逐通道恢复模式；alpha从小值开始 | features.py | verified | alpha初值.05，.2*tanh有界；通道模式非组常数 | 投影16到72 |  Local真实模型/noquery检查PASS，远端训练验证尚待。
| N04 | 一.4 | 原GN/普通残差/统计恢复匹配控制 | design/features | verified | 普通残差72到8到72，精确匹配新增1225参数 | 各方法实际计算时间独立报告 |  Local真实模型/noquery检查PASS，远端训练验证尚待。
| N05 | 一.4 | 验证冻结换头跨RX收益与RX可辨性 | probe.py/report | pending | source TX、留一RX普通head及RX probe | 不能仅凭source训练CE降低称恢复指纹 |
| N06 | 一.4 | RX变得可辨但身份跨RX未提高属于失败 | 报告 | pending | 保留机制失败与负结果 | 不根据target调参重排 |
| B01 | 二.1/3 | 依据真实图定义各路径与所有侧路 | trace/features | verified | Core.forward和ReferenceCVS.features已读 | Time/Freq/PA/Reference；频域有rho、PA统计等侧路 |
| B02 | 二.2 | 少量L_s步骤屏蔽一条实际路径 | features.py/native | verified | p=.1，5epoch warmup后均匀4组；无新增头 | 现有branch_drop_p实际为0 |  Local真实模型/noquery检查PASS，远端训练验证尚待。
| B03 | 二.3 | 不能只把日志feat_pa置零 | features.py | verified | 实际t/f/pa投影和reference_projection输出屏蔽 | 频域rho/统计侧路同时处理 |  Local真实模型/noquery检查PASS，远端训练验证尚待。
| B04 | 二.3 | U强学生和EMA教师方式保持原配方 | native/features/checks | verified | Scope只围绕out_l调用；teacher deepcopy测试 | U没有额外mask |  Local真实模型/noquery检查PASS，远端训练验证尚待。
| B05 | 二.2 | 推理所有路径恢复，没有RX路由 | features.py/checks | verified | eval与plain forward无hooks/no state changes | mask选择不读取身份、概率或domain |  Local真实模型/noquery检查PASS，远端训练验证尚待。
| B06 | 二.4 | 不用强行相等梯度；测单/部分分支与完整z | probe/report | pending | 冻结source feature消融；完整ordinary rehead主要指标 | 只提升缺失场景不算表征进展 |
| D01 | 三.1/3 | 局部分布RFF映射进入z，固定方向起步 | 后续独立矩阵 | deferred | 首轮先检验N/B，如用户原说明建议 | 不是物理IQ FFT |
| D02 | 三.2/4 | 均值/均值方差/非线性均值/RFF容量控制及RX对比 | 后续独立矩阵 | deferred | 固定投影下Source分布诊断待进行 | 有限维RFF不声称唯一嵌入 |
| S01 | 四.2 | 低维一阶和二阶反对称签名进入z | 后续独立矩阵 | deferred | 6维21分量、固定窗口及时间坐标对照 | 有限阶不声称信道/CFO不变 |
| S02 | 四.4 | 同投影平均/协方差对照；局部特征顺序打乱诊断 | 后续独立矩阵 | deferred | 仅冻结局部features，不打乱IQ当训练增强 | 无时间坐标不忽视重参数化限制 |
| U01 | 五.2 | 共享卷积字典固定步数残差修正和soft threshold | 第二梯队 | deferred | 先完成前四条且不全面替换CVS | 无TX专属字典、不在目标更新权重 |
| U02 | 五.4 | 温和阈值/原路径/字典范数步长控制和三组对照 | 第二梯队 | deferred | 普通残差/无阈值展开/有阈值展开 | 重构改善不等于识别改善 |
| R01 | 六.1/2 | MCR2直接作用实际单位z，无W，L真标签分组 | 条件性后续 | deferred | 先诊断特征几何过度集中 | 不默认叠加、不用U伪标签复杂几何 |
| R02 | 六.3 | 无临时丢弃projection头，无特殊子空间分类器 | 条件性后续 | deferred | 普通cosine统一判定、RX/day维度诊断 | 高秩不能代替身份/新类性能 |
| M01 | 七 | 首轮基线/N/B/N+B及容量控制 | design.py | pending | 五个条件；baseline固定复用对应seed合法oldgate E50 | 新增四个独立scratch50变体 |
| M02 | 七 | Stage2-C同row旧类前后及新类统一all registered | phase2 followup | deferred | support-only普通cos；匹配小矩阵优先 | 不会用Phase1目标成绩排序选择Phase2候选 |
| M03 | 科学边界 | 论文提供假说依据，IQ瓶颈未证实 | 报告 | verified | 已核验CVPR SNR和NeurIPS Gradient Starvation主来源 | 实现遵循用户统计恢复/屏蔽方案，不声称复现原论文完整算法 |


最高风险待办：实际L-only插入点及频域侧路/参考响应屏蔽必须用真实图验证，不能让metadata或模型副本传播临时hooks。未完成P0/P1与完整配置前不发布新实验。

原理来源：[SNR CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Jin_Style_Normalization_and_Restitution_for_Generalizable_Person_Re-Identification_CVPR_2020_paper.html)、[Gradient Starvation NeurIPS2021](https://papers.nips.cc/paper_files/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html)。

已执行：实际原模型CPU构建与源码计算图读取、native labeled out_l调用定位；没有新训练/target读取。其余路线按用户给定分阶段顺序延期，不宣称已实现或验证。

本地原型检查PASS：五模型普通原头initial一致、部署z160/W独立、新增投影与alpha有限非零梯度、精确1225参数匹配、strict state roundtrip；四路径实际投影/频域rho和统计侧路屏蔽、响应旁路屏蔽、退出恢复、教师U不受L临时hooks影响、U/query/target角色拒绝、native唯一out_l/x_l_main/y_l AST定位。原12模型旧矩阵已完成，主目标84flag false；新alpha/mask设计在读取其目标指标前已固定。

2026-10-13 正式闭环已实现并登记：20261013-phase1-restitution-competition-oldgate-e50-clean-manysig-m15-r01，五组×三seed，新增12 scratch50，复用3同seed固定历史对照；L_s-only普通cosine10轮及5留RX/1RX信息诊断，源V单组撤除保持固定W，双读出固定后fresh scoring。focused pipeline、release import、compile PASS；唯一独立P0/P1 review PASS。真实远端source训练/冻结换头/target评分尚待；Phase2匹配数据与同support旧新类联合证据仍未完成。

## 2026-10-13 当前交接与用户覆盖

分类头只使用网络默认头并随骨干联合训练，取消本记录 G05/N05/N06/B06 中任何单独拟合分类头的后续执行，以及 M02 中拟另训普通 cosine 头的旧设想。此前已完成的换头产物只作为历史诊断保留，不纳入正式性能或后续门槛；Phase2 如何使用默认判决机制尚待合法数据与方法匹配，不在此声明已经解决。

五条件×三 seed 原 run 15 行已全部冻结、预测和独立评分，无技术失败。12 个新增骨干共 600 轮、133,200 次更新；3 个历史对照复用 150 轮、33,300 次更新。默认头 clean 均值依次为 BASE_REF 80.5268%、ORDINARY 80.3075%、RESTORE 80.2532%、COMP 80.5663%、RESTORE_COMP 80.1298%；84% 目标未达成。完整默认头逐 seed、分层指标与负结果见原 report 的最新追加段。

对话报错为 stream disconnected before completion（连接层）；当前远端 run 已完整完成，不能将对话失败当作实验失败或重启理由。后续四条路线仍按原分阶段设计未执行，不依据本次 target 分数重排。

## 局部分布与路径顺序：下一组 Phase1 实现跟踪

本组继承用户六路线说明第三、第四节及最新默认分类头限制。使用既有合法源数据、旧伪标签门控＋EMA＋cos、scratch50（33＋17），先 seed 2026092701 做小矩阵，再按预登记源域健康条件决定扩展 02/03。仅 clean 自动预测和独立 truth-last 评分。目标仍为 clean 三 seed 均值 84%，不使用 target 分数更改方案或挑选初始化。

| ID | 来源 | 要求 | 目标文件 | 状态 | 验证 | 说明 |
|---|---|---|---|---|---|---|
| D01a | 三.3 | t2 中层局部投影6维、固定14方向 cos/sin 分布编码，4固定窗口进入实际z160 | experiments/cvs_feature_sequence50/features.py | verified | RFF区分同均值同方差分布；真实模型z/梯度 | 共享有界局部投影；不是物理IQ FFT |
| D02a | 三.4 | 均值、均值＋方差、均值后非线性、RFF控制 | features.py/design.py | verified | 数学边界与同训练/参数预算检查 | 固定映射使输出宽度相同，内在信息维度不宣称相同 |
| S01a | 四.2/4 | 一阶与二阶反对称路径量；无时间/有时间控制 | features.py | verified | 已知闭合有向面积、反向与顺序置换 | 6信号维＋可选时间；4固定窗口 |
| S02a | 四.4 | 相同局部投影均值＋协方差对照；冻结局部顺序诊断 | features.py/checks.py | verified | 同参数输出，RFF排列不变，signature顺序敏感 | 不打乱真实IQ作训练增强 |
| H01a | 用户最新指令 | 默认分类头原结构、原联合训练，无单独头拟合/诊断头 | model.py/source.py/predict.py | verified | 默认head权重初值、单次前向一致、训练后直接原头预测 | 原六路线换头要求由用户覆盖 |
| P01a | Phase1边界 | L/U/V权限、scratch来源、固定budget、逐row自动预测评分 | source.py/provenance.py/dispatch.py | implemented | 权限负测、真实checkpoint noquery smoke、远端读回 | Phase2不在本轮范围 |
| R01a | 对照与小矩阵 | BASE_REF/MEAN/MEAN_VAR/NONLINEAR_MEAN/RFF/COV/SIG/SIG_TIME | design.py/registration | verified | 固定seed/配置、source-only扩展，无target反馈 | 首种子8行，新增7独立scratch骨干 |

REJECTED_EXTRA_GATE：额外设计/计划审批与重复独立审查不属于项目八项最小流程；按既有用户方案与授权推进，不要求新的设计许可。

2026-10-14 本地实现更新：D01a/D02a/S01a/S02a/H01a 已在数学与真实模型中 verified；八组原生默认头结构/初值一致、新增方案参数精确18976、CE梯度覆盖新增特征与默认头、W不定义z、teacher deepcopy及strict checkpoint roundtrip PASS。P01a 本地协议负测与源码闭环 implemented，真实远端 source/score 尚待；R01a 实际八组×三seed与source-only非劣扩展配置、登记 verified。唯一 P1为native训练闭包中的RX/day/channel拟合诊断，已最小修复并通过针对性负测，定点复审尚待。

2026-10-14 唯一 P0/P1审查和一次定点复审 PASS。当前本组7项：D01a/D02a/S01a/S02a/H01a/R01a 本地 verified（6项），P01a 本地 implemented、远端真实checkpoint与启动尚待（1项）；无 deferred/rejected/blocked 项。原第5/6路线仍保留历史 deferred 状态，不在当前发布阻塞范围。

2026-10-14 N607真实checkpoint无query smoke、一次archive传输校验、compile与启动 VERIFIED。首种子七个新增source进程 live，2至3epoch；default_head_only和native_fitted_probes_disabled实际resolved为true。当前6项verified、1项implemented（完整真实source→prediction→score尚待），0项blocked；未声称84%目标完成。
