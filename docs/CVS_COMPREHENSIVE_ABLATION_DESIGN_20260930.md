# CVS 功能消融 域适应 新类注册与参数扫描实验设计

2026-09-30。状态：设计方案，尚未展开为运行配置、启动实验或执行新的目标域评分。本文使用当前协议和实现证据定义研究问题；所列扫描值是建议候选范围，不是已验证最优值，也不自动扩大运行授权。

建议把 CVS 分成 **8 个功能点：多分支射频表征、身份与域分离、弱标注可信学习、部署信道轨道学习、身份几何与开放世界准备、目标域适应、新类注册、未知拒识与多节点协同**。前 5 项属于 Phase1，后 2 项属于 Phase2，第 8 项属于 Phase3。资源成本贯穿全部阶段。每个功能点先检验必要性，再拆内部机制；只对有明确假设的模块组合开展交互实验。

## 1 设计依据与适用版本

### 1.1 已核对的事实

| 对象 | 当前证据 | 对设计的影响 |
|---|---|---|
| 科学协议 | 本机 `项目.md`，版本 2026-09-08，`p2_min_v1` | source-only 训练；Phase2 support 可拟合、query 只读；先固定预测，再独立连接 truth |
| 最近一组 CVS Phase1 | 20260927 五模型 seed 的 scratch、final200、DAOT A1＋RC4 practical residual 实验 | 用该版本的实际配置定义当前完整模型，不能用历史名称代替配置 |
| DAOT A1 | 发布快照的 `daot_ablation_overrides`：2 个教师视图、mean 聚合，tangent/nuisance/fingerprint 权重为 0 | 鲁棒三视图、切空间、指纹保留是扩展实验，不是当前 A1 的删除消融 |
| RC4 | correctness calibration、hard、partial-set、partial-conditional 启用；negative、anchor 关闭 | negative/anchor 属于加项研究；当前消融主要检验校准及 partial 路由 |
| 原生身份主干 | time/frequency/PA/statistics 启用，DAC 关闭；`feat_joint` 160 维；无 BatchNorm 模块 | DAC 不是现有贡献；AdaBN 在该基座上记 N/A，不能开一个无作用开关作为对照 |
| D92 E0 true256 | identity160＋FFT96 | 256 维是该部署表示的定义，不是全部分支缓存维数 |
| BranchLocalRidge | z_id/t_emb/f_emb/pa_local 各 160 维，FFT96，共 736 维输入；123616 维隐式交互；原始 received 单视图；lambda=1 | 必须分别消融输入信息、交互映射、径向核和求解规则 |
| 当前 LocalRidge 三阶段 | A 为原生分类器，B/C 为旧类/全注册类 support 独立重拟合 | B−A 混合分类头替换和适应贡献；B−C 混合重拟合与新增类竞争 |
| 微调候选 | 已有 channel/spectral/sequential-residual support 诊断；低秩 residual 草案曾暂缓 | 不能把草案当作已经部署的完整方法，也不能根据 query 历史重选候选 |

本文读取既有结果仅用于描述报告口径和已有证据边界，不据 target 历史排名选择模块、扫描方向、seed、RX 或类子集。后续研发只使用 source-only 或当前合法 support 证据。历史已评分目标集上的重复比较必须保留“透明重复基准”标记。

### 1.2 两条实验线

- **当前版本消融线**：以实际 DAOT A1＋RC4 final200 为 Phase1 完整参照；分别评价 D92 E0 true256 与 BranchLocalRidge，二者不能混称同一个注册模块。
- **扩展机制线**：DAOT A2–A8、negative 路由、PEFT、顺序注册和 Phase3 协同。先确定实现、合法输入、独立对照和固定配置，再评价；扩展结果不能追溯归因到当前 A1。

本机根目录与 Git 主检出存在不同版本/未提交修改。本次仅交付设计文档，不修改这些源码；真正执行时以登记的不可变代码提交、resolved config 和模型来源为准。

## 2 统一实验单位与科学边界

### 2.1 配对单位

Phase2 的最小比较单元建议为：

`(checkpoint, capsule_id, split_id, RX, day口径, scene, old/new类集合, K, support_draw, query_ID集合)`。

同单元的全部候选共享物理 support/query、received IQ、类别映射和评分规则。只改变预先声明的机制。若数据资产跨 day 混合，明确 day 是采样分层而非独立域；不可把混合池结果写成逐日迁移实验。

Phase1 的配对单位是固定物理 L/U/V 契约和模型随机重复。架构修改允许形状不同，公共张量尽量采用一致初始化；记录无法一一匹配的部分。相同随机 seed 并不保证不同计算图取得相同信道抽样，必要时用物理 ID/epoch 驱动的独立增强随机流实现共同随机数。

### 2.2 输入与隔离

1. Phase1 固定 source/target RX 互斥、物理 L/U/V 互斥；当前比例为 0.07/0.63/0.30。单一 source V 只做校准、评价和预先规定的选择，不能拆成新的训练角色或更新持久状态。
2. 每份 checkpoint 及其全部上游必须匹配实际训练物理 ID 与角色，且无目标参与训练/选模的污染。当前从零训练的消融不继承完整模型权重；改架构/标签率/数据契约不能借旧权重洗掉差异。
3. Phase2 只读取合法 bundle、固定 received IQ、当前 row support 及映射。冻结地面原型不可训练；不能在线由原型拟合协方差、LDA 或可训练分类头。可选量化聚合摘要必须在 Phase1 联合冻结，不能回读 source 样本补建。
4. 当前 query 及其确定性/随机计算 view 都不参与拟合、带宽、归一化、校准、阈值、早停、选择或回滚。每个 query 独立面对全部已注册类别。
5. 每个物理记录只绑定一份星地接收观测。同一 received IQ 的计算 view 不增加 K；不得重新调用信道模拟器生成第二个 LEO 观测。不同 scene 的物理集合保持互斥。
6. 已验证且未变的数据 capsule 跨方法复用。只有 received IQ、物理 ID、RX/TX、scene、K、support/query 或 schema 改变才处理相应数据验证，不因换方法或报告重复验证。

### 2.3 场景族分开报告

- 通用协议默认是 `leo_clear_weak / leo_low_elev_weak / leo_rain_weak`。
- 9 月 27 日实际实验有明确的 practical residual/post_sync/noeq 历史覆盖，使用 `practical_high / practical_mid / practical_low_urban`。沿用时保留原标识、原数据来源和对应授权，不能改名为三个 LEO_WEAK 场景。
- 两个族分别建表。新研究若改变族或生成输入，属于独立数据设计；不能为了配图对同一物理记录重生成多场景观测。
- clean 只作 Phase1 无星地增强对照。Phase2 新类和正式 unknown 指标遵循正式星地输入；无 LEO 的隔离诊断只在协议已有明确例外或新增明确授权下开展。

### 2.4 三类研究互不替代

| 研究类型 | 允许的选择依据 | 结果用途 |
|---|---|---|
| 机制研发/选参 | Phase1 source V；Phase2 合法 support 内部诊断或 source 代理任务 | 建立假设、固定算法与参数 |
| 冻结消融/敏感性评价 | 在目标访问前固定全部 arms/扫描值；整批预测固定后统一评分 | 描述差异和敏感性，不依据 target 曲线重选参数 |
| 独立确认 | 真正未用于研发的物理数据/目标域，且 checkpoint 的完整来源合规 | 有限范围内的独立泛化结论 |

只换 model/support/channel seed、换路径或改变 scene，不能使已暴露的物理 query 变成独立确认集。support 内调参也会对该有限 support 过拟合，因此采用预定的小网格；选择用内层折，选择后的诊断用外层折，当前 query 始终保持测试角色。嵌套 support 诊断并不保证独立 target 泛化。[模型选择偏差研究](https://www.jmlr.org/papers/v11/cawley10a.html)支持把选择与评价分开。

## 3 八个功能点及可检验假设

| 功能 | 科学问题 | 一级删除/替换 | 主要反证 |
|---|---|---|---|
| F1 多分支射频表征 | 时域、频域、PA、统计特征是否各有独立信息？ | 分别去掉分支；只保留 time；等容量替代 | 性能不降或同计算预算下收益消失 |
| F2 身份与域分离 | 域扰动建模和 DG 损失是否保留 TX、降低 RX 依赖？ | 去域相关训练；分别去 GRL、Fishr、group CE | RX 可分性下降但 TX 同时塌缩，或跨 RX 不改善 |
| F3 弱标注可信学习 | RC4 是否有效利用低标签率 U，避免错误监督？ | 无 U 身份监督、普通 PL、hard-only、无校准 | coverage 上升但 source 代理正确率/后续性能下降 |
| F4 信道轨道学习 | DAOT 是否提供卫星 CE 之外的稳定性？ | 去 DAOT，保留卫星 CE；去卫星 CE，保留 DAOT | 提升只由额外前向或训练预算解释 |
| F5 身份几何与开放世界准备 | 紧致、margin、尾部/proxy 是否改善迁移、注册或拒识？ | 几何组与 proxy 组分别去除 | closed-set 改善却新类/unknown 恶化，或表示塌缩 |
| F6 目标域适应 | 旧类 support 改变了什么：头、度量、表示还是三者？ | frozen NCM、解析头、缓存后 adapter、encoder SFT | 训练拟合改善但持出/冻结 query 不改善 |
| F7 新类注册 | 新类 support 能否形成可竞争的身份状态并保护旧类？ | 仅追加原型、独立重拟合、继承 B、联合训练 | 新类增长导致旧类竞争损失大，或新旧差靠降低旧类缩小 |
| F8 未知拒识与协同 | unknown 分数、多节点及相关性控制各贡献多少？ | 单节点、投票、等权、质量权重、相关性控制 | 全拒绝、复制节点伪增益、代理非同步数据被误称同步多星 |

F5 的“开放世界准备”是 Phase1 的表示/代理训练，F8 的“正式 unknown 拒识”是部署评价。两者必须分别完成证据，不能互相替代。

## 4 Phase1 消融矩阵

### 4.1 当前模型的一级矩阵

下面 16 个配置构成可先落地的完整一级设计。P01 是弱基线，不能用 P00−P01 归因某个单独模块。其余删除臂都从 P00 出发。

| ID | 配置差异 | 固定项与解释 |
|---|---|---|
| P00 | 当前完整 CVS A1＋RC4 | 固定实际参数、数据、final200、增强与优化日程 |
| P01 | 相同主干，仅 labeled TX CE | 去全部非 CE 训练目标；有明确固定步数/数据重复预算；用于总体增益参考 |
| P02 | 去 time 分支 | 真实停用前向并重训，保留 freq/PA/statistics；披露参数量变化 |
| P03 | 去 frequency 分支 | 同上；仅推理遮蔽另列为依赖诊断 |
| P04 | 去 PA 分支 | 当前 DAC 已关闭，不把 no-DAC 作为新消融 |
| P05 | 去 statistics 分支 | 从当前模型实际输入路径删除，不误用 domain 的 no_stats 开关 |
| P06 | 去身份与域分离/DG 训练组 | 去相关 GRL/confusion、domain/group/Fishr 目标，按 loss 依赖清单处理；不连带去 DAOT |
| P07 | 去 U 的 TX 身份监督 | hard/partial/self/satellite-U 及 DAOT-U logit/prototype 的 TX 监督信息全部为零；U 的无标签 feature/domain 目标单独保留并披露 |
| P08 | RC4 换普通 hard PL | EMA 保留；阈值、起始 epoch、loss 权重由 source 冻结；同时报告监督质量和接受量差异 |
| P09 | 去 correctness calibration | 换 source 预定的统一可信规则；其余 RC4 路由不变；不能读取 U 的隐藏 TX 真值 |
| P10 | RC4 hard-only | partial-set 和 partial-conditional 都关闭；hard、其余训练链不变 |
| P11 | 去 DAOT | 保留原卫星 CE 和信道日程；避免把信道增强同时删除 |
| P12 | 去 satellite CE | 卫星 CE 权重为零；保留 DAOT 需要的信道观测与教师路径 |
| P13 | 去紧致/原型/开放世界几何组 | `lambda_proto/lambda_zid_compact/lambda_open_world_feat=0`；其余 proxy/episode 不变 |
| P14 | 去 proxy unknown 训练组 | `lambda_proxy_unknown/lambda_soft_unknown_mixup=0`；几何和 source episode 保留 |
| P15 | 去 source episode 训练 | `lambda_source_episode=0`；其他全部不变 |

P06/P07 是功能组消融，后续细分才回答其中哪个项有效。P07 并不等于完全不用 U；真正 no-U 另设扩展臂，去所有 U 来源目标并保持预定 optimizer step 预算。不能让 no-U 改变 epoch 定义后直接称等预算。

建议主报告给原生 recipe 的相同数据/轮数对照，资源表给实际 steps、前向/反向和耗时。对收益较大的机制补一组等 optimizer steps，以及可行时的等墙钟/前向预算对照。相同 epoch 不等于相同计算；真删除和“保留前向但 loss 为零”的归因对照也不同。

### 4.2 F1 的内部拆解

对 time、frequency、PA、statistics 分别做 leave-one-out；额外保留 time-only、time＋freq、time＋PA、全部分支。若预算有限，只对已有 source 证据支持互补性的两对分支做 2×2。

容量控制使用相同输出 embedding 维度、尽量匹配参数量的单分支替代模型。主表同时报告准确率、Macro-F1、注册 C_old/C_new/H、trainable/total 参数、推理成本。推理临时置零分支只回答“当前网络依赖该分支多少”，不能替代从零重训的必要性消融。

时域前端另做 Sinc 参数化卷积 vs 同感受野普通卷积、可学习 Sinc vs 固定初始化滤波器，以及复值运算 vs 参数/计算尽量匹配的实值替代。它们属于建议的新架构对照，必须先验证当前具体前端和可替换边界，不能假定已有入口。冻结滤波器后从零训练其他层，与整网训练后的推理遮蔽是不同实验。若删除 time/frequency 会连带改变 PA 或融合输入，先补不删除依赖信息的桥接臂，或者明确报告为组合消融；不能仅凭开关名声称纯单分支效应。

F2 的 source RX 数量和 source day 多样性可另设数据规模实验，按预定 RX/day 子集顺序采样。它们会改变 Phase1 数据契约，必须从零训练、保持 target RX 隔离并披露有效样本量；不能继承全 source RX 模型后声称只用一个 source RX。跨日测试必须按实际物理日期划分，不能从混合 day 池推断。

### 4.3 F2/F3/F5 的内部拆解

| 组 | 建议子实验 | 特别限制 |
|---|---|---|
| F2 | no-GRL、no-Fishr、no-group-CE、去 domain head、只保留普通 domain discriminator | 单项 loss 关掉后检查是否还有等价梯度路径；z_dom 大小改变另列容量实验 |
| F3 | no-partial-set、no-partial-conditional、no-class-RX-cap、固定可信规则对比校准规则、去尾段 consolidation | 记录各路由覆盖、有效权重、source 代理可靠性；U 不可解封真值验证伪标签 |
| F3 扩展 | negative on、anchor on、无 EMA | negative/anchor 当前关闭；有 EMA 的其他机制同时受影响，属于 teacher 交互，不可只归因为 RC4 |
| F5 | 分别去 prototype、compact、open-world feature；去尾部/CVaR；proxy 实体 episode 对比 soft unknown mixup | source proxy 是相对 episode 类别表的代理，不是真正训练未见 TX |
| F5 扩展 | 几何半径、margin、proxy 数量与 loss 系数扫描 | threshold 只能 source 冻结；真实 target unknown 不用于选择 |

表征诊断至少包括 source 域内与跨域 TX 类内/类间距离、TX margin、特征范数和有效秩。RX 可预测性仅作代理诊断，须同时看 TX 判别，不能把抹除所有信息当作成功解耦。probe 只使用合法 source 训练数据拟合、V 评价，V 不更新主模型。

### 4.4 F4 的轨道与切空间扩展

| 对照 | 能回答的问题 | 不能单独回答的问题 |
|---|---|---|
| A0 vs A1 | 在同 RC4/卫星 CE 下增加两视图均值轨道目标的总贡献 | 鲁棒聚合或切空间贡献 |
| A1 vs A2 | 均值聚合下 2 vs 3 教师视图 | 额外视图与计算量的完全分离 |
| A2 vs A3 | 同三视图下 mean vs robust deployment 聚合 | 部署重要性、质量、Huber 各自贡献 |
| A3 vs A4 | 单参数 tangent 的加项效果 | 联合相关物理方向效果 |
| A4 vs A5 | 单参数 vs covariance tangent | 分支选择、nuisance head 效果 |
| A5 vs A6 | 现有实现中 branch-selective＋nuisance 的组合增量 | 两者独立贡献；须再加两个桥接臂 |
| A6 vs A7 | fingerprint keep 的增量 | 所有可能发射机扰动上的真实身份保持 |
| A7 vs A8 | temporal memory 方案总效率/性能变化 | memory 单项因果；教师视图数也改变，需视图数匹配桥接臂 |

在 A1 上用 `no_z/no_logit/no_proto` 核实三类有效 orbit 目标；先记录实际权重与梯度，原本为零的项不安排删除实验。relation 当前默认关闭，`relation_on` 属于加项。A3 鲁棒聚合内部可分别去 importance、reliability、coverage floor、Huber，但要固定其他归一化、教师视图与目标公式。

切空间只在 Phase1 的合法信道重放/物理干预中研究。Phase2 view 实验不能借用这条权限重新生成 LEO 观测。记录 sensitivity、R_select、N_eff、教师共识、dispersion 及实际计算成本；这些是解释性指标，不新增自动停止门槛。

### 4.5 少量交互设计

优先做三组 2×2：

1. RC4 身份监督 on/off × DAOT on/off，检验可信路由和轨道监督是否互补。
2. satellite CE on/off × DAOT on/off，检验 CE 与轨道目标是否互补。
3. 几何组 on/off × proxy unknown 组 on/off，检验可分几何与拒识准备是否互补。

对所有四态均有定义的指标 M，报告 `I=M11−M10−M01+M00`，及四态的绝对值。只做完整模型逐项删除得到的是条件贡献，不能当作模块贡献的可加分解。多因子全组合随因素数指数增长；小规模因子设计更适合这里的具体交互假设。[NIST 因子设计说明](https://www.itl.nist.gov/div898/handbook/pri/section3/pri333.htm)。

## 5 Phase2 域适应消融

### 5.1 固定 Phase1 及主干输入

域适应候选全部共享同一合规 Phase1 bundle，同一 row 的原始 received IQ；不重新训练地面模型，不混入源域样本或逐样本特征。先把特征/分类头变化与表示适配变化区分开。

| ID | 方法 | 更新对象 | 定位 |
|---|---|---|---|
| D00 | 原生冻结分类器 | 无 | A_native，当前原始参照 |
| D01 | 冻结地面原型 NCM | 无 | A_proto，揭示头形式差异；缺合规原型则 N/A |
| D02 | target support NCM | 当前 target support 类中心 | 最简单 few-shot 头适应 |
| D03 | D92 E0 true256 | 合法 target support 的既定解析状态 | 当前旧版部署表示参考；地面摘要仅按已授权冻结规则使用 |
| D04 | BranchRidge | target support 解析头 | 736 维分支信息的线性参考 |
| D05 | BranchInteraction | target support 交互解析头 | 在 D04 之上检验交互 |
| D06 | BranchLocalRidge | target support 局部径向核头 | 当前独立方法参照 |
| D07 | D06＋缓存后小 adapter | adapter＋同一最终头 | encoder 不反传；必须是可验证实现而非暂缓草案 |
| D08 | D06＋选定层/分支 PEFT | 层内低秩/通道参数＋同一最终头 | 需要 encoder support 前向/反向，成本另测 |
| D09 | 选定 backbone 的合法 support SFT | 预定范围参数＋统一注册头 | 完整/部分微调参照，不预设一定更优或更省算力 |

D03 到 D06 不是纯“某一个 loss 删除”：其输入、映射和分类器均可能不同。需要下一节的组件匹配，才能把收益归到具体机制。D07–D09 是扩展开发候选，是否采用不由历史 query 排名决定。

### 5.2 拆开头替换与表示适配

以同一个评分头形式比较：

- 原表示＋旧类 support 头：B_head。
- 适配表示＋同一类旧类 support 头：B_repr。
- 原生冻结头：A_native；另报告 A_proto。

`B_head−A_native` 是 support 头拟合与头替换的合并效果，不能称纯表示适应。`B_repr−B_head` 才是该统一头下的表示适配增量。若要拆纯头替换，使用合法且冻结的 source 阶段等价头，不能在 Phase2 从地面原型在线训练新分类器。

对 adapter 采用小型 2×2：表示适配 on/off × head recalibration on/off。若固定旧头无法兼容新表示，该格标 N/A，重定义一个兼容的共同头后再做交互，不能拼接异构指标。

### 5.3 插入点与监督对象

固定损失、步数和最终头，比较 z_id 缓存后、time、frequency、PA、融合层和预定最后 block。先做一种实现的小范围插入点研究，不能同时换 rank/步数/损失后归因为插入位置。

同预算比较 old-only support 监督与 old＋new support 监督，分别回答旧域适应、联合注册适配。B 只能用 old support；C 可用当前全部注册 support。只对 new support 微调的臂必须保留 old support 的正式输入口径，并说明其在优化中未使用；不能称完整旧类适应。

### 5.4 合法 support 验证

- K≥2 时，按物理 ID 分折；held 的所有 view 一起排除，normalization、带宽、anchor、adapter、头、校准只在 train-support 上估计。
- K=5 的 3 折 held 数通常不等，先汇总正确数/样本数，再计算每类和组宏平均，不能简单平均折准确率。
- K=1 每类只有一个物理 support，无法类内留一后仍保留该类训练样本。记独立 held 性能 N/A；仅做数值检查或用 source 代理冻结规则。
- 用 parent K≥5 的单 anchor proxy 研究实际 one-shot 拟合时，明确 parent K 只影响 anchor/held 池，实际 train K=1，不能写作 K=20 训练结果。
- 调参时内层折选值、外层折报告；最终部署用完整 K 个 support 重拟合。外层 held 不用于选本折参数或早停。OOF 结果必须注明每折实际 train K 小于标称 parent K。

## 6 Phase2 新类注册消融

### 6.1 真正没有旧类适应的 2×2

首先用统一 cosine/NCM 评分形式建立以下四态，旧类地面原型不可变；DA1 只建立独立 target old-support 原型，不在线修改地面原型。

| 状态 | 旧类决策状态 | 新类状态 | 输出 |
|---|---|---|---|
| DA0_REG0 | 冻结地面 old 原型 | 不注册 | old accuracy/floor；new/H 为 N/A |
| DA1_REG0 | target old-support 原型 | 不注册 | old accuracy/floor；new/H 为 N/A |
| DA0_REG1 | 冻结地面 old 原型 | 追加 target new-support 原型 | 全注册类共同竞争的 old/new/H |
| DA1_REG1 | target old-support 原型 | 追加 target new-support 原型 | 全注册类共同竞争的 old/new/H |

四态共用冻结表征、相同评分公式和归一化。DA0_REG1 的 old support 不进入决策拟合；new support 是注册所需输入。这组准确识别“旧类 support 原型适应×新类追加”的作用，不能概括为所有 LocalRidge/PEFT 算法的普遍效应。

LocalRidge 若在 DA0 中已使用 old support 拟合，其 DA0 只能表示“额外表示适配关闭”，不能冒称 zero-adaptation；另命名 `ADAPTER0/1 × REG0/1`，保留 A_native/B_head 作为外部参照。

四态均有定义的 old accuracy/floor、资源/延迟可做交互。REG0 的 new accuracy/H 记 N/A，不能补 0 后做差分。

### 6.2 当前注册表示的组件矩阵

| 组件 | 对照 | 固定要求 |
|---|---|---|
| z_id 与 FFT96 | identity-only、FFT-only、identity＋FFT | 同一 support、头与训练规则；维度差异披露 |
| 分支缓存 | z_id＋FFT、加 time、加 freq、加 PA、全部 736 | 主消融 leave-one-out；容量/信息量不能混称同一效应 |
| 交互 | 线性 concat vs interaction | 输入、物理权重、target 编码、求解精度相同 |
| 局部性 | interaction linear vs Gaussian local | 相同 interaction 距离与 support；比较径向核增量 |
| 核尺度 | trace matching on/off | 禁用后记录 Gram trace/有效正则比例，不能把幅值变化误称新几何 |
| support 带宽 | 原 train-only 规则 vs source 冻结常数/预定倍数 | 不使用 query 距离；零带宽按明确数学退化规则处理 |
| 物理样本权重 | 原求和目标 vs 预定均值目标的 matched lambda | 若均值目标除以 N，等价正则应除以 N；否则是在改变 K 依赖 |
| received 计算 view | 原始 received、确定性多 view、物理加权多 view | 多 view 总权重每物理样本为 1；支持/查询的使用规则分别固定 |
| 冻结地面摘要 | 不使用 vs 使用合法 int8 聚合摘要 | 有合规摘要才列此臂；缺失时不能临时回读 source 样本补建 |

“去交互＋去局部核＋换 FFT 权重”不是单项消融。每种核/目标的数值求解器保持可验证；求解失败属于技术失败，不能回退另一个头后继续标为本方法。

### 6.3 注册路径与状态继承

| ID | B→C 路径 | 主要问题 |
|---|---|---|
| R0 | B old-only、C all-class 独立重拟合 | 当前 LocalRidge 参照，非持续学习状态继承 |
| R1 | 保留 B 的适配表示，C 仅追加 new 状态 | 固定旧表示/旧头能否承受新增竞争？需统一 score 标尺 |
| R2 | 继承 B adapter，C 用全部注册 support 继续训练 | 是否保留旧类改善并学习 new？ |
| R3 | C adapter 从恒等/相同初始化重新开始 | R2−R3 揭示状态继承增量；固定 C 步数且披露总预算 |
| R4 | 从原表示在 all-class support 上一次联合训练 | 与顺序路径比较；额外补总更新量匹配对照 |
| R5 | 顺序路径＋support-only 的共享保护/一致性正则 | old/new 都用同一生成规则；不能加具体 TX 白名单或 ground 原型训练 |

同一组的最终 C 都使用相同 old/new support 物理 ID。B checkpoint、optimizer 是否继承、C 是否重设 optimizer、旧类 support 是否回放均写清。old target-support 回放与 source replay 不同：前者属于合法 row 输入，后者在主方法中禁止。

### 6.4 把旧类下降分成两部分

冻结 B 和 C 的完整分数后，独立分析器对同一 old query 计算：

- `B_old`：B 对 old 类竞争的准确率。
- `C_old_restricted`：仅在已固定的 C 分数中保留 old 类列的离线反事实准确率。
- `C_old_all`：C 对全部 old＋new 类竞争的正式准确率。

`B_old−C_old_all = (B_old−C_old_restricted) + (C_old_restricted−C_old_all)`。

第一项是 old 类内部决策变化，第二项是新增类竞争损失。采用一致 tie-break 时第二项非负。它是固定分数下的计数分解，不证明某个核或参数造成了因果变化。第一项若为负，表示 old 内部决策改善。

restricted 分数分析只发生在预测固定后的 scorer/分析器，绝不能让正式 predictor 按真实 old/new 角色选不同候选列。保存三态逐样本正确性转换计数，避免改善与退化抵消后被净均值掩盖。跨 B/C 的 raw logit 不同标尺时不能直接比较 margin；C 内的 old 最大分数−new 最大分数可以作为解释性边界指标。

### 6.5 注册规模与顺序

- 基本矩阵：old 类固定 6，`K={1,5,10,20}`，`N_new={0,2,5,10,20}`；保留全部 20 个 K×N_new 单元。
- 新类集合按预定随机/物理 ID 规则产生，建议 2⊂5⊂10⊂20，不能依据模型识别难度挑类。多个类集合重复记录独立 class-set seed；若未实现，写 N/A 而非把 support seed 当类集合重复。
- K 和 N_new 改变时，总 support `N=(6+N_new)K` 也改变。规模曲线同时列 N，不能把全部差异归为类别数量本身；固定 C 分数的列限制用于补充竞争诊断。
- 多次注册采用相同最终 20 类/最终 support，比较 20 一次加入、10＋10、5×4、2×10；类顺序预先固定并置换重复。每轮模型、optimizer、原型和头的继承规则固定。
- 多次注册保留最初 old 类、各历史 new cohort 和本轮 new cohort 的准确率/floor。分别报告最终旧类保留和新 cohort 后续遗忘，不只给最终 H。
- 当前 Phase1 只有 6 个已见 TX。扩大 old 类数量需重新定义 Phase1 训练契约并从零训练，不在固定基座上把 Phase1 未见 TX 标作 old。

## 7 参数扫描设计

### 7.1 扫描流程

1. 先完成一级机制必要性诊断；未有效启用的机制不扫描。
2. 每次只扫描一个机制内 1–2 个参数。对正则/LR 用对数网格；对有明确物理单位的 margin/radius/扰动幅度记录单位。
3. 第一轮使用 source-only 或完整合法 support 诊断，小网格预定后全部完成；不依据 partial 结果裁掉失败臂。
4. 若选择参数，以 source V 或内层 support 折执行预先规定规则；不以目标 query 准确率排序。参数空间大的方法与小空间基线披露搜索成本。
5. 第二轮只对研发证据支持的相邻区域做精扫。精扫的取值与预算在 target 访问前固定。
6. 正式 target 敏感性图保留全部冻结点，一批预测全部固定后统一评分；曲线最高点不自动成为下一次参数。

### 7.2 Phase1 扫描候选

以下都是建议，不代表当前参数必然落在这些范围的最优位置。lambda0、默认日程和物理尺度从实际生效配置取得。

| 参数族 | 建议值 | 目的/控制 |
|---|---|---|
| 标注率 rho_label | 0.005、0.01、0.02、0.05、0.10 | 分母固定为 L＋U；V 固定全池 0.30，因此 L 占全池比例是 0.70×rho，不是直接把 rho 当全池比例 |
| 主要非 CE loss | 0、0.25、0.5、1、2 倍 lambda0 | 每个功能组单独扫；先验开关 0 与一级消融复用 |
| RC4 total identity budget | 0.05、0.10、0.15、0.25 | 当前 0.15；记录约束后实际质量/有效监督质量，不只是名义值 |
| hard/partial 比例 | (1,0)、(0.75,0.25)、(0.6,0.4)、(0.4,0.6) | 总名义监督权重固定，再披露各路由接受率造成的有效质量差异 |
| 普通 PL threshold | 0.80、0.90、0.95、0.97 | 只用于普通 PL 对照；不能把 RC4 校准阈值与普通置信阈值当同一参数 |
| 教师视图数 | 2、3；0 为 no-DAOT | 当前 A1=2；view 数变化记录额外成本 |
| orbit z/logit/proto | 分别 0、0.5、1、2 倍实际权重 | 固定其他目标；relation_on 另列加项 |
| coverage floor | 0、0.05、0.15、0.30 | 仅 robust 三视图；A1 mean 不扫无作用参数 |
| importance clip | [0.5,2]、[0.25,4]、[0.125,8] | 固定 proxy 分布；无真实 LEO 资产不增加 empirical mixture |
| Huber beta | 0.15、0.30、0.60 | 记录 N_eff/dispersion，不能只看 train loss |
| tangent delta | 0.01、0.025、0.05、0.10 | 仅 tangent 扩展；固定共同随机数和合法干预方向 |
| channel augmentation | clean-only、原日程、预定中等固定强度 | 只改变 Phase1 训练，最终测试观测不重采样；时程与强度先分开扫 |
| 身份维度 | 64、128、160、256 | 独立架构/容量实验，从零训练，后续 schema/参数成本变化披露 |

当前 source 全池 L=0.07、U=0.63 对应 rho_label=0.10。做低标注率时可在原训练全池内把部分原 L 转为 U，绝不转入 V/target；nested 标签 mask 便于配对。数据契约改变后模型来源重新判断；不能继承高标签率已训练的 checkpoint 再称低标签率训练。

### 7.3 Phase2 扫描候选

| 参数族 | 建议值 | 限制 |
|---|---|---|
| K | 1、5、10、20 | 主矩阵；额外 K=2/3/50 需要足够物理 support 与新的合法 split，不自动扩容 |
| N_new | 0、2、5、10、20 | 主矩阵；N_new=0 的 new/H 为 N/A |
| FFT 倍权 | 0、1、2、4、8 | 当前 LocalRidge=4；归一化、其他分支权重与头固定 |
| ridge lambda | 0.01、0.1、1、10、100 | 原求和目标下扫描；0 不作默认点，避免把无正则奇异问题混为科学结论 |
| radial bandwidth | 0.25、0.5、1、2、4 倍 tau_train | tau_train 仅从当前 train-support；tau_train=0 时保持冻结等价核退化，不凭空加数 |
| trace matching | on/off | 二态机制消融；与 lambda 交互仅做小型网格 |
| adapter rank | 2、4、8、16 | 当前未采用的低秩方案为扩展；固定插入点、表示维度和优化目标 |
| adapter LR | 1e−4、3e−4、1e−3、3e−3、1e−2 | 根据具体参数化与精度选择预定子网格；不能跨方法直接比较相同 LR 的优劣 |
| adapter 更新数 | 0、16、32、64、128 | full-batch support 以更新数为主，0 复用基线；不按 query 早停 |
| 最大扰动/保护系数 | 0、0.25、0.5、1、2 倍设计初值 | 必须定义物理/几何意义；不能与任意温度混为一个尺度 |
| 计算 view 数 | 1、2、4、8 | 均从同一 received 构造；每物理样本总训练权重固定为 1 |
| B→C 继承 | adapter 继承/重置，optimizer 继承/重置 | 分开因素；比较时固定 C 更新数并额外列总更新数 |

解析 LocalRidge 的 lambda/带宽扫描需要新参数化入口，不能假定现有 `FROZEN_CONFIG` 已支持任意覆盖。参数扫描文件应保存输入值和实际生效值，实际不变的开关不能计为独立实验。

重点二维图只做 `FFT倍权×lambda`、`带宽倍数×lambda`、`rank×更新数`、`K×N_new`。前三组先在研发数据冻结，第四组作为完整正式规模矩阵；不建立全部参数的大笛卡尔积。

### 7.4 物理扰动与数据规模的压力扫描

SNR、残余 CFO/Doppler、Doppler rate、SFO、STO、multipath、phase noise、AGC 分开研究，再选一组预定的联合压力。建议从实际配置中的 nominal scale 出发使用 0.5/1/2 倍范围，SNR 使用该实现已有明确单位的 3 个预定档位；尚未核实实际单位和合理范围前不填绝对数。信道压制与发射机身份非理想性必须区分，不能任意抹除 PA 等身份来源后称为正常接收扰动。

Phase1 可以在合法 source 物理记录上生成训练干预。正式 Phase2/Phase3 压力输入则由 builder 在角色/truth 连接前对互斥物理记录分配档位，每记录仍只有一份 received 观测；要扫描新的输入字节/scene 就建立对应独立数据切片并完成一次适用验证。不得把同一 clean 记录生成多个档位供方法挑选或融合。由于档位物理集合不同，报告分配随机化、样本数及不确定性，不把跨档曲线冒称逐物理样本 matched 因果效应。既有 received 输入上的均衡/相位/归一化计算 view 另列算法敏感性，不称新的物理接收条件。

### 7.5 评价选择规则

建议预先指定可行域：无协议/数值错误，并在 source/外层 support 诊断中保留 old/new 各自准确率及 floor。选择规则可以是先排除明显支配配置，再按预定主指标、资源和稳定性排序；并写清 tie-break。不得用 target query 的 10/1/3 目标筛选或重跑。

理想方向继续保留：适应提升≥10 pp、注册旧类下降≤1 pp、新旧绝对差≤3 pp。它们是科学目标，不是每轮全部达标才允许继续的硬门槛，也不是停止健康实验的依据。达到三项时仍检查绝对准确率/floor，防止把两边都降低解释为平衡改善。

## 8 Phase3 可选消融与压力测试

本节是后续独立工作包，不声称目前已实现完整 Phase3 或拥有同步多星数据。

### 8.1 Unknown 与协同分开

先固定单节点表征与合法阈值，比较 cosine distance、energy、半径/尾部规则、合法 uncertainty 规则。只有模型确实输出该量才做候选；缺合规先验记 N/A。source proxy unknown 用于研发，真正未见 TX 的 target unknown 仅用于冻结评价。

多节点消融为：单节点、最高置信节点、多数投票、等权融合、质量权重、质量＋域可靠性、完整方法去相关性控制、完整方法去冲突/defer。复制同一冻结证据作为“冗余节点”是系统负测：不能凭重复计权提高置信度并被称作独立信息增益。

### 8.2 因果与压力矩阵

- Phase1 表征旧/新 × 单节点/多节点构成 2×2；同一观测口径下报告 B−A、C−A、D−B−C＋A。
- 节点数建议 1、2、3、4，只在真实可用节点及事件绑定范围内实施；节点子集不按 query 表现挑选。
- 缺失率 0%、25%、50%；延迟为 0、0.5、1、2 倍任务预定时窗；观测质量不均衡与高相关节点分别研究。
- 未注册比例建议 0%、10%、30%、50%，仅 scorer 依据固定池组成场景报告，predictor 不获真实比例/配额；不依据比例动态设阈值。
- same-event 多节点绑定必须有物理事件证据。非同步 WiSig/ManySig 只能叫多接收节点代理协同，不能通过 TX 真值把独立发射拼成一个 shot。

指标包括 registered old/new accuracy、Macro-F1、unknown AUROC/AUPR、预定 unknown 召回下的 registered 误拒率、OSCR、defer/coverage、通信字节和延迟。定义 unknown 为 AUPR 的正类，说明比例敏感性；reject/defer 的 registered 样本均在正式身份准确率中按错误计。没有真实匿名实体/确权标签就只做系统流程验证，不虚构实体关联准确率。

历史 unknown query 只作不可变检测证据；真正注册必须在合法确权后重新采集 K 个独立发射事件的 support，不追溯把 query 改成 support。

## 9 指标与统计分析

### 9.1 每个 row 必须有的三阶段表

| K | old 类数 | N_new | A_native_old | A_proto_old | B_head_old | B_method_old | C_method_old | C_method_new | B−A_native | B_method−B_head | B−C_old | 新旧绝对差 | H |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 实际 K | 实际 old | 实际 new | %/N/A | %/N/A | %/N/A | %/N/A | %/N/A | %/N/A | pp/N/A | pp/N/A | pp/N/A | pp/N/A | %/N/A |

A/B/C 在同一 old query 物理集合与同一 old support row 上配对。A 不消费 support 拟合；B/C old support 相同。缺失 A 的 support 诊断写 N/A，不能借其他方法的 A/B 或其他 cohort 拼补。保留完整 K×N_new、RX、scene、model seed、support draw 分层。

`G=B_old−A_old`；`F=B_old−C_old`；`D=|C_old−C_new|`；`H=2*C_old*C_new/(C_old+C_new)`。两侧均为 0 时 H 定义 0；无 new 时 H/D 为 N/A。每单元先算差、绝对差和 H，再按预定权重聚合；平均 accuracy 的 H 或绝对差不等于逐单元平均 H/D。

### 9.2 宏平均、floor 与错误方向

old/new 内优先每类宏平均，另列按样本计的 micro；列总注册类 Macro-F1。定义 `floor_all` 为全部实际注册类中最小 per-class accuracy，另列 old/new floor、10% 类别分位数和 RX 最差值。floor 易受样本数影响，必须同时列每类 query 数/区间；不先挑难类清单。

混淆方向包括 old→old、old→new、new→old、new→new；开放集另列 known→reject/defer 和 unknown→known。类数不同总 accuracy 的权重会改变，不能用全类 accuracy 掩盖 old/new 分化。

### 9.3 独立性与不确定性

1. 四个现有较新 model seed（2026092701 至 2026092704）作为可配对重复；392005 历史参考单列。“较新”不意味着其历史 target 已成为盲评集。
2. 同一 checkpoint 的多 support draw、多个 K、nested 类集合、共同 query 都有关联；4800 个拟合单元不等于 4800 次独立训练或独立数据重复。
3. 先在 model×RX 内按预定 scene/K/N_new/支持抽样权重形成配对差，列每 model、每 RX 的结果和均值/标准差。固定 RX 基准的 seed 区间只代表训练随机性，不代表未知 RX 总体泛化。
4. 若 model/RX/独立物理组数量足够，使用尊重交叉 model 与 RX 结构的 cluster bootstrap 或预定混合效应分析；共享 query 的重复视图、support draws 和任务格保持关联，不能逐 cell 独立重采样。cluster 少时以分层描述及区间限度为主，不堆叠很小的 p 值。
5. 用配对效应量与 95% CI 为主要报告；主要功能比较预先固定。如做多项显著性检验，对同一比较族进行 Holm 等校正，扩展/事后分析明确标记 exploratory。
6. “下降≤1 pp”与“差≤3 pp”若要作统计确认，应报告对应预定单侧上界/等效区间；均值达标或 p>0.05 不能证明等效。样本不足时只写描述性达标。
7. 失败和缺项计入覆盖，不删除坏 seed；低性能不作为技术失败。数值失败按预定规则留存，不择优重跑。

确认规模不机械固定为 5 seeds：先用合法研发证据估计配对波动与最小有意义效应，做预算/精度规划。小 seed/RX 数支持的是有限范围证据。

## 10 星载成本与训练日志

资源测量采用同硬件、同线程/BLAS、同精度、同 batch、相同模型状态和一致 warm-up。GPU 计时同步；同时列方法工作量和并发墙钟。CPU pilot 的时间不能写成星载实测时间。

| 项目 | 测量口径 |
|---|---|
| 参数 | encoder total、部署新增 trainable、解析头系数/非梯度拟合状态分别列；解析头不应因 optimizer_steps=0 被写成无适配状态 |
| 时间 | IQ 预处理/encoder、adapter、头拟合、单 query/整批推理、每次注册更新分别列；median/P95 与重复次数 |
| 计算 | optimizer updates、头拟合/Cholesky/三角求解次数、encoder 前向/反向次数；可用时 FLOPs/MACs，缺项 N/A |
| 内存 | 峰值 RSS/VRAM、常驻模型、support memory、Gram、adapter/optimizer、类别状态分别列 |
| 传输 | 冻结 bundle 大小、每次新增模型/摘要/support/类状态的实际序列化字节；日志不算部署 payload |
| 可扩展性 | 对 N=(old＋new)K 列经验成本曲线；精确径向核需存储随 support 数增长的状态，不能凭小参数量宣称恒定开销 |

缓存后 adapter 可节省重复 encoder 反传，但缓存建立、存储和部署推理仍计入总成本。层内 PEFT 需要经过主干反传，少量 trainable 参数不保证总计算更低；其意义以实际测量为准。

训练启动显示实际有效参数，逐步保存真实 loss 分量/权重、LR、梯度、路由/机制状态、source V、耗时与数值异常。保留详细文本和完整结构化记录，同时写 compact epoch JSONL/CSV，去掉大数组；缺失写 null/N/A。日志修订不停止、重启或热改健康运行。

## 11 预算与执行顺序

### 11.1 优先级

| 优先级 | 工作包 | 输出 |
|---|---|---|
| 第一批 | 固定当前 P00，完成 P01/P07/P10/P11/P12/P13/P14/P15 的 source 机制诊断；Phase2 D00–D06 口径匹配 | 总体训练贡献及头/表示/注册的统一参照 |
| 第二批 | P02–P06/P08/P09；少量 F3×F4、CE×DAOT、几何×proxy 交互 | 分支与 DG/可信监督的条件贡献、关键交互 |
| 第三批 | Phase2 DA×REG、LocalRidge 输入/核、R0–R4、完整 K×N_new | 旧类适应和新增竞争的独立解释 |
| 第四批 | source/support 参数粗扫与有限精扫、可实现 PEFT | 选择规则及性能/开销边界 |
| 第五批 | 冻结整批目标消融评价；如有合规新物理数据再独立确认；Phase3 单独建包 | 论文表格与声明范围 |

此顺序是设计建议，不授权现在启动任何批次。少量合法研发 pilot 可以先证伪，但不能由 target pilot 结果筛掉下一轮候选。已有冻结用户矩阵保持原要求，不以本文改写正在运行的任务。

### 11.2 计算量示例

Phase1 一级 16 arms×4 model seeds=64 次训练；若全部从零各 200 轮，共 12800 model-epochs。仅严格同代码/数据/增强/预算的已完成 P00 可引用，架构删除臂不能从 P00 续训。交互格与参数扫描去重后另外计费，不把所有因素全组合。

Phase2 一种冻结配置若覆盖 4 model×4 RX×3 scene×4 K×5 N_new×5 support draws，得到 4800 个配置单元。这是预算示例，不是新的固定矩阵，也不等于 4800 个独立样本。old-only B 对同一 RX/scene/K/draw 可复用，不因 N_new 重复拟合；A 可按实际固定 query 复用。若 C 的 support/状态不同，每格独立保存。现有不同 cohort/RX 数据不能拼成虚构独立新矩阵。

真实预算采用 `Σ每arm实际训练成本＋Σ每row拟合/特征成本＋Σ推理/评分成本`；不能仅按 epoch、参数量或计划 cell 数估算 GPU 时间。若资源有限先缩研发网格，保留主矩阵的完整性，不按 target 性能选择性删格。

### 11.3 落地时的必要验证

- 消融开关确实改变有效图/梯度/状态；关闭项的权重、更新和前向计数可读回。
- 改分支/维度后严格加载相应架构，不能靠 missing key 或部分加载旧权重伪装从零。
- query batch、顺序或无关 query 增减不改变当前样本决策；类列置换后语义输出相应置换，tie-break 规则一致。
- train/held 物理 ID 与 view 绑定、K=1 N/A、target 编码、求和/均值目标、退化带宽与浮点求解有聚焦检查。
- B/C 旧类 support/query 配对、N_new=0 C=B 复用、old-only restricted 分析只在固定 artifact 后运行。
- new run 不覆盖，启动 owner 唯一；技术失败只处理所属 row。按仓库八项最小流程完成本地版本、直接正确性检查、适用的一次 P0/P1 审查、预登记与启动读回；不增加审批/receipt 链。

只在真正启动时建立 `experiment.json/report.md/events.jsonl`，填实际逐行配置、六类 seed、checkpoint 完整来源、命令/commit/硬件、输出和日志。本文及配套 CSV 是研究设计，不是 launch-ready 实验登记。

## 12 论文与报告交付

建议至少形成以下主表/图：

1. **当前功能贡献表**：P00–P15，source 与各正式 target scene 分开，列 DG、old/new/H、资源及缺项。
2. **分支/可信/轨道细分表**：删除臂、桥接臂与扩展臂分开；标明实际开启状态。
3. **A/B/C 配对表**：完整 K×N_new，A_native/A_proto/B_head/B_method/C_old/C_new 及 G/F/D/H。
4. **DA×REG 四态图**：仅共定义指标的绝对值、效应和交互；REG0 new/H 留空并注明 N/A。
5. **注册损失分解图**：old 内部变化、new 竞争及总变化；旁边列 old→new/new→old。
6. **参数敏感性图**：全部冻结扫描点、区间、成本；二维图仅针对已声明交互。
7. **规模与持续注册图**：K×N_new 热图、cohort 遗忘轨迹、注册顺序波动。
8. **性能与成本图**：C_old/C_new/floor/H 对实际时间/内存/传输，不能只展示 trainable 参数。
9. **Phase3 专表**：unknown/known 责任、单节点/协同与相关性压力、证据同步性和代理声明。

每个结论限定到其设计：删除实验是条件贡献；加项实验是扩展增量；参数扫描是敏感性或 source/support 选参；反事实列限制是错误分解；透明重复基准不是独立泛化。所有比较保留绝对准确率、失败和缺项。

## 13 证据入口

以下本机文件是设计时实际读取的依据；Git 主检出可能未包含同一最新版本，执行前按所登记提交与数据证据确认，不从路径名推断。

- [当前科学协议](E:/type10-7/项目.md)
- [最小流程与 DA/REG 口径](E:/type10-7/tools/optimizer_workflow_contract.md)
- [最近 Phase1 五 seed 的运行报告](E:/type10-7/automation_reports/CV-SincNet/20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01/report.md)
- [实际 matched 配置](E:/type10-7/code/snapshots/daot_practical_three_20260918_wt/experiments/adv3b02_xuc/configs/matched_20260927/cvs-daot-rc4-s2026092701.json)
- [DAOT A0–A8 实际覆盖](E:/type10-7/code/snapshots/daot_practical_three_20260918_wt/experiments/adv3b02_xuc/code/cvsrffi/deployment_orbit.py)
- [固定 Phase1 分支元数据](E:/type10-7/docs/D92_FIXED_PHASE1_BRANCH_METADATA_20260929.json)
- [LocalRidge 冻结数学实现](E:/type10-7/code/cvsrffi/d92_branch_local_ridge.py)
- [已有三阶段指标口径](E:/type10-7/docs/D92_BRANCH_LOCAL_RIDGE_ABC_RESULT_20260930.md)
- [仅 support 的注册机制诊断](E:/type10-7/docs/D92_REGISTRATION_DIAGNOSTIC_RESULT_20260930.md)
- [PEFT 草案与暂缓边界](E:/type10-7/docs/D92_SUPPORT_PEFT_DESIGN_20260930.md)
- [Phase2 数据角色登记](E:/type10-7/automation_reports/CV-SincNet/20260927-phase2-practical-data-manytx-s2026092705-r01/experiment.json)
- [配套一级消融清单](CVS_ABLATION_CORE_MATRIX_20260930.csv)

外部方法学来源仅支持研究设计原则，不证明 CVS 性能：NIST 支持因子组合/交互设计；Cawley 与 Talbot 支持选择偏差的风险及选择/评价分离。本文所有扫描值、优先级和预算都是本次提案。
