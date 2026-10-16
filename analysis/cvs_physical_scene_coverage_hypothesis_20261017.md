# 下一唯一假设：按物理包均衡 practical 场景覆盖

2026-10-17。状态：HISTORICAL_REPLAY_VERIFIED／E20_PREREGISTERED_NOT_LAUNCHED。尚无新的真实 IQ 读取或训练；三组 E20 已实现并预登记。采样抖动的固定四周期估计入口已关闭，不以增加周期、换统计量或放宽拒绝率救跑。

## 为什么检验这个问题

当前 PairedChannel 为每个合法 L 包按 namespace＋物理元数据独立均匀抽取一个场景。model.py 的 namespace 包含 epoch 和 optimizer_step，所以每次出现刷新。它保证总体期望均衡，却不保证同一个物理包在有限 E20 中均衡见到六种场景。下一唯一假设是：在保持每包一次完整 practical、原有传播／噪声／接收状态依赖及全部监督的条件下，均衡同一包跨多次出现的场景覆盖，可能比独立抽样更有效。

这属于 A_s 的训练组织，不是新增 IQ 泛化增强 A_g，也不假定覆盖均匀等于识别提升。原 As 的独立／重复多视图及其他失败仍保持关闭；此方向每次只用一个完整星地算子，不增加视图，不增加损失或部署分支。

已读另一对话的 cvs_f_timegrid20/design.py：其方法是原 IQ 的有界时间网格重采样且 As=False，与这里的场景分配不是同一机制。另一对话当前普通增强分组实验也不在本任务中修改、重启或通信。

## 最小下一步

只读既有 sourceRX1／seed01 的原 As 训练记录，复算每个原 L 物理包的六场景曝光分布。已确认本地原 run 的 training_sequence.jsonl.gz 有 5,994 条角色记录，satmix_steps.jsonl.gz 有 4,440 条 step 记录并保存每步 scene_counts。下一步只消费 L 的已有 ID 元数据和对应步次；U 的记录不进入计算、不读取真值，不载入任何 IQ 或模型。

按实际 stable_seed／物理 tuple 顺序／namespace 复现的每步场景总数必须与原 4,440 条 scene_counts 对应，才把 replay 当作该 run 的历史事实。只统计，不重新生成 practical IQ。报告全部原 L 的出现次数、每场景次数、最少覆盖、同包场景最大与最小次数差，以及每 RX/day 的完整分布；不会把 L6300、L_train5580 或其他 run 的控制混进原 fitL5040。

若实际已对每个物理包实行均衡覆盖，或已有同定义的匹配识别实验，则不重复此方向。计数差异即使存在，也不是晋级证据；只有固定下一独立科学矩阵后才能检验识别。

## 若有依据，未来最小识别反证

候选分配规则可用每个合法物理 L 的出现计数：每 6 次使用一个私有随机排列的全部 6 场景。场景选择不依赖 TX／预测置信度，其他传播及接收参数仍由同一完整 practical 处理产生；不削弱信道、不提高测试 SNR、不修改正式 received IQ，也不要求 urban 高置信。这里尚未实现或登记该规则。

未来反证须以同一个 scratch E20／原 U／默认头正常联合训练／原始监督份额，对比额外 raw、独立随机单 As、逐包覆盖均衡单 As；视图数、损失尺度和更新数匹配。具体命令、唯一 commit、参数及数值判据在任何启动前固定。全部 row 自动做既定源 clean＋六 practical，只能以合法源证据决定后续，不能从历史 target 结果选范围。

若均衡覆盖在相同成本和 clean／suburban 保护下，不能比随机 As 提供真实 low/mid urban 额外识别收益，就放弃该假设。不以更均匀的计数、健康检查或实际前向非零替代性能证据，也不把单源折结果当作稳定 As 或联合目标完成。

底座保持 F＋跨 RX 同 TX／三 SIG_TIME／统一 Sinc／joint160／NoExplicitCFO／默认 cosine 头／E20／原隐藏 U。A_g 仍无有效候选，完整四因子、交互项、跨 seed 和正式 clean 净收益／low urban>60%／suburban 接近 clean 均未完成。联合目标 ACTIVE。


## 历史复算结果与本轮识别反证

原 B01 的全部4440步场景计数及最终scene汇总精确复现；568320次L观察、5040物理包。全部包见过全部6场景，缺失包×scene为0，故不能以未见过困难场景解释原结果。同包出现104至120次，中位113；六场景次数最小值中位14、最小6；最多与最少次数差中位11、最大27。TV距均匀分布中位0.08333。只读取已有L元数据，0新IQ／CP／NN／更新／信道IQ生成。完整RX/day与分布见 cvs_physical_scene_coverage_replay_20261017.json。

新反证实现 experiments/cvs_scene_cycle20：B00额外raw、C_IID原独立单As、B01逐包每6次覆盖6scene，均scratchE20／8L前向／总pairweight1.25／原U，已预登记，尚未发布。公共raw.75＋ordinary.25＋原浅中层SupCon完整保留；只加同预算.25R(view)，不增加新损失／分支。35负例、4114合成NN／3合成更新／520合成fullpractical生成通过，0真实IQ／checkpoint／target。B00损失和默认头梯度与既有D_raw完全相同，IID输入与原practical逐点一致，固定source-eval也逐点一致；默认头正常更新、joint160导出相同。

原始As总体边际概率不变；本候选只改变同一物理包多次出现的相关性，不是提高urban概率或重开旧系数探索。候选相对C_IID须lowurban与urban均值各≥+1pp，相对B00须lowurban≥+5pp／urban均值≥+3pp，同时两对照下clean与clean F1／low F1／midurban均不降、每practical及seenV≥−.5pp。固定最终20，不删包、不换概率／周期／seed／RX／系数／预算救跑。suburban距自身clean的2pp仍只是未达探索目标，联合成功未证明。
