# 同一普通增强 IQ 恢复 RMS：完整源反证

**ANALYZED／VERIFIED，固定候选 FAIL，关闭。** 两个冻结默认头的四次固定抽样均变差。没有 A_g、跨 RX clean、A_s 或联合目标的新增有效证据，不进入此算子的 E20。

原始条件见 [preregistration.md](preregistration.md)，矩阵与来源见 [experiment.json](experiment.json)。本次仅 source-fit 操作检验：原契约中的 fitL5040，RX3/4/6/8×day1/2/3×6TX，72 格各70包。两个既有 E20 final20 F＋跨 RX 同 TX 模型、三 SIG_TIME／统一 Sinc／joint160／NoExplicitCFO／原 cosine30 默认头保持冻结，无另训头。以下 raw 99.84% 是训练角色 fitL，不是跨 RX clean 测试准确率；四抽样重复同一批物理包，不能作为独立样本扩展。

## 实际识别结果

| 模型 seed | raw accuracy % | 原 ordinary % | 同 IQ RMS 恢复 % | RMS−ordinary pp | RMS−ordinary macro-F1 pp | 修复／新错 |
|---|---:|---:|---:|---:|---:|---|
| 2026092701 | 99.841270 | 92.450397 | 92.018849 | -0.431548 | -0.415679 | 82/169 |
| 2026092702 | 99.841270 | 92.390873 | 92.003968 | -0.386905 | -0.380170 | 97/175 |

预登记要求相对原 ordinary accuracy≥+0.2 pp、macro-F1不降，两模型均未满足。raw正确集合保留率分别为 92.150238%／92.135334%，低于99%；相对raw约−7.8 pp。不能将这解释为训练中的泛化下降幅度；它只反证冻结默认头下这个固定尺度恢复是否保真和修复。

| seed | draw1 pp | draw2 pp | draw3 pp | draw4 pp | 正增益抽样数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | -0.317460 | -0.396825 | -0.555556 | -0.456349 | 0/4 |
| 2026092702 | -0.317460 | -0.257937 | -0.654762 | -0.317460 | 0/4 |

两个模型的四次抽样全部负增益，且每次均低于−0.2 pp保护线；原要求至少3/4正增益。所有RX/TX/day和72单元格保留，未删难包。

| 固定条件 | seed01 | seed02 |
|---|---|---|
| repair_ordinary_accuracy | FAIL | FAIL |
| repair_ordinary_macroF1 | FAIL | FAIL |
| positive_draws | FAIL | FAIL |
| every_draw_protected | FAIL | FAIL |
| raw_accuracy | FAIL | FAIL |
| raw_macroF1 | FAIL | FAIL |
| raw_correct_retention | FAIL | FAIL |
| nontrivial_scale_fraction | PASS | PASS |
| packetpower_restored | PASS | PASS |
| positive_scalar_shape_preserved | PASS | PASS |
| every_RX_vs_raw | FAIL | FAIL |
| every_RX_vs_ordinary | FAIL | FAIL |
| every_TX_vs_raw | FAIL | FAIL |
| every_TX_vs_ordinary | FAIL | FAIL |
| every_day_vs_raw | FAIL | FAIL |
| every_day_vs_ordinary | FAIL | FAIL |

| RX | 01 RMS−ordinary pp | 02 RMS−ordinary pp |
|---|---:|---:|
| 3 | -0.357143 | -0.297619 |
| 4 | -0.714286 | -0.436508 |
| 6 | -0.396825 | -0.178571 |
| 8 | -0.257937 | -0.634921 |

## 算子与真实执行

每个源 batch 仅生成一次实际 ordinary：保留旧原生 PA/DAC/class-signature/phase/CFO/shift/multipath/AWGN 等开关和顺序，实际 PA on、DAC off、class-signature off。使用原 checkpoint native args、原不可变后端和 configure_augmentor_for_epoch 有效 E13；原日程 min(epoch,13)，不使用更强的 E14至20 外推。canonical 合成检查确认 return_pa_strength=True 与默认调用的 IQ 和随机状态一致。

n=o×sqrt(sum(x²)/sum(o²))，FP64功率比、原FP32输出；一包一个正实数，无新随机数/传播/噪声，不改原输入或原ordinary，不依据身份、RX、day、置信度分支，不做gain clip。原包RMS恢复不能撤销已产生的非线性、卷积、相位、时移或噪声损伤，也不等于提高SNR、更换RX。

两模型的20160个观察的全部记录尺度统计逐项相同；14.960317%观察的校正因子偏离1超过1%。相对包功率误差最大5.963520e−8，正实数形状误差最大3.765672e−8，均通过工程判据。原输入/RNG保持、状态与buffer保持、无grad及实际eval计数由运行断言记录。这说明算子正确接入，不能覆盖默认头身份收益失败。没有保存或传输IQ，故不冒称另一次IQ字节比较或独立重执行变换。

描述分层：每模型PA触发710/20160观察。PA未触发时RMS相对ordinary仅+0.041131／+0.051414 pp；PA触发时−13.380282／−12.394366 pp。这是既定观察的描述分层，不是独立PA因果实验，不据此改成“只对非PA归一化”或重开已失败的PA bypass E20。完整分层见 [descriptive_groups.json](descriptive_groups.json)。

## 复算、成本与状态

独立分析未导入本候选的score/qualify/restore或调用模型：核对全部90720硬判决与logit argmax、标量logsumexp的CE、accuracy、六类macro-F1/混淆矩阵、RX/TX/day、全部72格×三类视图×两模型、四抽样、修复/新错、raw正确集合。320增强batch完整（每draw39×128＋48），各包正gain、RMS关系、功率误差、形状统计、PA触发与总计数全部核对，与原结果一致。源物理ID唯一数5040，两模型元数据一致；原始logits/ID和逐包统计只存私有产物。

r02实际90722 NN＝90720源默认头＋2合成smoke；40320 ordinary生成、40320恢复视图、128合成canonical ordinary、2 checkpoint加载。新epoch/优化器更新/教师/另训头/target均0。历史两个模型40 model-E／8880更新／9373356 NN单列，未续训。既有缓存反序列化L6300、未用源V/历史特征；RX1原L1260在源推断前排除，U仅角色ID、无IQ/TX标签。没有heldRX1、U、V或target前向。

r01技术失败完整保留：在写配置日志时遇到原生SatViewStage dataclass，2合成NN／1CP加载／64合成ordinary，0源预测／更新。日志副本逐字段转换和strictJSON提前检查已修复，实际对象/参数/日程不变；原类AST复现与修复检查、原两P1回归及仅此运行缺陷的定向复审PASS。r02另建不可变输出根，科学配置独立比较保持一致。两次真实尝试合计90724 NN、3CP加载、192合成canonical ordinary，失败成本没有删除。

一个N607预检、一次archive传输SHA比较及remote compile均VERIFIED。发布器和首worker真实PID/CWD/argv/CUDA及日志0→109字节/源预测增长已观察；第二worker未取得live/proc采样，保留launch/active/resolved及完成证据，不为补采样重跑。最终dispatcher与两worker均自然退出，r02无failure，不改健康旧任务。两worker主体耗时23.274793／28.344119秒，顺序主体合计51.618912秒；总调度墙钟未单独测量，不把主体和当作全部成本。

本地19输入/配置负例＋8科学反例／96合成RMS操作；日志修复检查0IQ/CP/NN/更新。分析和绘图0源IQ/NN/更新。本地多个Python没有matplotlib，未装包；改用既有远端绘图环境生成PNG/PDF，首次仅图目录父层缺失、确认没有图后修正并生成，源研究未重跑、未修改原结果。PNG已实际查看。

实际执行commit：48a90f90a027b58a549ed499bde77e24e8f813d4。完整聚合见[source_analysis.json](source_analysis.json)，执行范围见[execution_summary.json](execution_summary.json)。图：[完整源反证图](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_ordinary_RMS_source_20261017_r02/final/figures/ordinary_RMS_source.png)；同目录PDF保留。原始日志/预测/尺度统计/物理ID及运行态保存在 local_artifacts/F_ordinary_RMS_source_20261017_r02/final，不提交Git或传回权重/IQ。

## 结论与下一问题

FAIL_FIXED_ORDINARY_POST_RMS_E13_FOURDRAW_CLOSED。关闭完整ordinary后置RMS／有效E13／四抽样这一固定方案；不换归一化位置、部分系数、clip、禁用某旧操作后仍作为本候选、PA条件、draw、RX或seed救跑。未启动此候选E20。否定的是该固定操作的源身份修复假设，不是所有功率归一化方法，也不能据此证明全部ordinary增强无训练价值。

下一步只优先研究一个独立问题：旧ordinary的通用AWGN/multipath是否造成不必要的判别损失，并与A_s重复施加信道损伤。最小源因果对照是固定原随机抽样、默认头、完整源样本，比较full／仅移除AWGN／仅移除multipath／同时移除两者；先沿实际调用路径区分参数匹配与功率变化导致的后续总效应。该问题属于旧IQ增强审计，不能冒称新增A_g。尚未实现、预登记或启动，也不自动进入训练；若固定控制无法在两个冻结模型恢复至少0.2 pp且保护源分层，则放弃相应固定bypass假设，不能靠换强度/seed修复。通过也必须由同预算匹配E20的跨RX clean结果证明训练价值。

联合目标继续ACTIVE。有效新增A_g、A_s跨seed稳定、完整契约四格/交互项、正式clean＋六practical、lowurban>60%及suburban距自身clean≤2pp探索目标仍未完成。基线保持F＋跨RX同TX／原默认头联合训练／E20；本方向研究普通IQ观测，区别于另一对话practical正负噪声方向。
