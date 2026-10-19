# 原 F 与 raw 补充特征共同学习：固定四行比较

状态：RUNNING / VERIFIED，真实父模型6NN无query smoke通过，四行PID/CWD/argv/GPU及日志增长已核实；尚未完成正式评分。

唯一新假设：冻结原 F 时，新增特征难以进入适合严重受损 IQ 的联合表示；让原 F 与同一补充编码器共同学习，可能产生普通 FULL 适应没有的有效特征。该判断只来自既有源结果：固定原 F 的 RAW/PAIR 源 urban 为 58.5556%/58.6389%，实际训练 low urban 面板为 54.8986%/54.3919%；原 FULL 对应 69.6250%/66.0473%。原 W 已联合更新，所以这次改变的是整个原特征提取器的可训练集合，不是再次放开头。新方案不以正式分数选择参数。

旧固定 2252/k5/width12/E10 冻结 F 方案已源失败并关闭。本次不延长或继承任何失败 E30。4 行均从共同合法 B0 FINAL20 独立分叉。新增参数、kernel、rank、seed、损失、观测和 E10 全部不变；唯一变化是原 F 编码器与补充路径在同一 backward/optimizer 中共同更新。旧反证针对冻结表示，不能直接否定本次联合表示假设；若本次失败，同样关闭该固定方案，不以改 kernel/rank/lambda/seed 救回。

## 机制和匹配控制

保留原 F 三路 SIG_TIME、Sinc、PA、参考响应、融合、真实 joint160 和原 6×160 cosine 分类头。原身份编码器全部合法特征参数、新 raw 编码器、原 W 联合训练；原无用辅助网络和外层未用分类头固定。不新增 logits、替换头或单独头阶段。

补充编码器严格复用上一轮公式：2→8 k5/s2/p2→SiLU→8→12 k1→DW12 k5/p2→SiLU；共享 h12×128。零初始化 B1 12→72 投影加入原 t1 的 preGN，B2 12→96 在 AvgPool2(h) 后加入 t2 preGN。其他新增层正常初始化，无第二个零门控。原 t1/t2 对象和 SIG_TIME hooks 保留。RAW 补充读取完整同一 IQ；PAIR 仅补充读取 AvgPool2(x).repeat2，原 F 始终完整 IQ。两者同参数、私有初始 seed20261019901、初始函数等于 B0。

部署新增2252、总902974参数；可训练234445=普通FULL232193+2252。原Sinc恢复与FULL一致的CENTER_UNITS物理AdamW和全局物理clip5；补充参数并入普通组，不另外改优化强度。reset AdamW2e-4/min1e-6/cos10/wd1e-4。每步实际记录Sinc学习率、物理clip及更新计数。补充卷积MAC214528/包，不代表全模型耗时。

## 矩阵、预算、数据和评估

4新行 RAW/PAIR×NoAs/As，model seed2026092701，共同B0 E20+新增E10/2220更新，累计E30。原FULL/TEMP联合W四行E30为匹配控制，直接复用。共同父成本不归零，无新模型种子。原B0 E20仅作不同成本的参照。

复用既有R02 bank284160、全部L索引、actual-first L1800、新L mid/low各两draw、源V两draw及fullV27000，新增物理模拟0。源L6300/U56700/V27000物理角色保持。按用户明确回复，只用L的三项目标：CE0(raw)+CE0(second)+mean_all128[parentCorrect KL(parent_raw||student_raw)]，tau1/lambda1；NoAs第二槽为同一raw。无U拟合/伪标签/EMA，U真值隐藏。全模型eval但学生autograd；没有dropout和增添数据增强。

每行新增568320学生NN、284160冻结父参考NN；初始和终末源读出各50400NN。固定FINAL30，逐行clean及六种practical各168000，单包原默认头预测。源决定固定后、该row全部prediction固定后，独立truth-last评分。既有控制不重训、不重新预测或评分。全部负结果保留。

## 事前源规则和停止条件

RAW与PAIR分别判断：源V mid/low urban两draw均值相对普通FULL_As至少+0.5pp；fullVclean相对FULL_As及各自NoAs分别不低于−0.2pp。RAW−PAIR是独立的细时间信息对照，不再把它等同于联合学习是否有效的主问题。报告NoAs−FULL_NoAs及As专属交互，区分一般容量增量和星地增量；不由此预先宣称As特有收益。

两者通过时，按源urban、再clean、再静态RAW顺序固定首选。源规则在本次新分数前固定，不能以本次结果回调。如果均失败，关闭固定原F+raw2252联合/k5/width12/E10，不扫尺寸/权重/seed。没有自动E50；有源支持也必须先固定最多+20的继承和匹配控制，累计不超过50。

唯一owner codex/root/F-raw-coadapt。每GPU所有owner最多2进程、free≥12000MB。仅自身技术/协议/数值/来源/接口/输出/无法产生合法prediction失败时保留失败，保护健康行，无自动重试。科学失败仍完成声明七景。使用普通N607账号，不触碰健康旧任务。

输出PROJECT/runs/20261019-phase1-F-rawCoAdapt-e20plus10-s2026092701-m4-r01；CWD=PROJECT/releases/cvs_F_rawCoAdapt_E30_20261019_r01，既有Torch2.1/FP32/TF32off/CPU2。命令、commit、逐row路径见experiment.json。首步真实父6NN无query smoke，PASS立即继续；单归档、一次预检/传输SHA/compile、一次独立P0/P1审查及启动后态。预期source/initial/final/steps、源决定、七景prediction/score、analysis/complete或failure。

## 实现追踪

| ID | 要求 | 位置 | 状态 | 验证 |
|---|---|---|---|---|
| R01 | 新源依据及旧反证范围 | 本报告 | verified | 固定F源失败记录 |
| R02 | 原F/2252特征/原W同一步联合 | model/optimizer | verified | 合成真实模型与原生更新 |
| R03 | 严格RAW/PAIR输入、零函数、hooks/joint160 | layers/checks | verified | 实际输入/深拷贝/特征读出 |
| R04 | 原Sinc物理更新与全局clip | optimizer/train | verified | 逐步计数和物理LR |
| R05 | 合法B0父及数据角色 | model/data | verified | 原strict loader/既有bank |
| R06 | 用户L三项目标/NoAs/同曝光 | train/config | implemented | 全量日志与计数 |
| R07 | 4新行E30/4旧匹配控制 | design/report | implemented | 成本和无重训重测 |
| R08 | 每候选相对FULL源增量/clean保持 | report/flow_checks | implemented | 只读source判定 |
| R09 | 固定终轮七景/独立truth-last | dispatch/evaluate | implemented | 逐row预测与评分 |
| R10 | F1/最差RX/修正新增错/seed边界 | report/analysis | implemented | 完整重计与配对 |
| R11 | commit/一次review/CPsmoke/发布/后态 | publish/checks | verified | 独立运行证据 |
| R12 | 不覆盖/不重启/不自动扫参或E50 | dispatch | implemented | immutable root/失败保留 |

严格实现固定结构与训练集合；PAIR是raw成对平均输入控制，并非非线性time_fuse内m/d操作的精确替代。最高风险仍是新增表示只提高源拟合或引入跨RX捷径，任何梯度/散布不代替正式识别证据。已暴露benchmark上的n1探索，不能声称新的盲测或可重复clean正收益。

本地34合成NN/7负测及0NN source flow PASS；实际原t1/t2/Sinc、原W和补充特征在同一步更新，物理CENTER_UNITS及clip计数通过，0真实CP/IQ/target。

唯一一次独立P0/P1审查PASS，15文件/2个发布内嵌脚本/4行配置及参数MAC、0NN源flow核对通过；无新增模型/CP/IQ/target/remote，见review.json。代码/config commit 5bbd30bde89f6e00c8fc108d4ce76e8a2293ceb5 已本地提交；push/OID状态 VERIFIED，见git_delivery.json。命令 CVS-RFFI/python -u experiments/cvs_raw_coadapt30/runner.py，CWD=PROJECT/releases/cvs_F_rawCoAdapt_E30_20261019_r01，普通账号既有Torch2.1/FP32/TF32off/CPU2；首步真实B0 checkpoint6NN无query smoke通过即继续复用bank并启动四行。逐行路径、资源、停止条件和产物见experiment.json。REJECTED_EXTRA_GATE：不把暂不可达的远端Git OID追加为实验发布门槛；交付仍继续推送/独立核实。

启动证据见launch_evidence.json：单归档传输与远端compile VERIFIED；四行复用既有实际As bank，新增物理模拟0。没有据运行状态声称性能目标已完成。
