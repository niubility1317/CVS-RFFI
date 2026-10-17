# 全源 E30：固定 LA16×真实 practical 四格

代码：`bfbef1d853ffe2eca5d3094b60b53e2ff8a5f7e8`；起点：`1b1d6cb4d02ea9086e8674bbb29123a51603524f`。唯一维护矩阵见 [experiment.json](experiment.json)。本轮新增 B10/B11，B00/B01 复用 20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01 已完成的同预算产物，不继承其权重，不重复测试。

现有 ordinary practical 把 urban 均值从 38.2673% 提高到 65.1583%，low urban 从 31.9857% 提高到 53.1375%，clean 从 85.1565% 降到 82.7970%（仅 seed2026092701）。训练边界混合反而使 clean 再降 0.9714 个百分点、urban 再降 1.1123 个百分点，已经关闭。两者都没有解决 clean 与星地共同提高。

这次唯一问题是：固定 LA16 的源 clean 正信号能否与完整星地训练结合。历史 LA16 在另一个 RX1/E20 源协议下 clean +2.555556 个百分点、seen V 健康检查通过，但不含 As 的 practical 保持失败；历史 FAIL 保留。当前全源 E30 四格是此前没有运行的交互实验，不能把旧源收益当成新模型收益，也不改窗口、幅度、概率、阈值或选择 seed。

B00=ordinary/ordinary；B10=ordinary/LA16(ordinary)；B01=ordinary/practical(ordinary)；B11=ordinary/practical(LA16(ordinary))。LA16 完整窗口16，起点0至240，正实数共同作用 I/Q，深度≤0.5、相对 L2≤0.1，私有 seed20261017111+step；不新增噪声、相位、剪裁或 RMS。保留第一半原视图监督，CE/RDrop KL 各半归一化为0.5，RDrop 两次使用同一个已生成 IQ；不新增一致性损失。As 只调用一次原 native practical，依赖关系、reflect、原生 AGC、residual/post_sync、EQ关闭G=1、reference30dB、25MHz/2.45GHz及相位/IQ补偿保持。新增映射只作用 L，不作用 U/V/正式测试。

F 固定 SIG_TIME_X3＋统一 Sinc CENTER_UNITS＋跨 RX 同 TX SupCon，900722参数/206状态，joint160、原 ResidualPhysicalHead 与固定 cosine scale30 联合训练。保留原 RDrop/ownEMA/旧门控 U；无新头、部署分支或显式 CFO。原源 RX1/3/4/6/8、day1/2/3、6TX，L6300/U56700隐藏/V27000、7/63/30、split392005，全源无留出 RX、无额外 L_dev。原 equalized1 IQ256 先 RMS(center=false)，普通增强后新增映射，As 后不加 RMS。实际导入路径与完整参数自动写入 module_files/actual_native_args。

两个新模型各从零 E30=20L10U，222步/epoch、6660更新、3409920有标签前向包。预计每组原训练＋源评价5407922前向包、额外冻结源读出12602包、正式预测1176002包，均报告实测计数和墙钟。新增合计60模型epoch/13320更新；参照原60模型epoch成本单列；不使用E50/E200或历史 LA16 权重。预算耗尽不称为收敛。

每组 FINAL30 冻结后，先用同CPU2在固定原V1800和P24缓存的2draw×3场景full reflect上补充源读出，无更新、无新信道。缓存加载V27000/L6300（L未使用）、U0，不新增源域留出。B10：完整V clean不得低于B00超过0.5pp；B11：完整V clean不得低于B01超过0.5pp，且2draw×2urban每格至少+1pp。仅提供后续重复实验的源证据，不证明Ag或授权自动扩种子。源阈值失败仍按固定矩阵测试；源读出技术异常保留错误并继续正式预测，最终标源缺项。

正式原固定168000样本的clean＋六种practical逐组自动完成全部prediction后，独立exec scorer才接truth。目标分数不回流其他组或源规则。报告四个差值及交互 B11−B10−B01＋B00、每场景/seed/RX/day/TX、macro F1、最差RX、固定B00 clean正确集合保持率、源状态/质量/间隔/门控、覆盖和训练曲线。首轮探索目标：urban相对B01+3pp、low+5pp且绝对>60%、联合clean超过B00、每个suburban距自身clean≤2pp；未达成如实否定，不做目标驱动重试。仅一个模型seed，不声称跨seed复现。

启动命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_LA16xpractical_E30_20261017_r01/experiments/cvs_fullsource_factorial27/runner.py`。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_LA16xpractical_E30_20261017_r01`；独占根：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-LA16xpractical-e30-s2026092701-m2-r01`；owner `/root`。普通N607账户，共享锁下每GPU最多3个所有owner实验进程且剩余≥12000MB，GPU由实时容量选择。首步同契约真实E20检查点仅2个synthetic no-query smoke后丢弃，PASS立即继续。预登记技术停止限本run协议/数据/路径/输出/非有限/跳步/预算或预测truth接线故障；保留产物、不自动重试、不触碰健康旧任务；低分不是停机条件。

本地4项聚焦检查PASS：24个配置负例、241个完整窗口位置/零输入/private RNG/非法step、256次synthetic native物理调用、512包synthetic完整F前向与默认头/Sinc/三SIG梯度，真实IQ/CP/优化更新均0；包编译与diff检查PASS。一次独立有界静态P0/P1审查PASS，源CPU2接线已核对。真实检查点smoke、远端传输/编译、进程与日志核验待实际启动后记录。

远端启动核验：VERIFIED。单包传输SHA比较一致、远端compile PASS；dispatcher500562，B10 PID500587/GPU0，B11 PID500666/GPU1；CWD/argv匹配本release，两个日志均从0增长至6489字节，GPU进程独立可见。真实同契约CP smoke PASS：strictload、2个synthetic包、真实IQ/query/更新0；检查点未用于初始化。训练结果、正式分数均尚未产生。

同 row 独立分析接续：分析代码提交 `fa7c7494722df03bcef2413a3d0a613c0b220e91` 已推送且远端OID一致；单包传输校验/编译PASS。分析进程 PID504365、CPU2/无GPU，只等待已核实 dispatcher500562 终局；全部四格冻结、七环境prediction与独立评分完成后才读取truth。输出预定 `analysis_r01`，当前尚未生成。分析无IQ/CP读取、无NN/更新；核对全部120个epoch、26640步日志、逐物理样本预测、四个差值与交互、Ag随机流/覆盖/幅度和源状态/added-noise分层，并生成图表。当前没有跨seed成功证据，也不据目标成绩安排扩种子。

共享普通增强的实际概率、算子顺序、RMS位置、循环接缝、PA/DAC/class-signature/CFO状态与叠加噪声边界见 [ordinary_audit.md](ordinary_audit.md)。四组基准配置相同，已完成的共同前9个epoch的ordinary摘要完全一致；终局仍待新组E30完成。

B10 训练终局已独立核实：E30/6660、FINAL30固定轮次、全源角色、scratch/无祖先/无目标训练/无另训头；原默认头与F联合更新。训练与原源评价墙钟1604.664535秒，student train/eval NN=3976920/864002，teacher train/eval NN=0/567000，合计5407922（其中L NN3409920）。Ag共6660次/852480包、6300个物理ID/241个起点，最大relativeL2=0.099999814、平均0.075320419，平均去除功率比例0.025466501；幅度与覆盖不作为识别收益。额外冻结源读出已完成，完整V clean相对B00+0.033333pp，仅满足健康条件，不证明跨RX DG；源urban两draw相对B00为mid+1.888889/+2.277778pp、low+1.888889/+0.555556pp，仅源结果。PID500587已自动exec到predict，正式七环境尚待全部预测与独立评分完成。B11保持原计划继续训练，未依据B10调整。

B10 七环境终局已核实：每视图168000个固定样本、全部prediction完成后独立truth-last评分，共126条overall/RX/day/TX记录；原默认头直接预测，无另训头/拟合query。预测NN1176002、墙钟49.322319秒；额外冻结源读出NN12602、墙钟8.107096秒。以下仅为model seed2026092701、同预算E30的Ag独立组，对照同契约B00；B10没有As训练。

| view | B10 accuracy(%) | B10 macro F1(%) | B10−B00 accuracy(pp) |
|---|---:|---:|---:|
| clean | 84.275000 | 83.961890 | -0.881548 |
| practical_high | 48.373214 | 44.755838 | 1.860119 |
| practical_mid | 46.310714 | 42.892592 | 1.632738 |
| practical_low_suburban | 44.425595 | 41.254079 | 1.711310 |
| practical_high_urban | 46.926190 | 43.564284 | 1.692857 |
| practical_mid_urban | 38.592857 | 36.205307 | 1.010119 |
| practical_low_urban | 32.773810 | 31.046523 | 0.788095 |

B10 urban均值39.430952%，clean最差RX72.316667%，low_urban最差RX26.787500%。在当前完整源域/E30/单seed下，LA16未产生独立clean正收益：85.156548%→84.275000%，下降0.881548个百分点。其源V健康条件通过不能替代跨RX识别收益。小幅practical变化是在无As训练条件下相对B00的描述，不能与有As的B01直接解释为联合机制成败。B11仍按原注册配置健康运行；完整四格、交互、源状态分层及图表待其完成，目标结果不回流训练、判据或选择性重跑。

B11 已完成并独立核实 own scratch FINAL30/6660、无祖先/目标训练/另训头、全原源角色。训练与原源评价墙钟2964.629931秒；student train/eval NN3976920/864002、teacher train/eval NN0/567000，总NN5407922，其中L3409920，与B00/B10/B01相同。Ag6660次/852480包、6300物理ID/241起点，最大relativeL2=0.099999814、均值0.075320419，与B10一致；不是收益证据。

读取任何B11正式分数前，固定源域判据结果已确定：完整V clean相对B01为−0.011111pp，健康条件PASS；两个draw中mid/low urban差值依次为[−0.888889,+0.444444]及[+0.388889,−0.611111]pp，4格均未达到预登记+1pp，联合源证据FAIL，**不扩seed**。SOURCE_QUALIFIED_READOUT只是源读出完成状态，不代表科学判据通过。该结论只使用原V1800/P24 full reflect缓存及完整V27000，不读取B11目标分数；阈值、模型及训练配置不变。PID500666已自动exec predict，全部七环境仍按原计划完成，源失败不取消正式报告。
