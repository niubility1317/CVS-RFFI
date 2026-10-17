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
