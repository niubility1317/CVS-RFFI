# 完整 practical：satellite CE 训练温度与减半权重对照

当前状态：ANALYZED；两条新增 E30 及全部七视图评分、完整日志/图表交付完成。源域四项 FAIL，关闭固定候选。以下保留原预登记，最终结果见 [final_analysis.md](final_analysis.md)。

最值得保留的是完整 nominal practical IQ 路径、明确的 clean 监督份额和默认分类头联合训练。当前矛盾是完整受损观测带来显著适应能力，但提高 As 剂量或额外约束未稳定兼顾 clean。课程、单侧间隔与 dropout 视图预算的源域失败均保持关闭，不微调救援。另一对话正在完成 Asweight E50 三种子，本矩阵不重复它的工作。

本次只检验一个假设：相对将 satellite CE 统一减半，训练温度 2 是否通过改变类别概率形状改善正确／不确定观测的梯度分配。已有源域 As30 logits 显示，mid/low urban 错样本的最大概率中位数约 0.376/0.359，超过 0.95 的错样本仅约 0.306%/0.204%。因此“主要修复高置信错判”的解释不成立。温度 2 是读取该描述统计前固定的候选；不扫描温度。两者错误样本的 logit 梯度方向很接近，主要差别可能来自正确样本；这仍需训练证伪。它们不是已测得的骨干梯度。

矩阵唯一维护于 experiment.json：CE_HALF=.5CE(z,y)；CE_T2=CE(z/2,y)，不乘 T²；原 label smoothing=.01 不变。原 As30、NoAs30 复用既有匹配冻结产物。两个新行从同一合规 P43 B01 scratchAs20 student＋ownEMA 独立追加 10 轮，各累计 E30/6660 updates；新增共 E20/4440 updates。部署头仍为 cosine30。所有 64 个有标签 As 都参与 CE，不加质量门控。原 RDrop 对称 KL 不变。

每行每步 6×128 有标签前向共 768，As64 两次；native U、教师、普通 IQ 与信道随机流不变，无额外模型前向或参数。mixed128 卫星占半，outer mixed weight=.5，所以增加修正 .25×(两次新 CE64 均值−两次原 CE64 均值)。损失均在各自两次 logits 上计算，不能先平均 logits。温度与半 CE 仅显式链式因子相同，不能宣称真实参数梯度范数匹配。

科学判据在正式分数前固定：T2 相对 NoAs30 的完整 V clean 不低于 −0.1 个百分点；源域两 draw 的 mid/low 均值相对 As30 至少 +0.5 个百分点、low 不降低；相对 HALF 的 mid/low 至少 +0.5 个百分点。HALF 仅用于归因，不按正式成绩晋级。任一失败关闭该固定候选；即使失败仍自动完成 clean＋六种 practical。没有自动扩大种子、E50 或参数扫描。

代码和命令：本地 code/worktrees/phase1_urban_sigtime_20261014；远端 PROJECT/releases/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01；普通 N607 现有 Python3.10.19/Torch2.1.0+cu121/NumPy2.2.5；命令 CVS-RFFI/python -u experiments/cvs_satellite_ce64/runner.py。code commit 发布前补入 experiment.json。

数据：L6300/U56700 TX 真值隐藏/V27000；RX1/3/4/6/8、day1/2/3、sixTX，split392005，无留出 RX/Ldev。采样率 25MHz，IQ256，既有 RMS center=false。完整 residual/post_sync，EQ 关闭 G=1、质量适配／相位跟踪／IQ 补偿保留、reference_snr_db30、reflect边界；实际模块 __file__ 与 resolved 值逐行输出。正式固定 IQ、六种信道和 SNR 不变。

输出：PROJECT/runs/20261018-phase1-FcrossRX-satelliteCE-temperature-e30-s2026092701-m2-r01 为独占 root，owner codex/root/satellite-ce64。GPU 总进程上限 2，计入其他 owner，剩余显存至少 12000MB；资源不足等待，不停止或重启其他任务。仅协议／来源／数值／运行或输出冲突等技术失败中止自身任务，保留产物，不自动补跑。

预期成本并非实际成本：新训练约 2×3162962 NN，源读出 2×12600，正式预测 2×1176002，远端真实 checkpoint smoke 2NN 一次，本地实际 F 合成 1536NN 单列。共享 E20 父训练只计一次；历史对照不重新训练／预测／评分。实际成本、墙钟和全部训练曲线在完成后报告。

最终交付 all4 logical rows×all7 的 acc/F1/最差RX、逐RX/day/TX、paired repairs/newerrors、NoAs30 clean正确集合保持率、默认头间隔／30、合法源状态／lock／质量／新增噪声比分层、覆盖与完整曲线。source 真值及源图不等于 target 物理因果识别；formal不回流调参。新模型只有一个 seed，不宣称跨seed双提升。

最小流程仅执行一次聚焦 P0/P1、必要负例与真实父权重 no-query smoke、一个 archive SHA 比较和远端编译。额外设计审批／文档哈希为 REJECTED_EXTRA_GATE，不作为发布条件。

本地验证已完成：37 协议负例、12 目标张量检查、4 流程检查与分析 12 toy 检查通过。两 arm 实际 F 共 1536NN／4.2500s，默认头及 Sinc 梯度有限非零。唯一独立 P0/P1 审查 PASS，无阻塞项。1.25 为原槽权重口径；HALF 有效 CE 系数和变为 1.125，原 KL 系数不变，这正是标量减弱控制，不能称总实际梯度相等。

发布已 VERIFIED：代码 commit 90fe8ac0e2577ce0038a8e5a7855a4d2d9c5efc6；archive 单次 SHA 比较 PASS，22 个 Python 文件远端编译 PASS。dispatcher 1340924，CE_HALF 1341087/GPU0、CE_T2 1341161/GPU1，独立核实 PID/CWD/argv/GPU 及日志增长。真实 ownAs20 checkpoint smoke 2NN、1CP、0IQ、0更新 PASS。两行均完成新增段第 1 轮，尚无冻结性能结果。实际导入文件和运行信道值见 actual_runtime.json。
