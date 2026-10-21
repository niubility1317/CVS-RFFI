# SARD 的 DURA 恢复后增强：从零 E50＝13L＋37U

状态：ANALYZED／最终 VERIFIED；两组训练、七景双读出预测与独立评分全部完成。实际代码／配置 commit `94b7f8eff785369030eeef94899c4f59e4ff0b51`，本地和远端分支OID一致。

用户提出把 SARD 的随机 practical residual 增强样本替换为 DURA 输出，并明确从零训练完整 E50、前13轮 L 阶段＋后37轮 U 阶段。后37轮依原 SARD 流程仍含有标签训练，并非只训练 U。为隔离增强替换的影响，固定一个 seed2026092701 的原 SARD 同日程对照；两个 row 的完整矩阵、预算和路径只在 experiment.json 维护。

变体的64个 L_s 增强槽采用 x→原随机 practical residual 信道→冻结 DURA v1.0→F。clean／普通增强／raw 分支保持，As 真标签 CE 和 As独立KL0保留，普通/raw KL、CROSS_RX SupCon、旧U门控、ownEMA及联合训练默认头保持。G 不接TX标签，不参与反向传播或任何参数/状态更新；U仍只走原半监督路径。两组训练从零，均50×222＝11100更新，使用单段 cosine50，不继承已暴露目标上的任何F权重。

DURA固定R69 stage2的源FINAL20/4940、266951参数；使用已验证commit47c264286ca689f060a3f36aafb763fe209a6f06的InferenceEngine(reuse=True,graph=True,native=False)，FP32/TF32off。加载时沿既有源数据角色与TX/F-free完整来源结论，核对固定source handles与payload，不重做数据验证。G构造不消耗F训练RNG。inference输出离开G后clone为独立普通detached tensor，允许F学习而不给G梯度；G失败不得回退raw增强。

原source7/63/30、split392005、全部5个sourceRX/day1/2/3保持，L6300／U56700隐藏／V27000。原有FINAL50固定，不按源V或目标分数挑epoch。每row训练冻结后立即自动预测原168000物理样本×clean及六种practical的raw/DURA两种读出，共14单元；原固定DURA恢复IQ可复用，全部预测固定后独立CPU scorer才读truth。报告Accuracy、Macro-F1、逐RX/day/TX、修复/新增错/净纠错与匹配差异。旧SARD E30+20、CRIC+DURA作注明日程差异的历史参考。

正向迹象预先定义为：DURA读出六环境均值较匹配SARD至少+0.5pp、clean不低于-0.2pp、lowurban正收益。无论是否满足均完成所有预登记测试，不依目标结果扩seed、调参或选择性重跑。本轮是已暴露ManySig工程基准、一个F seed共享一个G，不能声称跨seed稳健性、在轨或Phase2效果；本轮不把G缓存读出时间当作在线G+F时延。

N607普通账户，两张不同GPU各一条本run worker。启动时按所有owner进程量最少、显存余量最多选择，free≥12000MB、每卡总进程≤3、最多两卡；独立CPU scorer最多1。保留其他owner健康任务。真实源checkpoint无query smoke和SARD全六前向/默认头梯度检查为首步，PASS即进入训练。系统性技术失败保存partial、不自动重试，不因低性能停止。不可覆盖run `20261021-phase1-SARD-DURAaug-e50-13L37U-s2026092701-m2-r01`；精确命令/CWD/环境/预期产物见experiment.json。

本地4项聚焦边界负测先失败后通过；唯一独立P0/P1审查无阻塞。已只读核实普通N607连接、数据/来源句柄、磁盘与GPU容量。训练完整入口在恢复器依赖前加载，避免评测依赖的同名SSDG精简shim覆盖原训练参数解析器。后续只做一次归档传输比对/远端编译及启动后独立运行核验。

启动独立核验 VERIFIED：归档一次SHA比对及12文件远端编译通过。两row分别GPU3／GPU6，PID/CWD/argv/run-root和实际GPU均一致；真实源checkpoint无query smoke及768NN梯度probe均PASS；已各完成E1／222更新，训练步骤日志持续增长，初始化证实无父F、own student EMA、数据角色EXACT_MATCH。实际native args为50/13/37、lr0.0002、wd0.0001；DURA组首轮14208个As全部恢复、G更新0、raw fallback0。目标访问仍为0，所有训练后14单元预测及独立CPU评分流程已串联。

已设置当前对话完成回报心跳 `sard-dura-e50`，每15分钟只读检查本run，状态无变化时保持安静；完成后取回完整结果与日志、独立复算和报告，报告后关闭。训练后测试本身由已启动队列自动执行，不依赖心跳派发，不重复训练/预测/评分。

最终独立核验 VERIFIED／ANALYZED：两组各50轮／11100更新、raw/DURA共28单元（每单元168000）齐全；504个CM和252个配对单元复算一致。两组各11100步和50轮、全部结构化训练记录及完整stdout已解析，未见系统性技术错误，实际U标签始终隐藏；DURA训练处理710400个As且G更新0。新增4704000个F目标读出，目标G复用原恢复IQ。DURA读出六环境均值差-2.9949pp、clean差-2.4470pp、lowurban差-2.3042pp，预登记正向迹象条件未全部满足。本轮1个模型seed，不报告跨seedSD；详细指标、历史参考、训练/U成本及逐RX/day/TX见 [正式测试报告](test_results.md)。所属worker/controller均已自然退出，无重复训练或预测。

实验全部完成，完成回报心跳 `sard-dura-e50` 已删除，独立核实本地automation定义已不存在；不再继续轮询。本轮负结果完整保留，未扩seed或替换默认F/G。
