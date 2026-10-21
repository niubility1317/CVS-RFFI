# SARD 混合增强修复验证

状态：RUNNING／启动 VERIFIED；尚无新正式测试结果。实际代码／配置 commit `4f92ab1791999e0bfc1630201e602a8bf01fb675` 已 push，远端分支 OID 独立读回一致。

用户在完整负结果和源域机制分析后要求修复并重新发布。上一轮的全部替换实现没有发现技术错误。本轮以源域完整日志中的增强难度降低、原始受损覆盖丢失以及恢复视图未进入跨 RX 对比为依据，检验新的训练设计；有效性尚待正式测试。

两组均从零 E50＝13L＋37U、seed2026092701，后37轮仍联合 L/U。原64个 As 位置中，32个保留原 practical residual 输出，32个经过冻结 DURA v1.0。位置只按槽位和 step 奇偶交替，不读 TX/RX/day 来选择。场景和物理来源、标签和损失系数、六次 L 前向共768样本/step、As KL0、ordinary/raw RDrop、旧 U 门控、自身 EMA、联合训练默认头及 Sinc 优化均保持。每组实际 G 训练 IQ 为355200，原 As 总数仍为710400；G 无标签输入、无梯度、无更新。新增 G 时间记录包含32批图初始化，不包含模型加载，不能替代独立速度实验。

MIX 保留 ORD128 对比。MIX_XRX 使用原 SAT192 关系函数，复用同一次前向已有的普通128＋增强64特征，额外64包括32原受损＋32恢复视图。浅层／中层的同 TX、不同 RX 正对、温度和权重日程保持。该差异同时改变视图覆盖和 anchor 数128→192，不能解释成只改变恢复视图或 anchor 数匹配的消融；识别器前向预算相同不代表对比损失计算量相同。

匹配原 SARD 对照复用已完成 `20261021-phase1-SARD-DURAaug-e50-13L37U-s2026092701-m2-r01` 的 SARD-s2026092701。原全部替换负结果、旧 SARD E30＋20 和 CRIC＋DURA 仅作注明来源和日程的参考，不重复训练和预测。新行不继承任何 F 权重；DURA 的 sourceFINAL20/4940、完整源域 TX/F-free 来源、固定运行 commit47c264286ca689f060a3f36aafb763fe209a6f06 及同物理数据契约引用上一轮已验证结论。没有新的数据划分或目标访问训练权限。

两组固定 FINAL50，全部训练完成后各自自动预测原168000物理样本×clean及六种practical×raw/DURA，共14单元。复用固定 DURA 恢复 IQ，全部预测固定后独立 CPU scorer 连接 truth。完整结果报告 Accuracy、Macro-F1、逐 RX/day/TX、净纠错、实际 U 接受率和曝光及 G 成本。每组相对原匹配 SARD 的正向迹象门槛仍为 DURA 六环境均值≥+0.5个百分点、clean≥−0.2个百分点、low urban>0；无论是否达到均完成测试，不以目标成绩调参、选模、扩 seed 或选择性重跑。

唯一启动 owner；不可覆盖 run `20261021-phase1-SARD-DURAmix-e50-13L37U-s2026092701-m2-r01`、release `cvs_SARD_DURAmix_E50_13L37U_20261021_r01`。N607 两张不同卡按所有 owner 进程数最少、显存余量最多选取，每卡一条本 run worker；保持原容量限制和其他健康任务，独立 CPU scorer 最多1。精确逐行矩阵、命令、CWD、数据/模型来源、日志路径和停止规则以 experiment.json 为准。技术失败保留 partial、不盲重试，不因低性能停止。

本地8项相关测试通过。launcher 首步执行真实源 checkpoint 无 query smoke、六前向默认头梯度与 DURA 冻结检查，并检查恢复 map 的实际对比辅助梯度仅在 MIX_XRX 中出现，PASS 后立即训练。当前普通 N607 预检查已通过。唯一独立 P0/P1 审查已完成，未发现阻塞问题。归档传输比对／远端编译及实际 PID/CWD/argv/GPU/日志增长证据将在发布登记中记录。

本轮只有一个 F seed 和一个固定 G，属于已暴露 ManySig 工程基准的新方法验证，不能声称三种子稳定性、盲测、真实在轨或 Phase2 收益。保留当前固定 PAMF 识别器和 DURA 默认版本。

启动独立核验 VERIFIED：一个归档SHA比对和14个Python文件远端编译通过；两row实际GPU0／GPU6，PID/CWD/argv/visibleGPU与nvidia-smi映射一致。两组真实源checkpoint无query smoke及768NN联合默认头梯度probe均PASS；两组G始终冻结，恢复map的对比辅助梯度在MIX为false、MIX_XRX为true。已各完成E1／222更新，完整step日志与stdout较前次增长。首轮As总14208，其中7104保留原受损、7104经过G；G更新／梯度／fallback／U的G调用均为0。实际初始化为无父F、own student EMA、数据角色EXACT_MATCH；native参数50/13/37、lr0.0002、wd0.0001。源域训练未访问target。

已设置当前对话完成回报心跳 `sard-dura-mix-e50`，每15分钟短连接只读检查；健康状态无变化时保持安静。训练后的14单元预测和独立CPU评分由原队列自行执行，不依赖心跳启动；心跳负责取回完整日志、独立复算、报告和Git收尾，完成后删除。旧 `sard-dura-e50` 已关闭，本轮不会重新运行旧实验。
